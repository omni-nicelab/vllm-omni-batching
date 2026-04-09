# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Diffusion Model Runner for vLLM-Omni.

Handles model loading, compilation, caching, and execution of diffusion model
forward passes. This follows the AR pattern where the Runner handles all
model-related operations.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

import torch
from torch.profiler import record_function
from vllm.config import LoadConfig
from vllm.logger import init_logger
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import DeviceMemoryProfiler

from vllm_omni.diffusion.cache.cache_dit_backend import cache_summary
from vllm_omni.diffusion.cache.cache_manager import CacheManager
from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import supports_step_execution
from vllm_omni.diffusion.offloader import get_offload_backend
from vllm_omni.diffusion.registry import _NO_CACHE_ACCELERATION
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput
from vllm_omni.diffusion.worker.input_batch import InputBatch, scatter_latents
from vllm_omni.diffusion.worker.model_states import init_model_state
from vllm_omni.diffusion.worker.utils import DiffusionRequestState, RunnerOutput
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import OmniKVTransferManager
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


class DiffusionModelRunner:
    """
    Model runner that handles model loading and execution for diffusion models.

    This class follows the AR pattern where the Runner handles all model-related
    operations including loading, compilation, offloading, caching, and execution.
    The Worker only handles infrastructure (device, distributed env).
    """

    def __init__(
        self,
        vllm_config,
        od_config: OmniDiffusionConfig,
        device: torch.device,
    ):
        """
        Initialize the diffusion model runner.

        Args:
            vllm_config: vLLM configuration.
            od_config: OmniDiffusion configuration.
            device: The device to run on.
        """
        self.vllm_config = vllm_config
        self.od_config = od_config
        self.device = device
        self.pipeline = None
        self.cache_backend = None
        self.cache_manager: CacheManager | None = None
        self.offload_backend = None

        # Cache for per-request stepwise state.
        self.state_cache: dict[str, DiffusionRequestState] = {}
        # Persistent step-local batch view.
        self.input_batch: InputBatch | None = None
        self.model_state = None

        # Initialize KV cache manager for connector management
        self.kv_transfer_manager = OmniKVTransferManager.from_od_config(od_config)

    @staticmethod
    def _prompt_preview_for_log(prompts: list[Any] | None, max_length: int = 120) -> str:
        if not prompts:
            return "<none>"

        first_prompt = prompts[0]
        if isinstance(first_prompt, str):
            prompt_text = first_prompt
        elif isinstance(first_prompt, dict):
            prompt_text = first_prompt.get("prompt") or str(first_prompt)
        else:
            prompt_text = str(first_prompt)

        prompt_text = " ".join(prompt_text.split())
        if len(prompt_text) > max_length:
            prompt_text = f"{prompt_text[: max_length - 3]}..."
        if len(prompts) > 1:
            prompt_text = f"{prompt_text} (+{len(prompts) - 1} more)"
        return prompt_text

    @staticmethod
    def _sampling_seed_for_log(sampling: Any) -> str:
        seed = getattr(sampling, "seed", None)
        if seed is not None:
            return str(seed)
        if getattr(sampling, "generator", None) is not None:
            return "generator"
        return "auto"

    def _compile_transformer(self, attr_name: str) -> None:
        """Compile a transformer attribute on the pipeline with torch.compile."""
        model = getattr(self.pipeline, attr_name, None)
        if model is None:
            return
        try:
            setattr(self.pipeline, attr_name, regionally_compile(model, dynamic=True))
            logger.info("Model runner: %s compiled with torch.compile.", attr_name)
        except Exception as e:
            logger.warning(
                "Model runner: torch.compile for %s failed: %s. Using eager mode.",
                attr_name,
                e,
            )

    def _log_cache_dit_request_stats(self, req: OmniDiffusionRequest) -> None:
        if (
            self.pipeline is None
            or self.cache_backend is None
            or not self.cache_backend.is_enabled()
            or self.od_config.cache_backend != "cache_dit"
        ):
            return

        request_ids = getattr(req, "request_ids", None) or []
        request_id = request_ids[0] if request_ids else "unknown"
        if request_id == "dummy_req_id":
            return

        total_steps = int(getattr(req.sampling_params, "num_inference_steps", 0) or 0)
        prompt_preview = self._prompt_preview_for_log(req.prompts)
        seed_value = self._sampling_seed_for_log(req.sampling_params)
        seen_context_keys: set[tuple[int, str]] = set()
        found_stats = False

        candidate_modules = [
            self.pipeline,
            getattr(self.pipeline, "transformer", None),
            getattr(self.pipeline, "transformer_2", None),
            getattr(self.pipeline, "bagel", None),
        ]
        language_model = getattr(self.pipeline, "language_model", None)
        candidate_modules.extend([language_model, getattr(language_model, "model", None)])

        for module in candidate_modules:
            if module is None:
                continue
            context_manager = getattr(module, "_context_manager", None)
            context_names = tuple(getattr(module, "_context_names", ()) or ())
            if context_manager is None or not context_names:
                continue

            for context_name in context_names:
                context_key = (id(context_manager), context_name)
                if context_key in seen_context_keys:
                    continue
                seen_context_keys.add(context_key)
                try:
                    context = context_manager.get_context(context_name)
                except Exception:
                    continue
                if context is None:
                    continue
                found_stats = True

                context_total_steps = total_steps or (int(context.get_current_step()) + 1)
                cached_steps = list(context.get_cached_steps() or [])
                cfg_cached_steps = list(context.get_cfg_cached_steps() or [])
                skip_count = len(cached_steps)
                cfg_skip_count = len(cfg_cached_steps)
                skip_ratio = 100.0 * skip_count / context_total_steps if context_total_steps > 0 else 0.0
                cfg_skip_ratio = 100.0 * cfg_skip_count / context_total_steps if context_total_steps > 0 else 0.0

                logger.info(
                    "[Cache-DiT] Request %s seed=%s prompt=%s for %s: skipped %d / %d steps (%.2f%%).",
                    request_id,
                    seed_value,
                    prompt_preview,
                    context_name,
                    skip_count,
                    context_total_steps,
                    skip_ratio,
                )
                logger.info(
                    "[Cache-DiT] Request %s seed=%s prompt=%s for %s: skipped_step_ids=%s",
                    request_id,
                    seed_value,
                    prompt_preview,
                    context_name,
                    cached_steps,
                )
                if cfg_cached_steps:
                    logger.info(
                        "[Cache-DiT] Request %s seed=%s prompt=%s for %s: cfg_skipped %d / %d steps (%.2f%%),"
                        " cfg_skipped_step_ids=%s",
                        request_id,
                        seed_value,
                        prompt_preview,
                        context_name,
                        cfg_skip_count,
                        context_total_steps,
                        cfg_skip_ratio,
                        cfg_cached_steps,
                    )

        if not found_stats:
            logger.info("[Cache-DiT] Request %s: no live cache contexts found.", request_id)

    def _log_cache_dit_stepwise_request_stats(self, request_state: DiffusionRequestState) -> None:
        if (
            request_state.cache_slot is None
            or request_state.req_id == "dummy_req_id"
            or self.od_config.cache_backend != "cache_dit"
        ):
            return

        payload = request_state.cache_slot.payload
        if not isinstance(payload, tuple):
            return

        total_steps = request_state.total_steps or int(getattr(request_state.sampling, "num_inference_steps", 0) or 0)
        prompt_preview = self._prompt_preview_for_log(request_state.prompts)
        seed_value = self._sampling_seed_for_log(request_state.sampling)
        seen_context_ids: set[int] = set()
        found_stats = False

        for contexts in payload:
            if not isinstance(contexts, dict):
                continue
            for context_name, context in contexts.items():
                if context is None or id(context) in seen_context_ids:
                    continue
                seen_context_ids.add(id(context))
                found_stats = True

                cached_steps = list(context.get_cached_steps() or [])
                cfg_cached_steps = list(context.get_cfg_cached_steps() or [])
                skip_count = len(cached_steps)
                cfg_skip_count = len(cfg_cached_steps)
                skip_ratio = 100.0 * skip_count / total_steps if total_steps > 0 else 0.0
                cfg_skip_ratio = 100.0 * cfg_skip_count / total_steps if total_steps > 0 else 0.0

                logger.info(
                    "[Cache-DiT][stepwise] Request %s seed=%s prompt=%s for %s: skipped %d / %d steps (%.2f%%).",
                    request_state.req_id,
                    seed_value,
                    prompt_preview,
                    context_name,
                    skip_count,
                    total_steps,
                    skip_ratio,
                )
                logger.info(
                    "[Cache-DiT][stepwise] Request %s seed=%s prompt=%s for %s: skipped_step_ids=%s",
                    request_state.req_id,
                    seed_value,
                    prompt_preview,
                    context_name,
                    cached_steps,
                )
                if cfg_cached_steps:
                    logger.info(
                        "[Cache-DiT][stepwise] Request %s seed=%s prompt=%s for %s: cfg_skipped %d / %d steps (%.2f%%),"
                        " cfg_skipped_step_ids=%s",
                        request_state.req_id,
                        seed_value,
                        prompt_preview,
                        context_name,
                        cfg_skip_count,
                        total_steps,
                        cfg_skip_ratio,
                        cfg_cached_steps,
                    )

        if not found_stats:
            logger.info(
                "[Cache-DiT][stepwise] Request %s: no slot cache contexts found.",
                request_state.req_id,
            )

    def _should_log_on_this_rank(self) -> bool:
        try:
            return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        except Exception:
            return True

    def _log_stepwise_batch(
        self,
        scheduler_output: DiffusionSchedulerOutput,
        scheduled_states: list[DiffusionRequestState],
        new_req_ids: Iterable[str],
    ) -> None:
        if not scheduled_states or not self._should_log_on_this_rank():
            return

        new_req_id_set = set(new_req_ids)
        req_ids = [state.req_id for state in scheduled_states if state.req_id != "dummy_req_id"]
        if not req_ids:
            return

        new_batch_req_ids = [req_id for req_id in req_ids if req_id in new_req_id_set]
        cached_batch_req_ids = [req_id for req_id in req_ids if req_id not in new_req_id_set]
        per_req_progress = [
            f"{state.req_id}:{int(state.step_index) + 1}/{max(int(state.total_steps), 1)}"
            for state in scheduled_states
            if state.req_id != "dummy_req_id"
        ]
        request_meta = [
            f"{state.req_id}(seed={self._sampling_seed_for_log(state.sampling)}, "
            f"prompt={self._prompt_preview_for_log(state.prompts, max_length=64)})"
            for state in scheduled_states
            if state.req_id != "dummy_req_id"
        ]

        first_sampling = scheduled_states[0].sampling
        logger.info(
            "[StepBatch] scheduler_step=%d batch_size=%d req_ids=%s new_req_ids=%s cached_req_ids=%s "
            "progress=%s shape=%sx%s num_inference_steps=%s cache_backend=%s",
            scheduler_output.step_id,
            len(req_ids),
            req_ids,
            new_batch_req_ids,
            cached_batch_req_ids,
            per_req_progress,
            getattr(first_sampling, "width", None),
            getattr(first_sampling, "height", None),
            getattr(first_sampling, "num_inference_steps", None),
            self.od_config.cache_backend,
        )
        logger.info("[StepBatch] request_meta=%s", request_meta)

    def load_model(
        self,
        memory_pool_context_fn: callable | None = None,
        load_format: str | None = None,
        custom_pipeline_name: str | None = None,
    ) -> None:
        """
        Load the diffusion model, apply compilation and offloading.

        Args:
            memory_pool_context_fn: Optional function that returns a context manager
                for memory pool allocation (used for sleep mode).
            load_format: Format for loading model weights. Supported formats:
                - "default" (default): Automatically detect and use the default format based on configuration
                - "custom_pipeline": Init model from a custom pipeline class specified by `custom_pipeline_name`
                - "dummy": Skip actual weight loading, useful for testing and custom pipelines that
                    don't require default weights.
            custom_pipeline_name: Optional custom pipeline class name to use.
        """

        if load_format == "dummy":
            return

        load_device = (
            "cpu" if self.od_config.enable_cpu_offload or self.od_config.enable_layerwise_offload else str(self.device)
        )

        def get_memory_context():
            if memory_pool_context_fn is not None:
                return memory_pool_context_fn(tag="weights")
            return nullcontext()

        # Load model within forward context
        load_config = LoadConfig()
        model_loader = DiffusersPipelineLoader(load_config, od_config=self.od_config)
        time_before_load = time.perf_counter()

        with get_memory_context():
            with DeviceMemoryProfiler() as m:
                self.pipeline = model_loader.load_model(
                    od_config=self.od_config,
                    load_device=load_device,
                    load_format=load_format,
                    custom_pipeline_name=custom_pipeline_name,
                    device=self.device,
                )
        time_after_load = time.perf_counter()

        logger.info(
            "Model loading took %.4f GiB and %.6f seconds",
            m.consumed_memory / GiB_bytes,
            time_after_load - time_before_load,
        )
        logger.info("Model runner: Model loaded successfully.")
        self.model_state = init_model_state(self.od_config, self.pipeline, self.device)

        if getattr(self.od_config, "step_execution", False) and not self.supports_step_mode():
            raise ValueError(
                "step_execution=True requires a pipeline implementing "
                "prepare_encode(), denoise_step(), step_scheduler(), and post_decode(); "
                f"{self.od_config.model_class_name} does not support that contract."
            )

        # Apply CPU offloading
        self.offload_backend = get_offload_backend(self.od_config, device=self.device)
        if self.offload_backend is not None:
            logger.info(f" Enabling offloader backend: {self.offload_backend.__class__.__name__}")
            self.offload_backend.enable(self.pipeline)

        # Apply torch.compile if not in eager mode
        if not self.od_config.enforce_eager:
            if current_omni_platform.supports_torch_inductor():
                self._compile_transformer("transformer")
                self._compile_transformer("transformer_2")
            else:
                logger.warning(
                    "Model runner: Platform %s does not support torch inductor, skipping torch.compile.",
                    current_omni_platform.get_torch_device(),
                )

        # Setup cache backend
        self.cache_backend = get_cache_backend(self.od_config.cache_backend, self.od_config.cache_config)
        self.cache_manager = None

        if self.cache_backend is not None:
            if self.od_config.model_class_name in _NO_CACHE_ACCELERATION:
                logger.warning(
                    "Cache backend '%s' is not supported for %s; disabling cache acceleration.",
                    self.od_config.cache_backend,
                    self.od_config.model_class_name,
                )
                self.cache_backend = None
                self.od_config.cache_backend = None
            else:
                self.cache_backend.enable(self.pipeline)
                cache_state_driver = self.cache_backend.create_state_driver(self.pipeline)
                if cache_state_driver is not None:
                    self.cache_manager = CacheManager(cache_state_driver)

        logger.info("Model runner: Initialization complete.")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights into the pipeline."""
        return self.pipeline.load_weights(weights)

    def _record_peak_memory(self, output: DiffusionOutput) -> None:
        """Record peak GPU memory for the current forward pass into output.

        Must be called immediately after pipeline.forward(), with
        reset_peak_memory_stats() called just before it, so the measurement
        reflects this request only and not the global historical maximum.

        Uses max_memory_reserved (CUDA memory pool high-water mark) rather than
        max_memory_allocated so that allocator fragmentation is also visible.
        See: https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_reserved.html
        """
        peak_reserved_bytes = current_omni_platform.max_memory_reserved()
        peak_allocated_bytes = current_omni_platform.max_memory_allocated()

        output.peak_memory_mb = peak_reserved_bytes / (1024**2)
        peak_reserved_gb = peak_reserved_bytes / (1024**3)
        peak_allocated_gb = peak_allocated_bytes / (1024**3)
        pool_overhead_gb = peak_reserved_gb - peak_allocated_gb

        logger.info(
            "Peak GPU memory (this request): %.2f GB reserved, %.2f GB allocated, %.2f GB pool overhead (%.1f%%)",
            peak_reserved_gb,
            peak_allocated_gb,
            pool_overhead_gb,
            pool_overhead_gb / peak_reserved_gb * 100 if peak_reserved_gb > 0 else 0.0,
        )

    def execute_model(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """
        Execute a forward pass for the given requests.

        Args:
            req: A diffusion request containing a list of prompts to process.

        Returns:
            DiffusionOutput with generated results.

        Note:
            We use torch.no_grad() for HSDP because HSDP2's fully_shard requires access
            to tensor version counters in pre_forward hooks, which inference tensors do
            not track. For non-HSDP inference, we use torch.inference_mode() for better
            performance.
        """
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if len(req.prompts) == 0:
            raise ValueError("Cannot execute model with empty request list")

        # Use no_grad() for HSDP compatibility, inference_mode() otherwise for better perf
        use_hsdp = self.od_config.parallel_config.use_hsdp
        grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
        with grad_context:
            # The manager handles the check for need_recv_cache internally
            self.kv_transfer_manager.receive_multi_kv_cache_distributed(
                req,
                cfg_kv_collect_func=getattr(self.od_config, "cfg_kv_collect_func", None),
                target_device=getattr(self.pipeline, "device", None),
            )

            if req.sampling_params.generator is None and req.sampling_params.seed is not None:
                if req.sampling_params.generator_device is not None:
                    gen_device = req.sampling_params.generator_device
                elif self.device.type == "cpu":
                    gen_device = "cpu"
                else:
                    gen_device = self.device
                req.sampling_params.generator = torch.Generator(device=gen_device).manual_seed(req.sampling_params.seed)

            # Refresh cache context if needed
            if (
                not getattr(req, "skip_cache_refresh", False)
                and self.cache_backend is not None
                and self.cache_backend.is_enabled()
                and req.sampling_params.num_inference_steps is not None
            ):
                self.cache_backend.refresh(self.pipeline, req.sampling_params.num_inference_steps)

            is_primary = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            if is_primary:
                current_omni_platform.reset_peak_memory_stats()

            with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
                with record_function("pipeline_forward"):
                    output = self.pipeline.forward(req)

            if is_primary:
                self._record_peak_memory(output)

            # NOTE:
            if (
                self.cache_backend is not None
                and self.cache_backend.is_enabled()
                and self.od_config.cache_backend == "cache_dit"
                and self.od_config.enable_cache_dit_summary
            ):
                cache_summary(self.pipeline, details=True)
                self._log_cache_dit_request_stats(req)

            return output

    # ------------------------------------------------------------------
    # Step-wise execution
    # ------------------------------------------------------------------

    def supports_step_mode(self) -> bool:
        """Return whether current pipeline supports step execution."""
        return self.pipeline is not None and supports_step_execution(self.pipeline)

    def _make_new_state(
        self,
        req_id: str,
        req: OmniDiffusionRequest,
    ) -> DiffusionRequestState:
        if req_id in self.state_cache:
            raise ValueError(f"Received duplicate new-request payload for cached request {req_id}.")

        request_ids = req.request_ids or [req_id]
        if len(request_ids) != len(req.prompts):
            raise ValueError(
                f"request_ids length ({len(request_ids)}) does not match prompts length ({len(req.prompts)})"
            )

        state = DiffusionRequestState(
            req_id=req_id,
            sampling=copy.deepcopy(req.sampling_params),
            prompts=req.prompts,
        )
        self.state_cache[req_id] = state
        return state

    def _update_states(
        self,
        scheduler_output: DiffusionSchedulerOutput,
    ) -> tuple[list[DiffusionRequestState], set[str]]:
        """Update persistent request states in finish/add/update order."""
        if scheduler_output.num_scheduled_reqs == 0:
            raise ValueError("Stepwise execution requires at least one scheduled request.")

        for req_id in scheduler_output.finished_req_ids:
            state = self.state_cache.pop(req_id, None)
            if state is not None and self.cache_manager is not None:
                self.cache_manager.free(state)

        scheduled_states: list[DiffusionRequestState] = []
        new_req_ids: set[str] = set()
        try:
            for new_req_data in scheduler_output.scheduled_new_reqs:
                state = self._make_new_state(new_req_data.sched_req_id, new_req_data.req)
                scheduled_states.append(state)
                new_req_ids.add(state.req_id)

            for req_id in scheduler_output.scheduled_cached_reqs.sched_req_ids:
                state = self.state_cache.get(req_id)
                if state is None:
                    raise ValueError(f"Missing cached state for request {req_id}.")
                scheduled_states.append(state)
        except Exception:
            for req_id in new_req_ids:
                state = self.state_cache.pop(req_id, None)
                if state is not None and self.cache_manager is not None:
                    self.cache_manager.free(state)
            raise

        return scheduled_states, new_req_ids

    def _ensure_sampling_generator(self, state: DiffusionRequestState) -> None:
        sampling = state.sampling
        if sampling.generator is not None or sampling.seed is None:
            return

        if sampling.generator_device is not None:
            gen_device = sampling.generator_device
        elif self.device.type == "cpu":
            gen_device = "cpu"
        else:
            gen_device = self.device
        sampling.generator = torch.Generator(device=gen_device).manual_seed(sampling.seed)

    # ── stepwise profiling helpers ──────────────────────────────────

    def _stepwise_profiling_enabled(self) -> bool:
        return bool(getattr(self.od_config, "enable_diffusion_pipeline_profiler", False)) and hasattr(
            self.pipeline, "clear_profiler_records"
        )

    def _sync_and_time(self) -> float:
        if current_omni_platform.is_available():
            current_omni_platform.synchronize()
        return time.perf_counter()

    def _prepare_inputs(
        self,
        states: list[DiffusionRequestState],
        new_req_ids: set[str],
    ) -> InputBatch:
        profiling = self._stepwise_profiling_enabled()
        with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
            for state in states:
                self._ensure_sampling_generator(state)
                if state.req_id in new_req_ids:
                    if profiling:
                        self.pipeline.clear_profiler_records()
                        t0 = self._sync_and_time()

                    self.pipeline.prepare_encode(state)

                    if profiling:
                        elapsed = self._sync_and_time() - t0
                        sub_stages = self.pipeline.stage_durations
                        state.extra["_stage_durations"] = {
                            "prepare_encode": elapsed,
                            **sub_stages,
                        }

        input_batch = InputBatch.make_batch(
            states,
            cached_batch=getattr(self, "input_batch", None),
        )
        self.input_batch = input_batch
        return input_batch

    def _postprocess_step(
        self,
        scheduled_states: list[DiffusionRequestState],
        noise_pred: torch.Tensor | None,
    ) -> tuple[list[bool], list[DiffusionOutput | None]]:
        finished_flags = [False] * len(scheduled_states)
        results: list[DiffusionOutput | None] = [None] * len(scheduled_states)

        if noise_pred is None and getattr(self.pipeline, "interrupt", False):
            finished_flags = [True] * len(scheduled_states)
            results = [DiffusionOutput(error="stepwise denoise interrupted") for _ in scheduled_states]
            return finished_flags, results

        if noise_pred is None and self.cache_manager is not None:
            finished_flags = [True] * len(scheduled_states)
            results = [DiffusionOutput(error="stepwise denoise returned None") for _ in scheduled_states]
            return finished_flags, results
        if noise_pred is None:
            for idx, request_state in enumerate(scheduled_states):
                self.pipeline.step_scheduler(request_state, noise_pred)
                finished_flags[idx] = request_state.denoise_completed
            return finished_flags, results

        row_offset = 0
        for idx, request_state in enumerate(scheduled_states):
            num_rows = int(request_state.latents.shape[0])
            next_row_offset = row_offset + num_rows
            request_noise_pred = noise_pred[row_offset:next_row_offset]
            self.pipeline.step_scheduler(request_state, request_noise_pred)
            finished_flags[idx] = request_state.denoise_completed
            row_offset = next_row_offset

        if row_offset != int(noise_pred.shape[0]):
            raise ValueError(
                f"Stepwise noise_pred consumed {row_offset} rows, "
                f"but batched noise_pred has {int(noise_pred.shape[0])} rows."
            )

        return finished_flags, results

    def _update_states_after(
        self,
        scheduled_states: list[DiffusionRequestState],
        input_batch: InputBatch,
        finished_flags: list[bool],
        results: list[DiffusionOutput | None],
    ) -> list[DiffusionOutput | None]:
        """Step-after update: sync latent batch view, decode finished states, and clear cache."""
        gathered_latents = torch.cat([request_state.latents for request_state in scheduled_states], dim=0)
        if (
            tuple(input_batch.latents.shape) == tuple(gathered_latents.shape)
            and input_batch.latents.dtype == gathered_latents.dtype
            and input_batch.latents.device == gathered_latents.device
        ):
            input_batch.latents.copy_(gathered_latents)
        else:
            input_batch.latents = gathered_latents.clone()

        self.input_batch = input_batch
        scatter_latents(scheduled_states, input_batch)

        profiling = self._stepwise_profiling_enabled()
        for idx, (request_state, finished) in enumerate(zip(scheduled_states, finished_flags, strict=True)):
            if finished and results[idx] is None:
                if profiling:
                    self.pipeline.clear_profiler_records()
                    t0 = self._sync_and_time()

                results[idx] = self.pipeline.post_decode(request_state)

                if profiling:
                    elapsed = self._sync_and_time() - t0
                    d = request_state.extra.get("_stage_durations", {})
                    d["post_decode"] = elapsed
                    d.update(self.pipeline.stage_durations)
                    results[idx].stage_durations = d

        for request_state, finished in zip(scheduled_states, finished_flags, strict=True):
            if not finished:
                continue

            if self.od_config.enable_cache_dit_summary:
                self._log_cache_dit_stepwise_request_stats(request_state)
            if self.cache_manager is not None:
                self.cache_manager.free(request_state)
            self.state_cache.pop(request_state.req_id, None)
        if not self.state_cache:
            self.input_batch = None
        return results

    def _prepare_attn_metadata(self, input_batch: InputBatch) -> Any:
        model_state = getattr(self, "model_state", None)
        if model_state is None:
            return {}
        prepare_attn = getattr(model_state, "prepare_attn", None)
        if not callable(prepare_attn):
            return {}
        return prepare_attn(input_batch)

    def _build_runner_output(
        self,
        scheduled_states: list[DiffusionRequestState],
        finished_flags: list[bool],
        results: list[DiffusionOutput | None],
    ) -> RunnerOutput:
        if len(scheduled_states) == 1:
            request_state = scheduled_states[0]
            return RunnerOutput(
                req_id=request_state.req_id,
                step_index=request_state.step_index,
                finished=finished_flags[0],
                result=results[0],
            )
        return RunnerOutput(
            req_id=[request_state.req_id for request_state in scheduled_states],
            step_index=[request_state.step_index for request_state in scheduled_states],
            finished=finished_flags,
            result=results,
        )

    def _cleanup_states_after_failure(
        self,
        scheduled_states: list[DiffusionRequestState],
    ) -> None:
        for request_state in scheduled_states:
            if self.cache_manager is not None:
                self.cache_manager.free(request_state)
            self.state_cache.pop(request_state.req_id, None)
        self.input_batch = None

    def execute_stepwise(self, scheduler_output: DiffusionSchedulerOutput) -> RunnerOutput:
        """Execute one step with persistent request states and a step-local batch."""
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if not self.supports_step_mode():
            raise ValueError("Current pipeline does not support step execution.")
        if self.od_config.cache_backend not in (None, "none"):
            if self.cache_manager is None:
                raise ValueError(
                    f"Step mode cache backend '{self.od_config.cache_backend}' has no resident-state driver."
                )

        use_hsdp = self.od_config.parallel_config.use_hsdp
        grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
        with grad_context:
            scheduled_states: list[DiffusionRequestState] = []
            cache_activated: bool = False
            try:
                scheduled_states, new_req_ids = self._update_states(scheduler_output)
                self._log_stepwise_batch(scheduler_output, scheduled_states, new_req_ids)
                input_batch = self._prepare_inputs(scheduled_states, new_req_ids)

                # NOTE: This runner-level `attn_metadata` is intentionally kept as an
                # extension hook for future heterogeneous batching. The naming may be
                # slightly misleading: model-internal attention metadata is still built
                # inside the transformer path (e.g. QwenImage transformer). Here we
                # mainly pass a mask/supplementary context object through forward context.
                attn_metadata = self._prepare_attn_metadata(input_batch)

                if self.cache_manager is not None:
                    self.cache_manager.activate(scheduled_states)
                    cache_activated = True

                profiling = self._stepwise_profiling_enabled()

                with set_forward_context(
                    vllm_config=self.vllm_config,
                    omni_diffusion_config=self.od_config,
                    attn_metadata=attn_metadata,
                ):
                    if profiling:
                        t0 = self._sync_and_time()

                    noise_pred = self.pipeline.denoise_step(input_batch)

                    if profiling:
                        elapsed = self._sync_and_time() - t0
                        batch_size = len(scheduled_states)
                        amortized = elapsed / batch_size
                        for state in scheduled_states:
                            d = state.extra.setdefault("_stage_durations", {})
                            d["denoise_step"] = d.get("denoise_step", 0.0) + elapsed
                            d["denoise_step_amortized"] = d.get("denoise_step_amortized", 0.0) + amortized
                finished_flags, results = self._postprocess_step(
                    scheduled_states,
                    noise_pred,
                )
                results = self._update_states_after(
                    scheduled_states,
                    input_batch,
                    finished_flags,
                    results,
                )
                return self._build_runner_output(
                    scheduled_states,
                    finished_flags,
                    results,
                )
            except Exception:
                self._cleanup_states_after_failure(scheduled_states)
                raise
            finally:
                if cache_activated and self.cache_manager is not None:
                    self.cache_manager.deactivate(scheduled_states)
