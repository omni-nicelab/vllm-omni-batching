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
from vllm.utils.mem_utils import DeviceMemoryProfiler, GiB_bytes

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
        self.vllm_config = vllm_config
        self.od_config = od_config
        self.device = device
        self.pipeline = None
        self.cache_backend = None
        self.cache_manager: CacheManager | None = None
        self.offload_backend = None

        self.state_cache: dict[str, DiffusionRequestState] = {}
        self.input_batch: InputBatch | None = None
        self.model_state = None
        self.kv_transfer_manager = OmniKVTransferManager.from_od_config(od_config)

    def _compile_transformer(self, attr_name: str) -> None:
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

    def load_model(
        self,
        memory_pool_context_fn: callable | None = None,
        load_format: str | None = None,
        custom_pipeline_name: str | None = None,
    ) -> None:
        if load_format == "dummy":
            return

        load_device = (
            "cpu" if self.od_config.enable_cpu_offload or self.od_config.enable_layerwise_offload else str(self.device)
        )

        def get_memory_context():
            if memory_pool_context_fn is not None:
                return memory_pool_context_fn(tag="weights")
            return nullcontext()

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

        self.offload_backend = get_offload_backend(self.od_config, device=self.device)
        if self.offload_backend is not None:
            logger.info(" Enabling offloader backend: %s", self.offload_backend.__class__.__name__)
            self.offload_backend.enable(self.pipeline)

        if not self.od_config.enforce_eager:
            if current_omni_platform.supports_torch_inductor():
                self._compile_transformer("transformer")
                self._compile_transformer("transformer_2")
            else:
                logger.warning(
                    "Model runner: Platform %s does not support torch inductor, skipping torch.compile.",
                    current_omni_platform.get_torch_device(),
                )

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
        return self.pipeline.load_weights(weights)

    def _record_peak_memory(self, output: DiffusionOutput) -> None:
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
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if len(req.prompts) == 0:
            raise ValueError("Cannot execute model with empty request list")

        use_hsdp = self.od_config.parallel_config.use_hsdp
        grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
        with grad_context:
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

            if (
                self.cache_backend is not None
                and self.cache_backend.is_enabled()
                and self.od_config.cache_backend == "cache_dit"
                and self.od_config.enable_cache_dit_summary
            ):
                cache_summary(self.pipeline, details=True)

            return output

    def supports_step_mode(self) -> bool:
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

    def _prepare_inputs(
        self,
        states: list[DiffusionRequestState],
        new_req_ids: set[str],
    ) -> InputBatch:
        with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
            for state in states:
                self._ensure_sampling_generator(state)
                if state.req_id in new_req_ids:
                    self.pipeline.prepare_encode(state)

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

        for idx, (request_state, finished) in enumerate(zip(scheduled_states, finished_flags, strict=True)):
            if finished and results[idx] is None:
                results[idx] = self.pipeline.post_decode(request_state)

        for request_state, finished in zip(scheduled_states, finished_flags, strict=True):
            if not finished:
                continue
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
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if not self.supports_step_mode():
            raise ValueError("Current pipeline does not support step execution.")
        if self.od_config.cache_backend not in (None, "none") and self.cache_manager is None:
            raise ValueError(
                f"Step mode cache backend '{self.od_config.cache_backend}' has no resident-state driver."
            )

        use_hsdp = self.od_config.parallel_config.use_hsdp
        grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
        with grad_context:
            scheduled_states: list[DiffusionRequestState] = []
            cache_activated = False
            try:
                scheduled_states, new_req_ids = self._update_states(scheduler_output)
                input_batch = self._prepare_inputs(scheduled_states, new_req_ids)
                attn_metadata = self._prepare_attn_metadata(input_batch)

                if self.cache_manager is not None:
                    self.cache_manager.activate(scheduled_states)
                    cache_activated = True

                with set_forward_context(
                    vllm_config=self.vllm_config,
                    omni_diffusion_config=self.od_config,
                    attn_metadata=attn_metadata,
                ):
                    noise_pred = self.pipeline.denoise_step(input_batch)

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
