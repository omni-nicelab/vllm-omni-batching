# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Diffusion Model Runner for vLLM-Omni.

Handles model loading, compilation, caching, and execution of diffusion model
forward passes. This follows the AR pattern where the Runner handles all
model-related operations.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from contextlib import nullcontext

import torch
from torch.profiler import record_function
from vllm.config import LoadConfig
from vllm.logger import init_logger
from vllm.utils.mem_utils import DeviceMemoryProfiler, GiB_bytes

from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.offload import apply_offload_hooks
from vllm_omni.diffusion.request import DiffusionRequestState, OmniDiffusionRequest
from vllm_omni.diffusion.worker.step_batch import (
    BatchBuilder,
    FixedResolutionBatchBuilder,
    StepBatch,
    StepOutput,
    StepRunnerOutput,
    StepSchedulerOutput,
)

logger = init_logger(__name__)


class GPUDiffusionModelRunner:
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
        batch_builder: BatchBuilder | None = None,
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
        self.batch_builder = batch_builder or FixedResolutionBatchBuilder()
        self._request_state_cache: dict[str, DiffusionRequestState] = {}

    def load_model(
        self,
        memory_pool_context_fn: callable | None = None,
    ) -> None:
        """
        Load the diffusion model, apply compilation and offloading.

        Args:
            memory_pool_context_fn: Optional function that returns a context manager
                for memory pool allocation (used for sleep mode).
        """
        load_device = "cpu" if self.od_config.enable_cpu_offload else str(self.device)

        def get_memory_context():
            if memory_pool_context_fn is not None:
                return memory_pool_context_fn(tag="weights")
            return nullcontext()

        # Load model within forward context
        with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
            load_config = LoadConfig()
            model_loader = DiffusersPipelineLoader(load_config)
            time_before_load = time.perf_counter()

            with get_memory_context():
                with DeviceMemoryProfiler() as m:
                    self.pipeline = model_loader.load_model(
                        od_config=self.od_config,
                        load_device=load_device,
                    )
            time_after_load = time.perf_counter()

        logger.info(
            "Model loading took %.4f GiB and %.6f seconds",
            m.consumed_memory / GiB_bytes,
            time_after_load - time_before_load,
        )
        logger.info("Model runner: Model loaded successfully.")

        # Apply CPU offloading (DiT <-> encoders mutual exclusion)
        if self.od_config.enable_cpu_offload:
            for name in ["vae"]:
                module = getattr(self.pipeline, name, None)
                if module is None:
                    continue
                try:
                    module.to(self.device, non_blocking=True)
                except Exception as exc:
                    logger.debug("Failed to move %s to GPU: %s", name, exc)

            apply_offload_hooks(self.pipeline, self.od_config, device=self.device)

        # Apply torch.compile if not in eager mode
        if not self.od_config.enforce_eager:
            try:
                self.pipeline.transformer = regionally_compile(
                    self.pipeline.transformer,
                    dynamic=True,
                )
                logger.info("Model runner: Model compiled with torch.compile.")
            except Exception as e:
                logger.warning(f"Model runner: torch.compile failed with error: {e}. Using eager mode.")

        # Setup cache backend
        self.cache_backend = get_cache_backend(self.od_config.cache_backend, self.od_config.cache_config)

        if self.cache_backend is not None:
            self.cache_backend.enable(self.pipeline)

        logger.info("Model runner: Initialization complete.")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights into the pipeline."""
        return self.pipeline.load_weights(weights)

    @torch.inference_mode()
    def execute_model(self, reqs: list[OmniDiffusionRequest]) -> DiffusionOutput:
        """
        Execute a forward pass for the given requests.

        Args:
            reqs: List of diffusion requests to process.

        Returns:
            DiffusionOutput with generated results.
        """
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if not reqs or len(reqs) == 0:
            raise ValueError("Cannot execute model with empty request list")

        # TODO: dealing with first req for now
        req = reqs[0]

        if req.generator is None and req.seed is not None:
            req.generator = torch.Generator(device=self.device).manual_seed(req.seed)

        # Refresh cache context if needed
        if self.cache_backend is not None and self.cache_backend.is_enabled():
            self.cache_backend.refresh(self.pipeline, req.num_inference_steps)

        with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
            with record_function("pipeline_forward"):
                output = self.pipeline.forward(req)

        return output

    @torch.inference_mode()
    def execute_step(self, scheduler_output: StepSchedulerOutput) -> StepRunnerOutput:
        """Execute a single scheduled diffusion step."""
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."

        states = self._get_cached_request_states(scheduler_output)
        to_encode = [state for state in states if self._needs_prepare(state)]
        if to_encode:
            self._batch_encode(to_encode)

        step_outputs: list[StepOutput] = []
        denoise_states = [state for state in states if not state.denoise_completed]
        if denoise_states:
            batch = self.batch_builder.build(denoise_states)
            if batch is not None:
                step_outputs = self._denoise_batch(batch)

        decoded = {}
        decode_states = [state for state in states if state.denoise_completed]
        if decode_states:
            decoded = self._batch_decode(decode_states)

        output = StepRunnerOutput(
            step_id=scheduler_output.step_id,
            step_outputs=step_outputs,
            decoded=decoded,
        )

        # Clear large tensors from step_outputs before returning.
        # This avoids expensive IPC serialization overhead when using multiproc.
        # The actual latents are preserved in _request_state_cache.
        for out in output.step_outputs:
            out.latents = None
            out.noise_pred = None

        self._evict_finished_states(states)
        return output

    def _get_cached_request_states(self, scheduler_output: StepSchedulerOutput) -> list[DiffusionRequestState]:
        """Map scheduler output to cached request states using req_id."""
        resolved: list[DiffusionRequestState] = []
        for state in scheduler_output.req_states:
            cached = self._request_state_cache.get(state.req_id)
            if cached is None:
                cached = DiffusionRequestState(req_id=state.req_id, req=state.req)
                self._request_state_cache[state.req_id] = cached
            else:
                cached.req = state.req
            resolved.append(cached)
        return resolved

    def _evict_finished_states(self, states: list[DiffusionRequestState]) -> None:
        for state in states:
            if state.is_completed:
                self._request_state_cache.pop(state.req_id, None)

    def _needs_prepare(self, state: DiffusionRequestState) -> bool:
        if state.latents is None or state.timesteps is None:
            return True
        if state.prompt_embeds is None:
            return True
        return state.step_index == 0 and state.current_timestep is None

    def _batch_encode(self, states: list[DiffusionRequestState]) -> list[DiffusionRequestState]:
        """Prepare per-request encodings and initialize timesteps/latents."""
        if not states:
            return []

        assert self.pipeline is not None, "Model not loaded. Call load_model() first."

        encoded_states: list[DiffusionRequestState] = []
        with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
            for state in states:
                req = state.req
                if req.num_outputs_per_prompt != 1:
                    raise ValueError("Continuous batching requires num_outputs_per_prompt=1 per request.")
                if req.generator is None and req.seed is not None:
                    req.generator = torch.Generator(device=self.device).manual_seed(req.seed)

                scheduler_override = self.pipeline.scheduler.__class__.from_config(self.pipeline.scheduler.config)
                (
                    prompt_embeds,
                    prompt_embeds_mask,
                    negative_prompt_embeds,
                    negative_prompt_embeds_mask,
                    latents,
                    img_shapes,
                    txt_seq_lens,
                    negative_txt_seq_lens,
                    timesteps,
                    do_true_cfg,
                    guidance,
                    true_cfg_scale,
                    height,
                    width,
                ) = self.pipeline.prepare_encode(
                    req=req,
                    scheduler_override=scheduler_override,
                )

                if hasattr(scheduler_override, "set_begin_index"):
                    scheduler_override.set_begin_index(0)
                if latents.shape[0] != 1:
                    raise ValueError("Continuous batching requires a single latent batch per request.")

                state.prompt_embeds = prompt_embeds
                state.prompt_embeds_mask = prompt_embeds_mask
                state.negative_prompt_embeds = negative_prompt_embeds
                state.negative_prompt_embeds_mask = negative_prompt_embeds_mask
                state.latents = latents
                state.img_shapes = img_shapes
                state.txt_seq_lens = txt_seq_lens
                state.negative_txt_seq_lens = negative_txt_seq_lens
                state.timesteps = timesteps
                state.do_true_cfg = do_true_cfg
                state.guidance = guidance
                state.true_cfg_scale = float(true_cfg_scale) if true_cfg_scale is not None else 1.0
                state.scheduler = scheduler_override
                state.step_index = 0
                state.generator = req.generator if isinstance(req.generator, torch.Generator) else None

                if req.height is None:
                    req.height = height
                if req.width is None:
                    req.width = width

                encoded_states.append(state)

        return encoded_states

    def _denoise_batch(self, batch: StepBatch) -> list[StepOutput]:
        """Execute denoise + per-request scheduler_step."""
        if not batch.requests:
            return []

        assert self.pipeline is not None, "Model not loaded. Call load_model() first."

        base = batch.requests[0]

        if self.cache_backend is not None and self.cache_backend.is_enabled():
            steps_set = {state.total_steps for state in batch.requests}
            if len(steps_set) != 1:
                raise ValueError(
                    "Cache backend does not support mixed num_inference_steps. "
                    "Disable cache or ensure uniform step counts."
                )
            self.cache_backend.refresh(self.pipeline, steps_set.pop())

        with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
            noise_pred = self.pipeline.denoise_step(
                batch.prompt_embeds,
                batch.prompt_embeds_mask,
                batch.negative_prompt_embeds,
                batch.negative_prompt_embeds_mask,
                batch.latents,
                batch.img_shapes,
                batch.txt_seq_lens,
                batch.negative_txt_seq_lens,
                batch.timesteps,
                base.do_true_cfg,
                base.guidance,
                base.true_cfg_scale,
            )

        outputs: list[StepOutput] = []
        for bi, state in enumerate(batch.requests):
            if state.latents is None:
                raise ValueError("Request state latents are not initialized.")
            if state.current_timestep is None:
                raise ValueError("Request state timesteps are not initialized.")
            if state.scheduler is None:
                raise ValueError("Request state scheduler is not initialized.")

            t = state.current_timestep
            pred_i = None if noise_pred is None else noise_pred[bi : bi + 1]
            is_complete = (state.step_index + 1) >= state.total_steps
            state.latents, step_out = self.pipeline.scheduler_step(
                state.latents,
                pred_i,
                t,
                state.do_true_cfg,
                step_index=state.step_index,
                req_id=state.req_id,
                is_complete=is_complete,
                scheduler_override=state.scheduler,
            )
            state.step_index += 1
            state.timestep = t
            outputs.append(step_out)

        return outputs

    def _batch_decode(self, states: list[DiffusionRequestState]) -> dict[str, torch.Tensor]:
        """Decode final latents into images."""
        if not states:
            return {}

        assert self.pipeline is not None, "Model not loaded. Call load_model() first."

        # Detect VAE device to handle CPU offload case
        vae_device = next(self.pipeline.vae.parameters()).device

        decoded: dict[str, torch.Tensor] = {}
        with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
            for state in states:
                if state.latents is None:
                    raise ValueError("Request state latents are not initialized.")

                height = state.req.height
                width = state.req.width
                if height is None or width is None:
                    raise ValueError("Request height/width must be set before decode.")

                # Move latents to VAE device for decoding
                latents_for_decode = state.latents.to(device=vae_device)

                decoded[state.req_id] = self.pipeline.post_decode(
                    latents_for_decode,
                    int(height),
                    int(width),
                    output_type="pil",
                )

        return decoded
