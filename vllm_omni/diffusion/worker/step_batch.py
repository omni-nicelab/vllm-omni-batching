# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Step-level batch structures for continuous batching of diffusion requests.

This module defines the data structures used for step-level scheduling:
- StepBatch: Batch of requests for a single denoising step
- StepOutput: Output from processing one request in a step
- StepSchedulerOutput: Scheduler's output (batch + metadata)
- StepRunnerOutput: Runner's output (step results + decoded images)
- BatchBuilder: Interface for building batches with constraints
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vllm_omni.diffusion.request import DiffusionRequestState

if TYPE_CHECKING:
    pass


@dataclass
class StepBatch:
    """Batch of DiffusionRequestState for a single denoising step.

    Continuous batching can accept heterogeneous timesteps, but enforces
    same resolution in this version (v1).
    """

    req_ids: list[str]
    requests: list[DiffusionRequestState]
    latents: torch.Tensor
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor | None
    negative_prompt_embeds: torch.Tensor | None
    negative_prompt_embeds_mask: torch.Tensor | None
    timesteps: torch.Tensor
    img_shapes: list[list[tuple[int, int, int]]] | None = None
    txt_seq_lens: list[int] | None = None
    negative_txt_seq_lens: list[int] | None = None

    def __post_init__(self) -> None:
        if len(self.req_ids) != len(self.requests):
            raise ValueError("`req_ids` and `requests` must be the same length.")
        for rid, state in zip(self.req_ids, self.requests):
            if rid != state.req_id:
                raise ValueError("`req_ids` order must match `requests` order.")
        # NOTE: Resolution/CFG validation is delegated to BatchBuilder implementations
        # (e.g., FixedResolutionBatchBuilder for v1). This allows StepBatch to remain
        # a generic data structure for v2 which may support heterogeneous resolutions.

    def update_dynamic(self) -> None:
        """Update dynamic tensors (latents, timesteps) from request states.

        Call this instead of rebuilding the entire batch when only latents/timesteps change.
        This avoids torch.cat overhead by using in-place copy.
        """
        for i, state in enumerate(self.requests):
            if state.latents is not None:
                self.latents[i : i + 1].copy_(state.latents)
            if state.current_timestep is not None:
                self.timesteps[i] = state.current_timestep

    @classmethod
    def from_requests(cls, requests: list[DiffusionRequestState]) -> StepBatch:
        if not requests:
            raise ValueError("Cannot build StepBatch from empty request list.")

        req_ids = [r.req_id for r in requests]
        latents_list = [r.latents for r in requests]
        if any(lat is None for lat in latents_list):
            raise ValueError("All requests must have `latents` initialized.")
        latents = torch.cat([lat for lat in latents_list if lat is not None], dim=0)

        if any(r.prompt_embeds is None for r in requests):
            raise ValueError("All requests must have `prompt_embeds` initialized.")
        prompt_embeds = torch.cat([r.prompt_embeds for r in requests if r.prompt_embeds is not None], dim=0)

        prompt_masks = [r.prompt_embeds_mask for r in requests]
        if any(m is None for m in prompt_masks):
            if any(m is not None for m in prompt_masks):
                raise ValueError("Mixed prompt_embeds_mask in batch.")
            prompt_embeds_mask = None
        else:
            prompt_embeds_mask = torch.cat([m for m in prompt_masks if m is not None], dim=0)

        negative_prompt_embeds = None
        if all(r.negative_prompt_embeds is not None for r in requests):
            negative_prompt_embeds = torch.cat(
                [r.negative_prompt_embeds for r in requests if r.negative_prompt_embeds is not None],
                dim=0,
            )

        negative_masks = [r.negative_prompt_embeds_mask for r in requests]
        if any(m is None for m in negative_masks):
            if any(m is not None for m in negative_masks):
                raise ValueError("Mixed negative_prompt_embeds_mask in batch.")
            negative_prompt_embeds_mask = None
        else:
            negative_prompt_embeds_mask = torch.cat([m for m in negative_masks if m is not None], dim=0)

        if any(r.current_timestep is None for r in requests):
            raise ValueError("All requests must have `timesteps` initialized.")
        timesteps = torch.stack([r.current_timestep for r in requests], dim=0)

        # img_shapes: each request has shape [batch=1][frames] -> flatten to [requests][frames]
        img_shapes = None
        if any(r.img_shapes is not None for r in requests):
            img_shapes = [r.img_shapes[0] if r.img_shapes else [] for r in requests]

        # txt_seq_lens: prefer field (mask-based) over property (shape-based)
        txt_seq_lens = [r.txt_seq_lens[0] if r.txt_seq_lens else r.txt_seq_len for r in requests]

        # negative_txt_seq_lens: same logic
        negative_txt_seq_lens = None
        if any(r.negative_txt_seq_lens is not None for r in requests):
            negative_txt_seq_lens = [r.negative_txt_seq_lens[0] if r.negative_txt_seq_lens else 0 for r in requests]

        return cls(
            req_ids=req_ids,
            requests=requests,
            latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            timesteps=timesteps,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            negative_txt_seq_lens=negative_txt_seq_lens,
        )


@dataclass
class StepOutput:
    """Output of a single denoising step for a request.

    NOTE: `latents` may be None when returned through IPC to avoid
    serialization overhead. The actual latents are kept in Worker's
    _request_state_cache.
    """

    req_id: str
    step_index: int
    timestep: torch.Tensor | float | int | None
    latents: torch.Tensor | None = None
    noise_pred: torch.Tensor | None = None
    is_complete: bool = False


@dataclass
class StepSchedulerOutput:
    """Scheduler output for a single step.

    This is returned by DiffusionStepScheduler.schedule() and consumed by
    the worker/runner.
    """

    # Step ID for tracking
    step_id: int

    # Request states scheduled in this step
    req_stats: list[DiffusionRequestState] = field(default_factory=list)

    # Request IDs that finished in this scheduling cycle
    finished_req_ids: set[str] = field(default_factory=set)

    # Request IDs that were preempted in this scheduling cycle
    preempted_req_ids: set[str] = field(default_factory=set)

    # Total number of requests currently running
    num_running_reqs: int = 0

    # Total number of requests waiting
    num_waiting_reqs: int = 0


@dataclass
class StepRunnerOutput:
    """Runner output for a single scheduled step.

    This is returned by the worker/runner after executing a StepSchedulerOutput.
    """

    step_id: int
    step_outputs: list[StepOutput] = field(default_factory=list)
    decoded: dict[str, torch.Tensor] = field(default_factory=dict)


class BatchBuilder(ABC):
    """Build a StepBatch from denoising request states."""

    @abstractmethod
    def build(self, states: list[DiffusionRequestState]) -> StepBatch | None:
        """Build a batch for denoising."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any cached state. Called when batch composition changes."""
        pass


class FixedResolutionBatchBuilder(BatchBuilder):
    """v1: fixed resolution + consistent CFG config with batch caching.

    Validates that all requests in the batch have:
    - Same height/width
    - Same do_true_cfg and true_cfg_scale
    - Same guidance values (if present)

    Optimization: Caches the batch when request composition is unchanged,
    only updating dynamic tensors (latents, timesteps) via in-place copy.

    For v2 heterogeneous batching, implement a different BatchBuilder subclass.
    """

    def __init__(self) -> None:
        self._cached_batch: StepBatch | None = None
        self._cached_req_ids: tuple[str, ...] | None = None

    def reset(self) -> None:
        """Clear cached batch."""
        self._cached_batch = None
        self._cached_req_ids = None

    def _validate_states(self, states: list[DiffusionRequestState]) -> None:
        """Validate batch constraints (resolution, CFG config)."""
        if not states:
            return

        base = states[0]
        for state in states[1:]:
            # Resolution check
            if state.req.height != base.req.height or state.req.width != base.req.width:
                raise ValueError(
                    "FixedResolutionBatchBuilder: resolution mismatch "
                    f"({state.req.height}x{state.req.width} vs {base.req.height}x{base.req.width})"
                )
            # CFG config check
            if state.do_true_cfg != base.do_true_cfg:
                raise ValueError("FixedResolutionBatchBuilder: mixed do_true_cfg in batch.")
            if float(state.true_cfg_scale) != float(base.true_cfg_scale):
                raise ValueError("FixedResolutionBatchBuilder: mixed true_cfg_scale in batch.")
            # Guidance check
            if base.guidance is None:
                if state.guidance is not None:
                    raise ValueError("FixedResolutionBatchBuilder: mixed guidance (None vs tensor).")
            else:
                if state.guidance is None:
                    raise ValueError("FixedResolutionBatchBuilder: mixed guidance (tensor vs None).")
                if base.guidance.numel() != state.guidance.numel():
                    raise ValueError("FixedResolutionBatchBuilder: mixed guidance shape.")
                if not torch.allclose(base.guidance, state.guidance):
                    raise ValueError("FixedResolutionBatchBuilder: mixed guidance values.")

    def build(self, states: list[DiffusionRequestState]) -> StepBatch | None:
        if not states:
            self.reset()
            return None

        self._validate_states(states)

        # Check if we can reuse cached batch
        current_req_ids = tuple(s.req_id for s in states)
        if self._cached_batch is not None and self._cached_req_ids == current_req_ids:
            # Same requests - update dynamic tensors in place
            self._cached_batch.requests = states
            self._cached_batch.update_dynamic()
            return self._cached_batch

        # Request composition changed - rebuild full batch
        batch = StepBatch.from_requests(states)
        self._cached_batch = batch
        self._cached_req_ids = current_req_ids
        return batch
