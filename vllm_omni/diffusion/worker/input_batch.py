# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diffusion input-batch structures following the MRV2-style vLLM layout."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from vllm.utils import random_uuid

from vllm_omni.diffusion.worker.utils import DiffusionRequestState


def _normalize_prompt_embeds(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        return x
    raise ValueError(f"prompt_embeds must be 2D or 3D, got shape={tuple(x.shape)}")


def _normalize_mask(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim != 2:
        raise ValueError(f"prompt mask must be 1D or 2D, got shape={tuple(x.shape)}")
    if x.dtype != torch.bool:
        x = x != 0
    return x


def _pad_prompt_embeds(x: torch.Tensor, target_seq_len: int) -> torch.Tensor:
    x = _normalize_prompt_embeds(x)
    bsz, seq_len, hidden = x.shape
    if seq_len == target_seq_len:
        return x
    out = x.new_zeros((bsz, target_seq_len, hidden))
    out[:, :seq_len] = x
    return out


def _pad_mask(x: torch.Tensor, target_seq_len: int) -> torch.Tensor:
    x = _normalize_mask(x)
    bsz, seq_len = x.shape
    if seq_len == target_seq_len:
        return x
    out = torch.zeros((bsz, target_seq_len), dtype=torch.bool, device=x.device)
    out[:, :seq_len] = x
    return out


def _select_states(
    states: Sequence[DiffusionRequestState],
    idx_mapping: torch.Tensor | None,
) -> tuple[list[DiffusionRequestState], torch.Tensor, np.ndarray]:
    if not states:
        raise ValueError("Cannot build InputBatch from empty states.")

    if idx_mapping is None:
        device = states[0].latents.device if states[0].latents is not None else None
        idx_mapping = torch.arange(len(states), dtype=torch.int32, device=device)
    else:
        if idx_mapping.ndim != 1:
            raise ValueError("idx_mapping must be a 1D tensor.")
        idx_mapping = idx_mapping.to(dtype=torch.int32)

    selected_states: list[DiffusionRequestState] = []
    for batch_idx, state_idx in enumerate(idx_mapping.tolist()):
        if state_idx < 0 or state_idx >= len(states):
            raise ValueError(f"idx_mapping[{batch_idx}]={state_idx} is out of range for states.")
        selected_states.append(states[state_idx])
    return selected_states, idx_mapping, idx_mapping.detach().cpu().numpy()


def _prepare_req_ids(
    states: Sequence[DiffusionRequestState],
) -> list[str]:
    return [state.req_id for state in states]


def _prepare_padded_prompt_fields(
    states: Sequence[DiffusionRequestState],
    *,
    embeds_attr: str,
    mask_attr: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    embeds_values = [getattr(state, embeds_attr) for state in states]
    if not any(embeds is not None for embeds in embeds_values):
        return None, None
    if not all(embeds is not None for embeds in embeds_values):
        raise ValueError(f"Mixed {embeds_attr} in batch.")

    embeds_list = [_normalize_prompt_embeds(embeds) for embeds in embeds_values if embeds is not None]
    max_seq_len = max(embeds.shape[1] for embeds in embeds_list)

    mask_values = [getattr(state, mask_attr) for state in states]
    if any(mask is None for mask in mask_values):
        if any(mask is not None for mask in mask_values):
            raise ValueError(f"Mixed {mask_attr} in batch.")
        if any(embeds.shape[1] != max_seq_len for embeds in embeds_list):
            raise ValueError(
                f"Variable-length {embeds_attr} in batch but {mask_attr} is None. "
                f"Provide masks or ensure {embeds_attr} have the same seq_len."
            )
        return torch.cat(embeds_list, dim=0), None

    padded_embeds = torch.cat([_pad_prompt_embeds(embeds, max_seq_len) for embeds in embeds_list], dim=0)
    padded_masks = torch.cat([_pad_mask(mask, max_seq_len) for mask in mask_values if mask is not None], dim=0)
    return padded_embeds, padded_masks


def _prepare_latents(states: Sequence[DiffusionRequestState]) -> torch.Tensor:
    latents_values = [state.latents for state in states]
    if any(latents is None for latents in latents_values):
        raise ValueError("All requests must have `latents` initialized.")
    return torch.cat([latents for latents in latents_values if latents is not None], dim=0)


def _prepare_timesteps(states: Sequence[DiffusionRequestState]) -> torch.Tensor:
    timesteps = [state.current_timestep for state in states]
    if any(timestep is None for timestep in timesteps):
        raise ValueError("All requests must have a current timestep initialized.")
    if not all(torch.is_tensor(timestep) for timestep in timesteps):
        raise ValueError("InputBatch expects tensor timesteps; normalize them in prepare_inputs().")
    return torch.stack([timestep for timestep in timesteps if timestep is not None], dim=0)


def _prepare_seq_lens(states: Sequence[DiffusionRequestState], attr_name: str) -> list[int] | None:
    values = [getattr(state, attr_name) for state in states]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"Mixed {attr_name} in batch.")
    return [int(value[0]) for value in values if value is not None]


def _prepare_img_shapes(states: Sequence[DiffusionRequestState]) -> list | None:
    values = [state.img_shapes for state in states]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("Mixed img_shapes in batch.")
    return [value[0] if value else [] for value in values if value is not None]


def _prepare_prompt_embeds(
    states: Sequence[DiffusionRequestState],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    prompt_embeds, prompt_embeds_mask = _prepare_padded_prompt_fields(
        states,
        embeds_attr="prompt_embeds",
        mask_attr="prompt_embeds_mask",
    )
    if prompt_embeds is None:
        raise ValueError("All requests must have `prompt_embeds` initialized.")
    return prompt_embeds, prompt_embeds_mask


def _prepare_negative_prompt_embeds(
    states: Sequence[DiffusionRequestState],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    return _prepare_padded_prompt_fields(
        states,
        embeds_attr="negative_prompt_embeds",
        mask_attr="negative_prompt_embeds_mask",
    )


def _same_composition(
    cached_batch: "InputBatch" | None,
    req_ids: list[str],
    idx_mapping_np: np.ndarray,
) -> bool:
    if cached_batch is None:
        return False
    if cached_batch.req_ids != req_ids:
        return False
    return np.array_equal(cached_batch.idx_mapping_np, idx_mapping_np)


@dataclass
class InputBatch:
    """Ephemeral step-level batch view.

    This object intentionally does not own persistent request states. The
    runner remains the source of truth and passes those states back in when it
    wants to refresh the dynamic tensors.
    """

    req_ids: list[str]
    num_reqs: int
    num_reqs_after_padding: int
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray

    latents: torch.Tensor
    timesteps: torch.Tensor
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor | None
    negative_prompt_embeds: torch.Tensor | None
    negative_prompt_embeds_mask: torch.Tensor | None

    img_shapes: list | None = None
    txt_seq_lens: list[int] | None = None
    negative_txt_seq_lens: list[int] | None = None

    def __post_init__(self) -> None:
        if len(self.req_ids) != int(self.idx_mapping.numel()):
            raise ValueError("`req_ids` and `idx_mapping` must have the same length.")
        if self.num_reqs != len(self.req_ids):
            raise ValueError("`num_reqs` must match the number of request ids.")
        if self.num_reqs_after_padding < self.num_reqs:
            raise ValueError("`num_reqs_after_padding` must be >= `num_reqs`.")

    def _refresh_dynamic_fields(self, states: Sequence[DiffusionRequestState]) -> None:
        """Refresh dynamic tensors from persistent request states in-place."""
        selected_states, _, _ = _select_states(states, self.idx_mapping)
        for batch_idx, state in enumerate(selected_states):
            if state.latents is None:
                raise ValueError(f"Request {state.req_id} is missing latents.")
            self.latents[batch_idx : batch_idx + 1].copy_(state.latents)
        self.timesteps.copy_(_prepare_timesteps(selected_states))

    def _rebuild(
        self,
        states: Sequence[DiffusionRequestState],
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        req_ids: list[str],
    ) -> "InputBatch":
        self.req_ids = req_ids
        self.num_reqs = len(req_ids)
        self.num_reqs_after_padding = len(req_ids)
        self.idx_mapping = idx_mapping
        self.idx_mapping_np = idx_mapping_np
        self.latents = _prepare_latents(states)
        self.timesteps = _prepare_timesteps(states)
        self.prompt_embeds, self.prompt_embeds_mask = _prepare_prompt_embeds(states)
        (
            self.negative_prompt_embeds,
            self.negative_prompt_embeds_mask,
        ) = _prepare_negative_prompt_embeds(states)
        self.img_shapes = _prepare_img_shapes(states)
        self.txt_seq_lens = _prepare_seq_lens(states, "txt_seq_lens")
        self.negative_txt_seq_lens = _prepare_seq_lens(states, "negative_txt_seq_lens")
        self.__post_init__()
        return self

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        num_tokens: int,
        device: torch.device,
    ) -> "InputBatch":
        assert 0 < num_reqs <= num_tokens
        req_ids = [f"req_{i}_{random_uuid()}" for i in range(num_reqs)]
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
        idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
        timesteps = torch.zeros(num_reqs, dtype=torch.float32, device=device)
        empty_latents = torch.empty((num_reqs, 0), device=device)
        empty_prompt_embeds = torch.empty((num_reqs, 0, 0), device=device)
        return cls(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            latents=empty_latents,
            timesteps=timesteps,
            prompt_embeds=empty_prompt_embeds,
            prompt_embeds_mask=None,
            negative_prompt_embeds=None,
            negative_prompt_embeds_mask=None,
            img_shapes=None,
            txt_seq_lens=None,
            negative_txt_seq_lens=None,
        )

    @classmethod
    def from_states(
        cls,
        states: Sequence[DiffusionRequestState],
        idx_mapping: torch.Tensor | None = None,
        cached_batch: "InputBatch" | None = None,
    ) -> "InputBatch":
        """Gather a temporary batch view from already-prepared request states.

        This method expects the runner to have already checked and normalized any
        request-local defaults before calling into it.
        """
        selected_states, idx_mapping, idx_mapping_np = _select_states(states, idx_mapping)
        req_ids = _prepare_req_ids(selected_states)

        if _same_composition(cached_batch, req_ids, idx_mapping_np):
            assert cached_batch is not None
            cached_batch._refresh_dynamic_fields(selected_states)
            return cached_batch
        if cached_batch is not None:
            return cached_batch._rebuild(selected_states, idx_mapping, idx_mapping_np, req_ids)

        prompt_embeds, prompt_embeds_mask = _prepare_prompt_embeds(selected_states)
        negative_prompt_embeds, negative_prompt_embeds_mask = _prepare_negative_prompt_embeds(selected_states)
        return cls(
            req_ids=req_ids,
            num_reqs=len(selected_states),
            num_reqs_after_padding=len(selected_states),
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            latents=_prepare_latents(selected_states),
            timesteps=_prepare_timesteps(selected_states),
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            img_shapes=_prepare_img_shapes(selected_states),
            txt_seq_lens=_prepare_seq_lens(selected_states, "txt_seq_lens"),
            negative_txt_seq_lens=_prepare_seq_lens(selected_states, "negative_txt_seq_lens"),
        )


DiffusionInputBatch = InputBatch
