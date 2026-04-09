# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patch cache-dit CachePattern classes to support batched forward.

After ``cache_dit.enable_cache()`` patches the transformer, this module
further patches the class-level ``forward()`` on ``CachedBlocks_Pattern_Base``
and ``CachedBlocks_Pattern_3_4_5`` so that, when the ``CachedContextManager``
has ``_batch_contexts`` set, the forward runs a batch-aware path:

1. Fn blocks on the full batch
2. Per-request ``can_cache`` decisions (each request's own context + slice)
3. Mn blocks on the compute-group subset only (index_select / index_copy_)
4. Bn blocks on the full batch

All existing cache-dit methods (``can_cache``, ``similarity``, ``apply_cache``,
buffer read/write) are called *unmodified*; we simply switch
``_current_context`` before each call so they operate on the right request.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

_BATCH_ALIGNED_KWARG_NAMES = frozenset(
    {
        "temb",
        "modulate_index",
        "encoder_hidden_states_mask",
        "hidden_states_mask",
        "timestep",
        "guidance",
        "pooled_projections",
        "attention_mask",
        "encoder_attention_mask",
    }
)

# ---------------------------------------------------------------------------
# Batch-context helpers (operate on CachedContextManager instances)
# ---------------------------------------------------------------------------


def set_batch_contexts(
    context_manager: Any,
    contexts_list: list[dict[str, Any]],
    row_counts: list[int],
) -> None:
    """Activate batch mode on a ``CachedContextManager`` instance."""
    context_manager._batch_contexts = contexts_list
    context_manager._batch_row_counts = row_counts
    offsets = [0]
    for c in row_counts:
        offsets.append(offsets[-1] + c)
    context_manager._batch_row_offsets = offsets


def clear_batch_contexts(context_manager: Any) -> None:
    """Deactivate batch mode on a ``CachedContextManager`` instance."""
    context_manager._batch_contexts = None
    context_manager._batch_row_offsets = None
    context_manager._batch_row_counts = None
    context_manager._current_context = None


def _is_batch_mode(context_manager: Any) -> bool:
    return getattr(context_manager, "_batch_contexts", None) is not None


def _slice_compute_tensor(
    value: Any,
    compute_row_t: torch.Tensor,
    total_rows: int,
) -> Any:
    if not isinstance(value, torch.Tensor) or value.ndim == 0:
        return value

    index = compute_row_t.to(device=value.device)
    leading_dim = int(value.shape[0])

    if leading_dim == total_rows:
        return torch.index_select(value, 0, index)

    if leading_dim == total_rows * 2:
        doubled_index = torch.cat([index, index + total_rows], dim=0)
        return torch.index_select(value, 0, doubled_index)

    return value


def _slice_batch_aligned_kwargs(
    kwargs: dict[str, Any],
    compute_row_t: torch.Tensor,
    total_rows: int,
) -> dict[str, Any]:
    if not kwargs:
        return kwargs

    sliced_kwargs = dict(kwargs)
    for key in _BATCH_ALIGNED_KWARG_NAMES:
        if key not in sliced_kwargs:
            continue
        sliced_kwargs[key] = _slice_compute_tensor(
            sliced_kwargs[key],
            compute_row_t,
            total_rows,
        )
    return sliced_kwargs


def _get_request_encoder_seq_len(
    kwargs: dict[str, Any],
    request_index: int,
    row_start: int,
    row_end: int,
    fallback_seq_len: int,
) -> int:
    encoder_mask = kwargs.get("encoder_hidden_states_mask")
    if isinstance(encoder_mask, torch.Tensor) and encoder_mask.ndim >= 2 and int(encoder_mask.shape[0]) >= row_end:
        mask_slice = encoder_mask[row_start:row_end]
        if mask_slice.numel() > 0:
            seq_len = int(mask_slice.sum(dim=1, dtype=torch.int64).max().item())
            return max(0, min(seq_len, fallback_seq_len))

    txt_seq_lens = kwargs.get("txt_seq_lens")
    if isinstance(txt_seq_lens, (list, tuple)) and request_index < len(txt_seq_lens):
        try:
            seq_len = int(txt_seq_lens[request_index])
            return max(0, min(seq_len, fallback_seq_len))
        except (TypeError, ValueError):
            pass

    return fallback_seq_len


def _trim_encoder_slice(
    encoder_slice: torch.Tensor | None,
    kwargs: dict[str, Any],
    request_index: int,
    row_start: int,
    row_end: int,
) -> tuple[torch.Tensor | None, int | None]:
    if encoder_slice is None:
        return None, None

    seq_len = _get_request_encoder_seq_len(
        kwargs,
        request_index,
        row_start,
        row_end,
        int(encoder_slice.shape[1]),
    )
    return encoder_slice[:, :seq_len], seq_len


def _restore_encoder_slice(
    encoder_slice: torch.Tensor | None,
    reference_slice: torch.Tensor | None,
    seq_len: int | None,
) -> torch.Tensor | None:
    if encoder_slice is None or reference_slice is None:
        return encoder_slice
    if seq_len is None:
        return encoder_slice

    restored = reference_slice.clone()
    clipped_len = min(int(seq_len), int(restored.shape[1]), int(encoder_slice.shape[1]))
    restored[:, :clipped_len] = encoder_slice[:, :clipped_len]
    return restored


# ---------------------------------------------------------------------------
# Batched forward for Pattern 0/1/2 (CachedBlocks_Pattern_Base)
# ---------------------------------------------------------------------------


def _forward_batched_base(
    self: Any,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Batch-aware forward for ``CachedBlocks_Pattern_Base`` (Pattern 0/1/2).

    ``self`` is a ``CachedBlocks_Pattern_Base`` instance whose
    ``context_manager`` has ``_batch_contexts`` set.
    """
    cm = self.context_manager
    batch_contexts: list[dict[str, Any]] = cm._batch_contexts
    row_offsets: list[int] = cm._batch_row_offsets
    row_counts: list[int] = cm._batch_row_counts
    num_requests = len(batch_contexts)
    ctx_name: str = self.cache_context  # string key for this block group

    # --- Validate: set context from first request for _check_cache_params ---
    cm._current_context = batch_contexts[0][ctx_name]
    self._check_cache_params()

    # --- Stage 1: Fn blocks on full batch ---
    original_hidden_states = hidden_states
    hidden_states, encoder_hidden_states = self.call_Fn_blocks(
        hidden_states, encoder_hidden_states, *args, **kwargs,
    )
    fn_residual_full = self._get_Fn_residual(original_hidden_states, hidden_states)
    fn_hidden_states_full = hidden_states.clone() if cm.is_l1_diff_enabled() else None
    del original_hidden_states

    # --- Stage 2: per-request can_cache decisions ---
    use_l1 = cm.is_l1_diff_enabled()
    parallelized = self._is_parallelized()
    prefix_fn = (
        f"{self.cache_prefix}_Fn_hidden_states"
        if use_l1
        else f"{self.cache_prefix}_Fn_residual"
    )

    cache_decisions: list[bool] = []
    for i in range(num_requests):
        ctx = batch_contexts[i][ctx_name]
        cm._current_context = ctx
        cm.mark_step_begin()

        start = row_offsets[i]
        end = start + row_counts[i]
        check_tensor = hidden_states[start:end] if use_l1 else fn_residual_full[start:end]

        can_use = cm.can_cache(check_tensor, parallelized=parallelized, prefix=prefix_fn)
        cache_decisions.append(can_use)

    compute_indices = [i for i, c in enumerate(cache_decisions) if not c]
    cache_indices = [i for i, c in enumerate(cache_decisions) if c]

    # --- Stage 3a: compute group -> Mn blocks on physical subset ---
    if compute_indices:
        compute_rows: list[int] = []
        for i in compute_indices:
            compute_rows.extend(range(row_offsets[i], row_offsets[i] + row_counts[i]))
        compute_row_t = torch.tensor(compute_rows, dtype=torch.long, device=hidden_states.device)
        compute_kwargs = _slice_batch_aligned_kwargs(
            kwargs,
            compute_row_t,
            int(hidden_states.shape[0]),
        )

        compute_hs = torch.index_select(hidden_states, 0, compute_row_t)
        compute_enc = (
            torch.index_select(encoder_hidden_states, 0, compute_row_t)
            if encoder_hidden_states is not None
            else None
        )

        mn_hs, mn_enc, mn_hs_residual, mn_enc_residual = self.call_Mn_blocks(
            compute_hs, compute_enc, *args, **compute_kwargs,
        )

        # Write Mn outputs back into full-batch tensor
        hidden_states = hidden_states.clone()
        hidden_states.index_copy_(0, compute_row_t, mn_hs)
        if encoder_hidden_states is not None and mn_enc is not None:
            encoder_hidden_states = encoder_hidden_states.clone()
            encoder_hidden_states.index_copy_(0, compute_row_t, mn_enc)

        # Per-request buffer storage for compute group
        local_offset = 0
        for i in compute_indices:
            ctx = batch_contexts[i][ctx_name]
            cm._current_context = ctx
            nr = row_counts[i]
            bs = row_offsets[i]

            # Fn buffer (for next step's similarity check)
            cm.set_Fn_buffer(fn_residual_full[bs : bs + nr], prefix=f"{self.cache_prefix}_Fn_residual")
            if use_l1:
                assert fn_hidden_states_full is not None
                cm.set_Fn_buffer(
                    fn_hidden_states_full[bs : bs + nr],
                    f"{self.cache_prefix}_Fn_hidden_states",
                )

            # Bn buffer (for future cache-hit apply_cache)
            req_mn_residual = mn_hs_residual[local_offset : local_offset + nr]
            if cm.is_cache_residual():
                cm.set_Bn_buffer(req_mn_residual, prefix=f"{self.cache_prefix}_Bn_residual")
            else:
                cm.set_Bn_buffer(
                    mn_hs[local_offset : local_offset + nr],
                    prefix=f"{self.cache_prefix}_Bn_hidden_states",
                )

            # Encoder Bn buffer
            if mn_enc_residual is not None:
                req_enc_residual, _ = _trim_encoder_slice(
                    mn_enc_residual[local_offset : local_offset + nr],
                    kwargs,
                    i,
                    bs,
                    bs + nr,
                )
                if cm.is_encoder_cache_residual():
                    cm.set_Bn_encoder_buffer(req_enc_residual, prefix=f"{self.cache_prefix}_Bn_residual")
                else:
                    req_mn_enc, _ = _trim_encoder_slice(
                        mn_enc[local_offset : local_offset + nr],
                        kwargs,
                        i,
                        bs,
                        bs + nr,
                    )
                    cm.set_Bn_encoder_buffer(
                        req_mn_enc,
                        prefix=f"{self.cache_prefix}_Bn_hidden_states",
                    )

            local_offset += nr

    del fn_residual_full

    # --- Stage 3b: cache group -> apply cached residuals per request ---
    for i in cache_indices:
        ctx = batch_contexts[i][ctx_name]
        cm._current_context = ctx
        cm.add_cached_step()

        start = row_offsets[i]
        end = start + row_counts[i]
        encoder_slice = encoder_hidden_states[start:end] if encoder_hidden_states is not None else None
        trimmed_encoder_slice, seq_len = _trim_encoder_slice(
            encoder_slice,
            kwargs,
            i,
            start,
            end,
        )

        cached_hs, cached_enc = cm.apply_cache(
            hidden_states[start:end],
            trimmed_encoder_slice,
            prefix=(
                f"{self.cache_prefix}_Bn_residual"
                if cm.is_cache_residual()
                else f"{self.cache_prefix}_Bn_hidden_states"
            ),
            encoder_prefix=(
                f"{self.cache_prefix}_Bn_residual"
                if cm.is_encoder_cache_residual()
                else f"{self.cache_prefix}_Bn_hidden_states"
            ),
        )
        hidden_states[start:end] = cached_hs
        if encoder_hidden_states is not None and cached_enc is not None:
            encoder_hidden_states[start:end] = _restore_encoder_slice(
                cached_enc,
                encoder_slice,
                seq_len,
            )

    # --- Stage 4: Bn blocks on full batch ---
    hidden_states, encoder_hidden_states = self.call_Bn_blocks(
        hidden_states, encoder_hidden_states, *args, **kwargs,
    )

    cm._current_context = None
    return self._process_forward_outputs(hidden_states, encoder_hidden_states)


# ---------------------------------------------------------------------------
# Batched forward for Pattern 3/4/5 (CachedBlocks_Pattern_3_4_5)
# ---------------------------------------------------------------------------


def _forward_batched_345(
    self: Any,
    hidden_states: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Batch-aware forward for ``CachedBlocks_Pattern_3_4_5`` (Pattern 3/4/5).

    Differences from base:
    - ``call_Fn_blocks`` / ``call_Mn_blocks`` / ``call_Bn_blocks`` take
      ``(hidden_states, *args, **kwargs)`` — no separate encoder_hidden_states arg.
    - ``call_Mn_blocks`` returns a 3-tuple ``(hs, new_enc_hs, hs_residual)``.
    - encoder_hidden_states is extracted from block outputs, not passed in.
    """
    cm = self.context_manager
    batch_contexts: list[dict[str, Any]] = cm._batch_contexts
    row_offsets: list[int] = cm._batch_row_offsets
    row_counts: list[int] = cm._batch_row_counts
    num_requests = len(batch_contexts)
    ctx_name: str = self.cache_context

    # --- Validate ---
    cm._current_context = batch_contexts[0][ctx_name]
    self._check_cache_params()

    # --- Stage 1: Fn blocks on full batch ---
    original_hidden_states = hidden_states
    hidden_states, new_encoder_hidden_states = self.call_Fn_blocks(
        hidden_states, *args, **kwargs,
    )
    fn_residual_full = self._get_Fn_residual(original_hidden_states, hidden_states)
    fn_hidden_states_full = hidden_states.clone() if cm.is_l1_diff_enabled() else None
    del original_hidden_states

    # --- Stage 2: per-request can_cache decisions ---
    use_l1 = cm.is_l1_diff_enabled()
    parallelized = self._is_parallelized()
    prefix_fn = (
        f"{self.cache_prefix}_Fn_hidden_states"
        if use_l1
        else f"{self.cache_prefix}_Fn_residual"
    )

    cache_decisions: list[bool] = []
    for i in range(num_requests):
        ctx = batch_contexts[i][ctx_name]
        cm._current_context = ctx
        cm.mark_step_begin()

        start = row_offsets[i]
        end = start + row_counts[i]
        check_tensor = hidden_states[start:end] if use_l1 else fn_residual_full[start:end]

        can_use = cm.can_cache(check_tensor, parallelized=parallelized, prefix=prefix_fn)
        cache_decisions.append(can_use)

    compute_indices = [i for i, c in enumerate(cache_decisions) if not c]
    cache_indices = [i for i, c in enumerate(cache_decisions) if c]

    # --- Stage 3a: compute group -> Mn blocks on physical subset ---
    if compute_indices:
        compute_rows: list[int] = []
        for i in compute_indices:
            compute_rows.extend(range(row_offsets[i], row_offsets[i] + row_counts[i]))
        compute_row_t = torch.tensor(compute_rows, dtype=torch.long, device=hidden_states.device)
        compute_kwargs = _slice_batch_aligned_kwargs(
            kwargs,
            compute_row_t,
            int(hidden_states.shape[0]),
        )

        compute_hs = torch.index_select(hidden_states, 0, compute_row_t)

        # Pattern 3/4/5: call_Mn_blocks returns (hs, new_enc_hs, hs_residual)
        mn_hs, mn_enc, mn_hs_residual = self.call_Mn_blocks(compute_hs, *args, **compute_kwargs)

        # Write back
        hidden_states = hidden_states.clone()
        hidden_states.index_copy_(0, compute_row_t, mn_hs)

        # Per-request buffer storage
        local_offset = 0
        for i in compute_indices:
            ctx = batch_contexts[i][ctx_name]
            cm._current_context = ctx
            nr = row_counts[i]
            bs = row_offsets[i]

            # Fn buffer
            cm.set_Fn_buffer(fn_residual_full[bs : bs + nr], prefix=f"{self.cache_prefix}_Fn_residual")
            if use_l1:
                assert fn_hidden_states_full is not None
                cm.set_Fn_buffer(
                    fn_hidden_states_full[bs : bs + nr],
                    f"{self.cache_prefix}_Fn_hidden_states",
                )

            # Bn buffer
            req_mn_residual = mn_hs_residual[local_offset : local_offset + nr]
            if cm.is_cache_residual():
                cm.set_Bn_buffer(req_mn_residual, prefix=f"{self.cache_prefix}_Bn_residual")
            else:
                cm.set_Bn_buffer(
                    mn_hs[local_offset : local_offset + nr],
                    prefix=f"{self.cache_prefix}_Bn_hidden_states",
                )

            # Encoder Bn buffer
            if mn_enc is not None:
                # Compute encoder residual for this request
                old_enc_slice = (
                    new_encoder_hidden_states[bs : bs + nr]
                    if new_encoder_hidden_states is not None
                    else None
                )
                old_enc_slice, _ = _trim_encoder_slice(
                    old_enc_slice,
                    kwargs,
                    i,
                    bs,
                    bs + nr,
                )
                req_enc, _ = _trim_encoder_slice(
                    mn_enc[local_offset : local_offset + nr],
                    kwargs,
                    i,
                    bs,
                    bs + nr,
                )
                if old_enc_slice is not None:
                    req_enc_residual = req_enc - old_enc_slice
                else:
                    req_enc_residual = req_enc
                if cm.is_encoder_cache_residual():
                    cm.set_Bn_encoder_buffer(req_enc_residual, prefix=f"{self.cache_prefix}_Bn_residual")
                else:
                    cm.set_Bn_encoder_buffer(req_enc, prefix=f"{self.cache_prefix}_Bn_hidden_states")

            local_offset += nr

        # Update new_encoder_hidden_states from Mn output for compute group
        if mn_enc is not None:
            if new_encoder_hidden_states is not None:
                new_encoder_hidden_states = new_encoder_hidden_states.clone()
                new_encoder_hidden_states.index_copy_(0, compute_row_t, mn_enc)
            else:
                # Encoder HS only available for compute group; create full tensor
                new_encoder_hidden_states = torch.zeros_like(hidden_states)
                new_encoder_hidden_states.index_copy_(0, compute_row_t, mn_enc)

    del fn_residual_full

    # --- Stage 3b: cache group -> apply cached values per request ---
    for i in cache_indices:
        ctx = batch_contexts[i][ctx_name]
        cm._current_context = ctx
        cm.add_cached_step()

        start = row_offsets[i]
        end = start + row_counts[i]
        encoder_slice = new_encoder_hidden_states[start:end] if new_encoder_hidden_states is not None else None
        trimmed_encoder_slice, seq_len = _trim_encoder_slice(
            encoder_slice,
            kwargs,
            i,
            start,
            end,
        )

        cached_hs, cached_enc = cm.apply_cache(
            hidden_states[start:end],
            trimmed_encoder_slice,
            prefix=(
                f"{self.cache_prefix}_Bn_residual"
                if cm.is_cache_residual()
                else f"{self.cache_prefix}_Bn_hidden_states"
            ),
            encoder_prefix=(
                f"{self.cache_prefix}_Bn_residual"
                if cm.is_encoder_cache_residual()
                else f"{self.cache_prefix}_Bn_hidden_states"
            ),
        )
        hidden_states[start:end] = cached_hs
        if new_encoder_hidden_states is not None and cached_enc is not None:
            new_encoder_hidden_states[start:end] = _restore_encoder_slice(
                cached_enc,
                encoder_slice,
                seq_len,
            )

    # --- Stage 4: Bn blocks on full batch ---
    if cm.Bn_compute_blocks() > 0:
        hidden_states, new_encoder_hidden_states = self.call_Bn_blocks(
            hidden_states, *args, **kwargs,
        )

    cm._current_context = None
    return self._process_forward_outputs(hidden_states, new_encoder_hidden_states)


# ---------------------------------------------------------------------------
# Monkey-patch entry point
# ---------------------------------------------------------------------------


def patch_cache_dit_for_batching() -> None:
    """Monkey-patch cache-dit CachePattern classes to support batch mode.

    Safe to call multiple times (idempotent via ``_batch_patched`` flag).
    """
    try:
        from cache_dit.caching.cache_blocks.pattern_base import (
            CachedBlocks_Pattern_Base,
        )
        from cache_dit.caching.cache_blocks.pattern_3_4_5 import (
            CachedBlocks_Pattern_3_4_5,
        )
    except ImportError:
        logger.warning(
            "cache-dit CachePattern classes not found; "
            "batch-mode monkey-patch skipped."
        )
        return

    if not getattr(CachedBlocks_Pattern_Base, "_batch_patched", False):
        _orig_base = CachedBlocks_Pattern_Base.forward

        def _forward_with_batch_base(
            self: Any,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if _is_batch_mode(self.context_manager):
                return _forward_batched_base(
                    self, hidden_states, encoder_hidden_states, *args, **kwargs,
                )
            return _orig_base(self, hidden_states, encoder_hidden_states, *args, **kwargs)

        CachedBlocks_Pattern_Base.forward = _forward_with_batch_base  # type: ignore[assignment]
        CachedBlocks_Pattern_Base._batch_patched = True  # type: ignore[attr-defined]
        logger.info("Patched CachedBlocks_Pattern_Base.forward for batch mode.")

    if not getattr(CachedBlocks_Pattern_3_4_5, "_batch_patched", False):
        _orig_345 = CachedBlocks_Pattern_3_4_5.forward

        def _forward_with_batch_345(
            self: Any,
            hidden_states: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if _is_batch_mode(self.context_manager):
                return _forward_batched_345(self, hidden_states, *args, **kwargs)
            return _orig_345(self, hidden_states, *args, **kwargs)

        CachedBlocks_Pattern_3_4_5.forward = _forward_with_batch_345  # type: ignore[assignment]
        CachedBlocks_Pattern_3_4_5._batch_patched = True  # type: ignore[attr-defined]
        logger.info("Patched CachedBlocks_Pattern_3_4_5.forward for batch mode.")
