from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional, Tuple

import torch

try:
    from .interface import BlockSparseTensorsTorch, FlashAttnFunc
    from .cute_preprocess_QK import preprocess_QK
    from .cute_preprocess_V import preprocess_V
except:
    from .interface import BlockSparseTensorsTorch, FlashAttnFunc
    from .cute_preprocess_QK import preprocess_QK
    from .cute_preprocess_V import preprocess_V


@dataclass(frozen=True)
class QuantizedAttentionTensors:
    q_fp8: torch.Tensor
    q_scale: torch.Tensor
    k_fp8: torch.Tensor
    k_scale: torch.Tensor
    vt_fp8: torch.Tensor
    v_scale: torch.Tensor
    turn_on_int8: bool


@dataclass(frozen=True)
class QuantizedPagedKVCache:
    k_cache: torch.Tensor
    k_scale: torch.Tensor
    v_cache: torch.Tensor
    v_scale: torch.Tensor
    page_size: int
    page_table: Optional[torch.Tensor] = None
    turn_on_int8: bool = True


def _quantized_dtype(turn_on_int8: bool) -> torch.dtype:
    return torch.int8 if turn_on_int8 else torch.float8_e4m3fn


def _require_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be on CUDA")


def _check_page_size(page_size: int) -> None:
    if page_size != 128:
        raise ValueError(
            "Only page_size=128 is supported by the current quantized V layout. "
            "The existing V preprocessing emits one scale vector per 128-token page."
        )


def _validate_page_table(page_table: torch.Tensor, batch_size: int) -> None:
    if page_table.dtype != torch.int32:
        raise ValueError("page_table must be int32")
    if page_table.dim() != 2 or page_table.shape[0] != batch_size:
        raise ValueError("page_table must have shape [batch, max_num_pages]")
    _require_cuda_tensor("page_table", page_table)


def _num_pages_from_page_table(page_table: torch.Tensor, required_pages: int) -> int:
    valid_pages = page_table[:, :required_pages]
    valid_mask = valid_pages >= 0
    if not torch.any(valid_mask):
        return 0
    return int(valid_pages[valid_mask].max().item()) + 1


def quantize_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    turn_on_int8: bool = True,
) -> QuantizedAttentionTensors:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, and v must all be rank-4 tensors shaped [batch, seqlen, heads, dim]")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _require_cuda_tensor(name, tensor)
    chunk_size = q.shape[-1]
    if k.shape[-1] != chunk_size:
        raise ValueError("k must have the same head dimension as q for the current QK quantization path")
    q_fp8, q_scale = preprocess_QK(q, int8=turn_on_int8, chunk_size=chunk_size)
    k_fp8, k_scale = preprocess_QK(k, int8=turn_on_int8, chunk_size=chunk_size)
    vt_fp8, v_scale = preprocess_V(v, int8=turn_on_int8)
    return QuantizedAttentionTensors(
        q_fp8=q_fp8,
        q_scale=q_scale,
        k_fp8=k_fp8,
        k_scale=k_scale,
        vt_fp8=vt_fp8,
        v_scale=v_scale,
        turn_on_int8=turn_on_int8,
    )


def flash_attn_prequantized_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    quantized: QuantizedAttentionTensors,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    mask_mod: Optional[Callable] = None,
    full_block_cnt: Optional[torch.Tensor] = None,
    full_block_idx: Optional[torch.Tensor] = None,
    mask_block_cnt: Optional[torch.Tensor] = None,
    mask_block_idx: Optional[torch.Tensor] = None,
):
    block_sparse_tensors = None
    if any(
        tensor is not None
        for tensor in (full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx)
    ):
        block_sparse_tensors = BlockSparseTensorsTorch(
            full_block_cnt=full_block_cnt,
            full_block_idx=full_block_idx,
            mask_block_cnt=mask_block_cnt,
            mask_block_idx=mask_block_idx,
        )
    return FlashAttnFunc.apply(
        q,
        quantized.q_fp8,
        quantized.q_scale,
        k,
        quantized.k_fp8,
        quantized.k_scale,
        v,
        quantized.vt_fp8,
        quantized.v_scale,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        mask_mod,
        block_sparse_tensors.full_block_cnt if block_sparse_tensors is not None else None,
        block_sparse_tensors.full_block_idx if block_sparse_tensors is not None else None,
        block_sparse_tensors.mask_block_cnt if block_sparse_tensors is not None else None,
        block_sparse_tensors.mask_block_idx if block_sparse_tensors is not None else None,
    )


def allocate_quantized_paged_kv_cache(
    num_pages: int,
    page_size: int,
    num_heads: int,
    head_dim: int,
    device: torch.device | str,
    *,
    turn_on_int8: bool = True,
    page_table: Optional[torch.Tensor] = None,
) -> QuantizedPagedKVCache:
    _check_page_size(page_size)
    if num_pages <= 0:
        raise ValueError("num_pages must be positive")
    dtype = _quantized_dtype(turn_on_int8)
    k_cache = torch.empty(num_pages, page_size, num_heads, head_dim, device=device, dtype=dtype)
    k_scale = torch.empty(num_pages, page_size, num_heads, 1, device=device, dtype=torch.float32)
    v_cache = torch.empty(num_pages, page_size, num_heads, head_dim, device=device, dtype=dtype)
    v_scale = torch.empty(num_pages, 1, num_heads, head_dim, device=device, dtype=torch.float32)
    return QuantizedPagedKVCache(
        k_cache=k_cache,
        k_scale=k_scale,
        v_cache=v_cache,
        v_scale=v_scale,
        page_size=page_size,
        page_table=page_table,
        turn_on_int8=turn_on_int8,
    )


def build_quantized_paged_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    page_table: torch.Tensor,
    *,
    page_size: int = 128,
    turn_on_int8: bool = True,
) -> QuantizedPagedKVCache:
    if k.dim() != 4 or v.dim() != 4:
        raise ValueError("k and v must be rank-4 tensors shaped [batch, seqlen, heads, dim]")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes for the current cache builder")
    _check_page_size(page_size)
    _validate_page_table(page_table, k.shape[0])
    _require_cuda_tensor("k", k)
    _require_cuda_tensor("v", v)

    batch_size, seqlen, num_heads, head_dim = k.shape
    required_pages = math.ceil(seqlen / page_size)
    if page_table.shape[1] < required_pages:
        raise ValueError("page_table does not have enough logical pages for the provided sequence length")

    num_pages = _num_pages_from_page_table(page_table, required_pages)
    cache = allocate_quantized_paged_kv_cache(
        num_pages,
        page_size,
        num_heads,
        head_dim,
        k.device,
        turn_on_int8=turn_on_int8,
        page_table=page_table,
    )

    k_fp8, k_scale = preprocess_QK(k, int8=turn_on_int8, chunk_size=head_dim, padding_size=page_size)
    vt_fp8, v_scale = preprocess_V(v, int8=turn_on_int8)

    page_table_cpu = page_table[:, :required_pages].detach().cpu()
    for batch_idx in range(batch_size):
        for logical_page in range(required_pages):
            physical_page = int(page_table_cpu[batch_idx, logical_page].item())
            if physical_page < 0:
                raise ValueError("page_table contains a negative page id within the active range")
            start = logical_page * page_size
            end = start + page_size
            cache.k_cache[physical_page].copy_(k_fp8[batch_idx, start:end])
            cache.k_scale[physical_page].copy_(k_scale[batch_idx, start:end])
            cache.v_cache[physical_page].copy_(vt_fp8[batch_idx, start:end])
            cache.v_scale[physical_page].copy_(v_scale[batch_idx, logical_page : logical_page + 1])
    return cache


def append_quantized_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    cache: QuantizedPagedKVCache,
    *,
    slot_mapping: torch.Tensor,
    page_table: Optional[torch.Tensor] = None,
) -> QuantizedPagedKVCache:
    if k.dim() != 4 or v.dim() != 4:
        raise ValueError("k and v must be rank-4 tensors shaped [batch, seqlen, heads, dim]")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes for append_quantized_kv")
    if slot_mapping.shape != k.shape[:2]:
        raise ValueError("slot_mapping must have shape [batch, seqlen]")
    _require_cuda_tensor("k", k)
    _require_cuda_tensor("v", v)
    _require_cuda_tensor("slot_mapping", slot_mapping)

    page_table = cache.page_table if page_table is None else page_table
    if page_table is None:
        raise ValueError("page_table must be provided either on the cache object or as an argument")
    _validate_page_table(page_table, k.shape[0])
    if cache.page_size != 128:
        raise ValueError("append_quantized_kv only supports page_size=128")

    batch_size, _, _, head_dim = k.shape
    page_table_cpu = page_table.detach().cpu()
    slot_mapping_cpu = slot_mapping.detach().cpu()

    pages_to_write: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for batch_idx in range(batch_size):
        for token_idx in range(slot_mapping_cpu.shape[1]):
            slot = int(slot_mapping_cpu[batch_idx, token_idx].item())
            if slot < 0:
                continue
            logical_page = slot // cache.page_size
            offset = slot % cache.page_size
            if logical_page >= page_table_cpu.shape[1]:
                raise ValueError("slot_mapping references a logical page outside page_table")
            physical_page = int(page_table_cpu[batch_idx, logical_page].item())
            if physical_page < 0:
                raise ValueError("slot_mapping references a negative physical page")
            pages_to_write.setdefault((batch_idx, logical_page, physical_page), []).append(
                (offset, token_idx)
            )

    for (batch_idx, _, physical_page), assignments in pages_to_write.items():
        offsets = sorted(offset for offset, _ in assignments)
        if offsets != list(range(cache.page_size)):
            raise ValueError(
                "append_quantized_kv currently requires full page-aligned writes. "
                "This is because V uses one scale vector per 128-token page in the existing kernel path."
            )
        ordered_token_indices = [token_idx for _, token_idx in sorted(assignments)]
        page_k = k[batch_idx, ordered_token_indices].unsqueeze(0)
        page_v = v[batch_idx, ordered_token_indices].unsqueeze(0)
        page_k_fp8, page_k_scale = preprocess_QK(
            page_k,
            int8=cache.turn_on_int8,
            chunk_size=head_dim,
            padding_size=cache.page_size,
        )
        page_v_fp8, page_v_scale = preprocess_V(page_v, int8=cache.turn_on_int8)
        cache.k_cache[physical_page].copy_(page_k_fp8[0, : cache.page_size])
        cache.k_scale[physical_page].copy_(page_k_scale[0, : cache.page_size])
        cache.v_cache[physical_page].copy_(page_v_fp8[0, : cache.page_size])
        cache.v_scale[physical_page].copy_(page_v_scale[0, :1])
    return cache


__all__ = [
    "QuantizedAttentionTensors",
    "QuantizedPagedKVCache",
    "allocate_quantized_paged_kv_cache",
    "append_quantized_kv",
    "build_quantized_paged_kv_cache",
    "flash_attn_prequantized_func",
    "quantize_attention_inputs",
]
