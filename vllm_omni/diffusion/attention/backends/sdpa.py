# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)


class SDPABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [x for x in range(1024)]  # todo

    @staticmethod
    def get_name() -> str:
        return "SDPA"

    @staticmethod
    def get_impl_cls() -> type["SDPAImpl"]:
        return SDPAImpl


class SDPAImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        attention_mask = attn_metadata.attn_mask if attn_metadata else None
        attention_mask = self._normalize_attention_mask(attention_mask, query, key)

        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=self.causal,
            scale=self.softmax_scale,
        )
        out = output.permute(0, 2, 1, 3)
        return out

    @staticmethod
    def _normalize_attention_mask(
        attention_mask: torch.Tensor | None,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor | None:
        """Normalize padding masks to shapes SDPA can broadcast reliably.

        Diffusion batching currently produces 2D boolean masks of shape
        ``[batch_size, seq_len]``. PyTorch SDPA cannot directly broadcast this
        form when ``batch_size > 1`` because it interprets the leading
        dimension as the query length axis. Convert it to a key-padding mask
        ``[batch_size, 1, 1, seq_len]``.
        """

        if attention_mask is None:
            return None

        if attention_mask.ndim == 2:
            batch_size = query.shape[0]
            key_len = key.shape[-2]
            if attention_mask.shape == (batch_size, key_len):
                return attention_mask[:, None, None, :]

        if attention_mask.ndim == 3:
            batch_size = query.shape[0]
            query_len = query.shape[-2]
            key_len = key.shape[-2]
            if attention_mask.shape == (batch_size, query_len, key_len):
                return attention_mask[:, None, :, :]

        return attention_mask
