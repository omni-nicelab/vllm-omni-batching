# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl


def test_sdpa_accepts_batched_2d_padding_mask():
    batch_size = 2
    seq_len = 5
    num_heads = 3
    head_dim = 4

    torch.manual_seed(0)

    query = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    value = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    attn_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, False, False, False],
        ],
        dtype=torch.bool,
    )

    impl = SDPAImpl(
        num_heads=num_heads,
        head_size=head_dim,
        softmax_scale=1.0 / (head_dim**0.5),
        causal=False,
    )

    output_from_2d_mask = impl.forward(
        query=query.clone(),
        key=key.clone(),
        value=value.clone(),
        attn_metadata=AttentionMetadata(attn_mask=attn_mask),
    )

    output_from_explicit_4d_mask = impl.forward(
        query=query.clone(),
        key=key.clone(),
        value=value.clone(),
        attn_metadata=AttentionMetadata(attn_mask=attn_mask[:, None, None, :]),
    )

    assert output_from_2d_mask.shape == query.shape
    torch.testing.assert_close(output_from_2d_mask, output_from_explicit_4d_mask)
