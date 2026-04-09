# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.model_states.interface import ModelState


def _prepare_attention_metadata(
    embeds: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor | None,
    txt_seq_lens: list[int] | None,
) -> AttentionMetadata:
    batch_size, seq_len = embeds.shape[:2]
    device = embeds.device

    if encoder_hidden_states_mask is not None:
        encoder_hidden_states_mask = encoder_hidden_states_mask.to(
            device=device,
            dtype=torch.bool,
        ).contiguous()
        inferred_seq_lens = encoder_hidden_states_mask.sum(dim=1, dtype=torch.int32)
    else:
        inferred_seq_lens = None

    if txt_seq_lens is not None:
        seq_lens = torch.tensor(txt_seq_lens, dtype=torch.int32, device=device)
        if inferred_seq_lens is not None and not torch.equal(seq_lens, inferred_seq_lens):
            raise ValueError(
                "Attention metadata txt_seq_lens does not match prompt_embeds_mask."
            )
    elif inferred_seq_lens is not None:
        seq_lens = inferred_seq_lens
    else:
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    if encoder_hidden_states_mask is None:
        token_positions = torch.arange(seq_len, device=device, dtype=torch.int32).unsqueeze(0)
        encoder_hidden_states_mask = token_positions < seq_lens.unsqueeze(1)

    return AttentionMetadata(attn_mask=encoder_hidden_states_mask)


class DefaultModelState(ModelState):
    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        pipeline: object,
        device: torch.device,
    ) -> None:
        self.od_config = od_config
        self.pipeline = pipeline
        self.device = device

    def prepare_attn(
        self,
        input_batch: InputBatch,
    ) -> dict[str, AttentionMetadata]:
        attn_metadata: dict[str, AttentionMetadata] = {
            "positive": _prepare_attention_metadata(
                input_batch.prompt_embeds,
                input_batch.prompt_embeds_mask,
                input_batch.txt_seq_lens,
            )
        }
        if input_batch.negative_prompt_embeds is not None:
            attn_metadata["negative"] = _prepare_attention_metadata(
                input_batch.negative_prompt_embeds,
                input_batch.negative_prompt_embeds_mask,
                input_batch.negative_txt_seq_lens,
            )
        return attn_metadata
