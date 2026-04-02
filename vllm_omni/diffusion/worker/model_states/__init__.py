# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig


def init_model_state(
    od_config: OmniDiffusionConfig,
    pipeline: object,
    device: torch.device,
):
    from vllm_omni.diffusion.worker.model_states.default import DefaultModelState

    return DefaultModelState(od_config, pipeline, device)
