# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.worker.input_batch import InputBatch


class ModelState(ABC):
    @abstractmethod
    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        pipeline: object,
        device: torch.device,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def prepare_attn(
        self,
        input_batch: InputBatch,
    ) -> dict[str, AttentionMetadata]:
        raise NotImplementedError
