# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Payload dataclasses for QwenImage three-stage execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class EncodeOutput:
    req_id: str
    context: torch.Tensor
    context_mask: torch.Tensor
    context_null: torch.Tensor | None = None
    context_null_mask: torch.Tensor | None = None
    latents: torch.Tensor | None = None
    latent_shape: list[int] = field(default_factory=list)
    timesteps: torch.Tensor | None = None
    sigmas: torch.Tensor | list[float] | None = None
    num_inference_steps: int = 0
    height: int = 0
    width: int = 0
    img_shapes: list = field(default_factory=list)
    guidance: torch.Tensor | None = None
    guidance_scale: float = 1.0
    true_cfg_scale: float = 1.0
    do_true_cfg: bool = False
    txt_seq_lens: list[int] | None = None
    negative_txt_seq_lens: list[int] | None = None
    image_latents: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DenoiseOutput:
    req_id: str
    latents: torch.Tensor
    latent_shape: list[int]
    height: int
    width: int
    output_type: str = "pil"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodeOutput:
    req_id: str
    image: Any
    output_type: str = "pil"
    metadata: dict[str, Any] = field(default_factory=dict)
