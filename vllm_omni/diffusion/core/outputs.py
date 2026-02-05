# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from PIL import Image


class DiffusionRequestType(Enum):
    ADD = b"\x00"
    ABORT = b"\x01"
    UTILITY = b"\x02"


@dataclass
class DiffusionCoreOutput:
    """DiffusionCore output sent to clients."""

    request_id: str
    finished: bool = False
    images: list[Image.Image] | None = None
    latents: torch.Tensor | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
