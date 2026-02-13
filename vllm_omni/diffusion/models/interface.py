# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import (
    Any,
    ClassVar,
    Protocol,
    TYPE_CHECKING,
    runtime_checkable,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.request import OmniDiffusionRequest


@runtime_checkable
class SupportImageInput(Protocol):
    support_image_input: ClassVar[bool] = True
    color_format: ClassVar[str] = "RGB"  # Default color format


@runtime_checkable
class SupportAudioOutput(Protocol):
    support_audio_output: ClassVar[bool] = True


@runtime_checkable
class SupportsStepExecution(Protocol):
    """Step-level execution protocol for diffusion pipelines.

    This protocol intentionally keeps method signatures permissive to support
    both state-based and argument-based implementations.
    """

    supports_step_execution: ClassVar[bool] = True

    def prepare_encode(self, req: "OmniDiffusionRequest", **kwargs: Any) -> Any:
        """Prepare request-level inputs before denoise steps."""

    def denoise_step(self, *args: Any, **kwargs: Any) -> Any:
        """Run one denoise step."""

    def step_scheduler(self, *args: Any, **kwargs: Any) -> Any:
        """Run one scheduler step."""

    def post_decode(self, *args: Any, **kwargs: Any) -> "DiffusionOutput | Any":
        """Decode output after denoise loop."""


def supports_step_execution(pipeline: object) -> bool:
    """Return whether `pipeline` implements :class:`SupportsStepExecution`."""

    return isinstance(pipeline, SupportsStepExecution)
