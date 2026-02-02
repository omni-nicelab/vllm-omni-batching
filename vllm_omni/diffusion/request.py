# adapted from sglang and fastvideo
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pprint
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import PIL.Image
import torch


class DiffusionRequestStatus(str, Enum):
    """Request status in the scheduler."""

    WAITING = "waiting"  # In waiting queue, not yet started
    RUNNING = "running"  # Currently being processed
    PREEMPTED = "preempted"  # Preempted and moved back to waiting
    FINISHED_COMPLETED = "finished_completed"  # All steps completed
    FINISHED_ABORTED = "finished_aborted"  # Aborted by user
    FINISHED_ERROR = "finished_error"  # Error during processing

    @staticmethod
    def is_finished(status: "DiffusionRequestStatus") -> bool:
        return status in (
            DiffusionRequestStatus.FINISHED_COMPLETED,
            DiffusionRequestStatus.FINISHED_ABORTED,
            DiffusionRequestStatus.FINISHED_ERROR,
        )


@dataclass
class OmniDiffusionRequest:
    """
    Complete state passed through the pipeline execution.

    This dataclass contains all information needed during the diffusion pipeline
    execution, allowing methods to update specific components without needing
    to manage numerous individual parameters.
    """

    # TODO(will): double check that args are separate from server_args
    # properly. Also maybe think about providing an abstraction for pipeline
    # specific arguments.
    # data_type: DataType

    request_id: str | None = None

    generator: torch.Generator | list[torch.Generator] | None = None

    # Image inputs
    image_path: str | None = None
    # Image encoder hidden states
    image_embeds: list[torch.Tensor] = field(default_factory=list)
    pil_image: torch.Tensor | PIL.Image.Image | None = None
    pixel_values: torch.Tensor | PIL.Image.Image | None = None
    preprocessed_image: torch.Tensor | None = None

    # Text inputs
    prompt: str | list[str] | None = None
    negative_prompt: str | list[str] | None = None
    prompt_path: str | None = None
    output_path: str = "outputs/"
    # without extension
    output_file_name: str | None = None
    output_file_ext: str | None = None
    # Primary encoder embeddings
    prompt_embeds: list[torch.Tensor] | torch.Tensor = field(default_factory=list)
    negative_prompt_embeds: list[torch.Tensor] | None = None
    prompt_attention_mask: list[torch.Tensor] | None = None
    negative_attention_mask: list[torch.Tensor] | None = None
    clip_embedding_pos: list[torch.Tensor] | None = None
    clip_embedding_neg: list[torch.Tensor] | None = None

    pooled_embeds: list[torch.Tensor] = field(default_factory=list)
    neg_pooled_embeds: list[torch.Tensor] = field(default_factory=list)

    # Additional text-related parameters
    max_sequence_length: int | None = None
    prompt_template: dict[str, Any] | None = None
    do_classifier_free_guidance: bool = False

    # Batch info
    num_outputs_per_prompt: int = 1
    seed: int | None = None
    seeds: list[int] | None = None

    # layered info
    layers: int = 4

    # cfg info
    cfg_normalize: bool = False

    # caption language
    use_en_prompt: bool = False

    # different bucket in (640, 1024) to determine the condition and output resolution
    resolution: int = 640

    # Tracking if embeddings are already processed
    is_prompt_processed: bool = False

    # Latent tensors
    latents: torch.Tensor | None = None
    raw_latent_shape: torch.Tensor | None = None
    noise_pred: torch.Tensor | None = None
    image_latent: torch.Tensor | None = None

    # Latent dimensions
    height_latents: list[int] | int | None = None
    width_latents: list[int] | int | None = None
    num_frames: list[int] | int = 1  # Default for image models
    num_frames_round_down: bool = False  # Whether to round down num_frames if it's not divisible by num_gpus

    # Original dimensions (before VAE scaling)
    height: list[int] | int | None = None
    width: list[int] | int | None = None
    fps: list[int] | int | None = None
    height_not_provided: bool = False
    width_not_provided: bool = False

    # Timesteps
    timesteps: torch.Tensor | None = None
    timestep: torch.Tensor | float | int | None = None
    step_index: int | None = None
    boundary_ratio: float | None = None

    # Scheduler parameters
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    guidance_scale_provided: bool = False
    guidance_scale_2: float | None = None
    guidance_rescale: float = 0.0
    eta: float = 0.0
    sigmas: list[float] | None = None

    true_cfg_scale: float | None = None  # qwen-image specific now

    n_tokens: int | None = None

    # Other parameters that may be needed by specific schedulers
    extra_step_kwargs: dict[str, Any] = field(default_factory=dict)

    # Component modules (populated by the pipeline)
    modules: dict[str, Any] = field(default_factory=dict)

    return_trajectory_latents: bool = False
    return_trajectory_decoded: bool = False
    trajectory_timesteps: list[torch.Tensor] | None = None
    trajectory_latents: torch.Tensor | None = None

    # Extra parameters that might be needed by specific pipeline implementations
    extra: dict[str, Any] = field(default_factory=dict)

    # Misc
    save_output: bool = True
    return_frames: bool = False

    # STA parameters
    STA_param: list | None = None
    is_cfg_negative: bool = False
    mask_search_final_result_pos: list[list] | None = None
    mask_search_final_result_neg: list[list] | None = None

    # VSA parameters
    VSA_sparsity: float = 0.0
    # perf_logger: PerformanceLogger | None = None

    # stage logging
    # logging_info: PipelineLoggingInfo = field(default_factory=PipelineLoggingInfo)

    # profile
    profile: bool = False
    num_profiled_timesteps: int = 8

    # debugging
    debug: bool = False

    # results
    output: torch.Tensor | None = None

    @property
    def batch_size(self):
        # Determine batch size
        if isinstance(self.prompt, list):
            batch_size = len(self.prompt)
        elif self.prompt is not None:
            batch_size = 1
        else:
            batch_size = self.prompt_embeds[0].shape[0]

        # Adjust batch size for number of videos per prompt
        batch_size *= self.num_outputs_per_prompt
        return batch_size

    def __post_init__(self):
        """Initialize dependent fields after dataclass initialization."""
        # Set do_classifier_free_guidance based on guidance scale and negative prompt
        if self.guidance_scale > 1.0 and self.negative_prompt is not None:
            self.do_classifier_free_guidance = True
        if self.negative_prompt_embeds is None:
            self.negative_prompt_embeds = []
        if self.guidance_scale_2 is None:
            self.guidance_scale_2 = self.guidance_scale

    def __str__(self):
        return pprint.pformat(asdict(self), indent=2, width=120)


@dataclass
class DiffusionRequestState:
    """Per-request state for continuous batching.

    This contains request-level config plus mutable generation state
    (step index, per-request scheduler/sampler state, RoPE cache, etc.).
    """

    # Identity + source request
    req_id: str
    req: OmniDiffusionRequest

    # Encoded prompts (computed once in prepare phase)
    prompt_embeds: torch.Tensor | None = None  # [B, seq_len, hidden_dim]
    prompt_embeds_mask: torch.Tensor | None = None
    negative_prompt_embeds: torch.Tensor | None = None
    negative_prompt_embeds_mask: torch.Tensor | None = None

    # I2V mode
    latent_condition: torch.Tensor | None = None  # encoded image condition
    first_frame_mask: torch.Tensor | None = None  # mask for I2V blending

    # Timestep scheduling (mutable)
    timesteps: torch.Tensor | None = None  # all timesteps for this request
    boundary_timestep: float | None = None  # for dual-model switching
    step_index: int = 0
    timestep: torch.Tensor | float | int | None = None

    # Latent state
    latents: torch.Tensor | None = None

    # Generator state (for reproducibility)
    generator: torch.Generator | None = None

    # Per-request scheduler/sampler state
    scheduler: Any | None = None
    sampler: Any | None = None
    sampler_state: dict[str, Any] = field(default_factory=dict)

    # CFG / guidance
    do_true_cfg: bool = False
    guidance: torch.Tensor | None = None
    true_cfg_scale: float = 1.0

    # Per-request RoPE metadata/cache (for per-sample RoPE in future)
    img_shapes: list[list[tuple[int, int, int]]] | None = None
    txt_seq_lens: list[int] | None = None
    negative_txt_seq_lens: list[int] | None = None
    rope_state: dict[str, Any] = field(default_factory=dict)

    # Callback hooks
    callback_on_step_end: Callable | None = None

    @property
    def latent_shape(self) -> tuple[int, int, int, int, int]:
        """[B, C, T, H, W] for video latents"""
        if self.latents is None:
            raise ValueError("Latents not initialized.")
        return tuple(self.latents.shape)  # type: ignore[return-value]

    @property
    def total_steps(self) -> int:
        if self.timesteps is not None:
            return len(self.timesteps)
        return int(self.req.num_inference_steps)

    @property
    def denoise_completed(self) -> bool:
        return self.step_index >= self.total_steps

    @property
    def is_completed(self) -> bool:
        """
        NOTE:
        Under the current design, completion of the denoise stage
        implicitly indicates completion of the decode stage.
        """
        return self.denoise_completed

    @property
    def current_timestep(self) -> torch.Tensor | None:
        if self.timesteps is None:
            return None
        return self.timesteps[self.step_index]

    @property
    def img_seq_len(self) -> int:
        if self.latents is None:
            return 0
        return int(self.latents.shape[0])

    @property
    def txt_seq_len(self) -> int:
        if self.prompt_embeds is None:
            return 0
        if self.prompt_embeds.ndim == 3:
            return int(self.prompt_embeds.shape[1])
        if self.prompt_embeds.ndim == 2:
            return int(self.prompt_embeds.shape[0])
        return 0
