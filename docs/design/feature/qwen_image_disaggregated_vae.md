# Qwen-Image Disaggregated VAE

This document describes the Qwen-Image disaggregated VAE path in vLLM-Omni.
The feature splits the monolithic Qwen-Image diffusion pipeline into separate
encode, denoise, and decode stages while keeping the existing single-stage
diffusion path unchanged.

## Overview

The original Qwen-Image pipeline runs prompt encoding, DiT denoising, and VAE
decoding inside one diffusion stage. The disaggregated path separates these
steps so each stage loads only the components it needs:

```text
Encode stage -> Denoise stage -> Decode stage
GPU 0          GPU 1            GPU 0
```

This is intended to reduce per-stage memory pressure and make it possible to
place text encoder/VAE work separately from the transformer denoise loop.

## Goals

- Keep the existing `model_stage=None` and `model_stage=diffusion` behavior
  compatible with the monolithic Qwen-Image path.
- Load Qwen-Image components by stage instead of loading the full pipeline in
  every process.
- Reuse the existing multi-stage orchestrator and
  `custom_process_input_func` mechanism.
- Support both request-level `execute_model` and stepwise `execute_stepwise`
  for the denoise stage.
- Keep Qwen-specific tensor schema handling inside the Qwen stage input
  processors and pipeline code.

This feature is not a general-purpose arbitrary submodule graph. It provides a
minimal, model-specific encode -> denoise -> decode path for Qwen-Image.

## Stage Layout

The reference stage config is
`vllm_omni/model_executor/stage_configs/qwen_image_3stage.yaml`.

| Stage | `stage_type` | `model_stage` | Responsibility |
| :--- | :--- | :--- | :--- |
| 0 | `submodule` | `encode` | Tokenizer, text encoder, scheduler inputs, initial latents. |
| 1 | `diffusion` | `denoise` | Transformer denoise loop and scheduler steps. |
| 2 | `submodule` | `decode` | VAE decode and image postprocessing. |

`model_stage=diffusion` remains the full coupled path:

| `model_stage` | Components |
| :--- | :--- |
| `diffusion` | scheduler, text encoder, tokenizer, VAE, transformer |
| `encode` | scheduler, text encoder, tokenizer |
| `denoise` | scheduler, transformer |
| `decode` | VAE |

## Runtime Flow

The runtime treats `diffusion` and `submodule` stages as direct-output stages.
They return `OmniRequestOutput` directly instead of vLLM token
`EngineCoreOutputs`.

The orchestrator flow is:

1. Submit the user request to stage 0.
2. Store each finished stage output on its stage client.
3. Run the next stage's configured `custom_process_input_func`.
4. Submit the converted prompt to the next direct-output stage.
5. Return the final decode output to the caller.

The Qwen-Image conversion functions live in
`vllm_omni/model_executor/stage_input_processors/qwen_image.py`:

- `encode_to_denoise`
- `denoise_to_decode`

The orchestrator does not inspect the Qwen tensor schema.

## Data Contract

Intermediate tensors are passed through the existing multimodal output path:

```text
DiffusionOutput.multimodal_output
  -> OmniRequestOutput.multimodal_output
  -> custom_process_input_func
  -> OmniTokensPrompt.additional_information
  -> downstream pipeline
```

Encode output contains prompt embeddings, masks, initial latents, timesteps,
image shapes, and CFG metadata. Denoise output contains final latents, latent
shape, target image size, and output type.

The current implementation uses the existing ZMQ/msgpack path. Large tensors
therefore still take a device-to-host-to-device path. Connector-backed tensor
transfer is a future optimization, not part of this initial feature.

## Execution

`QwenImagePipeline` provides separate stage entry points:

- `execute_encode()` for the encode submodule stage.
- `prepare_encode()`, `denoise_step()`, and `step_scheduler()` for the denoise
  diffusion stage.
- `post_intermediate_output()` to emit denoised latents instead of decoding.
- `execute_decode()` for the decode submodule stage.

`DiffusionModelRunner` uses `post_intermediate_output()` when a stepwise stage
declares `produces_intermediate_stage_output`. The same intermediate-output
path is also used when `execute_model` runs a denoise-only pipeline to
completion.

## Compatibility

Existing Qwen-Image users do not need to change their configuration. Without a
multi-stage config, Qwen-Image runs as before and performs text encode,
denoise, and VAE decode inside the same diffusion stage.

Online serving also treats `submodule` stages as diffusion-shaped stages for
request sampling parameters, so request-level image options such as height,
width, seed, and step count are propagated across encode, denoise, and decode.

## Related Files

- `vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py`
- `vllm_omni/diffusion/models/qwen_image/stage_data.py`
- `vllm_omni/diffusion/stage_submodule_client.py`
- `vllm_omni/diffusion/stage_submodule_proc.py`
- `vllm_omni/diffusion/worker/vae_model_runner.py`
- `vllm_omni/model_executor/stage_input_processors/qwen_image.py`
- `vllm_omni/model_executor/stage_configs/qwen_image_3stage.yaml`

## Limitations

- Tensor transfer is still inline ZMQ/msgpack, not connector-backed D2D.
- The submodule runner is currently Qwen-Image encode/decode oriented.
- The reference topology is a fixed linear 1 encode : 1 denoise : 1 decode
  pipeline.
- Image-conditioning/VAE-encode expansion can be added on top of the same
  encode-stage boundary.
