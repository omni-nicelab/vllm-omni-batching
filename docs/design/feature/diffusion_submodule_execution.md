# Diffusion Submodule Execution

This document describes the diffusion submodule execution path in vLLM-Omni.
It lets a diffusion pipeline expose lightweight stages such as encode or decode
without introducing a new runtime stage family. Qwen-Image disaggregated VAE is
the first reference topology built on this mechanism.

## Overview

Some diffusion pipelines have expensive components that do not need to live in
the same process as the denoise loop. A submodule stage runs one direct
pipeline entry point and emits an intermediate `DiffusionOutput` payload for
the next stage:

```text
Submodule stage -> Denoise stage -> Submodule stage
GPU 0             GPU 1          GPU 0
```

For Qwen-Image, this maps to encode -> denoise -> decode. The same runtime
shape can be reused by future diffusion models that need to split
preprocessing, postprocessing, or model-owned helper components out of the main
denoise stage.

## Goals

- Keep existing single-stage diffusion behavior unchanged.
- Represent lightweight diffusion stages as `stage_type=diffusion` plus
  `worker_type=submodule`, not as a new `StageType`.
- Reuse the existing multi-stage orchestrator and
  `custom_process_input_func` mechanism.
- Support direct request-level submodule execution and stepwise denoise stages.
- Keep model-specific tensor schemas inside model pipeline code and stage input
  processors.

## Runtime Shape

`DIFFUSION_SUBMODULE` has the same architectural position for diffusion that
`LLM_GENERATION` has for LLM. The merged execution type expands to an existing
stage family plus a worker selector:

| Merged execution type | Runtime shape | StagePool view |
| :--- | :--- | :--- |
| `LLM_GENERATION` | `stage_type=llm`, `worker_type=generation` | LLM stage |
| `DIFFUSION_SUBMODULE` | `stage_type=diffusion`, `worker_type=submodule` | Diffusion stage |

The difference is the output contract. `LLM_GENERATION` runs through the vLLM
engine-core path and emits token `EngineCoreOutputs`; diffusion submodule
stages use the existing direct diffusion-output path and emit
`OmniRequestOutput`.

`worker_type=submodule` only changes how a diffusion replica is initialized:
`initialize_diffusion_stage()` builds `StageSubModuleClient` instead of the
full `StageDiffusionClient`. StagePool and the orchestrator still see a normal
`stage_type=diffusion` stage and poll it through `get_diffusion_output_nowait`.

## Data Contract

Intermediate tensors are passed through the existing multimodal output path:

```text
DiffusionOutput.multimodal_output
  -> OmniRequestOutput.multimodal_output
  -> custom_process_input_func
  -> OmniTokensPrompt.additional_information
  -> downstream pipeline
```

The orchestrator does not inspect model tensor schemas. Each model owns the
conversion functions that map one stage output to the next stage input.

The current implementation uses the existing ZMQ/msgpack path. Large tensors
therefore still take a device-to-host-to-device path. Connector-backed tensor
transfer is a future optimization.

## Orchestrator Flow

1. Submit the user request to the first stage.
2. Poll the direct diffusion output from the current stage pool.
3. Run the next stage's configured `custom_process_input_func`.
4. Submit the converted prompt to the next diffusion stage.
5. Return the final output to the caller.

## Qwen-Image Reference Topology

The initial reference config is
`vllm_omni/model_executor/stage_configs/qwen_image_3stage.yaml`.

| Stage | Runtime `stage_type` | Runtime `worker_type` | `model_stage` | Responsibility |
| :--- | :--- | :--- | :--- | :--- |
| 0 | `diffusion` | `submodule` | `encode` | Tokenizer, text encoder, scheduler inputs, initial latents. |
| 1 | `diffusion` | unset | `denoise` | Transformer denoise loop and scheduler steps. |
| 2 | `diffusion` | `submodule` | `decode` | VAE decode and image postprocessing. |

`model_stage=diffusion` remains the full coupled Qwen-Image path:

| `model_stage` | Components |
| :--- | :--- |
| `diffusion` | scheduler, text encoder, tokenizer, VAE, transformer |
| `encode` | scheduler, text encoder, tokenizer |
| `denoise` | scheduler, transformer |
| `decode` | VAE |

The Qwen-Image conversion functions live in
`vllm_omni/model_executor/stage_input_processors/qwen_image.py`:

- `encode_to_denoise`
- `denoise_to_decode`

`QwenImagePipeline` provides the model-specific entry points:

- `execute_encode()` for the encode submodule stage.
- `prepare_encode()`, `denoise_step()`, and `step_scheduler()` for the denoise
  diffusion stage.
- `post_intermediate_output()` to emit denoised latents instead of decoding.
- `execute_decode()` for the decode submodule stage.

Existing Qwen-Image users do not need to change their configuration. Without a
multi-stage config, Qwen-Image runs as before and performs text encode,
denoise, and VAE decode inside the same diffusion stage.

## Related Files

- `vllm_omni/diffusion/stage_submodule_client.py`
- `vllm_omni/diffusion/stage_submodule_proc.py`
- `vllm_omni/diffusion/worker/diffusion_submodule_runner.py`
- `vllm_omni/diffusion/worker/submodule_worker.py`
- `vllm_omni/model_executor/stage_input_processors/qwen_image.py`
- `vllm_omni/model_executor/stage_configs/qwen_image_3stage.yaml`
- `vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py`

## Limitations

- Tensor transfer is still inline ZMQ/msgpack, not connector-backed D2D.
- Submodule stages currently use the normal multi-stage startup path; the
  distributed `single_stage_mode` startup path is not supported yet.
- Submodule stages currently support stage-pool replicas, not intra-replica
  model parallelism.
- The Qwen-Image reference topology is a fixed linear 1 encode : 1 denoise :
  1 decode pipeline.
