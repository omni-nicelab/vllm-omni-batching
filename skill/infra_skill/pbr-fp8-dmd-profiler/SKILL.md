---
name: pbr-fp8-dmd-profiler
description: Use when working in the pbr-material-edit repo on Qwen Image FP8 or quantized inference, DMD2 distillation training metrics/logging, or profiler trace export. Trigger for tasks involving commits d445890da5947638fe1da5c5e8c0be6ce4094ae1, 62bc648810161dcf72b10fa1c837db644922c4fd, the dev-wjz-profile branch, mh_cute_ops, flashinfer rope, sageattention/FA3 attention backend selection, QwenImagePipeline.enable_inference, DMD2Loss.latest_metrics, TorchProfiler, DIFFSYNTH_TORCH_PROFILER_DIR, VLLM_TORCH_PROFILER_DIR, or exporting torch profiler/nsys traces for Qwen Image six-view inference.
---

# PBR FP8 DMD Profiler

## Scope

Use this skill as a project playbook for `pbr-material-edit` changes derived from:

- `d445890da5947638fe1da5c5e8c0be6ce4094ae1`: Wujiazhen/吴佳圳 FP8 and quantized Qwen Image inference path.
- `62bc648810161dcf72b10fa1c837db644922c4fd`: DMD2 loss metrics, training loop logging, checkpoint naming, and PEFT loading fixes.
- `dev-wjz-profile`: local Qwen profiling hooks and torch profiler trace export.

Prefer inspecting historical files without switching branches when the worktree is dirty:

```bash
git show d445890da5947638fe1da5c5e8c0be6ce4094ae1:<path>
git show 62bc648810161dcf72b10fa1c837db644922c4fd:<path>
git show dev-wjz-profile:<path>
```

## FP8 Inference Commit

The `d445890...` commit adds `diffsynth/core/layers/` and wires inference hooks into `diffsynth/pipelines/qwen_image.py`.

Core files and responsibilities:

- `diffsynth/core/layers/linear.py`: wraps `nn.Linear` with `mh_cute_ops.mlp.CuteInferLinear`; default config is `chunk_size=64`, `int8=True`, `token_divisibility=128`; `run_mh_cute_linear()` pads 3D `[B,S,C]` input and trims output.
- `diffsynth/core/layers/mlp.py`: wraps diffusers `FeedForward` with `mh_cute_ops.mlp.CuteInferMLP`; requires GELU approximate/tanh; stores fallbacks so modules can be enabled/disabled.
- `diffsynth/core/layers/attn.py`: selects attention backend by `DIFFSYNTH_QWEN_ATTN_BACKEND`; accepted values in this commit are `fa3`, `mh_cute`, `sage`; invalid values fall back to `fa3`.
- `diffsynth/core/layers/norm.py`: uses sglang fused q/k norm and scale-shift kernels when available, otherwise falls back to PyTorch modules.
- `diffsynth/core/layers/rope.py`: builds cos/sin cache and applies `flashinfer.rope.apply_rope_with_cos_sin_cache_inplace`.
- `diffsynth/core/layers/__init__.py`: exports the above helpers for pipeline/model wiring.

Inference hook flow in `QwenImagePipeline.enable_inference()`:

1. Require `self.dit` to be loaded; call `self.load_models_to_device(["dit"])`.
2. Preload requested kernels: mh_cute MLP, mh_cute linear, attention backend, and flashinfer rope.
3. Monkey-patch `dit.pos_embed.forward` and optional `forward_sampling` to return cos/sin caches.
4. For each transformer block, create fused norm/gate helper modules, optionally convert MLP and QKV projections, then monkey-patch block and attention `forward`.
5. Set `dit._qwen_image_pipeline_inference_enabled = True` and `pipe.use_inference = True`.

Important implementation details:

- `enable_fp8_attention` only affects the FA3 path in `core/layers/attn.py`; mh_cute attention uses `DIFFSYNTH_QWEN_MH_CUTE_INT8`, default true.
- In `d445890...`, `enable_inference()` quantizes image-side `img_mlp` and attention `to_q/to_k/to_v` only. Text-side projections (`add_q_proj/add_k_proj/add_v_proj`) and `txt_mlp` fall through unless later code extends the attr list.
- `--attn_backend auto` appears in the validation script, but the core backend parser in this commit only accepts `fa3`, `mh_cute`, and `sage`. Avoid `auto` unless the parser is updated.
- Call inference configuration after loading LoRA into `pipe.dit`, matching the validation script pattern.

Validation script flags from `examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit-2509-sixview-accelerate.py`:

```bash
--inference
--attn_backend mh_cute   # or fa3/sage
--quant                  # enables mh_cute MLP/QKV conversion in the hook
```

The helper sets `DIFFSYNTH_QWEN_ATTN_BACKEND` and calls:

```python
pipe.enable_inference(enable_mlp=quant, enable_qkv_linear=quant)
```

## DMD2 Metrics Commit

The `62bc648...` commit updates DMD2 training around generated latent monitoring and more consistent train/validation logging.

`diffsynth/diffusion/loss.py`:

- Add `DMD2Loss.latest_metrics`.
- Add `_update_latent_metrics(latents, generated_latents)` with keys:
  - `latent/std_mean_gt`
  - `latent/std_mean_generated`
  - `latent/std_mean_delta`
- Simulate generator samples with `pipe.scheduler.set_timesteps(num_denoising_steps)` and `pipe.step(...)` instead of hand-coded sigma Euler updates.
- Use `pipe.cfg_guided_model_fn(...)` with scale `1.0` for fake/student predictions and `self.guidance_scale` for teacher/real predictions.
- Pass `inputs_posi` and `inputs_nega` into guidance loss so CFG paths stay consistent.

`examples/qwen_image/model_training/train.py`:

- Split input preparation into `process_pipeline_inputs()` and `prepare_inputs()`.
- `forward()` now consumes processed inputs from `prepare_inputs()`.

`examples/qwen_image/model_training/train_dmd2.py`:

- Add `--log_steps`; clamp `generator_update_freq` and `log_steps` to at least 1.
- Add helpers `_set_training_state(task_name, adapter_name)`, `_optimizer_params(optimizer)`, and `_get_aux_metrics()`.
- Run generator updates every `generator_update_freq` batches; run guidance updates every batch, after generator update on generator turns.
- Log via `model_logger.log_metrics()` with keys such as `train/loss_guidance`, `train/loss_dm`, `train/epoch`, `train/global_step`, `perf/step_seconds`, plus `DMD2Loss.latest_metrics`.
- Save via `model_logger.save_model(...)` using `best.safetensors`, `step-{global_step}.safetensors`, and `epoch-{epoch + 1}.safetensors`.
- Validation and save checks use `(global_step + 1) % steps == 0`; `global_step` increments at the end of each batch.

`examples/qwen_image/model_training/train_dmd2_module.py`:

- Disable base-class LoRA injection by temporarily setting `lora_base_model=None`, not by nulling `lora_rank`.
- When loading LoRA checkpoints, call `mapping_lora_state_dict()`, normalize prefixes `pipe.dit.`, `base_model.model.`, and `model.`, and map adapter-specific keys through `.default.` for PEFT state matching.
- Print explicit warnings when no tensors or only a partial set of tensors load.

`train_sixviews_v3_dmd_distill.sh` establishes the reference DMD2 run shape: six-view albedo data, PEFT LoRA target modules including QKV/add-QKV/out/MLP/mod layers, rank 64, 4 denoising steps, generator LR `1e-4`, guidance LR `5e-5`, generator update frequency 5, guidance scale 1.0, and dynamic timestep rescale options.

## Profiler Export

`dev-wjz-profile` adds `diffsynth/core/profiler/torch_profiler.py` plus profile scopes in `qwen_image_dit.py`, `qwen_image_dit_inference.py`, and `pipelines/qwen_image.py`.

Torch profiler behavior:

- `get_torch_profiler_dir()` reads `VLLM_TORCH_PROFILER_DIR` first, then `DIFFSYNTH_TORCH_PROFILER_DIR`.
- `record_profile_scope(name)` is a no-op unless a profiler dir is set or `DIFFSYNTH_TORCH_PROFILER_RECORD_FUNCTIONS` / `VLLM_TORCH_PROFILER_RECORD_FUNCTIONS` is true.
- `TorchProfiler.start(trace_path_template)` writes `<template>_rank<RANK>.json`, gzip-compressed to `.json.gz` by default.
- The profiler records CPU and CUDA activities, with shapes, memory, stack, and flops enabled by default.
- `TorchProfiler.stop()` calls `step()` if no step was recorded, then exports the trace.

Useful environment variables:

```bash
DIFFSYNTH_TORCH_PROFILER_DIR=outputs/quant_run/torch_profiler
VLLM_TORCH_PROFILER_DIR=outputs/quant_run/torch_profiler
DIFFSYNTH_TORCH_PROFILER_TRACE_TEMPLATE=outputs/quant_run/torch_profiler/sample
DIFFSYNTH_TORCH_PROFILER_GZIP=1
DIFFSYNTH_TORCH_PROFILER_RECORD_SHAPES=1
DIFFSYNTH_TORCH_PROFILER_PROFILE_MEMORY=1
DIFFSYNTH_TORCH_PROFILER_WITH_STACK=1
DIFFSYNTH_TORCH_PROFILER_WITH_FLOPS=1
DIFFSYNTH_TORCH_PROFILER_RECORD_FUNCTIONS=1
```

Reference export pattern in the six-view validation script:

```python
profiler_dir = get_torch_profiler_dir()
if profiler_dir:
    expected_trace = TorchProfiler.start(trace_path_template)
try:
    with record_function("qwen_image_edit.generate"):
        generated_image = pipe(...)
    if profiler_dir:
        TorchProfiler.step()
finally:
    if profiler_dir:
        profiler_result = TorchProfiler.stop()
```

Reference `run_quant.sh` behavior:

- Defaults `QUANT=1`, `NUM_INFERENCE_STEPS=5`, `MAX_SAMPLES=1`, and `OUTPUT_DIR=outputs/quant_run_<utc timestamp>`.
- If `ENABLE_TORCH_PROFILER=1` and `TORCH_PROFILER_DIR` is empty, sets `TORCH_PROFILER_DIR=$OUTPUT_DIR/torch_profiler`.
- Adds `/home/wjz/tmp/mh_cute_ops` to `PYTHONPATH` when quant is enabled.
- Sets `DIFFSYNTH_MODEL_BASE_PATH`, `DIFFSYNTH_SKIP_DOWNLOAD=true`, `DIFFSYNTH_TORCH_PROFILER_DIR`, and `CUTE_DSL_ARCH` for the validation command.
- Passes `--profile_warmup_runs`, `--use_inference_dit`, and optional `--quant`.

The branch does not commit a dedicated `nsys` wrapper. For Nsight Systems, wrap the same command externally and keep torch profiler export separate, for example:

```bash
ENABLE_TORCH_PROFILER=0 nsys profile \
  -o outputs/nsys/qwen_quant \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --force-overwrite=true \
  bash run_quant.sh
```

If named Python scopes must appear in Nsight Systems, add explicit NVTX ranges or run with an NVTX-emitting PyTorch context; the committed profiler hooks are primarily for torch profiler Chrome traces.

## Porting Checklist

1. Inspect current code before patching; this repo often has dirty worktrees and branch-local changes.
2. For FP8 inference, port `core/layers/*` first, then pipeline imports/hooks, then validation flags.
3. Confirm optional runtime dependencies: `mh_cute_ops`, `flashinfer`, `sageattention`, `flash_attn_interface`, and sglang kernels.
4. For DMD2 logging, keep loss metric names stable so wandb/history comparisons survive.
5. For profiler export, add `core/profiler` and scopes before changing run scripts; validate that a trace file is produced per rank.
6. Run the smallest practical smoke test first: one sample, few inference steps, `QUANT=0`, then enable quant/profiling.
