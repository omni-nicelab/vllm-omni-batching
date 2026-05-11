# FLUX.2 Tileable PBR Material Inference

This example runs tileable PBR material map prediction with a FLUX.2-klein-base-9B
LoRA. It supports 2K MultiDiffusion inference and diffsynth-side inference
optimizations such as FlashAttention 3, mh_cute attention, and mh_cute Linear
quantized inference.

## Files

- `infer_tileable_pbr_flux2.py`: standalone batch inference entrypoint.
- `multidiffusion.py`: circular MultiDiffusion implementation for tileable 2K output.
- `run_inference.sh`: minimal example command.

## Input Format

The input directory is scanned for files named:

```text
<base>_<Key>.tga
<base>_<Key>.png
<base>_<Key>.jpg
<base>_<Key>.jpeg
```

Supported keys:

| Key | Meaning |
| --- | --- |
| `Diffuse` | RGB diffuse / basecolor |
| `Normal` | RGB tangent-space normal |
| `SMBE` | packed smoothness / metallic / blend / emissive |
| `Height` | scalar height map |
| `Smooth` | channel extracted from `SMBE` |
| `Metallic` | channel extracted from `SMBE` |
| `Blend` | channel extracted from `SMBE` |
| `Emissive` | channel extracted from `SMBE` |

Example:

```text
tiling3/
  muban0_Diffuse.tga
  muban0_Normal.tga
  muban0_SMBE.tga
  muban0_Height.tga
```

For `--input_keys Diffuse --output_keys Normal`, the script reads
`muban0_Diffuse.*` and writes `pred_Normal.png`.

## Recommended Command

Run from the repository root:

```bash
cd /mnt/upfs/user/jiaxiang.z/codes/pbr-material-edit

export PYTHONPATH=/mnt/upfs/user/jiaxiang.z/codes/pbr-material-edit
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/upfs/user/yueren.jiang/models
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope

python examples/flux2/material_tileable/infer_tileable_pbr_flux2.py \
  --backbone flux2 \
  --lora_path /mnt/upfs/user/yueren.jiang/checkpoints/smbe_flux2_1024/diffuse_to_normal_crop1024/lora/step-35600.safetensors \
  --test_image_dir /mnt/upfs/user/yueren.jiang/test_image/tiling3 \
  --output_dir outputs/tileable_pbr_flux2/diffuse_to_normal_step-35600_md2k_mhcute_quant \
  --input_keys Diffuse \
  --output_keys Normal \
  --multidiffusion \
  --md_height 2048 \
  --md_width 2048 \
  --window_size 1024 \
  --stride 768 \
  --num_inference_steps 8 \
  --dtype fp16 \
  --inference \
  --attn_backend mh_cute \
  --quant \
  --cfg_scale 4.0 \
  --embedded_guidance 4.0 \
  --seed 0
```

Do not hardcode `sys.path` in the script. Use `PYTHONPATH` as shown above.

## Outputs

For each sample base name, outputs are written to:

```text
<output_dir>/<base>/
  input_panel00.png
  pred_full_rgb.png
  pred_<Key>.png
```

For the recommended command above:

```text
outputs/tileable_pbr_flux2/diffuse_to_normal_step-35600_md2k_mhcute_quant/muban0/
  input_panel00.png
  pred_full_rgb.png
  pred_Normal.png
```

## MultiDiffusion Settings

For FLUX.2 2K tileable inference:

- `--multidiffusion`: enable tiled denoise loop.
- `--md_height 2048 --md_width 2048`: final output size.
- `--window_size 1024`: native tile size used by the LoRA inference window.
- `--stride 768`: overlap stride. This gives 9 windows per denoise step for 2048x2048.
- circular wrapping is enabled by default to improve tileability at image borders.
- `--no_circular`: disable circular wrapping.

Cost scales roughly with:

```text
num_steps * num_windows_per_step * cfg_passes
```

With `2048x2048`, `window_size=1024`, `stride=768`, and `cfg_scale=4.0`:

- 9 windows per step.
- CFG runs positive and negative passes, so 18 DiT forwards per step.

## Attention And Quantization

`--attn_backend` choices:

| Backend | Notes |
| --- | --- |
| `mh_cute` | Fastest in the current smoke profile when paired with `--quant`. |
| `flash_attention_3` | Good fallback; faster than FA2 in the current smoke profile. |
| `flash_attention_2` | Script default if no backend is specified. |
| `xformers` | Available option if installed. |
| `torch` | PyTorch SDPA fallback. |

Important details:

- `infer_tileable_pbr_flux2.py` defaults to `flash_attention_2`.
- `run_inference.sh` currently exports `DIFFSYNTH_ATTENTION_IMPLEMENTATION=flash_attention_3`.
- diffsynth's attention module auto-selects FA3 first when no explicit backend is set.
- The script writes `DIFFSYNTH_ATTENTION_IMPLEMENTATION` from `--attn_backend`, so the CLI argument wins.

Linear quantized inference:

- Enable with `--inference --quant`.
- Requires `--dtype fp16`.
- `--quant` converts FLUX.2 DiT Linear layers to mh_cute int8 inference.
- `--linear_chunk_size` defaults to `64`; tune this if Linear remains the bottleneck.

## Measured Smoke Results

Smoke case:

- sample: `muban0`
- task: `Diffuse -> Normal`
- output: `2048x2048`
- MultiDiffusion: `window_size=1024`, `stride=768`
- steps: 8 unless otherwise noted

| Config | dtype | steps | denoise time | wall time | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| FA2 baseline | bf16 | 8 | 3:36 | not recorded | no Linear quant |
| mh_cute attention-only | bf16 | 2 | 46.8s | 71s | no Linear quant |
| mh_cute attention + Linear quant | fp16 | 2 | 45.2s | 66s | `--quant` |
| mh_cute attention + Linear quant | fp16 | 8 | 2:03 | 144s | recommended |

Do not compare the 2-step and 8-step rows directly. The 8-step optimized run is
the comparable row against the 8-step FA2 baseline.

## Profile Summary

A 2-step profile of `mh_cute attention + Linear quant` shows that the bottleneck
is DiT forward, not MultiDiffusion slicing or scheduler work.

For `2048x2048`, `window_size=1024`, `stride=768`, `cfg_scale=4.0`:

| Module | Time |
| --- | ---: |
| `cfg_guided_model_fn` | 33.0s / 2 steps |
| DiT `single_stream_blocks` | 22.5s |
| DiT `double_stream_blocks` | 10.4s |
| `mh_cute_linear` ops | 16.9s |
| `attention_forward` ops | 8.9s |

Implications:

- Reducing tiles per step has direct impact, but may affect seam quality.
- Disabling or batching CFG would reduce repeated DiT forward overhead.
- Linear remains the largest operator-level cost after quantization.
- Attention backend still matters, but attention is not the only bottleneck.

Same 2-step profile with `fp16 + Linear quant` and only attention backend changed:

| Backend | 2-step wall | DiT model_fn total | attention_forward total |
| --- | ---: | ---: | ---: |
| FA2 | 41.6s | 39.8s | 15.7s |
| FA3 | 36.2s | 34.3s | 10.3s |
| mh_cute | 34.9s | 33.0s | 8.9s |

## Common Variants

Use FA3 instead of mh_cute attention:

```bash
--dtype fp16 \
--inference \
--quant \
--attn_backend flash_attention_3
```

Run without Linear quant:

```bash
--dtype bf16 \
--inference \
--attn_backend flash_attention_3
```

Limit to the first sample:

```bash
--max_samples 1
```

Overwrite existing predictions:

```bash
--overwrite
```

Predict multiple outputs packed into RGB panels:

```bash
--input_keys Diffuse \
--output_keys Smooth,Metallic,Blend
```

## Troubleshooting

`--quant` fails with a dtype assertion:

- Use `--dtype fp16`.
- mh_cute `CuteInferLinear` currently expects fp16 weights.

Import path errors:

- Run from repo root.
- Set `PYTHONPATH=/mnt/upfs/user/jiaxiang.z/codes/pbr-material-edit`.
- Do not add hardcoded `sys.path` entries to the script.

No samples found:

- Check that filenames match `<base>_<Key>.<ext>`.
- Check that every source key required by `--input_keys` exists.

Unexpected output width:

- `--md_width 0` means `md_height * output_panel_count`.
- For one RGB output key such as `Normal`, `--md_width 2048` is expected.

