"""Tileable PBR map inference with FLUX.2 LoRA and optional MultiDiffusion.

Expected input files are named like:
    <base>_Diffuse.tga
    <base>_Normal.tga
    <base>_SMBE.tga
    <base>_Height.tga

The script packs input keys into RGB reference panels, runs FLUX.2 Edit LoRA,
then splits predicted output panels back to pred_<Key>.png files.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


KEY_INFO: Dict[str, Dict] = {
    "Diffuse": {"channels": 3, "source": "Diffuse", "extract": None},
    "Normal": {"channels": 3, "source": "Normal", "extract": None},
    "SMBE": {"channels": 3, "source": "SMBE", "extract": None},
    "Height": {"channels": 1, "source": "Height", "extract": 0},
    "Smooth": {"channels": 1, "source": "SMBE", "extract": 0},
    "Metallic": {"channels": 1, "source": "SMBE", "extract": 1},
    "Blend": {"channels": 1, "source": "SMBE", "extract": 2},
    "Emissive": {"channels": 1, "source": "SMBE", "extract": 3},
}
USER_KEY_ALIASES = {k.lower(): k for k in KEY_INFO}
SOURCE_FILE_ALIASES = {
    "diffuse": "Diffuse",
    "normal": "Normal",
    "smbe": "SMBE",
    "height": "Height",
    "hight": "Height",
}
KEY_DESCRIPTION = {
    "Diffuse": "diffuse color (basecolor / albedo)",
    "Normal": "tangent-space normal map",
    "SMBE": "SMBE packed map (smoothness / metallic / blend / emissive)",
    "Height": "height (displacement) map",
    "Smooth": "smoothness scalar",
    "Metallic": "metallic mask",
    "Blend": "blend mask",
    "Emissive": "emissive mask",
}
CATEGORY_PROMPTS = {
    "地面": "floor / ground surface",
    "多合一": "multi-purpose combined material",
    "布料": "fabric / cloth",
    "木板": "wooden board / plank",
    "木纹": "wood grain / bark texture",
    "植物": "plant / vegetation",
    "石砖": "stone / brick",
    "金属": "metal / metallic surface",
}
FILENAME_RE = re.compile(
    r"^(?P<base>.+?)_(?P<key>[A-Za-z]+)\.(?:tga|png|jpg|jpeg)$",
    re.IGNORECASE,
)
PanelSpec = List[Tuple[str, int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora_path", required=True, help="FLUX.2 LoRA .safetensors path.")
    parser.add_argument("--test_image_dir", required=True, help="Directory with <base>_<Key>.tga/png files.")
    parser.add_argument("--output_dir", required=True, help="Directory to write predictions.")
    parser.add_argument("--input_keys", required=True, help="Comma-separated keys, e.g. Diffuse,Normal.")
    parser.add_argument("--output_keys", required=True, help="Comma-separated keys, e.g. Normal or Smooth.")
    parser.add_argument("--backbone", default="flux2", choices=("flux2", "flux2_2048"))
    parser.add_argument("--base_model_id", default="black-forest-labs/FLUX.2-klein-base-9B")
    parser.add_argument("--support_model_id", default="black-forest-labs/FLUX.2-klein-9B")
    parser.add_argument("--model_base_path", default=os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "models"))
    parser.add_argument("--download_source", default=os.environ.get("DIFFSYNTH_DOWNLOAD_SOURCE", "modelscope"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--inference", action="store_true", help="Enable FLUX.2 inference optimizations.")
    parser.add_argument(
        "--attn_backend",
        default=os.environ.get("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "flash_attention_2"),
        choices=("flash_attention_3", "flash_attention_2", "mh_cute", "xformers", "torch"),
        help="Attention backend. Use mh_cute to test mh_cute_ops attention.",
    )
    parser.add_argument("--quant", action="store_true", help="Convert FLUX.2 DiT Linear layers to mh_cute_ops int8.")
    parser.add_argument("--linear_chunk_size", type=int, default=64)
    parser.add_argument("--prompt", default=None, help="Override prompt. Default is built from keys.")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--height", type=int, default=2048, help="Single-pass output height.")
    parser.add_argument("--width", type=int, default=0, help="Single-pass output width. 0 = height * output panel count.")
    parser.add_argument("--num_inference_steps", type=int, default=8)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--embedded_guidance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multidiffusion", action="store_true")
    parser.add_argument("--md_height", type=int, default=2048)
    parser.add_argument("--md_width", type=int, default=0, help="0 = md_height * output panel count.")
    parser.add_argument("--window_size", type=int, default=0, help="0 = 1024 for flux2, 2048 for flux2_2048.")
    parser.add_argument("--stride", type=int, default=0, help="0 = window_size // 2.")
    parser.add_argument("--no_circular", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_full", action="store_true", default=True)
    parser.add_argument("--no_save_full", dest="save_full", action="store_false")
    return parser.parse_args()


def normalize_user_key(raw: str) -> str:
    key = USER_KEY_ALIASES.get(raw.strip().lower())
    if key is None:
        raise ValueError(f"Unknown key {raw!r}. Supported: {sorted(KEY_INFO)}")
    return key


def parse_keys_arg(raw: str) -> List[str]:
    keys = [normalize_user_key(x) for x in raw.split(",") if x.strip()]
    if not keys:
        raise ValueError("keys cannot be empty")
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate keys are not allowed: {keys}")
    return keys


def pack_keys_into_panels(keys: List[str]) -> List[PanelSpec]:
    panels: List[PanelSpec] = []
    current: PanelSpec = []
    used = 0
    for key in keys:
        n = int(KEY_INFO[key]["channels"])
        if used + n > 3:
            panels.append(current)
            current = []
            used = 0
        current.append((key, n, used))
        used += n
    if current:
        panels.append(current)
    return panels


def panel_layout_str(panel: PanelSpec) -> str:
    fills = ["0", "0", "0"]
    for key, n, off in panel:
        if n == 1:
            fills[off] = key
        else:
            for i in range(n):
                fills[off + i] = f"{key}.{['R', 'G', 'B'][i]}"
    return " ".join(f"{band}={val}" for band, val in zip(("R", "G", "B"), fills))


def build_prompt(input_panels: List[PanelSpec], output_panels: List[PanelSpec], category: Optional[str]) -> str:
    def desc(panels: List[PanelSpec]) -> str:
        parts = []
        for i, panel in enumerate(panels):
            keys_desc = " + ".join(KEY_DESCRIPTION[k] for k, _, _ in panel)
            parts.append(f"panel {i + 1} ({panel_layout_str(panel)}): {keys_desc}")
        return "; ".join(parts)

    base = (
        f"Stylized PBR material maps. Reference inputs - {desc(input_panels)}. "
        f"Generate the corresponding target - {desc(output_panels)}."
    )
    if category and category in CATEGORY_PROMPTS:
        return f"{base} Material category: {category} ({CATEGORY_PROMPTS[category]})."
    return base


def scan_samples(root: str) -> Dict[str, Dict[str, Tuple[str, str]]]:
    samples: Dict[str, Dict[str, Tuple[str, str]]] = defaultdict(dict)
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in sorted(filenames):
            match = FILENAME_RE.match(fname)
            if match is None:
                continue
            src_key = SOURCE_FILE_ALIASES.get(match.group("key").lower())
            if src_key is None:
                continue
            full_path = os.path.join(dirpath, fname)
            rel_dir = os.path.relpath(dirpath, root)
            category = rel_dir.split(os.sep)[0] if rel_dir != "." else ""
            samples[match.group("base")][src_key] = (full_path, category)
    return samples


def required_sources(keys: Iterable[str]) -> List[str]:
    out: List[str] = []
    for key in keys:
        src = KEY_INFO[key]["source"]
        if src not in out:
            out.append(src)
    return out


def load_source_image(path: str, size: Tuple[int, int]) -> Image.Image:
    image = Image.open(path)
    if image.size != size:
        image = image.resize(size, Image.LANCZOS)
    return image


def key_array(files: Dict[str, Tuple[str, str]], key: str, size: Tuple[int, int]) -> np.ndarray:
    info = KEY_INFO[key]
    src = info["source"]
    if src not in files:
        raise FileNotFoundError(f"Missing source {src} for key {key}")
    image = load_source_image(files[src][0], size)
    if info["extract"] is None:
        return np.array(image.convert("RGB"), dtype=np.uint8)
    arr = np.array(image.convert("RGBA"), dtype=np.uint8)
    channel = int(info["extract"])
    return arr[:, :, channel:channel + 1]


def make_panel_image(files: Dict[str, Tuple[str, str]], panel: PanelSpec, size: Tuple[int, int]) -> Image.Image:
    canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for key, n, off in panel:
        arr = key_array(files, key, size)
        canvas[:, :, off:off + n] = arr[:, :, :n]
    return Image.fromarray(canvas, mode="RGB")


def split_horizontal_panels(image: Image.Image, n_panels: int) -> List[Image.Image]:
    if n_panels == 1:
        return [image]
    width, height = image.size
    if width % n_panels != 0:
        raise ValueError(f"Output width {width} is not divisible by panel count {n_panels}")
    panel_w = width // n_panels
    return [image.crop((i * panel_w, 0, (i + 1) * panel_w, height)) for i in range(n_panels)]


def save_key_outputs(panel_images: List[Image.Image], output_panels: List[PanelSpec], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for panel_image, panel in zip(panel_images, output_panels):
        arr = np.array(panel_image.convert("RGB"), dtype=np.uint8)
        for key, n, off in panel:
            chunk = arr[:, :, off:off + n]
            if n == 3:
                Image.fromarray(chunk, mode="RGB").save(os.path.join(out_dir, f"pred_{key}.png"))
            else:
                Image.fromarray(chunk[:, :, 0], mode="L").save(os.path.join(out_dir, f"pred_{key}.png"))


def load_flux2_pipeline(args: argparse.Namespace):
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = args.model_base_path
    os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = args.download_source
    os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = args.attn_backend
    os.makedirs(args.model_base_path, exist_ok=True)

    import torch
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=dtype_map[args.dtype],
        device=args.device,
        model_configs=[
            ModelConfig(model_id=args.support_model_id, origin_file_pattern="text_encoder/*.safetensors"),
            ModelConfig(model_id=args.base_model_id, origin_file_pattern="transformer/*.safetensors"),
            ModelConfig(model_id=args.support_model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id=args.support_model_id, origin_file_pattern="tokenizer/"),
    )
    pipe.load_lora(pipe.dit, args.lora_path)
    if args.inference:
        pipe.enable_inference(
            enable_linear=args.quant,
            linear_int8=True,
            linear_chunk_size=args.linear_chunk_size,
        )
    return pipe


def sample_done(out_dir: str, output_keys: List[str]) -> bool:
    return all(os.path.isfile(os.path.join(out_dir, f"pred_{key}.png")) for key in output_keys)


def main() -> int:
    args = parse_args()
    if args.quant and args.dtype != "fp16":
        raise ValueError("--quant uses mh_cute_ops CuteInferLinear, which requires --dtype fp16.")
    input_keys = parse_keys_arg(args.input_keys)
    output_keys = parse_keys_arg(args.output_keys)
    input_panels = pack_keys_into_panels(input_keys)
    output_panels = pack_keys_into_panels(output_keys)

    out_panel_count = len(output_panels)
    single_width = args.width or args.height * out_panel_count
    md_width = args.md_width or args.md_height * out_panel_count
    native_window = 1024 if args.backbone == "flux2" else 2048
    window_size = args.window_size or native_window
    stride = args.stride or window_size // 2

    all_samples = scan_samples(args.test_image_dir)
    needed = set(required_sources(input_keys))
    samples = [(base, files) for base, files in sorted(all_samples.items()) if needed.issubset(files.keys())]
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    if not samples:
        raise RuntimeError(f"No valid samples found in {args.test_image_dir}; required sources: {sorted(needed)}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[info] samples={len(samples)} input_keys={input_keys} output_keys={output_keys}")
    print(f"[info] lora={args.lora_path}")
    print(f"[info] multidiffusion={args.multidiffusion} window={window_size} stride={stride}")

    pipe = load_flux2_pipeline(args)

    if args.multidiffusion:
        from multidiffusion import multidiffusion_infer

    for index, (base, files) in enumerate(samples, start=1):
        out_dir = os.path.join(args.output_dir, base)
        if not args.overwrite and sample_done(out_dir, output_keys):
            print(f"[skip] {base}")
            continue
        os.makedirs(out_dir, exist_ok=True)

        category = next((cat for _path, cat in files.values() if cat), "")
        prompt = args.prompt or build_prompt(input_panels, output_panels, category)
        ref_size = (md_width, args.md_height) if args.multidiffusion else (single_width, args.height)
        edit_images = [make_panel_image(files, panel, ref_size) for panel in input_panels]
        edit_arg = edit_images[0] if len(edit_images) == 1 else edit_images

        print(f"[{index}/{len(samples)}] {base} prompt={prompt!r}")
        if args.multidiffusion:
            pred = multidiffusion_infer(
                pipe,
                args.backbone,
                edit_image=edit_arg,
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=args.md_height,
                width=md_width,
                window_size=window_size,
                stride=stride,
                circular=not args.no_circular,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                embedded_guidance=args.embedded_guidance,
                seed=args.seed,
            )
        else:
            pred = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                edit_image=edit_arg,
                edit_image_auto_resize=False,
                height=args.height,
                width=single_width,
                seed=args.seed,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                embedded_guidance=args.embedded_guidance,
            )

        if args.save_full:
            pred.save(os.path.join(out_dir, "pred_full_rgb.png"))
            for i, image in enumerate(edit_images):
                image.save(os.path.join(out_dir, f"input_panel{i:02d}.png"))
        panel_images = split_horizontal_panels(pred, len(output_panels))
        save_key_outputs(panel_images, output_panels, out_dir)
        print(f"[ok] {base} -> {out_dir}")

    print(f"[done] results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
