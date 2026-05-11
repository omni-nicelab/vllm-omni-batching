"""Build a FLUX.2 validation grid for teacher, SCM, and 1-step DMD checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


DEFAULT_DATASET_DIR = Path(
    "/mnt/upfs/user/yueren.jiang/dataset/smbe_dataset_0428/processed/diffuse_to_normal"
)
DEFAULT_TEACHER_LORA = Path(
    "/mnt/upfs/user/yueren.jiang/checkpoints/full_aug/flux2/"
    "diffuse_to_normal_crop1024_rank64/lora/step-84000.safetensors"
)
DEFAULT_SCM_LORA = Path(
    "/mnt/upfs/jiazhen.wu/wjz/work/pbr-material-edit-dmd/pbr-material-edit/"
    "outputs/flux2_scm_0428_diffuse_to_normal_crop1024_rank64_20260509044224/"
    "step-3699.safetensors"
)
DEFAULT_DMD_LORA = Path(
    "/mnt/upfs/jiazhen.wu/wjz/work/pbr-material-edit-dmd-feat-tiling/"
    "outputs/flux2_dmd_1step_cfg_0428_diffuse_to_normal_crop1024_rank64_"
    "20260509205629_continue100_from_step-3699/step-7399.safetensors"
)
DEFAULT_MODEL_BASE = Path("/mnt/upfs/user/yueren.jiang/models")


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    role: str
    steps: int
    cfg_scale: float


VARIANTS = [
    Variant("teacher_30step_cfg4", "teacher 30-step cfg4", "teacher", 30, 4.0),
    Variant("teacher_4step_cfg4", "teacher 4-step cfg4", "teacher", 4, 4.0),
    Variant("teacher_1step_cfg4", "teacher 1-step cfg4", "teacher", 1, 4.0),
    Variant("teacher_1step_cfg1", "teacher 1-step cfg1", "teacher", 1, 1.0),
    Variant("scm_1step_cfg1", "scm 1-step cfg1", "scm", 1, 1.0),
    Variant("scm_1step_cfg4", "scm 1-step cfg4", "scm", 1, 4.0),
    Variant("dmd_1step_cfg4", "dmd 1-step cfg4", "dmd", 1, 4.0),
    Variant("dmd_1step_cfg1", "dmd 1-step cfg1", "dmd", 1, 1.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--teacher_lora", type=Path, default=DEFAULT_TEACHER_LORA)
    parser.add_argument("--scm_lora", type=Path, default=DEFAULT_SCM_LORA)
    parser.add_argument("--dmd_lora", type=Path, default=DEFAULT_DMD_LORA)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--model_base_path", type=Path, default=DEFAULT_MODEL_BASE)
    parser.add_argument("--download_source", default="modelscope")
    parser.add_argument("--base_model_id", default="black-forest-labs/FLUX.2-klein-base-9B")
    parser.add_argument("--support_model_id", default="black-forest-labs/FLUX.2-klein-9B")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedded_guidance", type=float, default=4.0)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max_tile_height", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def safe_sample_name(entry: dict, index: int) -> str:
    stem = Path(entry["image"]).stem
    stem = stem.replace("__out0__Normal", "").replace("__out0", "").replace("__out", "_out")
    return f"{index:02d}_{stem}"


def resolve_dataset_path(dataset_dir: Path, value) -> Path:
    if isinstance(value, list):
        value = value[0]
    path = Path(value)
    return path if path.is_absolute() else dataset_dir / path


def load_plan(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    config_path = args.dataset_dir / "dataset_config.json"
    metadata_path = args.dataset_dir / "val.json"
    require_file(config_path, "dataset_config.json")
    require_file(metadata_path, "val.json")
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    with open(metadata_path, "r", encoding="utf-8") as file:
        samples = json.load(file)
    if args.num_samples > 0:
        samples = samples[: args.num_samples]
    return config, samples


def torch_dtype(dtype_name: str):
    import torch

    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype_name]


def load_flux2_pipeline(args: argparse.Namespace, lora_path: Path):
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(args.model_base_path)
    os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = args.download_source
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch_dtype(args.dtype),
        device=args.device,
        model_configs=[
            ModelConfig(model_id=args.support_model_id, origin_file_pattern="text_encoder/*.safetensors"),
            ModelConfig(model_id=args.base_model_id, origin_file_pattern="transformer/*.safetensors"),
            ModelConfig(model_id=args.support_model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id=args.support_model_id, origin_file_pattern="tokenizer/"),
    )
    print(f"[load] LoRA: {lora_path}", flush=True)
    pipe.load_lora(pipe.dit, str(lora_path))
    return pipe


def unload_pipeline(pipe) -> None:
    try:
        pipe.load_models_to_device([])
    except Exception:
        pass
    del pipe
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def run_pipe(pipe, sample: dict, sample_index: int, args: argparse.Namespace, config: dict, variant: Variant) -> Image.Image:
    edit_path = resolve_dataset_path(args.dataset_dir, sample["edit_image"])
    edit_image = Image.open(edit_path).convert("RGB")
    prompt = sample.get("prompt") or config.get("prompt") or ""
    height = int(config.get("image_height", config.get("resolution_per_panel", 1024)))
    width = int(config.get("image_width", config.get("resolution_per_panel", 1024)))
    return pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        edit_image=edit_image,
        edit_image_auto_resize=False,
        cfg_scale=variant.cfg_scale,
        embedded_guidance=args.embedded_guidance,
        height=height,
        width=width,
        seed=args.seed + sample_index,
        rand_device="cpu",
        num_inference_steps=variant.steps,
        progress_bar_cmd=lambda x: x,
    ).convert("RGB")


def generate_role(
    args: argparse.Namespace,
    config: dict,
    samples: list[dict],
    *,
    role: str,
    lora_path: Path,
    variants: Iterable[Variant],
) -> None:
    role_variants = list(variants)
    if not role_variants:
        return
    print(f"[role] {role}: {len(role_variants)} variants", flush=True)
    pipe = load_flux2_pipeline(args, lora_path)
    try:
        for sample_index, sample in enumerate(samples):
            sample_name = safe_sample_name(sample, sample_index)
            sample_dir = args.output_dir / sample_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            for variant in role_variants:
                out_path = sample_dir / f"{variant.name}_Normal.png"
                if out_path.is_file() and not args.overwrite:
                    print(f"[skip] {sample_name} {variant.name}", flush=True)
                    continue
                image = run_pipe(pipe, sample, sample_index, args, config, variant)
                image.save(out_path, format="PNG", compress_level=3)
                print(
                    f"[ok] {sample_name} {variant.name} "
                    f"steps={variant.steps} cfg={variant.cfg_scale:g}",
                    flush=True,
                )
    finally:
        unload_pipeline(pipe)


def copy_gt(args: argparse.Namespace, samples: list[dict]) -> None:
    for sample_index, sample in enumerate(samples):
        sample_name = safe_sample_name(sample, sample_index)
        sample_dir = args.output_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        gt_path = resolve_dataset_path(args.dataset_dir, sample["image"])
        edit_path = resolve_dataset_path(args.dataset_dir, sample["edit_image"])
        require_file(gt_path, "GT image")
        require_file(edit_path, "edit image")
        Image.open(gt_path).convert("RGB").save(sample_dir / "gt_Normal.png", format="PNG", compress_level=3)
        Image.open(edit_path).convert("RGB").save(sample_dir / "input_Diffuse.png", format="PNG", compress_level=3)


def font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def title_strip(width: int, text: str, size: int = 18) -> Image.Image:
    strip = Image.new("RGB", (width, 32), (22, 22, 22))
    draw = ImageDraw.Draw(strip)
    draw.text((8, 5), text, fill=(240, 240, 240), font=font(size))
    return strip


def tile(label: str, image: Image.Image, max_height: int) -> Image.Image:
    image = image.convert("RGB")
    if max_height > 0 and image.height > max_height:
        width = round(image.width * max_height / image.height)
        image = image.resize((width, max_height), Image.LANCZOS)
    label_image = title_strip(image.width, label, size=16)
    out = Image.new("RGB", (image.width, label_image.height + image.height), (35, 35, 35))
    out.paste(label_image, (0, 0))
    out.paste(image, (0, label_image.height))
    return out


def make_row(args: argparse.Namespace, sample: dict, sample_index: int) -> Image.Image:
    sample_name = safe_sample_name(sample, sample_index)
    sample_dir = args.output_dir / sample_name
    columns = [("GT", "gt_Normal.png")] + [(variant.label, f"{variant.name}_Normal.png") for variant in VARIANTS]
    tiles = []
    for label, filename in columns:
        path = sample_dir / filename
        require_file(path, f"{sample_name} {filename}")
        tiles.append(tile(label, Image.open(path), args.max_tile_height))
    gap = 6
    row_w = sum(item.width for item in tiles) + gap * (len(tiles) - 1)
    row_h = max(item.height for item in tiles)
    row = Image.new("RGB", (row_w, row_h), (42, 42, 42))
    x = 0
    for item in tiles:
        row.paste(item, (x, 0))
        x += item.width + gap
    header = title_strip(row.width, sample_name, size=18)
    out = Image.new("RGB", (row.width, header.height + row.height), (28, 28, 28))
    out.paste(header, (0, 0))
    out.paste(row, (0, header.height))
    return out


def build_contact_sheet(args: argparse.Namespace, samples: list[dict]) -> Path:
    rows = [make_row(args, sample, idx) for idx, sample in enumerate(samples)]
    gap = 10
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + gap * (len(rows) - 1)
    sheet = Image.new("RGB", (width, height), (30, 30, 30))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + gap
    out_path = args.output_dir / "ALL_samples_teacher_scm_dmd_1step_contact_sheet.png"
    sheet.save(out_path, format="PNG", compress_level=3)
    return out_path


def main() -> int:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = Path("outputs") / (
            f"flux2_teacher_scm_dmd_1step_grid_seed{args.seed}"
        )
    args.output_dir = args.output_dir.resolve()

    require_file(args.teacher_lora, "teacher LoRA")
    require_file(args.scm_lora, "SCM LoRA")
    require_file(args.dmd_lora, "DMD LoRA")
    config, samples = load_plan(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plan] dataset={args.dataset_dir}", flush=True)
    print(f"[plan] samples={len(samples)} output={args.output_dir}", flush=True)
    copy_gt(args, samples)

    role_to_lora = {
        "teacher": args.teacher_lora,
        "scm": args.scm_lora,
        "dmd": args.dmd_lora,
    }
    for role, lora_path in role_to_lora.items():
        generate_role(
            args,
            config,
            samples,
            role=role,
            lora_path=lora_path,
            variants=[variant for variant in VARIANTS if variant.role == role],
        )

    sheet_path = build_contact_sheet(args, samples)
    print(f"[done] contact sheet: {sheet_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
