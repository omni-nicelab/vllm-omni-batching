from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def build_prompt_payload(prompt: str, negative_prompt: str | None = None) -> dict[str, str]:
    payload = {"prompt": prompt}
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    return payload


def extract_images(outputs: list[Any]) -> list[Image.Image]:
    if not outputs:
        raise ValueError("No outputs were produced.")

    for output in outputs:
        images = getattr(output, "images", None)
        if images:
            return images

        request_output = getattr(output, "request_output", None)
        if request_output is None:
            continue

        stage_outputs = request_output if isinstance(request_output, list) else [request_output]
        for stage_output in stage_outputs:
            images = getattr(stage_output, "images", None)
            if images:
                return images

    raise ValueError(f"No images found in outputs: {outputs!r}")


def save_images(images: list[Image.Image], output_path: str) -> list[str]:
    if not images:
        raise ValueError("No images to save.")

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    suffix = output.suffix or ".png"
    stem = output.stem or "qwen_image_output"
    saved_paths: list[str] = []

    if len(images) == 1:
        images[0].save(output)
        return [str(output)]

    for idx, image in enumerate(images):
        save_path = output.parent / f"{stem}_{idx}{suffix}"
        image.save(save_path)
        saved_paths.append(str(save_path))
    return saved_paths


def summarize_generation_times(times_s: list[float]) -> dict[str, Any]:
    if not times_s:
        raise ValueError("times_s must be non-empty.")

    return {
        "count": len(times_s),
        "times_s": [round(t, 6) for t in times_s],
        "avg_s": round(sum(times_s) / len(times_s), 6),
        "min_s": round(min(times_s), 6),
        "max_s": round(max(times_s), 6),
    }


def _stage_clients(omni: Any) -> list[Any]:
    if hasattr(omni, "stage_list"):
        return list(getattr(omni, "stage_list") or [])
    engine = getattr(omni, "engine", None)
    return list(getattr(engine, "stage_clients", []) or [])


def collect_stage_summaries(omni: Any) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []

    for stage in _stage_clients(omni):
        engine_outputs = getattr(stage, "engine_outputs", None)
        outputs_list = engine_outputs if isinstance(engine_outputs, list) else [engine_outputs]
        outputs_list = [output for output in outputs_list if output is not None]

        multimodal_keys: list[str] = []
        if outputs_list:
            multimodal_output = getattr(outputs_list[0], "multimodal_output", None)
            if isinstance(multimodal_output, dict):
                multimodal_keys = sorted(multimodal_output.keys())

        summaries.append(
            {
                "stage_id": getattr(stage, "stage_id", None),
                "stage_type": getattr(stage, "stage_type", None),
                "model_stage": getattr(stage, "model_stage", None),
                "final_output": bool(getattr(stage, "final_output", False)),
                "final_output_type": getattr(stage, "final_output_type", None),
                "has_engine_outputs": bool(outputs_list),
                "multimodal_keys": multimodal_keys,
            }
        )

    return summaries


def print_stage_summaries(stage_summaries: list[dict[str, Any]]) -> None:
    if not stage_summaries:
        return

    print("Stage summary:")
    for summary in stage_summaries:
        mm_keys = summary["multimodal_keys"] or []
        mm_display = ", ".join(mm_keys) if mm_keys else "-"
        print(
            "  "
            f"stage={summary['stage_id']} "
            f"type={summary['stage_type']} "
            f"model_stage={summary['model_stage']} "
            f"final={summary['final_output']} "
            f"final_output_type={summary['final_output_type']} "
            f"has_engine_outputs={summary['has_engine_outputs']} "
            f"multimodal_keys=[{mm_display}]"
        )


def image_diff_stats(lhs: Image.Image, rhs: Image.Image) -> dict[str, float]:
    lhs_rgb = np.asarray(lhs.convert("RGB"), dtype=np.float32) / 255.0
    rhs_rgb = np.asarray(rhs.convert("RGB"), dtype=np.float32) / 255.0
    if lhs_rgb.shape != rhs_rgb.shape:
        raise ValueError(f"Image shapes differ: {lhs_rgb.shape} vs {rhs_rgb.shape}")

    abs_diff = np.abs(lhs_rgb - rhs_rgb)
    return {
        "mean_abs_diff": float(abs_diff.mean()),
        "max_abs_diff": float(abs_diff.max()),
    }


def write_report_json(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return

    report_path = Path(path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cleanup_runtime() -> None:
    gc.collect()

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return
    except Exception:
        pass

    try:
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()
            torch.npu.synchronize()
            return
    except Exception:
        pass

    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
    except Exception:
        pass
