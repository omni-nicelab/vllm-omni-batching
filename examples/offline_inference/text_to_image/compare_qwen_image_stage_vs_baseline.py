"""Benchmark Qwen-Image baseline vs 3-stage disaggregated VAE in isolated subprocesses."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qwen_image_offline_utils import image_diff_stats, write_report_json  # noqa: E402

DEFAULT_STAGE_YAML = "vllm_omni/model_executor/stage_configs/qwen_image_3stage.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Qwen-Image baseline and disaggregated VAE performance.")
    parser.add_argument("--model", default="Qwen/Qwen-Image", help="Model name or local path.")
    parser.add_argument("--stage-configs-path", default=DEFAULT_STAGE_YAML)
    parser.add_argument("--prompt", default="a photo of an astronaut riding a horse on the moon")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of prompts per generate() call.")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--step-execution", action="store_true")
    parser.add_argument("--max-stage-ratio", type=float, default=1.1)
    parser.add_argument("--baseline-devices", default="0")
    parser.add_argument("--stage-devices", default="0,1")
    parser.add_argument("--output-dir", default="outputs/qwen_image_disagg_compare")
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--no-enforce", action="store_true", help="Do not fail when the ratio exceeds the threshold.")
    return parser.parse_args()


def _common_args(args: argparse.Namespace) -> list[str]:
    result = [
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--seed",
        str(args.seed),
        "--cfg-scale",
        str(args.cfg_scale),
        "--guidance-scale",
        str(args.guidance_scale),
        "--steps",
        str(args.steps),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--batch-size",
        str(args.batch_size),
        "--repeat",
        str(args.repeat),
        "--warmup",
        str(args.warmup),
    ]
    if args.negative_prompt:
        result.extend(["--negative-prompt", args.negative_prompt])
    return result


def _run_child(cmd: list[str], *, devices: str) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices
    print(f"[subprocess] CUDA_VISIBLE_DEVICES={devices} {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_report_path = output_dir / "baseline_report.json"
    stage_report_path = output_dir / "stage_report.json"
    baseline_output = output_dir / "baseline.png"
    stage_output = output_dir / "stage.png"

    baseline_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_qwen_image_baseline.py"),
        *_common_args(args),
        "--output",
        str(baseline_output),
        "--report-json",
        str(baseline_report_path),
    ]
    if args.step_execution:
        baseline_cmd.append("--step-execution")
    stage_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_qwen_image_3stage.py"),
        *_common_args(args),
        "--stage-configs-path",
        args.stage_configs_path,
        "--output",
        str(stage_output),
        "--report-json",
        str(stage_report_path),
    ]

    _run_child(baseline_cmd, devices=args.baseline_devices)
    _run_child(stage_cmd, devices=args.stage_devices)

    baseline = _load_report(baseline_report_path)
    stage = _load_report(stage_report_path)
    baseline_avg = float(baseline["generation"]["avg_s"])
    stage_avg = float(stage["generation"]["avg_s"])
    stage_ratio = stage_avg / baseline_avg if baseline_avg > 0 else float("inf")

    diff_stats = None
    if baseline_output.exists() and stage_output.exists():
        diff_stats = image_diff_stats(Image.open(baseline_output), Image.open(stage_output))

    print("[compare]")
    print(f"  baseline_avg_generate={baseline_avg:.3f}s")
    print(f"  stage_avg_generate={stage_avg:.3f}s")
    print(f"  stage_over_baseline={stage_ratio:.3f}x")
    print(f"  threshold={args.max_stage_ratio:.3f}x")
    if diff_stats is not None:
        print(f"  image_mean_abs_diff={diff_stats['mean_abs_diff']:.6f}")
        print(f"  image_max_abs_diff={diff_stats['max_abs_diff']:.6f}")

    report = {
        "model": args.model,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "guidance_scale": args.guidance_scale,
        "steps": args.steps,
        "height": args.height,
        "width": args.width,
        "batch_size": args.batch_size,
        "step_execution": args.step_execution,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "baseline_devices": args.baseline_devices,
        "stage_devices": args.stage_devices,
        "baseline": baseline,
        "stage": stage,
        "compare": {
            "stage_over_baseline": round(stage_ratio, 6),
            "max_stage_ratio": args.max_stage_ratio,
            **(diff_stats or {}),
        },
    }
    write_report_json(args.report_json or str(output_dir / "compare_report.json"), report)

    if not args.no_enforce and stage_ratio > args.max_stage_ratio:
        raise SystemExit(
            f"3-stage avg_generate {stage_avg:.3f}s is {stage_ratio:.3f}x baseline "
            f"{baseline_avg:.3f}s, above threshold {args.max_stage_ratio:.3f}x"
        )


if __name__ == "__main__":
    main()
