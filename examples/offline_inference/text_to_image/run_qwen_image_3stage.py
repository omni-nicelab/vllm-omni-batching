"""Offline runner for Qwen-Image disaggregated VAE execution."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qwen_image_offline_utils import (  # noqa: E402
    build_prompt_payload,
    cleanup_runtime,
    collect_stage_summaries,
    extract_images,
    print_stage_summaries,
    save_images,
    summarize_generation_times,
    write_report_json,
)
from vllm_omni.entrypoints.omni import Omni  # noqa: E402

DEFAULT_STAGE_YAML = "vllm_omni/model_executor/stage_configs/qwen_image_3stage.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen-Image with encode -> denoise -> decode stages.")
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
    parser.add_argument("--output", default="outputs/qwen_image_3stage.png")
    parser.add_argument("--init-timeout", type=int, default=900)
    parser.add_argument("--batch-timeout", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--report-json", default=None)
    return parser.parse_args()


def build_stage_sampling_params(omni: Omni, args: argparse.Namespace) -> list[Any]:
    params_list = [params.clone() for params in omni.default_sampling_params_list]
    for params in params_list:
        params.seed = args.seed
        params.height = args.height
        params.width = args.width
        params.num_inference_steps = args.steps
        params.true_cfg_scale = args.cfg_scale
        params.guidance_scale = args.guidance_scale
        params.guidance_scale_provided = True
        params.num_outputs_per_prompt = 1
    return params_list


def run_once(omni: Omni, args: argparse.Namespace) -> tuple[list[Any], float]:
    prompts = [build_prompt_payload(args.prompt, args.negative_prompt) for _ in range(args.batch_size)]
    sampling_params_list = build_stage_sampling_params(omni, args)

    t0 = time.perf_counter()
    outputs = list(omni.generate(prompts=prompts, sampling_params_list=sampling_params_list, use_tqdm=False))
    return outputs, time.perf_counter() - t0


def main() -> None:
    args = parse_args()

    print("Qwen-Image 3-stage disaggregated VAE")
    print(
        f"model={args.model} stage_config={args.stage_configs_path} "
        f"steps={args.steps} size={args.width}x{args.height} batch_size={args.batch_size} "
        f"warmup={args.warmup} repeat={args.repeat}"
    )

    omni: Omni | None = None
    outputs: list[Any] | None = None
    generation_times: list[float] = []
    init_s = 0.0

    try:
        init_t0 = time.perf_counter()
        omni = Omni(
            model=args.model,
            stage_configs_path=args.stage_configs_path,
            init_timeout=args.init_timeout,
            batch_timeout=args.batch_timeout,
            worker_backend="multi_process",
            diffusion_batch_size=args.batch_size,
        )
        init_s = time.perf_counter() - init_t0
        print(f"[init] 3-stage ready in {init_s:.2f}s")

        for idx in range(args.warmup):
            _, warmup_s = run_once(omni, args)
            print(f"[warmup {idx + 1}/{args.warmup}] generate() = {warmup_s:.2f}s")

        for idx in range(args.repeat):
            outputs, run_s = run_once(omni, args)
            generation_times.append(run_s)
            print(f"[run {idx + 1}/{args.repeat}] generate() = {run_s:.2f}s")

        assert outputs is not None
        images = extract_images(outputs)
        saved_paths = save_images(images, args.output)
        stage_summaries = collect_stage_summaries(omni)
        timing_summary = summarize_generation_times(generation_times)

        print(f"[summary] init={init_s:.2f}s avg_generate={timing_summary['avg_s']:.2f}s")
        for path in saved_paths:
            print(f"[save] {path}")
        print_stage_summaries(stage_summaries)

        write_report_json(
            args.report_json,
            {
                "mode": "3stage",
                "model": args.model,
                "stage_configs_path": args.stage_configs_path,
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
                "seed": args.seed,
                "cfg_scale": args.cfg_scale,
                "guidance_scale": args.guidance_scale,
                "steps": args.steps,
                "height": args.height,
                "width": args.width,
                "batch_size": args.batch_size,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "init_s": round(init_s, 6),
                "generation": timing_summary,
                "saved_paths": saved_paths,
                "stage_summaries": stage_summaries,
            },
        )
    finally:
        if omni is not None:
            omni.close()
        cleanup_runtime()


if __name__ == "__main__":
    main()
