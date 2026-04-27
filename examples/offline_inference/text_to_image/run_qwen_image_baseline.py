"""Offline baseline runner for monolithic Qwen-Image generation."""

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
    extract_images,
    save_images,
    summarize_generation_times,
    write_report_json,
)
from vllm_omni.diffusion.data import DiffusionParallelConfig  # noqa: E402
from vllm_omni.entrypoints.omni import Omni  # noqa: E402
from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run monolithic Qwen-Image baseline inference.")
    parser.add_argument("--model", default="Qwen/Qwen-Image", help="Model name or local path.")
    parser.add_argument("--prompt", default="a photo of an astronaut riding a horse on the moon")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of prompts per generate() call.")
    parser.add_argument("--output", default="outputs/qwen_image_baseline.png")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--cfg-parallel-size", type=int, default=1, choices=[1, 2])
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument("--vae-patch-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--step-execution", action="store_true")
    parser.add_argument("--report-json", default=None)
    return parser.parse_args()


def build_sampling_params(args: argparse.Namespace) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        seed=args.seed,
        height=args.height,
        width=args.width,
        true_cfg_scale=args.cfg_scale,
        guidance_scale=args.guidance_scale,
        guidance_scale_provided=True,
        num_inference_steps=args.steps,
        num_outputs_per_prompt=1,
    )


def run_once(omni: Omni, args: argparse.Namespace) -> tuple[list[Any], float]:
    prompts = [build_prompt_payload(args.prompt, args.negative_prompt) for _ in range(args.batch_size)]
    sampling_params = build_sampling_params(args)

    t0 = time.perf_counter()
    outputs = omni.generate(prompts, sampling_params, use_tqdm=False)
    return outputs, time.perf_counter() - t0


def main() -> None:
    args = parse_args()
    parallel_config = DiffusionParallelConfig(
        tensor_parallel_size=args.tensor_parallel_size,
        cfg_parallel_size=args.cfg_parallel_size,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
    )

    print("Qwen-Image monolithic baseline")
    print(
        f"model={args.model} steps={args.steps} size={args.width}x{args.height} "
        f"batch_size={args.batch_size} step_execution={args.step_execution} "
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
            parallel_config=parallel_config,
            enforce_eager=args.enforce_eager,
            step_execution=args.step_execution,
            diffusion_batch_size=args.batch_size,
            max_num_seqs=args.batch_size,
        )
        init_s = time.perf_counter() - init_t0
        print(f"[init] baseline ready in {init_s:.2f}s")

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
        timing_summary = summarize_generation_times(generation_times)
        print(f"[summary] init={init_s:.2f}s avg_generate={timing_summary['avg_s']:.2f}s")
        for path in saved_paths:
            print(f"[save] {path}")

        write_report_json(
            args.report_json,
            {
                "mode": "baseline",
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
                "init_s": round(init_s, 6),
                "generation": timing_summary,
                "saved_paths": saved_paths,
            },
        )
    finally:
        if omni is not None:
            omni.close()
        cleanup_runtime()


if __name__ == "__main__":
    main()
