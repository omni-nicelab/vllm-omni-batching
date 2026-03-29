#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/diffusion_benchmark_serving.py"

# Edit these values directly.
MODEL="Qwen/Qwen-Image"
BASE_URL="http://127.0.0.1:8091"
TASK="t2i"
DATASET="random"
RESOLUTION=512
NUM_PROMPTS=20
NUM_INFERENCE_STEPS=20
WARMUP_REQUESTS=2
WARMUP_NUM_INFERENCE_STEPS=20
MAX_CONCURRENCY=2
REQUEST_RATE="inf"
OUTPUT_FILE="${SCRIPT_DIR}/qwen_image_baching_${RESOLUTION}_c${MAX_CONCURRENCY}.json"

CMD=(
    python3 "${BENCHMARK_SCRIPT}"
    --base-url "${BASE_URL}"
    --model "${MODEL}"
    --task "${TASK}"
    --dataset "${DATASET}"
    --num-prompts "${NUM_PROMPTS}"
    --width "${RESOLUTION}"
    --height "${RESOLUTION}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    --warmup-requests "${WARMUP_REQUESTS}"
    --warmup-num-inference-steps "${WARMUP_NUM_INFERENCE_STEPS}"
    --max-concurrency "${MAX_CONCURRENCY}"
    --request-rate "${REQUEST_RATE}"
    --output-file "${OUTPUT_FILE}"
    --disable-tqdm
)

printf 'Running continuous batching benchmark\n'
printf 'CMD: %q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
