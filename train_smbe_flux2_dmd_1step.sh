#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Fixed 0428 diffuse_to_normal FLUX2 1-step CFG-DMD run.
READONLY_ROOT="/mnt/upfs/user/yueren.jiang"
DATA_DIR="${READONLY_ROOT}/dataset/smbe_dataset_0428/processed/diffuse_to_normal"
TEACHER_LORA_PATH="${READONLY_ROOT}/checkpoints/full_aug/flux2/diffuse_to_normal_crop1024_rank64/lora/step-84000.safetensors"
STUDENT_INIT_LORA_PATH="${STUDENT_INIT_LORA_PATH:-/mnt/upfs/jiazhen.wu/wjz/work/tilling_generation/checkpoints/dmd2_full_aug_0428_bs1_rescale2_resume_e8/flux2/diffuse_to_normal_crop1024_rank64/lora/step-1299.safetensors}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d%H%M%S)}"
OUTPUT_PATH="${OUTPUT_PATH:-${ROOT_DIR}/outputs/flux2_dmd_1step_cfg_0428_diffuse_to_normal_crop1024_rank64_${RUN_ID}}"

export DIFFSYNTH_MODEL_BASE_PATH="${READONLY_ROOT}/models"
export DIFFSYNTH_DOWNLOAD_SOURCE="modelscope"
export DIFFSYNTH_SKIP_DOWNLOAD="true"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"
export WANDB_MODE="${WANDB_MODE:-disabled}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR_VALUE="${MASTER_ADDR_VALUE:-127.0.0.1}"
MASTER_PORT_VALUE="${MASTER_PORT_VALUE:-29501}"

BASE_MODEL_ID="black-forest-labs/FLUX.2-klein-base-9B"
SUPPORT_MODEL_ID="black-forest-labs/FLUX.2-klein-9B"
MODEL_ID_PATHS="${SUPPORT_MODEL_ID}:text_encoder/*.safetensors,${BASE_MODEL_ID}:transformer/*.safetensors,${SUPPORT_MODEL_ID}:vae/diffusion_pytorch_model.safetensors"
TOKENIZER_PATH="${SUPPORT_MODEL_ID}:tokenizer/"

CROP_SIZE=1024
MAX_PIXELS=2097152
LORA_RANK=64
NUM_EPOCHS="${NUM_EPOCHS:-50}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-0}"
SAVE_STEPS="${SAVE_STEPS:-20}"
LOG_STEPS="${LOG_STEPS:-10}"
VAL_STEPS="${VAL_STEPS:-100}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LEARNING_RATE_GEN="${LEARNING_RATE_GEN:-3e-6}"
LEARNING_RATE_GUIDANCE="${LEARNING_RATE_GUIDANCE:-1e-5}"
GENERATOR_UPDATE_FREQ="${GENERATOR_UPDATE_FREQ:-10}"
DMD_GUIDANCE_SCALE="${DMD_GUIDANCE_SCALE:-4.0}"
DMD_EMBEDDED_GUIDANCE="${DMD_EMBEDDED_GUIDANCE:-4.0}"
DMD2_GENERATOR_OBJECTIVE="${DMD2_GENERATOR_OBJECTIVE:-decoupled_hybrid}"
CFG_AUX_LOSS_WEIGHT="${CFG_AUX_LOSS_WEIGHT:-0.1}"
CA_LOSS_WEIGHT="${CA_LOSS_WEIGHT:-1.0}"
CA_TIMESTEP_MAX_RATIO="${CA_TIMESTEP_MAX_RATIO:-1.0}"
GRAD_CLIP_VALUE="${GRAD_CLIP_VALUE:-0.0}"
TIMESTEP_SAMPLING_STRATEGY="${TIMESTEP_SAMPLING_STRATEGY:-logit_normal}"
LOGIT_MEAN="${LOGIT_MEAN:-1.0}"
LOGIT_STD="${LOGIT_STD:-1.0}"
DYNAMIC_RESCALE_T_STEPS="${DYNAMIC_RESCALE_T_STEPS:-500}"
RESCALE_T_VAL="${RESCALE_T_VAL:-2.0}"
STUDENT_SCHEDULE="${STUDENT_SCHEDULE:-linear}"
CFG_DISTILL="${CFG_DISTILL:-1}"
ONE_STEP_CONDITIONING_TIMESTEP="${ONE_STEP_CONDITIONING_TIMESTEP:-}"
SEED="${SEED:-42}"

LORA_TARGET_MODULES="to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj"
for i in $(seq 0 23); do
  LORA_TARGET_MODULES+=",single_transformer_blocks.${i}.attn.to_out"
done

DEEPSPEED_ARGS=()
if [[ -f "${ROOT_DIR}/ds_config.json" ]]; then
  DEEPSPEED_ARGS=(--deepspeed_config_path "${ROOT_DIR}/ds_config.json")
fi
CFG_DISTILL_ARGS=()
if [[ "${CFG_DISTILL}" == "0" || "${CFG_DISTILL,,}" == "false" || "${CFG_DISTILL,,}" == "off" ]]; then
  CFG_DISTILL_ARGS=(--disable_cfg_distill)
fi
ONE_STEP_ARGS=()
if [[ -n "${ONE_STEP_CONDITIONING_TIMESTEP}" ]]; then
  ONE_STEP_ARGS=(--one_step_conditioning_timestep "${ONE_STEP_CONDITIONING_TIMESTEP}")
fi

[[ -d "${DATA_DIR}" ]] || { echo "ERROR: DATA_DIR does not exist: ${DATA_DIR}" >&2; exit 1; }
[[ -f "${DATA_DIR}/train.json" ]] || { echo "ERROR: train.json does not exist: ${DATA_DIR}/train.json" >&2; exit 1; }
[[ -f "${DATA_DIR}/val.json" ]] || { echo "ERROR: val.json does not exist: ${DATA_DIR}/val.json" >&2; exit 1; }
[[ -f "${TEACHER_LORA_PATH}" ]] || { echo "ERROR: TEACHER_LORA_PATH does not exist: ${TEACHER_LORA_PATH}" >&2; exit 1; }
[[ -f "${STUDENT_INIT_LORA_PATH}" ]] || { echo "ERROR: STUDENT_INIT_LORA_PATH does not exist: ${STUDENT_INIT_LORA_PATH}" >&2; exit 1; }

echo "----------------------------------------------------------------"
echo "FLUX.2 1-step CFG-DMD LoRA distillation: 0428 diffuse_to_normal"
echo "DATA_DIR: ${DATA_DIR}"
echo "TEACHER_LORA_PATH: ${TEACHER_LORA_PATH}"
echo "STUDENT_INIT_LORA_PATH: ${STUDENT_INIT_LORA_PATH}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "GPUS: ${NPROC_PER_NODE}"
echo "NUM_EPOCHS: ${NUM_EPOCHS}"
echo "NUM_DENOISING_STEPS: 1"
echo "GENERATOR_UPDATE_FREQ: ${GENERATOR_UPDATE_FREQ}"
echo "LR_GEN/GUIDANCE: ${LEARNING_RATE_GEN}/${LEARNING_RATE_GUIDANCE}"
echo "DMD2_GENERATOR_OBJECTIVE: ${DMD2_GENERATOR_OBJECTIVE}"
echo "STUDENT_SCHEDULE: ${STUDENT_SCHEDULE}"
echo "CFG_DISTILL: ${CFG_DISTILL}"
echo "CFG_AUX_LOSS_WEIGHT: ${CFG_AUX_LOSS_WEIGHT}"
echo "CA_TIMESTEP_MAX_RATIO: ${CA_TIMESTEP_MAX_RATIO}"
echo "ONE_STEP_CONDITIONING_TIMESTEP: ${ONE_STEP_CONDITIONING_TIMESTEP:-<student_schedule>}"
echo "GRAD_CLIP_VALUE: ${GRAD_CLIP_VALUE}"
echo "AUGMENT: random circular crop + tileable rotate/flip"
echo "----------------------------------------------------------------"

mkdir -p "${OUTPUT_PATH}"

PYTHONPATH=. torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR_VALUE}" \
  --master_port="${MASTER_PORT_VALUE}" \
  examples/flux2/model_training/train_dmd_1step.py \
  --dataset_base_path "${DATA_DIR}" \
  --dataset_metadata_path "${DATA_DIR}/train.json" \
  --val_dataset_metadata_path "${DATA_DIR}/val.json" \
  --val_steps "${VAL_STEPS}" \
  --max_val_batches "${MAX_VAL_BATCHES}" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image,edit_image_auto_resize" \
  --max_pixels "${MAX_PIXELS}" \
  --dataset_repeat "${DATASET_REPEAT}" \
  --dataset_num_workers "${DATASET_NUM_WORKERS}" \
  --model_id_with_origin_paths "${MODEL_ID_PATHS}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --lora_base_model "dit" \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --lora_rank "${LORA_RANK}" \
  --lora_checkpoint "${TEACHER_LORA_PATH}" \
  --student_lora_checkpoint "${STUDENT_INIT_LORA_PATH}" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --learning_rate "${LEARNING_RATE}" \
  --learning_rate_gen "${LEARNING_RATE_GEN}" \
  --learning_rate_guidance "${LEARNING_RATE_GUIDANCE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --log_steps "${LOG_STEPS}" \
  --num_denoising_steps 1 \
  --generator_update_freq "${GENERATOR_UPDATE_FREQ}" \
  --dmd2_guidance_scale "${DMD_GUIDANCE_SCALE}" \
  --dmd2_embedded_guidance "${DMD_EMBEDDED_GUIDANCE}" \
  --dmd2_generator_objective "${DMD2_GENERATOR_OBJECTIVE}" \
  --cfg_aux_loss_weight "${CFG_AUX_LOSS_WEIGHT}" \
  --ca_loss_weight "${CA_LOSS_WEIGHT}" \
  --ca_timestep_schedule focused \
  --ca_timestep_max_ratio "${CA_TIMESTEP_MAX_RATIO}" \
  --grad_clip_value "${GRAD_CLIP_VALUE}" \
  --student_schedule "${STUDENT_SCHEDULE}" \
  --timestep_sampling_strategy "${TIMESTEP_SAMPLING_STRATEGY}" \
  --logit_mean "${LOGIT_MEAN}" \
  --logit_std "${LOGIT_STD}" \
  --dynamic_rescale_t_steps "${DYNAMIC_RESCALE_T_STEPS}" \
  --rescale_t_val "${RESCALE_T_VAL}" \
  --seed "${SEED}" \
  --backward_simulation \
  --random_circular_crop \
  --crop_size "${CROP_SIZE}" \
  --augment_tileable \
  --augment_prob_original "0.5" \
  "${CFG_DISTILL_ARGS[@]}" \
  "${ONE_STEP_ARGS[@]}" \
  "${DEEPSPEED_ARGS[@]}"
