#!/bin/bash
# DMD2 Distillation Training Script
# Distills the teacher model into a fast few-step generator
export DIFFSYNTH_MODEL_BASE_PATH="/mnt/project/user/lan.chen/pbr/DiffSynth-Studio/models"
export DIFFSYNTH_SKIP_DOWNLOAD=true

DATA_ROOT="/mnt/project/train/MHTexture/User/lan.chen/topo_pbr_valid_score_17544_intersect_ref"
METADATA_PATH="/mnt/project/user/lan.chen/data_jsons/six_view_albedo_v3/train_filtered_high.json"
VAL_METADATA_PATH="/mnt/project/user/lan.chen/data_jsons/six_view_albedo_v3/val.json"

# Teacher LoRA checkpoint (trained SFT model to distill)
MODEL_PATH="/mnt/project/user/lan.chen/models/"
TEACHER_LORA_PATH="/mnt/project/user/lan.chen/outputs/data_albedo_six_view_v2_filtered_high_sp_20260228112539/step-2900.safetensors"

TIMESTAMP=$MLP_ID
OUTPUT_PATH="./outputs/dmd2_distill_${TIMESTAMP}"

# GPUS_PER_NODE=8
# NUM_PROCESSES=$((PET_NNODES * GPUS_PER_NODE))

echo "----------------------------------------------------------------"
echo "PHASE: Starting DMD2 Distillation Training"
# echo "CONFIG: Nodes=${PET_NNODES}, GPUs_Per_Node=${GPUS_PER_NODE}, Total_Processes=${NUM_PROCESSES}"
echo "TEACHER_LORA: ${TEACHER_LORA_PATH}"
echo "OUTPUT: ${OUTPUT_PATH}"
echo "----------------------------------------------------------------"

set -x
PYTHONPATH=. torchrun \
  ${PET_NPROC_PER_NODE:+--nproc_per_node=$PET_NPROC_PER_NODE} \
  ${PET_NNODES:+--nnodes=$PET_NNODES} \
  ${PET_NODE_RANK:+--node_rank=$PET_NODE_RANK} \
  ${MASTER_ADDR:+--master_addr=$MASTER_ADDR} \
  ${MASTER_PORT:+--master_port=$MASTER_PORT} \
  \
  examples/qwen_image/model_training/train_dmd2.py \
  --dataset_base_path "${DATA_ROOT}" \
  --dataset_metadata_path "${METADATA_PATH}" \
  --val_dataset_metadata_path "${VAL_METADATA_PATH}" \
  --val_steps 500 \
  --max_val_batches 10 \
  --num_val_samples 2 \
  --val_inference_steps 4 \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 6291456 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2509:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 64 \
  --lora_checkpoint "${TEACHER_LORA_PATH}" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --save_steps 100 \
  --sp_degree 1 \
  --deepspeed_config_path ds_config.json \
  --remove_prefix_in_ckpt "pipe.dit." \
  --num_epochs 5 \
  --learning_rate 1e-5 \
  --learning_rate_gen 1e-5 \
  --learning_rate_guidance 1e-5 \
  --num_denoising_steps 4 \
  --generator_update_freq 5 \
  --dmd2_guidance_scale 4.0 \
  --timestep_sampling_strategy logit_normal \
  --logit_mean 1.0 \
  --logit_std 1.0 \
  --dynamic_rescale_t_steps 500 \
  --rescale_t_val 5.0 \
  --backward_simulation \
  --log_steps 5

{ set +x; } 2>/dev/null

echo "----------------------------------------------------------------"
echo "DMD2 Distillation Training Completed"
echo "Output saved to: ${OUTPUT_PATH}"
echo "----------------------------------------------------------------"
