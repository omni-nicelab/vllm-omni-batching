#!/bin/bash
# Material Decomposition Training Script

cd /mnt/project/user/lan.chen/0227/pbr/DiffSynth-Studio

DATA_ROOT="/mnt/project/train/MHTexture/User/lan.chen/topo_pbr_valid_score_17544_intersect_ref"
METADATA_PATH="/mnt/project/user/lan.chen/data_jsons/six_view_albedo_v3/train_filtered_high.json"
VAL_METADATA_PATH="/mnt/project/user/lan.chen/data_jsons/six_view_albedo_v3/val.json"
MODEL_PATH="/mnt/project/user/lan.chen/models/"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
OUTPUT_PATH="/mnt/project/user/lan.chen/outputs/data_albedo_six_view_v2_filtered_high_sp1_${TIMESTAMP}"

GPUS_PER_NODE=8
NUM_PROCESSES=$((PET_NNODES * GPUS_PER_NODE))

echo "----------------------------------------------------------------"
echo "PHASE: Starting Training"
echo "CONFIG: Nodes=${PET_NNODES}, GPUs_Per_Node=${GPUS_PER_NODE}, Total_Processes=${NUM_PROCESSES}"
echo "----------------------------------------------------------------"

set -x
PYTHONPATH=. torchrun \
  --nproc_per_node=$PET_NPROC_PER_NODE \
  --nnodes=$PET_NNODES \
  --node_rank=$PET_NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  examples/qwen_image/model_training/train.py \
  --dataset_base_path "${DATA_ROOT}" \
  --dataset_metadata_path "${METADATA_PATH}" \
  --val_dataset_metadata_path "${VAL_METADATA_PATH}" \
  --val_steps 100 \
  --max_val_batches 10 \
  --num_val_samples 2 \
  --val_inference_steps 30 \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 6291456 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2509:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 10 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 64 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --save_steps 100 \
  --sp_degree 1 \
  --deepspeed_config_path ds_config.json

{ set +x; } 2>/dev/null