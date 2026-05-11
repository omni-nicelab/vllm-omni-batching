export DIFFSYNTH_MODEL_BASE_PATH=/mnt/upfs/user/yueren.jiang/models
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export DIFFSYNTH_ATTENTION_IMPLEMENTATION=flash_attention_3

PYTHONPATH=/mnt/upfs/user/jiaxiang.z/codes/pbr-material-edit \
python examples/flux2/material_tileable/infer_tileable_pbr_flux2.py \
    --backbone flux2 \
    --lora_path /mnt/upfs/user/yueren.jiang/checkpoints/smbe_flux2_1024/diffuse_to_normal_crop1024/lora/step-35600.safetensors \
    --test_image_dir /mnt/upfs/user/yueren.jiang/test_image/tiling3 \
    --output_dir outputs/tileable_pbr_flux2/diffuse_to_normal_step-35600_md2k \
    --input_keys Diffuse \
    --output_keys Normal \
    --multidiffusion \
    --md_height 2048 \
    --md_width 2048 \
    --window_size 1024 \
    --stride 768 \
    --num_inference_steps 8 \
    --cfg_scale 4.0 \
    --embedded_guidance 4.0 \
    --seed 0