import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import json
import os

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=None,
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)

step = 72
# LORA_PATH = "/upfs/user/lanchen/MaterialEdit/outputs/material_lora/epoch-0.safetensors"
 
LORA_PATH = f"/upfs/user/lanchen/MaterialEdit/outputs/data_manually_filtered1e-4/step-{step}.safetensors"

pipe.load_lora(pipe.dit, LORA_PATH)

# test_json = "/upfs/user/lanchen/MaterialEdit/data/val.json"
# test_json = "/upfs/user/lanchen/MaterialEdit/data/test.json"
# test_json = "/upfs/user/lanchen/MaterialEdit/data_topo_pbr_valid_score_17544_intersect_ref/val.json"
test_json = "/upfs/user/lanchen/MaterialEdit/data_manually_filtered/val.json"
images = []
prompts = []
edit_images = []

with open(test_json, 'r') as f:
    data = json.load(f)
    for item in data:
        prompt = item['prompt']
        image = item['image']
        # if 'realistic' not in image:
        #     continue
        edit_image = item['edit_image']
        images.append(image)
        prompts.append(prompt)
        edit_images.append(edit_image) 
# output_dir = "/upfs/user/lanchen/MaterialEdit/outputs/material_lora/val_epoch0/"



output_dir = f"/upfs/user/lanchen/MaterialEdit/outputs/data_manually_filtered1e-4/val_step-{step}/"
# output_dir = f"/upfs/user/lanchen/MaterialEdit/outputs/original/" 
os.makedirs(output_dir, exist_ok=True)
# DATA_ROOT="/upfs/MHTexture/stage1/objaverse/ortho_topofilter_26k_1024x1024_PBR_gtscale098"
# DATA_ROOT="/upfs/user/lanchen/render/data/cg_wh_300"
DATA_ROOT="/upfs/user/lanchen/render/data/topo_pbr_valid_score_17544_intersect_ref/"
# DATA_ROOT="/upfs/user/lanchen/MaterialEdit/test/"

for i in range(len(prompts)):
    if i % 1 == 0:
        print(f"Processing {i}/{len(prompts)}")
    else:
        continue
    prompt = prompts[i]
    image_path = images[i]
    edit_image_path = edit_images[i]
    
    # 加载HDRI原始图
    print("hdri_image:", edit_image_path)
    hdri_image = Image.open(os.path.join(DATA_ROOT, edit_image_path)).convert("RGB").resize((1024, 1024))
    # 加载参考albedo
    albedo_image = Image.open(os.path.join(DATA_ROOT, image_path)).convert("RGB").resize((1024, 1024))
    
    # 生成结果
    # 重要：在 Qwen-Image-Edit 中
    # - input_image: 训练目标（Albedo），推理时不传或传 None（从噪声生成）
    # - edit_image: 条件图像（HDRI），推理时传入作为参考
    generated_image = pipe(
        prompt, 
        edit_image=hdri_image,  # HDRI 作为条件输入
        seed=123, 
        num_inference_steps=30, 
        height=1024, 
        width=1024
    )
    
    # 拼接三张图：HDRI原始图 | 生成结果 | 参考albedo
    concat_image = Image.new("RGB", (1024 * 3, 1024))
    concat_image.paste(hdri_image, (0, 0))
    concat_image.paste(generated_image, (1024, 0))
    concat_image.paste(albedo_image, (1024 * 2, 0))
    
    output_path = os.path.join(output_dir, image_path.replace("/", "_"))
    concat_image.save(output_path)
    print(output_path)
