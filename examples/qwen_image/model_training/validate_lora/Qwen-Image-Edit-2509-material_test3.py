import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import json
import os
import numpy as np


def load_rgba_with_alpha_multiply(image_path):
    """
    加载 RGBA 图像，用 alpha 通道乘以 RGB 通道
    透明区域会变成黑色
    
    Args:
        image_path: 图像路径
    
    Returns:
        PIL Image (RGB)
    """
    img = Image.open(image_path)
    
    if img.mode == 'RGBA':
        # 转换为 numpy 数组
        img_array = np.array(img, dtype=np.float32)
        
        # 分离 RGB 和 Alpha
        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3:4] / 255.0  # 归一化到 0-1
        
        # RGB * Alpha (预乘)
        rgb_premultiplied = rgb * alpha
        
        # 转回 uint8 并创建 RGB 图像
        result = Image.fromarray(rgb_premultiplied.astype(np.uint8), mode='RGB')
        return result
    else:
        # 如果不是 RGBA，直接转换为 RGB
        return img.convert("RGB")


def pad_and_resize(image, target_size=1024, pad_color=(0, 0, 0), content_ratio=0.8):
    """
    将图片先 pad 成正方形，再 resize 到目标尺寸
    保持原始长宽比，用黑色填充，图片只占画布的指定比例
    
    Args:
        image: PIL Image
        target_size: 目标尺寸
        pad_color: 填充颜色 (默认黑色)
        content_ratio: 图片内容占画布的比例 (默认 0.8，即 80%)
    
    Returns:
        处理后的 PIL Image
    """
    w, h = image.size
    
    # 计算需要 pad 成的正方形尺寸（原图最大边 / content_ratio）
    max_side = max(w, h)
    canvas_size = int(max_side / content_ratio)
    
    # 创建正方形画布
    padded = Image.new("RGB", (canvas_size, canvas_size), pad_color)
    
    # 将原图居中粘贴
    paste_x = (canvas_size - w) // 2
    paste_y = (canvas_size - h) // 2
    padded.paste(image, (paste_x, paste_y))
    
    # resize 到目标尺寸
    resized = padded.resize((target_size, target_size), Image.LANCZOS)
    
    return resized

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

# step = 176
# LORA_PATH = "/upfs/user/lanchen/MaterialEdit/outputs/material_lora/epoch-0.safetensors"
step = 176
LORA_PATH = f"/upfs/user/lanchen/MaterialEdit/outputs/data_manually_filtered1e-4/step-{step}.safetensors"

# LORA_PATH = f"/upfs/user/lanchen/MaterialEdit/outputs/material_lora_topo_pbr_valid_score_17544_intersect_ref_1e-4/step-{step}.safetensors"
pipe.load_lora(pipe.dit, LORA_PATH)


test_json = "/upfs/user/lanchen/MaterialEdit/data/test.json"
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
# output_dir = f"/upfs/user/lanchen/MaterialEdit/outputs/material_lora_topo_pbr_valid_score_17544_intersect_ref_1e-4/val_step-{step}/"
# output_dir = f"/upfs/user/lanchen/MaterialEdit/outputs/original/" 
os.makedirs(output_dir, exist_ok=True)
# DATA_ROOT="/upfs/MHTexture/stage1/objaverse/ortho_topofilter_26k_1024x1024_PBR_gtscale098"
# DATA_ROOT="/upfs/user/lanchen/render/data/cg_wh_300"
# DATA_ROOT="/upfs/user/lanchen/render/data/topo_pbr_valid_score_17544_intersect_ref/"
DATA_ROOT="/upfs/user/lanchen/MaterialEdit/test/"

for i in range(len(prompts)):
    if i % 1 == 0:
        print(f"Processing {i}/{len(prompts)}")
    else:
        continue
    prompt = prompts[i]
    image_path = images[i]
    # if 'realistic' not in image_path and 'chibi' not in image_path:
    #     continue
    if 'building' not in image_path:
        continue
    if 'chibi' not in image_path:# and 'chibi' not in image_path:
        continue
    edit_image_path = edit_images[i]
    
    # 加载HDRI原始图 (RGBA -> RGB * Alpha)
    print("hdri_image:", edit_image_path)
    hdri_raw = load_rgba_with_alpha_multiply(os.path.join(DATA_ROOT, edit_image_path))
    print(f"  原始尺寸: {hdri_raw.size}")
    hdri_image = pad_and_resize(hdri_raw, target_size=1024)
    
    # 加载参考albedo (RGBA -> RGB * Alpha)
    albedo_raw = load_rgba_with_alpha_multiply(os.path.join(DATA_ROOT, image_path))
    albedo_image = pad_and_resize(albedo_raw, target_size=1024)
    
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
    concat_image = Image.new("RGB", (1024 * 2, 1024))
    concat_image.paste(hdri_image, (0, 0))
    concat_image.paste(generated_image, (1024, 0)) 
    
    output_path = os.path.join(output_dir, image_path.replace("/", "_"))
    concat_image.save(output_path)
    print(output_path)
