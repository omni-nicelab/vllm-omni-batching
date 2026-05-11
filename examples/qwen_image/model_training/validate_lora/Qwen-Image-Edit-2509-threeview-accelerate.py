"""
三视图 Albedo 推理脚本 - 使用 accelerate 进行多GPU推理

输入 (条件):
  - hdri/{视角}.png (单视角 hdri 渲染图)
  - albedo/{视角}.png (单视角 albedo)
  - three_view/normal.png (三视角 normal 拼接图)
输出: 三视角 albedo (3072x1024)

使用方法:
1. 运行脚本:
   accelerate launch --num_processes=8 Qwen-Image-Edit-2509-threeview-accelerate.py \
       --lora_path /path/to/lora.safetensors \
       --test_json /path/to/test.json \
       --output_dir /path/to/output \
       --data_root /path/to/data

2. 指定GPU:
   CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 ...
"""

import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import json
import os
import argparse
import numpy as np
from accelerate import Accelerator


def load_rgba_with_alpha_multiply(image_path):
    """
    加载 RGBA 图像，用 alpha 通道乘以 RGB 通道
    透明区域会变成黑色
    """
    img = Image.open(image_path)
    
    if img.mode == 'RGBA':
        img_array = np.array(img, dtype=np.float32)
        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3:4] / 255.0
        rgb_premultiplied = rgb * alpha
        result = Image.fromarray(rgb_premultiplied.astype(np.uint8), mode='RGB')
        return result
    else:
        return img.convert("RGB")


def pad_and_resize(image, target_size=1024, pad_color=(0, 0, 0), content_ratio=0.8):
    """
    将图片先 pad 成正方形，再 resize 到目标尺寸
    """
    w, h = image.size
    max_side = max(w, h)
    canvas_size = int(max_side / content_ratio)
    
    padded = Image.new("RGB", (canvas_size, canvas_size), pad_color)
    paste_x = (canvas_size - w) // 2
    paste_y = (canvas_size - h) // 2
    padded.paste(image, (paste_x, paste_y))
    
    resized = padded.resize((target_size, target_size), Image.LANCZOS)
    return resized


def image_crop_and_resize(image, max_pixels=1048576, height_division_factor=16, width_division_factor=16):
    """
    和训练时 ImageCropAndResize 一样的处理逻辑:
    1. 如果超过 max_pixels，按比例缩小
    2. 对齐到 16 的倍数
    3. 缩放并 center crop 到目标尺寸
    """
    width, height = image.size
    
    # 计算目标尺寸
    if width * height > max_pixels:
        scale = (width * height / max_pixels) ** 0.5
        height, width = int(height / scale), int(width / scale)
    
    # 对齐到 16 的倍数
    target_height = height // height_division_factor * height_division_factor
    target_width = width // width_division_factor * width_division_factor
    
    # 缩放并 center crop
    orig_width, orig_height = image.size
    scale = max(target_width / orig_width, target_height / orig_height)
    new_width = round(orig_width * scale)
    new_height = round(orig_height * scale)
    
    image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # center crop
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    image = image.crop((left, top, left + target_width, top + target_height))
    
    return image, target_width, target_height


def parse_args():
    parser = argparse.ArgumentParser(description="Three-view albedo generation with multi-GPU")
    
    # 必需参数
    parser.add_argument("--lora_path", type=str, required=True,
                        help="LoRA checkpoint 路径")
    parser.add_argument("--test_json", type=str, required=True,
                        help="测试数据 JSON 文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--data_root", type=str, required=True,
                        help="数据根目录")
    
    # 可选参数
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="推理步数 (default: 30)")
    parser.add_argument("--seed", type=int, default=123,
                        help="随机种子 (default: 123)")
    parser.add_argument("--max_pixels", type=int, default=1048576,
                        help="最大像素数，和训练时保持一致 (default: 1048576 = 1024*1024)")
    parser.add_argument("--image_size", type=int, default=1024,
                        help="hdri 图像尺寸 (default: 1024)")
    parser.add_argument("--content_ratio", type=float, default=0.8,
                        help="pad 时内容占比 (default: 0.8)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="跳过已存在的文件")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大处理样本数 (用于测试)")
    parser.add_argument("--save_grid", action="store_true", default=True,
                        help="保存网格可视化图")
    
    return parser.parse_args()


def load_image(image_path, target_size=None):
    """加载图像并可选地resize"""
    img = Image.open(image_path).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.LANCZOS)
    return img


def create_grid_visualization(hdri_img, albedo_img, normal_3v, generated_3v, gt_3v=None):
    """
    创建网格可视化 (固定 3072x1024 尺寸):
    Row 1: hdri + albedo + empty
    Row 2: normal_three_view
    Row 3: generated_three_view
    Row 4: gt_three_view (如果有)
    """
    row_width = 3072
    row_height = 1024
    single_size = 1024
    
    num_rows = 3 if gt_3v is None else 4
    grid_img = Image.new('RGB', (row_width, row_height * num_rows), (0, 0, 0))
    
    # Row 1: hdri + albedo + empty
    grid_img.paste(hdri_img.resize((single_size, single_size)), (0, 0))
    grid_img.paste(albedo_img.resize((single_size, single_size)), (single_size, 0))
    
    # Row 2: normal_three_view (放大到 3072x1024)
    grid_img.paste(normal_3v.resize((row_width, row_height)), (0, row_height))
    
    # Row 3: generated_three_view (放大到 3072x1024)
    grid_img.paste(generated_3v.resize((row_width, row_height)), (0, row_height * 2))
    
    # Row 4: gt_three_view
    if gt_3v is not None:
        grid_img.paste(gt_3v.resize((row_width, row_height)), (0, row_height * 3))
    
    return grid_img


def main():
    args = parse_args()
    
    # 初始化 accelerate
    accelerator = Accelerator()
    
    device = accelerator.device
    process_index = accelerator.process_index
    num_processes = accelerator.num_processes
    is_main_process = accelerator.is_main_process
    
    if is_main_process:
        print(f"使用 {num_processes} 个 GPU 进行推理")
        print(f"LoRA: {args.lora_path}")
        print(f"测试数据: {args.test_json}")
        print(f"输出目录: {args.output_dir}")
        print(f"max_pixels: {args.max_pixels}")
    
    # 创建目录
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "pred"), exist_ok=True)
        if args.save_grid:
            os.makedirs(os.path.join(args.output_dir, "grid"), exist_ok=True)
    accelerator.wait_for_everyone()
    
    # 加载数据
    with open(args.test_json, 'r') as f:
        data = json.load(f)
    
    # 整理数据
    all_data = []
    for item in data:
        sample = {
            'prompt': item['prompt'],
            'edit_image': item['edit_image'],  # [hdri, albedo, normal_3v]
            'object_name': item.get('object_name', ''),
            'view_name': item.get('view_name', ''),
        }
        # GT 可能存在也可能不存在
        if 'image' in item:
            sample['gt_image'] = item['image']
        all_data.append(sample)
    
    if args.max_samples:
        all_data = all_data[:args.max_samples]
    
    if is_main_process:
        print(f"总共 {len(all_data)} 个样本")
    
    # 按进程分割数据
    my_data = all_data[process_index::num_processes]
    print(f"[进程 {process_index}/{num_processes}] 分配到 {len(my_data)} 个样本，设备: {device}")
    
    # 加载模型
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=None,
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    
    # 加载 LoRA
    pipe.load_lora(pipe.dit, args.lora_path)
    print(f"[进程 {process_index}] 模型加载完成")
    
    accelerator.wait_for_everyone()
    
    # 处理数据
    for idx, item in enumerate(my_data):
        prompt = item['prompt']
        edit_image_paths = item['edit_image']  # [hdri_path, albedo_path, normal_3v_path]
        object_name = item['object_name']
        view_name = item['view_name']
        gt_image_path = item.get('gt_image', None)
        
        # 输出文件名
        output_name = f"{object_name}_{view_name}.png" if object_name else edit_image_paths[0].replace("/", "_")
        pred_path = os.path.join(args.output_dir, "pred", output_name)
        
        # 跳过已存在
        if args.skip_existing and os.path.exists(pred_path):
            continue
        
        print(f"[进程 {process_index}] 处理 {idx+1}/{len(my_data)}: {object_name}/{view_name}")
        
        try:
            # 加载输入图像
            # hdri 用 single view 一样的处理方式 (load_rgba_with_alpha_multiply + pad_and_resize)
            hdri_raw = load_rgba_with_alpha_multiply(os.path.join(args.data_root, edit_image_paths[0]))
            hdri_img = pad_and_resize(hdri_raw, target_size=args.image_size, content_ratio=args.content_ratio)
            
            # albedo 直接加载，不处理
            albedo_img = load_image(os.path.join(args.data_root, edit_image_paths[1]))
            
            # normal 三视图用 ImageCropAndResize 处理 (和训练时一样)
            normal_raw = load_image(os.path.join(args.data_root, edit_image_paths[2]))
            normal_3v, output_width, output_height = image_crop_and_resize(
                normal_raw, max_pixels=args.max_pixels
            )
            
            print(f"[进程 {process_index}] normal: {normal_raw.size} -> {normal_3v.size}")
            
            # 构建多图输入
            edit_images = [hdri_img, albedo_img, normal_3v]
            
            # 推理 (使用 normal 处理后的尺寸)
            generated_image = pipe(
                prompt,
                edit_image=edit_images,
                seed=args.seed,
                num_inference_steps=args.num_inference_steps,
                height=output_height,
                width=output_width,
            )
            
            # 保存预测结果
            generated_image.save(pred_path)
            print(f"[进程 {process_index}] 保存预测: {pred_path}")
            
            # 保存网格可视化
            if args.save_grid:
                gt_img = None
                if gt_image_path:
                    gt_full_path = os.path.join(args.data_root, gt_image_path)
                    if os.path.exists(gt_full_path):
                        gt_img = load_image(gt_full_path)
                
                grid_img = create_grid_visualization(hdri_img, albedo_img, normal_3v, generated_image, gt_img)
                grid_path = os.path.join(args.output_dir, "grid", output_name)
                grid_img.save(grid_path)
                print(f"[进程 {process_index}] 保存网格: {grid_path}")
                
        except Exception as e:
            print(f"[进程 {process_index}] 处理失败 {object_name}/{view_name}: {e}")
            import traceback
            traceback.print_exc()
    
    accelerator.wait_for_everyone()
    
    if is_main_process:
        print("所有进程处理完成!")


if __name__ == "__main__":
    main()
