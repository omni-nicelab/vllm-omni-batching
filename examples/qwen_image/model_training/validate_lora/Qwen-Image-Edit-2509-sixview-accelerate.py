"""
六视图 Albedo 推理脚本 - 使用 accelerate 进行多GPU推理

输入 (条件):
  - hdri/{视角}.png (单视角 hdri 渲染图)
  - albedo/{视角}.png (单视角 albedo)
  - six_view/normal.png (六视角 normal 拼接图, 2行3列, 3072x2048)
输出: 六视角 albedo (3072x2048)

使用方法:
1. 运行脚本:
   accelerate launch --num_processes=8 Qwen-Image-Edit-2509-sixview-accelerate.py \
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


def enable_distill_scheduler(pipe):
    def set_timesteps_distill(num_inference_steps=100, denoising_strength=1.0, **kwargs):
        sigma_min = 0.0
        sigma_max = 1.0
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        timesteps = sigmas * pipe.scheduler.num_train_timesteps
        return sigmas, timesteps

    pipe.scheduler.set_timesteps_fn = set_timesteps_distill
    return pipe


def configure_inference(pipe, enabled: bool, attn_backend: str, quant: bool):
    if not enabled:
        return

    os.environ["DIFFSYNTH_QWEN_ATTN_BACKEND"] = attn_backend
    pipe.enable_inference(enable_mlp=quant, enable_qkv_linear=quant)


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
    parser = argparse.ArgumentParser(description="Six-view albedo generation with multi-GPU")
    
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
    parser.add_argument("--cfg_scale", type=float, default=4.0,
                        help="CFG scale for inference (default: 4.0)")
    parser.add_argument("--seed", type=int, default=123,
                        help="随机种子 (default: 123)")
    parser.add_argument("--max_pixels", type=int, default=1048576,
                        help="最大像素数，和训练时保持一致 (default: 1048576 = 1024*1024)")
    parser.add_argument("--image_size", type=int, default=1024,
                        help="hdri 图像尺寸 (default: 1024)")
    parser.add_argument("--content_ratio", type=float, default=0.8,
                        help="pad 时内容占比 (default: 0.8)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="跳过已存在的文件")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大处理样本数 (用于测试)")
    parser.add_argument("--save_grid", action="store_true",
                        help="保存网格可视化图")
    parser.add_argument("--normal_alpha", type=float, default=0.3,
                        help="normal 叠加透明度 (0-1, default: 0.3)")
    parser.add_argument("--distill", action="store_true",
                        help="使用蒸馏 few-step scheduler，复用 QwenImagePipeline 原始推理流程")
    parser.add_argument("--inference", action="store_true",
                        help="启用 qwen_image pipeline inference hook")
    parser.add_argument("--attn_backend", type=str, default="mh_cute",
                        choices=("auto", "fa3", "mh_cute", "sage"),
                        help="inference attention backend (default: mh_cute)")
    parser.add_argument("--quant", action="store_true",
                        help="enable mh_cute MLP/QKV quant path for inference hook")
    
    return parser.parse_args()


def load_image(image_path, target_size=None):
    """加载图像并可选地resize"""
    img = Image.open(image_path).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.LANCZOS)
    return img


def blend_images(base_img, overlay_img, alpha=0.3):
    """
    将 overlay_img 以一定透明度叠加在 base_img 上
    
    Args:
        base_img: 底图 (生成结果)
        overlay_img: 叠加图 (normal)
        alpha: 叠加透明度 (0-1, 越大 overlay 越明显)
    
    Returns:
        混合后的图像
    """
    # 确保尺寸一致
    if base_img.size != overlay_img.size:
        overlay_img = overlay_img.resize(base_img.size, Image.LANCZOS)
    
    # 转为 numpy 进行混合
    base_arr = np.array(base_img, dtype=np.float32)
    overlay_arr = np.array(overlay_img, dtype=np.float32)
    
    # 混合: result = base * (1 - alpha) + overlay * alpha
    blended_arr = base_arr * (1 - alpha) + overlay_arr * alpha
    blended_arr = np.clip(blended_arr, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended_arr, mode='RGB')


def create_grid_visualization(hdri_img, albedo_img, normal_6v, generated_6v, gt_6v=None, normal_alpha=0.3):
    """
    创建网格可视化:
    Row 1: hdri + albedo + empty (输入, 1024高度)
    Row 2-3: normal_six_view (六视图, 2048高度)
    Row 4-5: generated_six_view (生成结果, 2048高度)
    Row 6-7: normal 叠加在结果上 (方便对比, 2048高度)
    
    Args:
        hdri_img: hdri 单视角图
        albedo_img: albedo 单视角图
        normal_6v: normal 六视图
        generated_6v: 生成的六视图
        gt_6v: GT 六视图 (未使用，保留接口兼容)
        normal_alpha: normal 叠加透明度 (default: 0.3)
    """
    row_width = 3072
    single_height = 1024
    six_view_height = 2048  # 2行
    
    # 总高度: 1024 (输入) + 2048 (normal) + 2048 (结果) + 2048 (叠加) = 7168
    grid_height = single_height + six_view_height * 3
    grid_img = Image.new('RGB', (row_width, grid_height), (0, 0, 0))
    
    current_y = 0
    
    # Row 1: hdri + albedo + empty (输入, 1024高度)
    grid_img.paste(hdri_img.resize((single_height, single_height)), (0, current_y))
    grid_img.paste(albedo_img.resize((single_height, single_height)), (single_height, current_y))
    # 第三格留空 (黑色背景)
    current_y += single_height
    
    # Row 2-3: normal_six_view (放大到 3072x2048)
    normal_resized = normal_6v.resize((row_width, six_view_height))
    grid_img.paste(normal_resized, (0, current_y))
    current_y += six_view_height
    
    # Row 4-5: generated_six_view (放大到 3072x2048)
    generated_resized = generated_6v.resize((row_width, six_view_height))
    grid_img.paste(generated_resized, (0, current_y))
    current_y += six_view_height
    
    # Row 6-7: normal 叠加在结果上 (方便对比)
    blended = blend_images(generated_resized, normal_resized, alpha=normal_alpha)
    grid_img.paste(blended, (0, current_y))
    
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
        print(f"cfg_scale: {args.cfg_scale}")
        print(f"distill: {args.distill}")
        print(f"inference: {args.inference}")
        print(f"quant: {args.quant}")
        print(f"attn_backend: {args.attn_backend}")
    
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
            'edit_image': item['edit_image'],  # [hdri, albedo, normal_6v]
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

    if args.distill:
        args.cfg_scale = 1.0  # 蒸馏时不使用 CFG
        enable_distill_scheduler(pipe)

    # 加载 LoRA
    pipe.load_lora(pipe.dit, args.lora_path)
    configure_inference(pipe, args.inference, args.attn_backend, args.quant)
    print(f"[进程 {process_index}] 模型加载完成")

    accelerator.wait_for_everyone()

    # 处理数据
    for idx, item in enumerate(my_data):
        prompt = item['prompt']
        edit_image_paths = item['edit_image']  # [hdri_path, albedo_path, normal_6v_path]
        object_name = item['object_name']
        view_name = item['view_name']
        gt_image_path = item.get('gt_image', None)
        
        # 输出文件名
        output_name = f"{object_name}_{view_name}.png" if object_name else edit_image_paths[0].replace("/", "_")
        pred_path = os.path.join(args.output_dir, "pred", output_name)
        print(f"pred_path: {pred_path}")

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
            
            # normal 六视图用 ImageCropAndResize 处理 (和训练时一样)
            normal_raw = load_image(os.path.join(args.data_root, edit_image_paths[2]))
            normal_6v, output_width, output_height = image_crop_and_resize(
                normal_raw, max_pixels=args.max_pixels
            )
            
            print(f"[进程 {process_index}] normal: {normal_raw.size} -> {normal_6v.size}")
            
            # 构建多图输入
            edit_images = [hdri_img, albedo_img, normal_6v]
            
            # 推理 (使用 normal 处理后的尺寸)
            generated_image = pipe(
                prompt,
                edit_image=edit_images,
                seed=args.seed,
                cfg_scale=args.cfg_scale,
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
                
                grid_img = create_grid_visualization(hdri_img, albedo_img, normal_6v, generated_image, gt_img, normal_alpha=args.normal_alpha)
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
