"""
通用单视图推理脚本 - 使用 accelerate 进行多GPU推理

适用于: roughness, albedo, metallic 等单输入单输出任务
任务类型由 test_json 中的 prompt 决定

使用方法:
1. 运行脚本:
   accelerate launch --num_processes=4 Qwen-Image-Edit-2509-singleview-accelerate.py \
       --lora_path /path/to/lora.safetensors \
       --test_json /path/to/test.json \
       --output_dir /path/to/output \
       --data_root /path/to/data

2. 指定GPU:
   CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 ...

3. 简单启动（自动检测GPU数量）:
   accelerate launch --multi_gpu Qwen-Image-Edit-2509-singleview-accelerate.py ...
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


def parse_args():
    parser = argparse.ArgumentParser(description="Single-view inference with multi-GPU (roughness/albedo/metallic etc.)")
    
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
    parser.add_argument("--image_size", type=int, default=1024,
                        help="图像尺寸 (default: 1024)")
    parser.add_argument("--content_ratio", type=float, default=0.8,
                        help="pad 时内容占比 (default: 0.8)")
    parser.add_argument("--save_concat", action="store_true", default=True,
                        help="保存拼接图 (input | output)")
    parser.add_argument("--save_pred_only", action="store_true", default=False,
                        help="只保存预测结果")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="跳过已存在的文件")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大处理样本数 (用于测试)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 初始化 accelerate
    accelerator = Accelerator()
    
    # 获取当前进程信息
    device = accelerator.device
    process_index = accelerator.process_index
    num_processes = accelerator.num_processes
    is_main_process = accelerator.is_main_process
    
    if is_main_process:
        print(f"使用 {num_processes} 个 GPU 进行推理")
        print(f"LoRA: {args.lora_path}")
        print(f"测试数据: {args.test_json}")
        print(f"输出目录: {args.output_dir}")
        print(f"数据根目录: {args.data_root}")
    
    # 只在主进程创建目录
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.save_pred_only:
            os.makedirs(os.path.join(args.output_dir, "pred"), exist_ok=True)
    accelerator.wait_for_everyone()
    
    # 加载数据
    with open(args.test_json, 'r') as f:
        data = json.load(f)
    
    # 整理数据
    all_data = []
    for item in data:
        prompt = item['prompt']
        image = item['image']
        edit_image = item['edit_image']
        all_data.append({
            'prompt': prompt,
            'image': image,
            'edit_image': edit_image
        })
    
    # 限制样本数
    if args.max_samples:
        all_data = all_data[:args.max_samples]
    
    if is_main_process:
        print(f"总共 {len(all_data)} 个样本")
    
    # 按进程分割数据
    my_data = all_data[process_index::num_processes]
    
    print(f"[进程 {process_index}/{num_processes}] 分配到 {len(my_data)} 个样本，设备: {device}")
    
    # 在当前设备上加载模型
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
    
    # 等待所有进程加载完成
    accelerator.wait_for_everyone()
    
    # 处理数据
    processed_count = 0
    for idx, item in enumerate(my_data):
        prompt = item['prompt']
        image_path = item['image']
        edit_image_path = item['edit_image']
        
        output_filename = image_path.replace("/", "_")
        output_path = os.path.join(args.output_dir, output_filename)
        
        # 跳过已存在的
        if args.skip_existing and os.path.exists(output_path):
            continue
        
        print(f"[进程 {process_index}] 处理 {idx+1}/{len(my_data)}: {image_path}")
        
        try:
            # 加载HDRI原始图
            hdri_raw = load_rgba_with_alpha_multiply(os.path.join(args.data_root, edit_image_path))
            hdri_image = pad_and_resize(hdri_raw, target_size=args.image_size, content_ratio=args.content_ratio)
            
            # 生成结果
            generated_image = pipe(
                prompt, 
                edit_image=hdri_image,
                seed=args.seed, 
                num_inference_steps=args.num_inference_steps, 
                height=args.image_size, 
                width=args.image_size
            )
            
            if args.save_pred_only:
                # 只保存预测结果
                pred_path = os.path.join(args.output_dir, "pred", output_filename)
                generated_image.save(pred_path)
                print(f"[进程 {process_index}] 保存到: {pred_path}")
            
            if args.save_concat:
                # 拼接两张图
                concat_image = Image.new("RGB", (args.image_size * 2, args.image_size))
                concat_image.paste(hdri_image, (0, 0))
                concat_image.paste(generated_image, (args.image_size, 0)) 
                concat_image.save(output_path)
                print(f"[进程 {process_index}] 保存到: {output_path}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"[进程 {process_index}] 处理失败 {image_path}: {e}")
    
    # 等待所有进程完成
    accelerator.wait_for_everyone()
    
    if is_main_process:
        print("所有进程处理完成!")


if __name__ == "__main__":
    main()
