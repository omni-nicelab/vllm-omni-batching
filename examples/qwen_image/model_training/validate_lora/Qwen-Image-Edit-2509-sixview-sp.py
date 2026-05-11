"""
六视图 Albedo 推理脚本 - 使用 torchrun + Sequence Parallel (SP) 进行多GPU推理

基于 Qwen-Image-Edit-2509-sixview-accelerate.py 改造，
使用与训练一致的 SP (mh_parallel_ext) 进行序列并行推理。

SP 模式下:
  - sp_degree 个 GPU 协作处理同一个样本（序列维度切分）
  - 数据并行度 = total_gpus / sp_degree
  - 需要用 torchrun 启动（而非 accelerate launch）

使用方法:
  torchrun --nproc_per_node=8 \
      Qwen-Image-Edit-2509-sixview-sp.py \
      --lora_path /path/to/lora.safetensors \
      --test_json /path/to/test.json \
      --output_dir /path/to/output \
      --data_root /path/to/data \
    --sp_degree 2 \
    [--cfg_scale 4.0 | --distill]
"""

import torch
import torch.distributed as dist
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import json
import os
import argparse
import numpy as np


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

    if width * height > max_pixels:
        scale = (width * height / max_pixels) ** 0.5
        height, width = int(height / scale), int(width / scale)

    target_height = height // height_division_factor * height_division_factor
    target_width = width // width_division_factor * width_division_factor

    orig_width, orig_height = image.size
    scale = max(target_width / orig_width, target_height / orig_height)
    new_width = round(orig_width * scale)
    new_height = round(orig_height * scale)

    image = image.resize((new_width, new_height), Image.LANCZOS)

    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    image = image.crop((left, top, left + target_width, top + target_height))

    return image, target_width, target_height


def parse_args():
    parser = argparse.ArgumentParser(description="Six-view albedo generation with SP (Sequence Parallel)")

    # 必需参数
    parser.add_argument("--lora_path", type=str, required=True,
                        help="LoRA checkpoint 路径")
    parser.add_argument("--test_json", type=str, required=True,
                        help="测试数据 JSON 文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--data_root", type=str, required=True,
                        help="数据根目录")

    # SP 参数
    parser.add_argument("--sp_degree", type=int, default=2,
                        help="Sequence Parallel degree (default: 2)")

    # 可选参数
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="推理步数 (default: 30)")
    parser.add_argument("--cfg_scale", type=float, default=4.0,
                        help="CFG scale for inference (default: 4.0)")
    parser.add_argument("--seed", type=int, default=123,
                        help="随机种子 (default: 123)")
    parser.add_argument("--max_pixels", type=int, default=6291456,
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
    parser.add_argument("--normal_alpha", type=float, default=0.3,
                        help="normal 叠加透明度 (0-1, default: 0.3)")
    parser.add_argument("--distill", action="store_true",
                        help="使用蒸馏 few-step scheduler；开启后会强制 cfg_scale=1.0")
    parser.add_argument("--enable_profile", action="store_true",
                        help="启用 torch profiler")

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
    """
    if base_img.size != overlay_img.size:
        overlay_img = overlay_img.resize(base_img.size, Image.LANCZOS)

    base_arr = np.array(base_img, dtype=np.float32)
    overlay_arr = np.array(overlay_img, dtype=np.float32)

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
    """
    row_width = 3072
    single_height = 1024
    six_view_height = 2048

    grid_height = single_height + six_view_height * 3
    grid_img = Image.new('RGB', (row_width, grid_height), (0, 0, 0))

    current_y = 0

    grid_img.paste(hdri_img.resize((single_height, single_height)), (0, current_y))
    grid_img.paste(albedo_img.resize((single_height, single_height)), (single_height, current_y))
    current_y += single_height

    normal_resized = normal_6v.resize((row_width, six_view_height))
    grid_img.paste(normal_resized, (0, current_y))
    current_y += six_view_height

    generated_resized = generated_6v.resize((row_width, six_view_height))
    grid_img.paste(generated_resized, (0, current_y))
    current_y += six_view_height

    blended = blend_images(generated_resized, normal_resized, alpha=normal_alpha)
    grid_img.paste(blended, (0, current_y))

    return grid_img


def run_inference_loop(args, my_data, rank, dp_rank, sp_rank, pipe):
    for idx, item in enumerate(my_data):
        prompt = item['prompt']
        edit_image_paths = item['edit_image']
        object_name = item['object_name']
        view_name = item['view_name']
        gt_image_path = item.get('gt_image', None)

        output_name = f"{object_name}_{view_name}.png" if object_name else edit_image_paths[0].replace("/", "_")
        pred_path = os.path.join(args.output_dir, "pred", output_name)

        if args.skip_existing and os.path.exists(pred_path):
            dist.barrier()
            continue

        if sp_rank == 0:
            print(f"[dp_rank={dp_rank}] 处理 {idx+1}/{len(my_data)}: {object_name}/{view_name}")

        try:
            hdri_raw = load_rgba_with_alpha_multiply(os.path.join(args.data_root, edit_image_paths[0]))
            hdri_img = pad_and_resize(hdri_raw, target_size=args.image_size, content_ratio=args.content_ratio)

            albedo_img = load_image(os.path.join(args.data_root, edit_image_paths[1]))

            normal_raw = load_image(os.path.join(args.data_root, edit_image_paths[2]))
            normal_6v, output_width, output_height = image_crop_and_resize(
                normal_raw, max_pixels=args.max_pixels
            )

            if sp_rank == 0:
                print(f"[dp_rank={dp_rank}] normal: {normal_raw.size} -> {normal_6v.size}")

            edit_images = [hdri_img, albedo_img, normal_6v]

            generated_image = pipe(
                prompt,
                edit_image=edit_images,
                seed=args.seed,
                cfg_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                height=output_height,
                width=output_width,
                edit_image_auto_resize=False,
            )

            if sp_rank == 0:
                generated_image.save(pred_path)
                print(f"[dp_rank={dp_rank}] 保存预测: {pred_path}")

                if args.save_grid:
                    gt_img = None
                    if gt_image_path:
                        gt_full_path = os.path.join(args.data_root, gt_image_path)
                        if os.path.exists(gt_full_path):
                            gt_img = load_image(gt_full_path)

                    grid_img = create_grid_visualization(
                        hdri_img, albedo_img, normal_6v, generated_image, gt_img,
                        normal_alpha=args.normal_alpha
                    )
                    grid_path = os.path.join(args.output_dir, "grid", output_name)
                    grid_img.save(grid_path)
                    print(f"[dp_rank={dp_rank}] 保存网格: {grid_path}")

        except Exception as e:
            print(f"[rank={rank}] 处理失败 {object_name}/{view_name}: {e}")
            import traceback
            traceback.print_exc()

        dist.barrier()


def main():
    args = parse_args()

    if args.distill:
        args.cfg_scale = 1.0

    # ==================== 初始化分布式 + SP ====================
    # torchrun 会自动设置环境变量，dist.init_process_group 会读取
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    sp_degree = args.sp_degree
    assert world_size % sp_degree == 0, \
        f"world_size ({world_size}) must be divisible by sp_degree ({sp_degree})"

    # 初始化 mh_parallel_ext 的模型并行（建立 SP group）
    from mh_parallel_ext import parallel_state
    parallel_state.initialize_model_parallel(sequence_parallel_size=sp_degree)

    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_world_size()
    sp_rank = parallel_state.get_sequence_parallel_rank()
    sp_size = parallel_state.get_sequence_parallel_world_size()

    is_main = (rank == 0)

    if is_main:
        print(f"SP Inference Config:")
        print(f"  world_size={world_size}, sp_degree={sp_degree}")
        print(f"  data_parallel_size={dp_size}")
        print(f"  LoRA: {args.lora_path}")
        print(f"  Test JSON: {args.test_json}")
        print(f"  Output: {args.output_dir}")
        print(f"  max_pixels: {args.max_pixels}")
        print(f"  cfg_scale: {args.cfg_scale}")
        print(f"  distill: {args.distill}")

    # 创建目录（仅主进程）
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "pred"), exist_ok=True)
        if args.save_grid:
            os.makedirs(os.path.join(args.output_dir, "grid"), exist_ok=True)
    dist.barrier()

    # ==================== 加载数据 ====================
    with open(args.test_json, 'r') as f:
        data = json.load(f)

    all_data = []
    for item in data:
        sample = {
            'prompt': item['prompt'],
            'edit_image': item['edit_image'],
            'object_name': item.get('object_name', ''),
            'view_name': item.get('view_name', ''),
        }
        if 'image' in item:
            sample['gt_image'] = item['image']
        all_data.append(sample)

    if args.max_samples:
        all_data = all_data[:args.max_samples]

    if is_main:
        print(f"总共 {len(all_data)} 个样本")

    # 按 data parallel rank 分割数据（SP group 内的 GPU 处理同一个样本）
    my_data = all_data[dp_rank::dp_size]
    print(f"[rank={rank}, dp_rank={dp_rank}, sp_rank={sp_rank}] 分配到 {len(my_data)} 个样本，设备: {device}")

    # ==================== 加载模型 ====================
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
        enable_distill_scheduler(pipe)

    # 加载 LoRA
    pipe.load_lora(pipe.dit, args.lora_path)

    # 启用 SP：注入 SP attention forward，设置 use_sequence_parallel=True
    pipe.enable_sp_training()
    print(f"[rank={rank}] 模型加载完成, SP enabled (sp_size={sp_size})")

    dist.barrier()

    # ==================== 推理循环 ====================
    if args.enable_profile:
        from torch.profiler import profile, record_function, ProfilerActivity

        trace_path = os.path.join(args.output_dir, f"torch_profile_rank{rank}")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with record_function("inference_loop"):
                run_inference_loop(args, my_data, rank, dp_rank, sp_rank, pipe)
            prof.step()
    else:
        run_inference_loop(args, my_data, rank, dp_rank, sp_rank, pipe)

    dist.barrier()

    if is_main:
        print("所有进程处理完成!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
