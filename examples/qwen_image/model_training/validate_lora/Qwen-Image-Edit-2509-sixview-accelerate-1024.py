"""
六视图 Albedo 推理脚本 - 适配训练逻辑 (1024版本)
使用 accelerate 进行多GPU推理

输入 (条件):
  - albedo/{视角}.png (单视角 albedo)
  - six_view/normal.png (六视角 normal 拼接图, 2行3列, 3072x2048)
  - six_view/position.png (六视角 position 拼接图, 2行3列, 3072x2048)
输出: 六视角 albedo (3072x2048)

适配训练逻辑:
  - 使用 --skip_image_resize 跳过 ImageCropAndResize
  - 使用 --max_pixels 1048576 (1024x1024) 保持单视角图像原始尺寸

使用方法:
1. 运行脚本:
   accelerate launch --num_processes=8 Qwen-Image-Edit-2509-sixview-accelerate-1024.py \
       --lora_path /path/to/lora.safetensors \
       --test_json /path/to/test.json \
       --output_dir /path/to/output \
       --data_root /path/to/data \
      --skip_image_resize \
      --max_pixels 1048576

2. 指定GPU:
   CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 ...
"""

import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import json
import os
import argparse
from accelerate import Accelerator
import warnings
import torch.distributed as dist
import sys
# 导入训练脚本中的函数和模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from train_sixview_single import split_six_view_image, QwenImageSixViewSingleTrainingModule, ProcessV3DataOperator

# 抑制 PyTorch 分布式训练的网络警告
os.environ.setdefault('GLOO_SOCKET_IFNAME', 'eth0')
os.environ.setdefault('NCCL_SOCKET_IFNAME', 'eth0')
# 抑制 IPv6 相关警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*IPv6.*')


def load_image(image_path, target_size=None):
    """加载图像并可选地resize"""
    img = Image.open(image_path).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.LANCZOS)
    return img




def parse_args():
    parser = argparse.ArgumentParser(description="Six-view albedo generation with multi-GPU (1024 version, adapted for training)")
    
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
    parser.add_argument("--skip_image_resize", action="store_true", default=True,
                        help="跳过 ImageCropAndResize，直接使用原始图片（用于单视角图像，1024x1024）")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大处理样本数 (用于测试)")
    parser.add_argument("--save_grid", action="store_true", default=True,
                        help="保存网格可视化图")
    
    # Tiled 解码 (高分辨率必须开启)
    parser.add_argument("--tiled", action="store_true", default=False,
                        help="启用 tiled 解码，用于高分辨率推理")
    parser.add_argument("--tile_size", type=int, default=128,
                        help="tile 大小 (default: 128)")
    parser.add_argument("--tile_stride", type=int, default=64,
                        help="tile 步长 (default: 64)")
    
    # LoRA checkpoint 前缀处理（自动检测并移除，如果 checkpoint 中有前缀）
    parser.add_argument("--add_prefix_in_ckpt", type=str, default="pipe.dit.",
                        help="如果 checkpoint 中有此前缀，会自动移除以匹配模型中的键名。训练时已移除前缀，通常不需要此参数 (default: 'pipe.dit.')")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 初始化 accelerate
    accelerator = Accelerator()
    
    device = accelerator.device
    process_index = accelerator.process_index
    num_processes = accelerator.num_processes
    is_main_process = accelerator.is_main_process
    
    if is_main_process:
        print(f"使用 {num_processes} 个 GPU 进行推理 (1024版本，适配训练逻辑)")
        print(f"LoRA: {args.lora_path}")
        print(f"测试数据: {args.test_json}")
        print(f"输出目录: {args.output_dir}")
        print(f"max_pixels: {args.max_pixels}")
        print(f"skip_image_resize: {args.skip_image_resize}")
    
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
            'edit_image': item['edit_image'],  # [albedo, normal_6v, position_6v]
            'object_name': item.get('object_name', ''),
            'view_name': item.get('view_name', ''),
        }
        all_data.append(sample)
    
    if args.max_samples:
        all_data = all_data[:args.max_samples]
    
    if is_main_process:
        print(f"总共 {len(all_data)} 个样本")
    
    # 注意：如果使用 CUDA_VISIBLE_DEVICES=6,7，那么 process_index 会是 0 和 1
    # 但实际的物理 GPU 是 6 和 7
    # 这里我们直接使用所有进程（因为已经通过 CUDA_VISIBLE_DEVICES 限制了）
    
    # 检查是否需要启用 cross-view attention（6卡模式）
    enable_cross_view_attention = (
        num_processes >= 6 and
        dist.is_initialized() and
        dist.get_world_size() == 6
    )
    
    # 数据分配逻辑
    if enable_cross_view_attention:
        # 6卡模式：所有 GPU 处理相同的样本，但每个 GPU 处理不同的视角
        # 主进程（rank 0）决定处理哪些样本，其他 GPU 同步
        if is_main_process:
            # 主进程决定所有样本
            my_data = all_data
        else:
            # 其他进程等待主进程
            my_data = []
        # 同步所有进程
        accelerator.wait_for_everyone()
        # 广播样本列表（简化处理：所有 GPU 处理所有样本）
        my_data = all_data
        print(f"[进程 {process_index}/{num_processes}] Cross-view attention 模式：所有 GPU 处理所有 {len(my_data)} 个样本，每个 GPU 处理一个视角")
    else:
        # 非6卡模式：将不同样本分配到不同GPU
        my_data = all_data[process_index::num_processes]
        print(f"[进程 {process_index}/{num_processes}] 分配到 {len(my_data)} 个样本，设备: {device}")
    
    # 如果当前进程没有数据，直接返回
    if len(my_data) == 0:
        print(f"[进程 {process_index}] 没有分配到数据，跳过")
        accelerator.wait_for_everyone()
        return
    
    # 创建数据处理器（复用训练时的逻辑）
    data_processor = ProcessV3DataOperator(
        base_path=args.data_root,
        height=None,  # 不进行 resize，保持 1024x1024
        width=None,
        max_pixels=None,
        gpu_rank=process_index if enable_cross_view_attention else None,
        world_size=num_processes if enable_cross_view_attention else None,
    )
    
    # 直接使用训练模块加载模型和 LoRA（复用训练时的逻辑）
    # 但是需要先处理 checkpoint 的键名：训练时保存的 checkpoint 没有 pipe.dit. 前缀
    # 但模型中的键名有 pipe.dit. 前缀，所以需要添加前缀
    from diffsynth.core import load_state_dict
    
    # 自动检测并处理 checkpoint 前缀（训练时已移除前缀，但为了兼容性，如果检测到前缀则自动移除）
    lora_checkpoint_path = args.lora_path
    temp_checkpoint_path = None
    if args.add_prefix_in_ckpt and args.add_prefix_in_ckpt.strip():
        # 只在主进程检查并处理
        if is_main_process:
            # 加载原始 checkpoint 检查是否有前缀
            original_state_dict = load_state_dict(args.lora_path)
            sample_keys = list(original_state_dict.keys())[:5]
            has_prefix = any(k.startswith(args.add_prefix_in_ckpt) for k in sample_keys)
            
            if has_prefix:
                # Checkpoint 中有前缀，需要移除（训练时应该已经移除了，但为了兼容性保留此逻辑）
                temp_checkpoint_path = args.lora_path.replace(".safetensors", "_no_prefix.safetensors")
                no_prefix_state_dict = {}
                for key, value in original_state_dict.items():
                    if key.startswith(args.add_prefix_in_ckpt):
                        new_key = key[len(args.add_prefix_in_ckpt):]
                    else:
                        new_key = key
                    no_prefix_state_dict[new_key] = value
                
                # 保存临时 checkpoint（移除前缀的版本）
                from safetensors.torch import save_file
                save_file(no_prefix_state_dict, temp_checkpoint_path)
                
                # 确保文件完全写入磁盘
                import time
                time.sleep(0.1)
                
                if not os.path.exists(temp_checkpoint_path):
                    raise FileNotFoundError(f"Failed to create temporary checkpoint: {temp_checkpoint_path}")
                
                lora_checkpoint_path = temp_checkpoint_path
                
                print(f"[LoRA Loading] 检测到 checkpoint 中有前缀 '{args.add_prefix_in_ckpt}'，已自动移除")
                print(f"[LoRA Loading] 原始键名示例: {sample_keys[:3]}")
                print(f"[LoRA Loading] 移除前缀后键名示例: {list(no_prefix_state_dict.keys())[:3]}")
            else:
                # Checkpoint 中没有前缀，直接使用原始文件（正常情况，训练时已移除前缀）
                print(f"[LoRA Loading] Checkpoint 键名无前缀，直接使用（训练时已移除前缀）")
                temp_checkpoint_path = None
        
        # 所有进程等待主进程处理完成
        if dist.is_initialized():
            dist.barrier()
        
        # 如果创建了临时文件，所有进程都等待文件存在
        if temp_checkpoint_path is not None:
            import time
            max_wait = 30
            wait_time = 0
            while not os.path.exists(temp_checkpoint_path) and wait_time < max_wait:
                time.sleep(0.1)
                wait_time += 0.1
            
            if not os.path.exists(temp_checkpoint_path):
                raise FileNotFoundError(f"Temporary checkpoint file not found after waiting: {temp_checkpoint_path}")
            
            lora_checkpoint_path = temp_checkpoint_path
    
    # 训练模块会自动加载模型和 LoRA
    # 注意：必须设置 lora_base_model="dit" 才能加载 LoRA
    training_module = QwenImageSixViewSingleTrainingModule(
        model_paths=None,
        model_id_with_origin_paths="Qwen/Qwen-Image-Edit-2509:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors",
        tokenizer_path=None,
        processor_path=None,
        trainable_models=None,
        lora_base_model="dit",  # 必须设置才能加载 LoRA
        lora_target_modules="to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1",  # 和训练时保持一致
        lora_rank=64,  # LoRA rank（从 checkpoint 中读取，这里只是占位符）
        lora_checkpoint=lora_checkpoint_path,  # 使用处理后的 checkpoint 路径
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device=device,
        task="sft",
        zero_cond_t=False,
    )
    
    # 获取训练模块的 pipe（已加载模型和 LoRA）
    pipe = training_module.pipe
    
    # 验证 LoRA 是否正确加载
    if is_main_process and args.lora_path:
        print(f"\n{'='*60}")
        print(f"[LoRA Verification] 检查 LoRA 是否正确加载")
        print(f"{'='*60}")
        
        # 检查模型中的 LoRA 参数
        dit_model = pipe.dit
        lora_params = []
        lora_param_count = 0
        for name, param in dit_model.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                lora_params.append(name)
                lora_param_count += param.numel()
        
        print(f"[LoRA Verification] 找到 {len(lora_params)} 个 LoRA 参数层")
        print(f"[LoRA Verification] LoRA 参数总数: {lora_param_count:,}")
        
        # 检查 checkpoint 中的键名
        if args.add_prefix_in_ckpt and args.add_prefix_in_ckpt.strip():
            checkpoint_state_dict = load_state_dict(lora_checkpoint_path)
            checkpoint_keys = set(checkpoint_state_dict.keys())
            print(f"[LoRA Verification] Checkpoint 中的键名数量: {len(checkpoint_keys)}")
            print(f"[LoRA Verification] Checkpoint 键名示例 (前5个): {list(checkpoint_keys)[:5]}")
        
        # 检查模型中的键名
        model_state_dict = dit_model.state_dict()
        model_keys = set(model_state_dict.keys())
        lora_model_keys = {k for k in model_keys if "lora" in k.lower()}
        print(f"[LoRA Verification] 模型中的 LoRA 键名数量: {len(lora_model_keys)}")
        print(f"[LoRA Verification] 模型 LoRA 键名示例 (前5个): {list(lora_model_keys)[:5]}")
        
        # 检查键名匹配
        if args.add_prefix_in_ckpt and args.add_prefix_in_ckpt.strip():
            matched_keys = checkpoint_keys & lora_model_keys
            missing_in_model = checkpoint_keys - lora_model_keys
            missing_in_checkpoint = lora_model_keys - checkpoint_keys
            
            print(f"[LoRA Verification] 匹配的键名数量: {len(matched_keys)}")
            if len(missing_in_model) > 0:
                print(f"[LoRA Verification] ⚠ Checkpoint 中有但模型中没有的键名数量: {len(missing_in_model)}")
                print(f"[LoRA Verification]   示例 (前5个): {list(missing_in_model)[:5]}")
            if len(missing_in_checkpoint) > 0:
                print(f"[LoRA Verification] ⚠ 模型中有但 Checkpoint 中没有的键名数量: {len(missing_in_checkpoint)}")
                print(f"[LoRA Verification]   示例 (前5个): {list(missing_in_checkpoint)[:5]}")
        
        if len(lora_params) > 0:
            print(f"[LoRA Verification] LoRA 参数示例 (前5个):")
            for name in lora_params[:5]:
                param = dict(dit_model.named_parameters())[name]
                print(f"  - {name}: shape={param.shape}, requires_grad={param.requires_grad}, "
                      f"mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
            
            # 检查是否有非零的 LoRA 参数（如果全是零，可能没有正确加载）
            non_zero_count = 0
            zero_count = 0
            for name in lora_params[:10]:  # 检查前10个
                param = dict(dit_model.named_parameters())[name]
                if param.data.abs().sum() > 1e-8:
                    non_zero_count += 1
                else:
                    zero_count += 1
            
            if non_zero_count > 0:
                print(f"[LoRA Verification] ✓ LoRA 参数包含非零值（检查了 {non_zero_count + zero_count} 个参数，{non_zero_count} 个非零）")
            else:
                print(f"[LoRA Verification] ⚠ 警告：检查的 LoRA 参数全为零，可能未正确加载！")
        else:
            print(f"[LoRA Verification] ⚠ 警告：未找到 LoRA 参数！")
        
        print(f"{'='*60}\n")
    
    
    print(f"[进程 {process_index}] 模型加载完成")
    
    # 设置模型为评估模式（推理时不需要梯度）
    pipe.eval()
    for param in pipe.parameters():
        param.requires_grad = False
    
    # 设置 cross-view attention（如果启用）
    original_model_fn = None
    if enable_cross_view_attention:
        # 使用训练模块的 cross-view attention 方法
        original_model_fn = pipe.model_fn
        pipe.model_fn = training_module._create_cross_view_model_fn()
        if is_main_process:
            print(f"[Cross-View Attention] 已启用 cross-view attention（6卡模式）")
    
    accelerator.wait_for_everyone()
    
    # 处理数据
    # 在6卡模式下，需要确保所有GPU同步处理同一个样本
    if enable_cross_view_attention:
        # 6卡模式：所有GPU处理所有样本，但每个样本都需要同步
        # 使用 all_data 而不是 my_data，确保所有GPU处理相同的样本列表
        samples_to_process = all_data
    else:
        # 非6卡模式：每个GPU处理不同的样本
        samples_to_process = my_data
    
    for idx, item in enumerate(samples_to_process):
        # 在6卡模式下，每个样本处理前都需要同步，确保所有GPU同时处理同一个样本
        if enable_cross_view_attention:
            accelerator.wait_for_everyone()
        
        # 使用 ProcessV3DataOperator 处理数据（复用训练时的逻辑）
        data = {
            "prompt": item['prompt'],
            "image": item.get('image', ''),  # 六视角 albedo 拼接图路径（如果有）
            "edit_image": item['edit_image'],  # [albedo_path, normal_6v_path, position_6v_path]
        }
        
        # 使用数据处理器处理数据（会自动分割视角）
        processed_data = data_processor(data)
        views = processed_data.get("views", [])
        
        if not views:
            print(f"[进程 {process_index}] 警告: 无法处理数据 {item.get('object_name', '')}")
            continue
        
        object_name = item.get('object_name', '')
        view_name = item.get('view_name', '')
        
        try:
            if enable_cross_view_attention:
                # 6卡模式：每个GPU处理一个视角（支持 cross-view attention）
                # views 应该只有一个元素（当前GPU对应的视角）
                if len(views) != 1:
                    print(f"[进程 {process_index}] 警告: 6卡模式下应该有1个视角，得到 {len(views)} 个")
                    continue
                
                view_data = views[0]
                view_id = view_data["view_id"]
                albedo_view = view_data["albedo_input"]  # 输入 albedo（作为条件）
                normal_view = view_data["normal"]
                position_view = view_data["position"]
                
                # 构建单视角输入（1024x1024）
                edit_images = [albedo_view, normal_view, position_view]
                
                print(f"[进程 {process_index}] 处理样本 {idx+1}/{len(samples_to_process)}, view {view_id}: {object_name}/{view_name}")
                print(f"[进程 {process_index}] 输入尺寸: albedo={albedo_view.size}, normal={normal_view.size}, position={position_view.size}")
                
                # 在 cross-view attention 模式下，确保所有 GPU 同步后再开始推理
                # 这是关键：所有GPU必须同时进入 pipe() 调用，才能进行跨视角通信
                accelerator.wait_for_everyone()
                
                # 推理单视角 (1024x1024，和训练时一致)
                # 在6卡模式下，所有GPU同时调用 pipe()，在 model_fn 内部通过 dist.all_gather 进行跨视角通信
                # 推理时使用 no_grad，避免计算梯度
                pipe.eval()  # 设置为评估模式
                with torch.no_grad():
                    try:
                        generated_view = pipe(
                            processed_data["prompt"],
                            edit_image=edit_images,
                            seed=args.seed + idx,  # 不同样本使用不同种子
                            num_inference_steps=args.num_inference_steps,
                            height=1024,
                            width=1024,
                            edit_image_auto_resize=False,  # 与训练保持一致，禁用 auto_resize
                            tiled=args.tiled,
                            tile_size=args.tile_size,
                            tile_stride=args.tile_stride,
                        )
                    except RuntimeError as e:
                        if "CUDA" in str(e) or "illegal memory access" in str(e).lower():
                            print(f"[进程 {process_index}] CUDA 错误，尝试清理缓存并重试...")
                            torch.cuda.empty_cache()
                            # 确保模型在正确的设备上
                            pipe = pipe.to(device)
                            pipe.eval()  # 设置为评估模式
                            generated_view = pipe(
                                processed_data["prompt"],
                                edit_image=edit_images,
                                seed=args.seed + idx,
                                num_inference_steps=args.num_inference_steps,
                                height=1024,
                                width=1024,
                                edit_image_auto_resize=False,  # 与训练保持一致，禁用 auto_resize
                                tiled=args.tiled,
                                tile_size=args.tile_size,
                                tile_stride=args.tile_stride,
                            )
                        else:
                            raise
                
                # 在 cross-view attention 模式下，推理完成后需要同步，确保所有GPU都完成后再继续
                accelerator.wait_for_everyone()
                
                # 保存单视角结果（像训练时一样）
                output_name = f"{object_name}_{view_id}_output_albedo.png" if object_name else f"sample_{idx}_view_{view_id}.png"
                pred_path = os.path.join(args.output_dir, "pred", output_name)
                generated_view.save(pred_path)
                print(f"[进程 {process_index}] 保存视角 {view_id} 结果: {pred_path}")
                
                # 保存输入图片
                input_dir = os.path.join(args.output_dir, "pred")
                albedo_input_name = f"{object_name}_{view_id}_input_albedo.png" if object_name else f"sample_{idx}_view_{view_id}_input_albedo.png"
                albedo_input_path = os.path.join(input_dir, albedo_input_name)
                albedo_view.save(albedo_input_path)
                
                normal_input_name = f"{object_name}_{view_id}_input_normal.png" if object_name else f"sample_{idx}_view_{view_id}_input_normal.png"
                normal_input_path = os.path.join(input_dir, normal_input_name)
                normal_view.save(normal_input_path)
                
                position_input_name = f"{object_name}_{view_id}_input_position.png" if object_name else f"sample_{idx}_view_{view_id}_input_position.png"
                position_input_path = os.path.join(input_dir, position_input_name)
                position_view.save(position_input_path)
                
                print(f"[进程 {process_index}] 保存视角 {view_id} 输入: albedo, normal, position")
                
                # 拼接并保存到 grid 目录
                if args.save_grid:
                    SINGLE_VIEW_SIZE = 1024
                    grid_width = SINGLE_VIEW_SIZE * 4
                    grid_height = SINGLE_VIEW_SIZE
                    grid_img = Image.new("RGB", (grid_width, grid_height), (0, 0, 0))
                    
                    # 确保所有图片都是 1024x1024
                    if albedo_view.size != (SINGLE_VIEW_SIZE, SINGLE_VIEW_SIZE):
                        albedo_view = albedo_view.resize((SINGLE_VIEW_SIZE, SINGLE_VIEW_SIZE), Image.LANCZOS)
                    if normal_view.size != (SINGLE_VIEW_SIZE, SINGLE_VIEW_SIZE):
                        normal_view = normal_view.resize((SINGLE_VIEW_SIZE, SINGLE_VIEW_SIZE), Image.LANCZOS)
                    if position_view.size != (SINGLE_VIEW_SIZE, SINGLE_VIEW_SIZE):
                        position_view = position_view.resize((SINGLE_VIEW_SIZE, SINGLE_VIEW_SIZE), Image.LANCZOS)
                    if generated_view.size != (SINGLE_VIEW_SIZE, SINGLE_VIEW_SIZE):
                        generated_view = generated_view.resize((SINGLE_VIEW_SIZE, SINGLE_VIEW_SIZE), Image.LANCZOS)
                    
                    # 拼接：albedo | normal | position | generated
                    grid_img.paste(albedo_view, (0, 0))
                    grid_img.paste(normal_view, (SINGLE_VIEW_SIZE, 0))
                    grid_img.paste(position_view, (SINGLE_VIEW_SIZE * 2, 0))
                    grid_img.paste(generated_view, (SINGLE_VIEW_SIZE * 3, 0))
                    
                    grid_name = f"{object_name}_{view_id}_grid.png" if object_name else f"sample_{idx}_view_{view_id}_grid.png"
                    grid_path = os.path.join(args.output_dir, "grid", grid_name)
                    grid_img.save(grid_path)
                    print(f"[进程 {process_index}] 保存视角 {view_id} 拼接图: {grid_path}")
            else:
                # 少于6卡模式：每个GPU处理所有6个视角（不支持 cross-view attention）
                for view_data in views:
                    view_id = view_data["view_id"]
                    albedo_view = view_data["albedo_input"]  # 输入 albedo（作为条件）
                    normal_view = view_data["normal"]
                    position_view = view_data["position"]
                    
                    # 构建单视角输入（1024x1024）
                    edit_images = [albedo_view, normal_view, position_view]
                    
                    print(f"[进程 {process_index}] 处理样本 {idx+1}/{len(samples_to_process)}, view {view_id}: {object_name}/{view_name}")
                    print(f"[进程 {process_index}] 输入尺寸: albedo={albedo_view.size}, normal={normal_view.size}, position={position_view.size}")
                    
                    # 推理单视角 (1024x1024，和训练时一致)
                    # 注意：pipe.__call__ 已经有 @torch.no_grad() 装饰器，但为了确保，我们显式设置 eval 模式
                    pipe.eval()
                    try:
                        generated_view = pipe(
                            processed_data["prompt"],
                            edit_image=edit_images,
                            seed=args.seed + idx,  # 不同样本使用不同种子
                            num_inference_steps=args.num_inference_steps,
                            height=1024,
                            width=1024,
                            edit_image_auto_resize=False,  # 与训练保持一致，禁用 auto_resize
                            tiled=args.tiled,
                            tile_size=args.tile_size,
                            tile_stride=args.tile_stride,
                        )
                    except RuntimeError as e:
                        if "CUDA" in str(e) or "illegal memory access" in str(e).lower():
                            print(f"[进程 {process_index}] CUDA 错误，尝试清理缓存并重试...")
                            torch.cuda.empty_cache()
                            pipe = pipe.to(device)
                            pipe.eval()
                            generated_view = pipe(
                                processed_data["prompt"],
                                edit_image=edit_images,
                                seed=args.seed + idx,
                                num_inference_steps=args.num_inference_steps,
                                height=1024,
                                width=1024,
                                edit_image_auto_resize=False,  # 与训练保持一致，禁用 auto_resize
                                tiled=args.tiled,
                                tile_size=args.tile_size,
                                tile_stride=args.tile_stride,
                            )
                        else:
                            raise
                    
                    # 保存单视角结果（像训练时一样）
                    output_name = f"{object_name}_{view_id}_output_albedo.png" if object_name else f"sample_{idx}_view_{view_id}.png"
                    pred_path = os.path.join(args.output_dir, "pred", output_name)
                    generated_view.save(pred_path)
                    print(f"[进程 {process_index}] 保存视角 {view_id} 结果: {pred_path}")
                    
                    # 保存输入图片
                    input_dir = os.path.join(args.output_dir, "pred")
                    albedo_input_name = f"{object_name}_{view_id}_input_albedo.png" if object_name else f"sample_{idx}_view_{view_id}_input_albedo.png"
                    albedo_input_path = os.path.join(input_dir, albedo_input_name)
                    albedo_view.save(albedo_input_path)
                    
                    normal_input_name = f"{object_name}_{view_id}_input_normal.png" if object_name else f"sample_{idx}_view_{view_id}_input_normal.png"
                    normal_input_path = os.path.join(input_dir, normal_input_name)
                    normal_view.save(normal_input_path)
                    
                    position_input_name = f"{object_name}_{view_id}_input_position.png" if object_name else f"sample_{idx}_view_{view_id}_input_position.png"
                    position_input_path = os.path.join(input_dir, position_input_name)
                    position_view.save(position_input_path)
                    
                    print(f"[进程 {process_index}] 保存视角 {view_id} 输入: albedo, normal, position")
                    
                    # 拼接并保存到 grid 目录
                    if args.save_grid:
                        SINGLE_VIEW_SIZE = 1024
                        grid_width = SINGLE_VIEW_SIZE * 4
                        grid_height = SINGLE_VIEW_SIZE
                        grid_img = Image.new("RGB", (grid_width, grid_height), (0, 0, 0))
                        
                        # 确保所有图片都是 1024x1024
                        if albedo_view.size != (SINGLE_VIEW_SIZE, SINGLE_VIEW_SIZE):
                            albedo_view = albedo_view.resize((SINGLE_VIEW_SIZE, SINGLE_VIEW_SIZE), Image.LANCZOS)
                        if normal_view.size != (SINGLE_VIEW_SIZE, SINGLE_VIEW_SIZE):
                            normal_view = normal_view.resize((SINGLE_VIEW_SIZE, SINGLE_VIEW_SIZE), Image.LANCZOS)
                        if position_view.size != (SINGLE_VIEW_SIZE, SINGLE_VIEW_SIZE):
                            position_view = position_view.resize((SINGLE_VIEW_SIZE, SINGLE_VIEW_SIZE), Image.LANCZOS)
                        if generated_view.size != (SINGLE_VIEW_SIZE, SINGLE_VIEW_SIZE):
                            generated_view = generated_view.resize((SINGLE_VIEW_SIZE, SINGLE_VIEW_SIZE), Image.LANCZOS)
                        
                        # 拼接：albedo | normal | position | generated
                        grid_img.paste(albedo_view, (0, 0))
                        grid_img.paste(normal_view, (SINGLE_VIEW_SIZE, 0))
                        grid_img.paste(position_view, (SINGLE_VIEW_SIZE * 2, 0))
                        grid_img.paste(generated_view, (SINGLE_VIEW_SIZE * 3, 0))
                        
                        grid_name = f"{object_name}_{view_id}_grid.png" if object_name else f"sample_{idx}_view_{view_id}_grid.png"
                        grid_path = os.path.join(args.output_dir, "grid", grid_name)
                        grid_img.save(grid_path)
                        print(f"[进程 {process_index}] 保存视角 {view_id} 拼接图: {grid_path}")
                
        except Exception as e:
            print(f"[进程 {process_index}] 处理失败 {object_name}/{view_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 恢复原始的 model_fn（如果之前设置了 cross-view attention）
    if enable_cross_view_attention and original_model_fn is not None:
        pipe.model_fn = original_model_fn
        if is_main_process:
            print("[Cross-View Attention] 已恢复原始 model_fn")
    
    accelerator.wait_for_everyone()
    
    if is_main_process:
        print("所有进程处理完成!")


if __name__ == "__main__":
    main()
