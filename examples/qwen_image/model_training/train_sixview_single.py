"""
六视图 Albedo 训练脚本 - 单视角训练（从拼接图中分割）

支持 v3 格式数据（prepare_six_view_albedo_v3.py 生成）：
- image: 六视角 albedo 拼接图路径
- edit_image[0]: 单视角 albedo 路径（如 "物体名/albedo/000.png"）
- edit_image[1]: 六视角 normal 拼接图路径
- edit_image[2]: 六视角 position 拼接图路径

训练时：
1. 从 edit_image[0] 的文件名中提取视角ID（如 "000"）
2. 从拼接图中分割出对应的单视角
3. 对单个视角进行训练

每个训练样本对应一个视角，训练时从拼接图中自动分割出该视角的数据。
"""

import torch, os, argparse, accelerate, json, socket
from datetime import datetime
from PIL import Image
from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.core.data.operators import *
from einops import rearrange
from torch import nn
import torch.distributed as dist
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def save_training_config(args, dataset, val_dataset, accelerator):
    """保存训练配置到 output_path/train_config.json"""
    if not accelerator.is_main_process:
        return
    
    os.makedirs(args.output_path, exist_ok=True)
    config_path = os.path.join(args.output_path, "train_config.json")
    
    # 获取 GPU 信息
    try:
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    except:
        gpu_count = 0
        gpu_names = []
    
    config = {
        "experiment_info": {
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": socket.gethostname(),
            "gpu_count": gpu_count,
            "gpu_names": gpu_names,
        },
        "data": {
            "data_root": args.dataset_base_path,
            "train_metadata": args.dataset_metadata_path,
            "val_metadata": args.val_dataset_metadata_path,
            "train_samples": len(dataset) if dataset else 0,
            "val_samples": len(val_dataset) if val_dataset else 0,
            "dataset_repeat": args.dataset_repeat,
            "data_file_keys": args.data_file_keys,
            "max_pixels": args.max_pixels,
        },
        "model": {
            "model_id_with_origin_paths": args.model_id_with_origin_paths,
            "lora_checkpoint": args.lora_checkpoint,
            "lora_rank": args.lora_rank,
            "lora_base_model": args.lora_base_model,
            "lora_target_modules": args.lora_target_modules,
        },
        "training": {
            "learning_rate": getattr(args, 'learning_rate', None),
            "num_epochs": getattr(args, 'num_epochs', None),
            "batch_size": getattr(args, 'batch_size', 1),
            "gradient_accumulation_steps": getattr(args, 'gradient_accumulation_steps', 1),
            "save_steps": getattr(args, 'save_steps', None),
            "val_steps": getattr(args, 'val_steps', None),
            "use_gradient_checkpointing": getattr(args, 'use_gradient_checkpointing', False),
            "task": getattr(args, 'task', 'sft'),
            "extra_inputs": getattr(args, 'extra_inputs', None),
        },
        "output": {
            "output_path": args.output_path,
        },
        "raw_args": vars(args),
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)
    
    print("=" * 50)
    print(f"训练配置已保存到: {config_path}")
    print(f"训练数据: {config['data']['train_samples']} 样本")
    print(f"验证数据: {config['data']['val_samples']} 样本")
    print(f"学习率: {args.learning_rate}")
    print(f"LoRA Rank: {args.lora_rank}")
    if args.lora_checkpoint:
        print(f"继续训练自: {args.lora_checkpoint}")
    print("=" * 50)


class QwenImageSixViewSingleTrainingModule(DiffusionTrainingModule):
    """
    六视图单独训练模块
    每个样本包含6个视角，对每个视角分别计算 loss
    """
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        zero_cond_t=False,
    ):
        super().__init__()
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/") if processor_path is None else ModelConfig(processor_path)
        self.pipe = QwenImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, processor_config=processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
    
        # 清理缓存
        torch.cuda.empty_cache()
        
        # Other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.zero_cond_t = zero_cond_t
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
        }
    
    def _create_cross_view_model_fn(self):
        """创建自定义的 model_fn，在每个 transformer block 后进行跨 GPU 通信"""
        original_model_fn = self.pipe.model_fn
        
        def cross_view_model_fn(**kwargs):
            """自定义的 model_fn，实现跨视角 attention"""
            from diffsynth.core import gradient_checkpoint_forward
            
            # 获取参数
            dit = kwargs.get("dit", self.pipe.dit)
            latents = kwargs.get("latents")
            timestep = kwargs.get("timestep")
            prompt_emb = kwargs.get("prompt_emb")
            prompt_emb_mask = kwargs.get("prompt_emb_mask")
            height = kwargs.get("height")
            width = kwargs.get("width")
            edit_latents = kwargs.get("edit_latents", None)
            
            # 准备输入
            layer_num = kwargs.get("layer_num", None)
            if layer_num is None:
                layer_num = 1
                img_shapes = [(1, latents.shape[2]//2, latents.shape[3]//2)]
            else:
                layer_num = layer_num + 1
                img_shapes = [(1, latents.shape[2]//2, latents.shape[3]//2)] * layer_num
            
            txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
            timestep_scaled = timestep / 1000
            
            # 获取 rank 和 world_size（用于调试打印和后续处理）
            import torch.distributed as dist
            if dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            else:
                world_size = 1
                rank = 0
            
            # 根据 latents 形状决定 reshape 方式
            if layer_num > 1 and latents.shape[0] % layer_num == 0:
                B = latents.shape[0] // layer_num
                image = rearrange(latents, "(B N) C (H P) (W Q) -> B (N H W) (C P Q)", H=height//16, W=width//16, P=2, Q=2, B=B, N=layer_num)
            else:
                image = rearrange(latents, "B C (H P) (W Q) -> B (H W) (C P Q)", H=height//16, W=width//16, P=2, Q=2)
                layer_num = 1
            
            image_seq_len = image.shape[1]
            
            # 调试打印：查看目标图像 latents 的尺寸
            # if rank == 0:
            #     print(f"[DEBUG] Target image latents:")
            #     print(f"  latents.shape = {latents.shape}")
            #     print(f"  latents latent space: H={latents.shape[2]}, W={latents.shape[3]}")
            #     print(f"  image_seq_len = {image_seq_len}")
            #     print(f"  height={height}, width={width}")
            
            if edit_latents is not None:
                edit_latents_list = edit_latents if isinstance(edit_latents, list) else [edit_latents]
                
                # 检查 edit_latents[1] 和 edit_latents[2] 是否是拼接图（包含所有6个视角）
                # 如果是拼接图，需要根据当前 rank 分割出对应视角
                # 拼接图布局：2行3列，每个视角在 latent 空间是 128x128
                processed_edit_latents = []
                for i, e in enumerate(edit_latents_list):
                    # edit_latents[0] 是单视角 albedo，不需要分割
                    # edit_latents[1] 和 edit_latents[2] 可能是拼接图（normal 和 position）
                    if i == 0:
                        # edit_latents[0]: 单视角 albedo，直接使用
                        processed_edit_latents.append(e)
                    elif i >= 1 and dist.is_initialized() and dist.get_world_size() >= 6 and rank < 6:
                        # edit_latents[1] 和 edit_latents[2]: 可能是拼接图，需要分割
                        # 检查是否是拼接图：latent 空间尺寸应该是 256x384（2行3列，每个视角128x128）
                        if e.shape[2] == 256 and e.shape[3] == 384:
                            # 这是拼接图，需要分割
                            view_idx = rank
                            NUM_COLS = 3
                            row = view_idx // NUM_COLS
                            col = view_idx % NUM_COLS
                            
                            # 每个视角在 latent 空间是 128x128
                            view_latent_h = 128
                            view_latent_w = 128
                            
                            y_start = row * view_latent_h
                            y_end = y_start + view_latent_h
                            x_start = col * view_latent_w
                            x_end = x_start + view_latent_w
                            
                            # 分割出对应视角的 latent
                            view_latent = e[:, :, y_start:y_end, x_start:x_end]  # [B, C, 128, 128]
                            processed_edit_latents.append(view_latent)
                            
                            # if rank == 0:
                            #     print(f"[DEBUG] Split edit_latents[{i}] from concatenated latent:")
                            #     print(f"  Original shape: {e.shape}")
                            #     print(f"  View idx: {view_idx} (row={row}, col={col})")
                            #     print(f"  Split region: H[{y_start}:{y_end}], W[{x_start}:{x_end}]")
                            #     print(f"  Split shape: {view_latent.shape}")
                        else:
                            # 不是拼接图，直接使用
                            processed_edit_latents.append(e)
                    else:
                        # 非6卡模式或其他情况，直接使用
                        processed_edit_latents.append(e)
                
                # 更新 img_shapes（使用分割后的尺寸）
                img_shapes += [(1, e.shape[2]//2, e.shape[3]//2) for e in processed_edit_latents]  # (frame, height, width)，frame=1
                
                # 调试打印：查看处理后的 edit_latents 的实际尺寸
                if rank == 0:
                    print(f"[DEBUG] edit_latents shapes (after split if needed):")
                    for i, e in enumerate(processed_edit_latents):
                        print(f"  edit_latents[{i}].shape = {e.shape}")
                        print(f"  edit_latents[{i}] latent space: H={e.shape[2]}, W={e.shape[3]}")
                        print(f"  img_shapes[{i+len(img_shapes)-len(processed_edit_latents)}] = {img_shapes[i+len(img_shapes)-len(processed_edit_latents)]}")
                        # 计算预期的序列长度
                        H_latent = e.shape[2] // 2
                        W_latent = e.shape[3] // 2
                        expected_seq_len = H_latent * W_latent
                        print(f"  Expected seq_len from rearrange: H={H_latent}, W={W_latent}, seq_len={expected_seq_len}")
                
                edit_image = [rearrange(e, "B C (H P) (W Q) -> B (H W) (C P Q)", H=e.shape[2]//2, W=e.shape[3]//2, P=2, Q=2) for e in processed_edit_latents]
                
                # 调试打印：查看 rearrange 后的实际序列长度
                # if rank == 0:
                #     print(f"[DEBUG] edit_image after rearrange:")
                #     for i, e in enumerate(edit_image):
                #         print(f"  edit_image[{i}].shape = {e.shape}, seq_len = {e.shape[1]}")
                
                image = torch.cat([image] + edit_image, dim=1)
            
            image = dit.img_in(image)
            conditioning = dit.time_text_embed(timestep_scaled, image.dtype)
            text = dit.txt_in(dit.txt_norm(prompt_emb))
            
            # 遍历所有 transformer blocks，在每个 block 后进行跨 GPU 通信
            # rank 和 world_size 已经在前面定义了
            num_views = 6
            cross_attention_interval = 4  # 每隔 4 层做一次 cross attention
            
            # 为 self attention 阶段生成多视角的位置编码
            # 在6卡模式下，每个 rank 应该使用对应视角的位置编码
            if dist.is_initialized() and dist.get_world_size() >= 6 and rank < num_views:
                # 6卡模式：为所有视角生成位置编码，然后根据 rank 提取对应的部分
                if len(img_shapes) > 0:
                    _, height, width = img_shapes[0]  # 单个视角的尺寸
                    # 计算拼接图的尺寸（1行6列布局）
                    layout_rows, layout_cols = 1, 6
                    concat_height = height * layout_rows  # 1 * height = height
                    concat_width = width * layout_cols   # 6 * width
                    
                    # 为整个拼接图生成位置编码（只针对目标图像部分）
                    concat_img_shapes = [(1, concat_height, concat_width)]
                    concat_rotary_emb = dit.pos_embed(concat_img_shapes, txt_seq_lens, device=latents.device)
                    img_freqs_concat, txt_freqs_concat = concat_rotary_emb
                    
                    # 计算 latent 空间尺寸
                    latent_h = height // 16
                    latent_w = width // 16
                    concat_latent_h = concat_height // 16
                    concat_latent_w = concat_width // 16
                    
                    # 将位置编码 reshape 到空间维度
                    img_freqs_spatial = img_freqs_concat.view(concat_latent_h, concat_latent_w, -1)  # (H_concat, W_concat, rope_dim)
                    
                    # 根据当前 rank 提取对应视角的位置编码
                    row = rank // layout_cols  # 都是 0（只有一行）
                    col = rank % layout_cols   # 0->0, 1->1, 2->2, 3->3, 4->4, 5->5
                    
                    y_start = row * latent_h
                    y_end = y_start + latent_h
                    x_start = col * latent_w
                    x_end = x_start + latent_w
                    
                    # 提取当前视角的位置编码
                    view_freqs = img_freqs_spatial[y_start:y_end, x_start:x_end, :]  # (latent_h, latent_w, rope_dim)
                    view_freqs = view_freqs.reshape(-1, img_freqs_concat.shape[-1])  # (latent_h * latent_w, rope_dim)
                    
                    # 调试打印
                    # if rank == 0 and edit_latents is not None:
                    #     print(f"[DEBUG rank={rank}] Position encoding calculation:")
                    #     print(f"  image_seq_len = {image_seq_len}")
                    #     print(f"  view_freqs.shape[0] = {view_freqs.shape[0]}")
                    #     print(f"  latent_h={latent_h}, latent_w={latent_w}")
                    #     print(f"  height={height}, width={width}")
                    #     print(f"  img_shapes[0] = {img_shapes[0] if len(img_shapes) > 0 else 'None'}")
                    #     print(f"  latents.shape = {latents.shape}")
                    #     if len(edit_image) > 0:
                    #         print(f"  edit_image[0].shape = {edit_image[0].shape}")
                    
                    # 如果有 edit_image，需要为它们生成位置编码
                    # edit_image 结构: [albedo_input, normal, position]
                    # - edit_image[0]: 输入 albedo（单视角，可能是任意视角，应该使用单视角的位置编码）
                    # - edit_image[1]: normal（目标视角的 normal，对应当前 rank 的视角，使用当前视角的位置编码）
                    # - edit_image[2]: position（目标视角的 position，对应当前 rank 的视角，使用当前视角的位置编码）
                    if edit_latents is not None:
                        # edit_image 的数量
                        num_edit_images = len(edit_latents_list)
                        
                        # 为 edit_image[0] 生成单视角的位置编码（输入 albedo 是单视角的）
                        edit_img_shapes_0 = img_shapes[1:2] if len(img_shapes) > 1 else [(1, height, width)]  # 只取第一个 edit_image 的形状
                        edit_rotary_emb_0 = dit.pos_embed(edit_img_shapes_0, txt_seq_lens, device=latents.device)
                        edit_img_freqs_0, _ = edit_rotary_emb_0
                        
                        # 计算每个 edit_image 的实际序列长度
                        edit_seq_lens = [e.shape[1] for e in edit_image]
                        
                        # 调试打印
                        # if rank == 0:
                        #     print(f"[DEBUG rank={rank}] Edit image position encoding:")
                        #     print(f"  edit_img_shapes_0 = {edit_img_shapes_0}")
                        #     print(f"  edit_img_freqs_0.shape[0] = {edit_img_freqs_0.shape[0]}")
                        #     print(f"  edit_seq_lens = {edit_seq_lens}")
                        #     for i, e in enumerate(edit_latents_list):
                        #         print(f"  edit_latents[{i}].shape = {e.shape}")
                        #     for i, e in enumerate(edit_image):
                        #         print(f"  edit_image[{i}].shape = {e.shape}")
                        
                        # 为其他 edit_image（normal, position）使用当前视角的位置编码
                        # edit_image[1] 和 edit_image[2] 是对应目标视角的，使用当前视角的位置编码
                        edit_freqs_list = [edit_img_freqs_0]  # edit_image[0] 使用单视角位置编码
                        if num_edit_images > 1:
                            # edit_image[1] 和之后的都使用当前视角的位置编码
                            edit_freqs_list.extend([view_freqs] * (num_edit_images - 1))
                        
                        # 拼接目标图像和 edit_image 的位置编码
                        # 目标图像使用当前视角的位置编码
                        # edit_image[0] 使用单视角的位置编码
                        # edit_image[1] 和之后的使用当前视角的位置编码
                        img_freqs_combined = torch.cat([view_freqs] + edit_freqs_list, dim=0)
                        
                        # 验证总长度
                        expected_total_len = image_seq_len + sum(edit_seq_lens)
                        actual_total_len = img_freqs_combined.shape[0]
                        # if rank == 0:
                        #     print(f"[DEBUG rank={rank}] Total position encoding length check:")
                        #     print(f"  actual_total_len = {actual_total_len}")
                        #     print(f"  expected_total_len = {expected_total_len}")
                        #     print(f"  image_seq_len = {image_seq_len}")
                        #     print(f"  sum(edit_seq_lens) = {sum(edit_seq_lens)}")
                        #     print(f"  view_freqs.shape[0] = {view_freqs.shape[0]}")
                        #     print(f"  edit_freqs_list lengths = {[f.shape[0] for f in edit_freqs_list]}")
                        if actual_total_len != expected_total_len:
                            raise ValueError(f"Position encoding total length ({actual_total_len}) != expected total length ({expected_total_len}). "
                                           f"image_seq_len={image_seq_len}, edit_seq_lens={edit_seq_lens}")
                        
                        # 文本位置编码保持不变
                        txt_freqs = txt_freqs_concat
                        image_rotary_emb = (img_freqs_combined, txt_freqs)
                        
                        del concat_rotary_emb, img_freqs_concat, img_freqs_spatial, view_freqs, edit_rotary_emb_0, edit_img_freqs_0, edit_freqs_list
                    else:
                        # 没有 edit_image，只使用当前视角的位置编码
                        txt_freqs = txt_freqs_concat
                        image_rotary_emb = (view_freqs, txt_freqs)
                        
                        del concat_rotary_emb, img_freqs_concat, img_freqs_spatial, view_freqs
                    torch.cuda.empty_cache()
                else:
                    # 如果没有 img_shapes，使用原始方式
                    image_rotary_emb = dit.pos_embed(img_shapes, txt_seq_lens, device=latents.device)
            else:
                # 非6卡模式：使用原始方式
                image_rotary_emb = dit.pos_embed(img_shapes, txt_seq_lens, device=latents.device)
            
            attention_mask = None
            
            for block_id, block in enumerate(dit.transformer_blocks):
                if rank < num_views:
                    text, image = gradient_checkpoint_forward(
                        block,
                        kwargs.get("use_gradient_checkpointing", False),
                        kwargs.get("use_gradient_checkpointing_offload", False),
                        image=image,
                        text=text,
                        temb=conditioning,
                        image_rotary_emb=image_rotary_emb,
                        attention_mask=attention_mask,
                        enable_fp8_attention=kwargs.get("enable_fp8_attention", False),
                        modulate_index=None,
                    )
                
                # 跨视角 Cross Attention：每隔 N 层做一次，减少显存占用
                should_do_cross_attention = (block_id + 1) % cross_attention_interval == 0
                
                if rank < num_views and should_do_cross_attention:
                    # 收集所有视角的 features
                    image_view = image[:, :image_seq_len].clone()
                    gathered_images = [torch.zeros_like(image_view) for _ in range(num_views)]
                    dist.all_gather(gathered_images, image_view)
                    kv_all = torch.cat(gathered_images, dim=1)  # [B, seq_len * 6, dim]
                    
                    from diffsynth.models.qwen_image_dit import qwen_image_flash_attention, apply_rotary_emb_qwen
                    num_heads = block.attn.num_heads
                    
                    # 为所有视角生成 rotary embedding
                    # 注意：kv_all 是在序列维度上直接 cat 的：[view0, view1, view2, view3, view4, view5]
                    # 为了位置编码对齐，将6个视角看作一个大的拼接图（1行6列布局）来计算位置编码
                    # 这样位置编码的顺序直接对应序列拼接的顺序
                    if image_rotary_emb is not None:
                        img_freqs, _ = image_rotary_emb
                        if len(img_shapes) > 0:
                            _, height, width = img_shapes[0]  # 单个视角的尺寸
                            # 计算拼接图的尺寸（1行6列布局）
                            layout_rows, layout_cols = 1, 6
                            concat_height = height * layout_rows  # 1 * height = height
                            concat_width = width * layout_cols   # 6 * width
                            
                            # 为整个拼接图生成位置编码
                            concat_img_shapes = [(1, concat_height, concat_width)]
                            concat_rotary_emb = dit.pos_embed(concat_img_shapes, txt_seq_lens, device=latents.device)
                            img_freqs_concat, _ = concat_rotary_emb
                            
                            # 计算 latent 空间尺寸
                            latent_h = height // 16
                            latent_w = width // 16
                            concat_latent_h = concat_height // 16
                            concat_latent_w = concat_width // 16
                            
                            # 将位置编码 reshape 到空间维度
                            img_freqs_spatial = img_freqs_concat.view(concat_latent_h, concat_latent_w, -1)  # (H_concat, W_concat, rope_dim)
                            
                            # 提取每个视角在拼接图中的位置编码
                            # gathered_images 的顺序是 [rank0(view000), rank1(view001), rank2(view002), rank3(view003), rank4(view004), rank5(view005)]
                            # 对应1行6列布局：[000, 001, 002, 003, 004, 005]
                            view_freqs_list = []
                            for view_idx in range(num_views):
                                row = view_idx // layout_cols  # 都是 0（只有一行）
                                col = view_idx % layout_cols   # 0->0, 1->1, 2->2, 3->3, 4->4, 5->5
                                
                                y_start = row * latent_h
                                y_end = y_start + latent_h
                                x_start = col * latent_w
                                x_end = x_start + latent_w
                                
                                # 提取该视角对应的位置编码
                                view_freqs = img_freqs_spatial[y_start:y_end, x_start:x_end, :]  # (latent_h, latent_w, rope_dim)
                                view_freqs = view_freqs.reshape(-1, img_freqs_concat.shape[-1])  # (latent_h * latent_w, rope_dim)
                                view_freqs_list.append(view_freqs)
                            
                            # 拼接所有视角的位置编码（按照 gathered_images 的顺序：rank 0-5，对应 view 000-005）
                            img_freqs_kv = torch.cat(view_freqs_list, dim=0)  # (seq_len * 6, rope_dim)
                            
                            del concat_rotary_emb, img_freqs_concat, img_freqs_spatial, view_freqs_list
                            torch.cuda.empty_cache()
                        else:
                            # 如果没有 img_shapes，回退到重复模式
                            img_freqs_kv = img_freqs[:image_seq_len].repeat(num_views, 1)
                    else:
                        img_freqs_kv = None
                    
                    # 计算所有视角的 key/value
                    kv_all_norm = block.img_norm1(kv_all)
                    k_all = rearrange(block.attn.to_k(kv_all_norm), 'b s (h d) -> b h s d', h=num_heads)
                    v_all = rearrange(block.attn.to_v(kv_all_norm), 'b s (h d) -> b h s d', h=num_heads)
                    k_all = block.attn.norm_k(k_all)
                    if img_freqs_kv is not None:
                        k_all = apply_rotary_emb_qwen(k_all, img_freqs_kv)
                    
                    # 计算当前视角的 query
                    view_i_norm = block.img_norm1(image_view)
                    q_i = rearrange(block.attn.to_q(view_i_norm), 'b s (h d) -> b h s d', h=num_heads)
                    q_i = block.attn.norm_q(q_i)
                    if img_freqs_kv is not None and image_rotary_emb is not None:
                        # 根据当前 rank 提取对应视角的位置编码
                        # img_freqs_kv 的顺序是 [view0, view1, view2, view3, view4, view5]
                        view_start_idx = rank * image_seq_len
                        view_end_idx = view_start_idx + image_seq_len
                        img_freqs_q = img_freqs_kv[view_start_idx:view_end_idx]  # 当前视角的位置编码
                        q_i = apply_rotary_emb_qwen(q_i, img_freqs_q)
                        del img_freqs_q
                        torch.cuda.empty_cache()
                    
                    # Cross attention
                    attn_out_i = qwen_image_flash_attention(q_i, k_all, v_all, num_heads=num_heads, attention_mask=None, enable_fp8_attention=False)
                    attn_out_i = block.attn.to_out(attn_out_i)
                    image[:, :image_seq_len] = image_view + attn_out_i
                    
                    # 清理内存
                    del q_i, view_i_norm, kv_all_norm, k_all, v_all, attn_out_i, gathered_images, kv_all
                    if img_freqs_kv is not None:
                        del img_freqs_kv
                    torch.cuda.empty_cache()
            
            # 完成 DiT forward
            image = dit.norm_out(image, conditioning)
            image = dit.proj_out(image)
            image = image[:, :image_seq_len]
            
            # 重新排列回 latents 格式
            H_latent, W_latent = latents.shape[2], latents.shape[3]
            H_W, W_W = H_latent // 2, W_latent // 2
            
            if layer_num > 1 and latents.shape[0] % layer_num == 0:
                B = latents.shape[0] // layer_num
                latents = rearrange(image, "B (N H W) (C P Q) -> (B N) C (H P) (W Q)", H=H_W, W=W_W, P=2, Q=2, B=B, N=layer_num)
            else:
                latents = rearrange(image, "B (H W) (C P Q) -> B C (H P) (W Q)", H=H_W, W=W_W, P=2, Q=2)
            return latents
        
        return cross_view_model_fn
    
    def _build_view_inputs(self, data, view_data):
        """构建单个视角的输入"""
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        
        inputs_shared = {
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "edit_image_auto_resize": False,  # 禁用 auto_resize，因为 ProcessV3DataOperator 已经确保所有图像都是 1024x1024
            "zero_cond_t": self.zero_cond_t,
        }
        
        # 设置 input_image（目标 albedo）
        target_image = view_data["albedo_target"]
        if isinstance(target_image, list):
            inputs_shared.update({
                "input_image": target_image,
                "height": target_image[0].size[1],
                "width": target_image[0].size[0],
            })
        else:
            inputs_shared.update({
                "input_image": target_image,
                "height": target_image.size[1],
                "width": target_image.size[0],
            })
        
        # 添加 edit_image（输入条件）
        inputs_shared["edit_image"] = [view_data["albedo_input"], view_data["normal"], view_data["position"]]
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        
        return self.transfer_data_to_device((inputs_shared, inputs_posi, inputs_nega), self.pipe.device, self.pipe.torch_dtype)
    
    def forward(self, data, inputs=None):
        """
        处理 v3 格式数据：对所有视角分别计算 loss，然后累加
        
        data 应该包含（已由 ProcessV3DataOperator 处理）:
        - views: 视角数据列表，每个元素包含 {"albedo_target", "albedo_input", "normal", "position", "view_id"}
        """
        views = data.get("views", [])
        # 打印每张卡处理的样本和视角信息（用于验证）
        if dist.is_initialized() and dist.get_world_size() >= 6:
            rank = dist.get_rank()
            prompt = data.get("prompt", "unknown")
            prompt_str = prompt[:50] if isinstance(prompt, str) and len(prompt) > 50 else str(prompt)
            view_ids = [v.get("view_id", "unknown") for v in views] if views else []
            
            # 获取 albedo 文件路径信息
            # 优先从保存的路径信息获取（ProcessV3DataOperator 保存的）
            albedo_info = "unknown"
            if "_albedo_path" in data:
                albedo_info = data["_albedo_path"]
            elif "_albedo_concat_path" in data:
                albedo_info = data["_albedo_concat_path"]
            elif "edit_image" in data and isinstance(data["edit_image"], list) and len(data["edit_image"]) > 0:
                albedo_path = data["edit_image"][0]
                if isinstance(albedo_path, str):
                    albedo_info = albedo_path
                else:
                    albedo_info = f"type={type(albedo_path).__name__}"
            elif "image" in data:
                image_path = data["image"]
                if isinstance(image_path, str):
                    albedo_info = image_path
                else:
                    albedo_info = f"type={type(image_path).__name__}"
            
            # print(f"[Training Module GPU {rank}] Albedo='{albedo_info}', Prompt: {prompt_str}, Views: {view_ids}")
        
        # 处理视角数据
        if len(views) == 1:
            # 分布式模式：只处理当前GPU对应的视角
            view_data = views[0]
            inputs = self._build_view_inputs(data, view_data)
            
            # 运行 pipeline units
            for unit in self.pipe.units:
                inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
            
            # 检查是否启用跨视角 attention（6个GPU且是分布式模式）
            enable_cross_view_attention = (
                dist.is_initialized() and 
                dist.get_world_size() == 6 and 
                len(views) == 1
            )
            
            if enable_cross_view_attention:
                original_model_fn = self.pipe.model_fn
                self.pipe.model_fn = self._create_cross_view_model_fn()
            
            view_loss = self.task_to_loss[self.task](self.pipe, *inputs)
            
            if enable_cross_view_attention:
                self.pipe.model_fn = original_model_fn
            
            return view_loss
        else:
            # 单GPU模式：处理所有视角并累加 loss
            total_loss = 0.0
            for view_data in views:
                inputs = self._build_view_inputs(data, view_data)
                
                # 运行 pipeline units
                for unit in self.pipe.units:
                    inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
                
                total_loss += self.task_to_loss[self.task](self.pipe, *inputs)
            
            return total_loss


def qwen_image_sixview_single_parser():
    from diffsynth.diffusion.parsers import add_general_config, add_image_size_config
    
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit Six-View Single Training")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor. If provided, the processor will be used for image editing.")
    parser.add_argument("--zero_cond_t", default=False, action="store_true", help="A special parameter introduced by Qwen-Image-Edit-2511. Please enable it for this model.")
    parser.add_argument("--edit_image_auto_resize", default=True, action="store_true", help="Skip ImageCropAndResize for image and edit_image fields (useful for six-view concatenated images)")
    return parser


def split_six_view_image(concat_img: Image.Image, view_id: str):
    """
    从六视图拼接图中分割出单个视角
    
    Args:
        concat_img: 拼接图 (3072x2048, 2行3列)
        view_id: 视角ID ("000", "001", ..., "005")
    
    Returns:
        单视角图像 (1024x1024)
    """
    view_ids = ["000", "001", "002", "003", "004", "005"]
    view_idx = view_ids.index(view_id)
    
    SINGLE_VIEW_SIZE = 1024
    NUM_COLS = 3
    
    row = view_idx // NUM_COLS
    col = view_idx % NUM_COLS
    
    x = col * SINGLE_VIEW_SIZE
    y = row * SINGLE_VIEW_SIZE
    
    # 裁剪单视角
    single_view = concat_img.crop((x, y, x + SINGLE_VIEW_SIZE, y + SINGLE_VIEW_SIZE))
    return single_view


class SaveOriginalPathsOperator(DataProcessingOperator):
    """保存原始路径信息，用于后续打印和调试
    只在 __root__ operator 中使用，接收 data 字典
    注意：当 edit_image 已经被处理成 Image 对象时，我们需要从原始 metadata 中获取路径
    """
    def __init__(self, dataset=None):
        """
        Args:
            dataset: UnifiedDataset 实例，用于访问原始 metadata（可选）
        """
        self.dataset = dataset
    
    def __call__(self, data):
        # 首先尝试从当前 data 中获取路径（如果 edit_image 还是字符串）
        edit_image = data.get("edit_image", [])
        if edit_image and len(edit_image) >= 3:
            # 检查 edit_image[0] 是否是字符串
            if isinstance(edit_image[0], str):
                # 还是字符串，直接保存
                data["_original_edit_image_paths"] = [edit_image[0], edit_image[1], edit_image[2]]
            else:
                # 已经被处理成 Image，尝试从原始 metadata 中获取
                original_metadata = data.get("_original_metadata", {})
                if original_metadata and "edit_image" in original_metadata:
                    original_edit_image = original_metadata["edit_image"]
                    if isinstance(original_edit_image, list) and len(original_edit_image) >= 3:
                        if isinstance(original_edit_image[0], str):
                            data["_original_edit_image_paths"] = [original_edit_image[0], original_edit_image[1], original_edit_image[2]]
                        else:
                            data["_original_edit_image_paths"] = [None, None, None]
                    else:
                        data["_original_edit_image_paths"] = [None, None, None]
                else:
                    data["_original_edit_image_paths"] = [None, None, None]
        
        # 保存 image 的原始路径
        image = data.get("image")
        if isinstance(image, str):
            data["_original_image_path"] = image
        else:
            # 已经被处理成 Image，尝试从原始 metadata 中获取
            original_metadata = data.get("_original_metadata", {})
            if original_metadata and "image" in original_metadata:
                original_image = original_metadata["image"]
                if isinstance(original_image, str):
                    data["_original_image_path"] = original_image
                else:
                    data["_original_image_path"] = None
            else:
                data["_original_image_path"] = None
        
        return data


class ProcessV3DataOperator(DataProcessingOperator):
    """处理 v3 格式数据：从拼接图中分割出所有6个视角，或根据GPU rank只处理对应视角"""
    def __init__(self, base_path, height, width, max_pixels, gpu_rank=None, world_size=None):
        self.base_path = base_path
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.image_operator = ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, 16, 16)
        self.view_ids = ["000", "001", "002", "003", "004", "005"]
        self.gpu_rank = gpu_rank  # GPU rank (0-5)
        self.world_size = world_size  # 总GPU数量
    
    def __call__(self, data):
        """
        处理 v3 格式数据：
        - image: 六视角 albedo 拼接图路径
        - edit_image[0]: 单视角 albedo 路径（用于确定物体）
        - edit_image[1]: 六视角 normal 拼接图路径
        - edit_image[2]: 六视角 position 拼接图路径
        
        返回处理后的数据，包含所有6个视角的图像：
        - image: 第一个视角的 albedo（用于 UnifiedDataset 兼容性）
        - edit_image: [第一个视角的 albedo, normal, position]（用于 UnifiedDataset 兼容性）
        - views: 所有6个视角的数据列表
          - 每个元素: {"albedo": ..., "normal": ..., "position": ..., "view_id": "000"}
        """
        edit_image = data.get("edit_image", [])
        if not edit_image or len(edit_image) < 3:
            return data
        
        # 获取原始路径信息（从 SaveOriginalPathsOperator 保存的）
        # 优先使用保存的原始路径
        if "_original_edit_image_paths" in data and data["_original_edit_image_paths"]:
            if data["_original_edit_image_paths"][0] is not None:
                single_albedo_path = data["_original_edit_image_paths"][0]
                data["_albedo_path"] = single_albedo_path
            else:
                # 路径是 None，说明已经被处理成 Image 了，无法获取路径
                single_albedo_path = edit_image[0]
                data["_albedo_path"] = "unknown_image"
        elif isinstance(edit_image[0], str):
            # 如果 edit_image[0] 还是字符串，直接使用
            single_albedo_path = edit_image[0]
            data["_albedo_path"] = single_albedo_path
        else:
            # edit_image[0] 已经是 Image 对象，无法获取路径
            single_albedo_path = edit_image[0]
            data["_albedo_path"] = "unknown_image"
        
        # 获取 image 的原始路径
        if "_original_image_path" in data:
            image_path = data["_original_image_path"]
            data["_albedo_concat_path"] = image_path
        elif isinstance(data.get("image"), str):
            image_path = data["image"]
            data["_albedo_concat_path"] = image_path
        else:
            image_path = data.get("image")
            data["_albedo_concat_path"] = "unknown_image"
        
        # 加载原始单视角 albedo（edit_image[0]）
        if isinstance(single_albedo_path, str):
            single_albedo = Image.open(os.path.join(self.base_path, single_albedo_path)).convert("RGB")
        else:
            single_albedo = single_albedo_path if isinstance(single_albedo_path, Image.Image) else None
        
        # 加载拼接图
        image_path = data.get("image")
        print(f"image_path {image_path}")
        if isinstance(image_path, str) and image_path.strip():  # 检查路径不为空字符串
            albedo_concat = Image.open(os.path.join(self.base_path, image_path)).convert("RGB")
        elif isinstance(image_path, Image.Image):
            albedo_concat = image_path
        else:
            albedo_concat = None
        
        normal_path = edit_image[1]
        if isinstance(normal_path, str):
            normal_concat = Image.open(os.path.join(self.base_path, normal_path)).convert("RGB")
        else:
            normal_concat = normal_path if isinstance(normal_path, Image.Image) else None
        
        position_path = edit_image[2]
        if isinstance(position_path, str):
            position_concat = Image.open(os.path.join(self.base_path, position_path)).convert("RGB")
        else:
            position_concat = position_path if isinstance(position_path, Image.Image) else None
        
        # 检查必需的数据：single_albedo, normal_concat, position_concat 必须存在
        # albedo_concat 可以为空（当 image 为空时，使用 single_albedo 作为目标）
        if not single_albedo or not normal_concat or not position_concat:
            return data
        
        # 根据GPU rank决定处理哪些视角
        # 如果 world_size >= 6，每个GPU处理一个视角；否则每个GPU处理所有视角
        views = []  # 初始化 views 列表
        if self.world_size is not None and self.world_size >= 6 and self.gpu_rank is not None and self.gpu_rank < 6:
            # 分布式模式：每个GPU只处理一个视角
            view_idx = self.gpu_rank % len(self.view_ids)
            view_id = self.view_ids[view_idx]
            views_to_process = [view_id]
        else:
            # 单GPU或GPU数量不足6：处理所有视角
            views_to_process = self.view_ids
        
        # 从拼接图中分割出需要处理的视角
        for view_id in views_to_process:
            # 分割每个视角 1024×1024 PIL.Image.Image
            if albedo_concat is None:
                # image 为空时，使用 single_albedo 作为所有视角的目标 albedo
                albedo_view_target = single_albedo.copy()
            else:
                albedo_view_target = split_six_view_image(albedo_concat, view_id)  # 目标albedo（从拼接图分割）
            normal_view = split_six_view_image(normal_concat, view_id)
            position_view = split_six_view_image(position_concat, view_id)
            
            # 调试打印：查看分割后的图像尺寸
            # if self.gpu_rank == 0:
            #     print(f"[DEBUG ProcessV3DataOperator] After split, view_id={view_id}:")
            #     print(f"  albedo_view_target.size = {albedo_view_target.size}")
            #     print(f"  normal_view.size = {normal_view.size}")
            #     print(f"  position_view.size = {position_view.size}")
            
            # 对于 edit_image[0]，使用原始单视角albedo（如果当前视角匹配）
            # 否则使用从拼接图分割出来的albedo
            # 这里我们总是使用原始单视角albedo作为输入条件
            albedo_view_input = single_albedo.copy()
            
            # 不进行 resize，保持原始1024x1024分辨率
            # 如果设置了 height 和 width，进行 resize（但通常不设置，保持1024）
            if self.height and self.width:
                albedo_view_target = albedo_view_target.resize((self.width, self.height), Image.LANCZOS)
                albedo_view_input = albedo_view_input.resize((self.width, self.height), Image.LANCZOS)
                normal_view = normal_view.resize((self.width, self.height), Image.LANCZOS)
                position_view = position_view.resize((self.width, self.height), Image.LANCZOS)
                # print(f"edit size {albedo_view_target.size}, {albedo_view_input.size}, {normal_view.size}, {position_view.size}")
            
            views.append({
                "albedo_target": albedo_view_target,  # 目标albedo（用于计算loss）
                "albedo_input": albedo_view_input,    # 输入albedo（作为条件）
                "normal": normal_view,
                "position": position_view,
                "view_id": view_id,
            })
            print(f"views {views}")
        
        # 添加视角的数据（可能是1个或6个，取决于是否分布式）
        # 这是 forward 方法实际使用的数据
        data["views"] = views
        
        # 确保路径信息已保存（如果之前没有保存，使用当前获取的路径）
        if "_albedo_path" not in data:
            if isinstance(single_albedo_path, str):
                data["_albedo_path"] = single_albedo_path
            elif "_original_edit_image_paths" in data and data["_original_edit_image_paths"]:
                if data["_original_edit_image_paths"][0] is not None:
                    data["_albedo_path"] = data["_original_edit_image_paths"][0]
        
        if "_albedo_concat_path" not in data:
            if isinstance(image_path, str):
                data["_albedo_concat_path"] = image_path
            elif "_original_image_path" in data:
                data["_albedo_concat_path"] = data["_original_image_path"]
        
        return data


def create_dataset(args, metadata_path=None, repeat=1, accelerator=None):
    """创建数据集 - 支持 v3 格式（从拼接图中分割单视角）
    
    Args:
        args: 训练参数
        metadata_path: 元数据路径
        repeat: 数据集重复次数
        accelerator: Accelerator实例（用于获取GPU rank和world_size）
    """
    if metadata_path is None:
        metadata_path = args.dataset_metadata_path
    
    # 获取GPU信息（如果提供了accelerator）
    gpu_rank = None
    world_size = None
    if accelerator is not None:
        gpu_rank = accelerator.process_index
        world_size = accelerator.num_processes
    
    # 保存原始路径的 operator（需要在其他 operator 之前执行）
    save_paths_operator = SaveOriginalPathsOperator()
    
    # v3 数据处理器：从拼接图中分割单视角
    # 如果world_size >= 6，每个GPU只处理一个视角
    v3_processor = ProcessV3DataOperator(
        args.dataset_base_path, 
        args.height, 
        args.width, 
        args.max_pixels,
        gpu_rank=gpu_rank,
        world_size=world_size,
    )
    
    # 如果跳过 ImageCropAndResize，为 image 和 edit_image 提供只加载图片的 operator
    special_operator_map = {
        "__root__": save_paths_operator >> v3_processor,
    }
    
    if getattr(args, 'skip_image_resize', False):
        # 只加载图片，不进行 resize
        # 注意：SaveOriginalPathsOperator 需要在 LoadImage 之前执行，以保存原始路径
        load_image_only = RouteByType(operator_map=[
            (str, ToAbsolutePath(args.dataset_base_path) >> LoadImage()),
            (list, SequencialProcess(ToAbsolutePath(args.dataset_base_path) >> LoadImage())),
        ])
        # 注意：当 edit_image 被处理时，它已经是列表了，SaveOriginalPathsOperator 无法访问 data 字典
        # 所以路径信息会在 __root__ operator 的 SaveOriginalPathsOperator 中处理
        # 但此时 edit_image 可能已经被处理成 Image 了，所以会显示 "unknown_image"
        special_operator_map["image"] = load_image_only
        special_operator_map["edit_image"] = load_image_only
    
    return UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=metadata_path,
        repeat=repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        ),
        special_operator_map=special_operator_map,
    )


if __name__ == "__main__":
    parser = qwen_image_sixview_single_parser()
    args = parser.parse_args()
    
    # 检查是否是6卡模式（用于 cross-view attention）
    import torch.distributed as dist
    is_six_gpu_mode = dist.is_initialized() and dist.get_world_size() >= 6
    
    # 在6卡模式下，设置 split_batches=False，确保所有GPU处理相同的batch
    # 这样 SameSampleSampler 可以确保所有GPU处理相同的样本
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        split_batches=False if is_six_gpu_mode else None,  # 6卡模式：不分割batch，所有GPU处理相同样本
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    
    # 创建训练数据集（传入accelerator以支持分布式视角分配）
    dataset = create_dataset(args, accelerator=accelerator)
    
    # 创建验证数据集（如果提供了验证数据路径）
    # 验证时使用和训练时相同的数据处理逻辑
    val_dataset = None
    if args.val_dataset_metadata_path is not None:
        val_dataset = create_dataset(args, metadata_path=args.val_dataset_metadata_path, repeat=1, accelerator=accelerator)
    
    model = QwenImageSixViewSingleTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device=accelerator.device,
        zero_cond_t=args.zero_cond_t,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    
    # 保存训练配置
    save_training_config(args, dataset, val_dataset, accelerator)
    
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
    }
    
    # 对于训练任务，传入验证数据集
    if args.task in ["sft", "sft:train"]:
        launcher_map[args.task](accelerator, dataset, model, model_logger, val_dataset=val_dataset, args=args)
    else:
        launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
