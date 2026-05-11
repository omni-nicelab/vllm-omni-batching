import torch, os, argparse, accelerate
from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.core.data.operators import *

import deepspeed
from diffsynth.diffusion.runner import launch_sp_training_task


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class QwenImageTrainingModule(DiffusionTrainingModule):
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
        edit_image_auto_resize=True,
        sp_degree=1,
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
        
        if sp_degree > 1:
            self.pipe.enable_sp_training()

        # Other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.zero_cond_t = zero_cond_t
        self.edit_image_auto_resize = edit_image_auto_resize
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "edit_image_auto_resize": self.edit_image_auto_resize,
            "zero_cond_t": self.zero_cond_t,
        }
        # Assume you are using this pipeline for inference,
        # please fill in the input parameters.
        if isinstance(data["image"], list):
            inputs_shared.update({
                "input_image": data["image"],
                "height": data["image"][0].size[1],
                "width": data["image"][0].size[0],
            })
        else:
            inputs_shared.update({
                "input_image": data["image"],
                "height": data["image"].size[1],
                "width": data["image"].size[0],
            })
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor. If provided, the processor will be used for image editing.")
    parser.add_argument("--zero_cond_t", default=False, action="store_true", help="A special parameter introduced by Qwen-Image-Edit-2511. Please enable it for this model.")
    parser.add_argument("--edit_image_auto_resize", default=False, action="store_true", help="Skip ImageCropAndResize for image and edit_image fields (useful for six-view concatenated images)")
    parser.add_argument("--sp_degree", type=int, default=1, help="Sequence parallel degree. Must divide world_size. Requires mh_parallel_ext")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    parser.add_argument("--deepspeed_config_path", type=str, default=None, help="Path to DeepSpeed config file.")
    return parser


def create_dataset(args, metadata_path=None, repeat=1):
    """创建数据集的辅助函数"""
    if metadata_path is None:
        metadata_path = args.dataset_metadata_path
    
     # 如果跳过 ImageCropAndResize，为 image 和 edit_image 提供只加载图片的 operator
    special_operator_map = {
    }

    if getattr(args, 'edit_image_auto_resize', False):
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
        print("we use edit_image_auto_resize")

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
        special_operator_map=special_operator_map
    )

def patch_accelerator_with_deepspeed_mpu(accelerator: accelerate.Accelerator, parallel_state):

    _saved_accelerator_prepare_deepspeed = accelerator._prepare_deepspeed
    _saved_ds_initialize = deepspeed.initialize

    def patched_ds_initialize(*args, **kwargs):

        dp_world_size = parallel_state.get_data_parallel_world_size()
        batch_per_gpu = kwargs.get("train_micro_batch_size_per_gpu", 1)
        grad_acc_steps = kwargs.get("gradient_accumulation_steps", 1)

        print("Patching deepspeed.initialize to inject parallel_state for SP training.")
        kwargs["mpu"] = parallel_state
        kwargs["config_params"].update(
            {"train_batch_size": batch_per_gpu * grad_acc_steps * dp_world_size}
        )
        return _saved_ds_initialize(*args, **kwargs)

    def patched_prepare_deepspeed(*args, **kwargs):
        import deepspeed

        deepspeed.initialize = patched_ds_initialize
        result = _saved_accelerator_prepare_deepspeed(*args, **kwargs)
        deepspeed.initialize = _saved_ds_initialize
        return result

    accelerator._prepare_deepspeed = patched_prepare_deepspeed

if __name__ == "__main__":
    parser = qwen_image_parser()
    args = parser.parse_args()

    accelerator_init_kwargs = dict(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )

    if args.deepspeed_config_path is not None:
        accelerator_init_kwargs["deepspeed_plugin"] = accelerate.DeepSpeedPlugin(
            hf_ds_config=args.deepspeed_config_path
        )

    accelerator = accelerate.Accelerator(**accelerator_init_kwargs)

    seed = args.seed
    if args.sp_degree > 1:
        from mh_parallel_ext import parallel_state
        parallel_state.initialize_model_parallel(sequence_parallel_size=args.sp_degree)
        patch_accelerator_with_deepspeed_mpu(accelerator, parallel_state=parallel_state)
        seed = seed + parallel_state.get_data_parallel_rank()  # Ensure different seed for each data parallel rank

    accelerate.utils.set_seed(seed)

    # 创建训练数据集
    dataset = create_dataset(args)

    # 创建验证数据集（如果提供了验证数据路径）
    val_dataset = None
    if args.val_dataset_metadata_path is not None:
        val_dataset = create_dataset(args, metadata_path=args.val_dataset_metadata_path, repeat=1)

    model = QwenImageTrainingModule(
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
        edit_image_auto_resize=args.edit_image_auto_resize,
        sp_degree=args.sp_degree,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        wandb_project=f"qwen-image-edit-lora-sp",
        wandb_run_name=f"{os.path.basename(args.output_path)}_sp{args.sp_degree}_seed{args.seed}",
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task if args.sp_degree == 1 else launch_sp_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }

    # 对于训练任务，传入验证数据集
    if args.task in ["sft", "sft:train", "direct_distill", "direct_distill:train"]:
        launcher_map[args.task](accelerator, dataset, model, model_logger, val_dataset=val_dataset, args=args)
    else:
        launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
