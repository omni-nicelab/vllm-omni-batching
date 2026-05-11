"""
DMD2 (Distribution Matching Distillation v2) Training Script for Qwen-Image-Edit-LoRA

This script implements alternating training between generator (student) and guidance model
for DMD2 distillation using PEFT hotswap for memory efficiency.

Key features:
- Single base model shared by student/teacher/guidance adapters
- Fast adapter switching (<1ms) via PEFT hotswap
- Memory efficient: ~41.5GB vs 120GB with 3 separate models
- Correct gradient isolation verified
"""

import torch
import os
import argparse
import accelerate
import time
from tqdm import tqdm
from accelerate.utils import send_to_device

from train import (
    qwen_image_parser,
    create_dataset,
    patch_accelerator_with_deepspeed_mpu
)
from train_dmd2_module import QwenImageDMD2TrainingModule
from diffsynth.diffusion import ModelLogger
from diffsynth.diffusion.runner import run_validation, generate_val_images


def dmd2_parser():
    """Extend qwen_image_parser with DMD2-specific arguments."""
    parser = qwen_image_parser()

    # DMD2 hyperparameters
    parser.add_argument("--num_denoising_steps", type=int, default=4,
                        help="Number of denoising steps for student model")
    parser.add_argument("--generator_update_freq", type=int, default=5,
                        help="Update generator every N batches (guidance updates on other batches)")
    parser.add_argument("--dmd2_guidance_scale", type=float, default=4.0,
                        help="CFG scale for teacher model in DMD2 loss. Should match teacher inference cfg_scale.")
    parser.add_argument("--timestep_sampling_strategy", type=str, default="logit_normal",
                        choices=["uniform", "beta", "logit_normal"],
                        help="Timestep sampling strategy for DMD2. logit_normal matches the successful external distillation setup.")
    parser.add_argument("--backward_simulation", action="store_true",
                        help="Use ODE simulation for backward sampling (slower but may be more accurate)")
    parser.add_argument("--beta_alpha", type=float, default=4.0,
                        help="Beta distribution alpha parameter for timestep sampling")
    parser.add_argument("--beta_beta", type=float, default=1.2,
                        help="Beta distribution beta parameter for timestep sampling")
    parser.add_argument("--logit_mean", type=float, default=1.0,
                        help="Mean of the Gaussian before sigmoid for logit-normal timestep sampling")
    parser.add_argument("--logit_std", type=float, default=1.0,
                        help="Std of the Gaussian before sigmoid for logit-normal timestep sampling")
    parser.add_argument("--dynamic_rescale_t_steps", type=int, default=500,
                        help="Steps to anneal rescale_t toward 1.0 for all sampling strategies; beta mode also anneals Beta parameters over the same schedule")
    parser.add_argument("--rescale_t_val", type=float, default=1.0,
                        help="Initial non-linear timestep rescaling factor, annealed toward 1.0 by dynamic_rescale_t_steps")
    parser.add_argument("--learning_rate_gen", type=float, default=None,
                        help="Generator (student) learning rate (defaults to --learning_rate)")
    parser.add_argument("--learning_rate_guidance", type=float, default=None,
                        help="Guidance model learning rate (defaults to --learning_rate)")
    parser.add_argument("--log_steps", type=int, default=10,
                        help="Log metrics to wandb every N training steps")

    return parser


def launch_dmd2_training_task(
    accelerator,
    dataset,
    model,
    model_logger,
    generator_update_freq=5,
    learning_rate_gen=1e-5,
    learning_rate_guidance=1e-5,
    weight_decay=1e-2,
    num_workers=1,
    save_steps=None,
    num_epochs=1,
    val_dataset=None,
    val_steps=100,
    max_val_batches=10,
    num_val_samples=4,
    val_inference_steps=4,
    log_steps = 10,
    args=None
):
    """
    DMD2-specific training loop with alternating generator and guidance model updates.

    Training alternates between:
    1. Generator (student) updates: Every `generator_update_freq` batches
    2. Guidance updates: All other batches

    Args:
        accelerator: Accelerate Accelerator instance
        dataset: Training dataset
        model: QwenImageDMD2TrainingModule with PEFT adapters
        model_logger: Logger for metrics and checkpoints
        generator_update_freq: Update generator every N batches
        learning_rate_gen: Learning rate for generator (student adapter)
        learning_rate_guidance: Learning rate for guidance adapter
        weight_decay: Weight decay for both optimizers
        num_workers: Number of dataloader workers
        save_steps: Save checkpoint every N steps
        num_epochs: Number of training epochs
        val_dataset: Validation dataset (optional)
        val_steps: Run validation every N steps
        max_val_batches: Maximum batches for validation
        num_val_samples: Number of validation samples to generate
        val_inference_steps: Inference steps for validation generation
        args: Command-line arguments
    """

    # Parse args if provided
    if args is not None:
        learning_rate_gen = args.learning_rate_gen if args.learning_rate_gen is not None else args.learning_rate
        learning_rate_guidance = args.learning_rate_guidance if args.learning_rate_guidance is not None else args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        generator_update_freq = max(1, getattr(args, 'generator_update_freq', 5))
        if hasattr(args, 'val_steps'):
            val_steps = args.val_steps
        if hasattr(args, 'max_val_batches'):
            max_val_batches = args.max_val_batches
        if hasattr(args, 'num_val_samples'):
            num_val_samples = args.num_val_samples
        if hasattr(args, 'val_inference_steps'):
            val_inference_steps = args.val_inference_steps
        log_steps = max(1, getattr(args, 'log_steps', 1))

    # DeepSpeed via Accelerate only supports a single optimizer/scheduler pair.
    # Keep separate learning rates with parameter groups instead of separate optimizers.
    print("[DMD2] Creating optimizers...")
    print(f"  Generator (student) LR: {learning_rate_gen}")
    print(f"  Guidance LR: {learning_rate_guidance}")
    print(f"  Generator update frequency: every {generator_update_freq} batches")

    # Generator optimizer (student adapter parameters)
    generator_params = list(model.trainable_modules())
    print(f"  Generator trainable params: {sum(p.numel() for p in generator_params):,}")

    # Guidance optimizer (guidance adapter parameters)
    guidance_params = list(model.get_guidance_trainable_modules())
    print(f"  Guidance trainable params: {sum(p.numel() for p in guidance_params):,}")

    overlap = {id(param) for param in generator_params} & {id(param) for param in guidance_params}
    if overlap:
        raise RuntimeError(f"[DMD2] Generator and guidance parameter groups overlap: {len(overlap)} tensors")

    optimizer = torch.optim.AdamW(
        [
            {"params": generator_params, "lr": learning_rate_gen},
            {"params": guidance_params, "lr": learning_rate_guidance},
        ],
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    dataloader_kwargs = {
        "collate_fn": lambda x: x[0],
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 2

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        **dataloader_kwargs,
    )

    # Prepare with Accelerate (multiple optimizers)
    print("[DMD2] Preparing model and optimizers with Accelerate...")
    (
        model,
        optimizer,
        dataloader,
        scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        dataloader,
        scheduler,
    )

    def _student_adapter_name():
        return accelerator.unwrap_model(model).student_adapter_name

    def _set_training_state(task_name, adapter_name):
        base_model = accelerator.unwrap_model(model)
        base_model.task = task_name
        base_model.set_adapter_for_training(adapter_name)

    def _get_aux_metrics():
        base_model = accelerator.unwrap_model(model)
        return dict(getattr(base_model.dmd2_loss, "latest_metrics", {}))

    def _save_student_checkpoint(file_name):
        _set_training_state("dmd2:generator", _student_adapter_name())
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            base_model = accelerator.unwrap_model(model)
            state_dict = base_model.export_trainable_state_dict(
                state_dict,
                remove_prefix=model_logger.remove_prefix_in_ckpt,
            )
            state_dict = model_logger.state_dict_converter(state_dict)
            os.makedirs(model_logger.output_path, exist_ok=True)
            path = os.path.join(model_logger.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)

    # Prepare validation dataloader if provided
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            **dataloader_kwargs,
        )
        val_dataloader = accelerator.prepare(val_dataloader)

    # Training loop
    print(f"[DMD2] Starting training for {num_epochs} epochs...")
    if accelerator.is_main_process:
        base_model = accelerator.unwrap_model(model)
        print(
            "[DMD2] Runtime config: "
            f"backward_simulation={base_model.dmd2_loss.backward_simulation}, "
            f"num_denoising_steps={base_model.dmd2_loss.num_denoising_steps}, "
            f"vram_management_enabled={base_model.pipe.vram_management_enabled}"
        )

    _set_training_state("dmd2:generator", _student_adapter_name())
    # if accelerator.is_main_process:
    #     print("[DMD2] Saving initial student checkpoint before training...")
    #     model_logger.num_steps = 0
    # _save_student_checkpoint("step-0-initial.safetensors")

    global_step = 0
    best_val_loss = float('inf')
    last_logged_aux_metrics = {}

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=not accelerator.is_main_process
        )

        for batch_idx, data in enumerate(progress_bar):
            step_start_time = time.perf_counter()

            # Move data to device
            data = send_to_device(data, accelerator.device, non_blocking=True)

            is_generator_turn = (batch_idx % generator_update_freq == 0)
            loss_dm_value = None
            processed_inputs = accelerator.unwrap_model(model).prepare_inputs(data=data)[1]

            if is_generator_turn:
                # Generator update happens first so guidance sees the updated generator.
                _set_training_state("dmd2:generator", _student_adapter_name())
                optimizer.zero_grad()
                with accelerator.accumulate(model):
                    loss_dm = model(processed_inputs=processed_inputs)
                    accelerator.backward(loss_dm)
                    optimizer.step()
                    scheduler.step()

                loss_dm_value = loss_dm.item()

            # Guidance updates every batch and runs after generator when generator_turn is true.
            _set_training_state("dmd2:guidance", "guidance")
            optimizer.zero_grad()
            with accelerator.accumulate(model):
                loss_guidance = model(processed_inputs=processed_inputs)
                accelerator.backward(loss_guidance)
                optimizer.step()
                scheduler.step()

            loss_guidance_value = loss_guidance.item()
            step_time = time.perf_counter() - step_start_time

            # Update progress bar
            progress_bar.set_postfix({
                'step_s': f"{step_time:.1f}",
            })

            # Logging
            should_log = is_generator_turn or (global_step % log_steps == 0)
            if accelerator.is_main_process and should_log:
                metrics = {
                    "train/loss_guidance": loss_guidance_value,
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                    "perf/step_seconds": step_time,
                }
                if loss_dm_value is not None:
                    metrics["train/loss_dm"] = loss_dm_value

                metrics.update(_get_aux_metrics())
                model_logger.num_steps = global_step
                model_logger.log_metrics(metrics)

            # Validation
            if val_dataloader is not None and (global_step + 1) % val_steps == 0:
                if accelerator.is_main_process:
                    print(f"\n[DMD2] Running validation at step {global_step}...")

                # Switch to student adapter for validation
                _set_training_state("dmd2:generator", _student_adapter_name())

                val_loss = run_validation(
                    accelerator=accelerator,
                    model=model,
                    val_dataloader=val_dataloader,
                    model_logger=model_logger,
                    load_from_cache=False,
                    max_val_batches=max_val_batches
                )

                if accelerator.is_main_process:
                    print(f"[DMD2] Validation loss: {val_loss:.4f}")
                    model_logger.num_steps = global_step
                    model_logger.log_metrics({
                        "val/loss": val_loss,
                        "train/global_step": global_step,
                    })

                # Generate validation images on all ranks because the helper
                # contains distributed synchronization internally.
                if num_val_samples > 0:
                    _set_training_state("dmd2:generator", _student_adapter_name())
                    generate_val_images(
                        accelerator=accelerator,
                        model=model,
                        val_dataset=val_dataset,
                        model_logger=model_logger,
                        global_step=global_step,
                        num_val_samples=num_val_samples,
                        num_inference_steps=val_inference_steps,
                        height=getattr(args, 'height', None),
                        width=getattr(args, 'width', None),
                    )

                should_save_best = val_loss < best_val_loss
                if should_save_best:
                    best_val_loss = val_loss
                    if accelerator.is_main_process:
                        print(f"[DMD2] New best validation loss: {val_loss:.4f}, saving checkpoint...")
                        model_logger.num_steps = global_step
                    _save_student_checkpoint("best.safetensors")

                _set_training_state("dmd2:guidance", "guidance")
                model.train()

            # Save checkpoint
            if save_steps is not None and (global_step + 1) % save_steps == 0:
                if accelerator.is_main_process:
                    print(f"\n[DMD2] Saving checkpoint at step {global_step}...")
                    model_logger.num_steps = global_step
                _save_student_checkpoint(f"step-{global_step}.safetensors")

            global_step += 1

        # End of epoch
        if accelerator.is_main_process:
            print(f"\n[DMD2] Epoch {epoch + 1} completed. Saving checkpoint...")
            model_logger.num_steps = max(global_step - 1, 0)
        _save_student_checkpoint(f"epoch-{epoch + 1}.safetensors")

    # Final validation
    if val_dataloader is not None:
        if accelerator.is_main_process:
            print("\n[DMD2] Running final validation...")

        _set_training_state("dmd2:generator", _student_adapter_name())
        val_loss = run_validation(
            accelerator=accelerator,
            model=model,
            val_dataloader=val_dataloader,
            model_logger=model_logger,
            load_from_cache=False,
            max_val_batches=max_val_batches
        )

        if accelerator.is_main_process:
            print(f"[DMD2] Final validation loss: {val_loss:.4f}")
            model_logger.num_steps = max(global_step - 1, 0)
            model_logger.log_metrics({
                "val/final_loss": val_loss,
                "train/global_step": max(global_step - 1, 0),
            })

    if accelerator.is_main_process:
        print("[DMD2] Training completed!")


if __name__ == "__main__":
    parser = dmd2_parser()
    args = parser.parse_args()

    # Force DMD2 task if not specified
    if not hasattr(args, 'task') or not args.task or not args.task.startswith("dmd2"):
        args.task = "dmd2:generator"
        print(f"[DMD2] Setting task to '{args.task}'")

    # Setup accelerator
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

    # Set seed
    seed = args.seed
    if args.sp_degree > 1:
        from mh_parallel_ext import parallel_state
        parallel_state.initialize_model_parallel(sequence_parallel_size=args.sp_degree)
        patch_accelerator_with_deepspeed_mpu(accelerator, parallel_state=parallel_state)
        seed = seed + parallel_state.get_data_parallel_rank()

    accelerate.utils.set_seed(seed)

    # Create datasets
    print("[DMD2] Creating datasets...")
    dataset = create_dataset(args)
    val_dataset = None
    if args.val_dataset_metadata_path is not None:
        val_dataset = create_dataset(args, metadata_path=args.val_dataset_metadata_path, repeat=1)

    # Create model with PEFT multi-adapter setup
    print("[DMD2] Creating model with PEFT multi-adapter setup...")
    model = QwenImageDMD2TrainingModule(
        # Base model parameters (from train.py)
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
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
        # DMD2 parameters
        num_denoising_steps=args.num_denoising_steps,
        guidance_scale=args.dmd2_guidance_scale,
        backward_simulation=args.backward_simulation,
        timestep_sampling_strategy=args.timestep_sampling_strategy,
        beta_alpha=args.beta_alpha,
        beta_beta=args.beta_beta,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        dynamic_rescale_t_steps=args.dynamic_rescale_t_steps,
        rescale_t_val=args.rescale_t_val,
        # LoRA checkpoint to initialize all adapters (student/teacher/guidance)
        lora_checkpoint=args.lora_checkpoint,
    )

    # Create logger
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        wandb_project=f"qwen-image-edit-dmd2",
        wandb_run_name=f"{os.path.basename(args.output_path)}_dmd2_sp{args.sp_degree}_seed{args.seed}",
    )

    # Launch DMD2 training
    launch_dmd2_training_task(
        accelerator=accelerator,
        dataset=dataset,
        model=model,
        model_logger=model_logger,
        val_dataset=val_dataset,
        args=args,
        val_inference_steps=args.num_denoising_steps,  # Match validation inference steps to training denoising steps for consistency
    )
