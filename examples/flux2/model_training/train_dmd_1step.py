"""FLUX.2 1-step CFG-guided DMD distillation.

This entry is intentionally separate from train_dmd2.py. It keeps the existing
4-step DMD2 path untouched while adding a focused 1-step experiment:

- student/default: single conditional forward, 1 denoising step
- teacher: frozen CFG-guided real score
- guidance: fake score model tracking the student's no-CFG distribution
- generator loss: decoupled DM + CFG augmentation, with optional CFG regression
"""

from __future__ import annotations

import argparse
import os
import sys

import accelerate

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from train_dmd2 import (  # noqa: E402
    DMD2Loss,
    Flux2DMD2TrainingModule,
    ModelLogger,
    build_dataset,
    build_dataset_for_metadata,
    dmd2_parser,
    launch_dmd2_training_task,
)

import torch  # noqa: E402


class DMD1StepCFGLoss(DMD2Loss):
    """1-step DMD loss with separated DM and CFG-augmentation timesteps."""

    def __init__(
        self,
        cfg_aux_loss_weight: float = 0.1,
        ca_loss_weight: float = 1.0,
        ca_timestep_schedule: str = "focused",
        ca_timestep_max_ratio: float = 1.0,
        generator_objective: str = "decoupled_hybrid",
        grad_clip_value: float = 0.0,
        one_step_conditioning_timestep: float | None = None,
        **kwargs,
    ):
        kwargs["num_denoising_steps"] = 1
        super().__init__(**kwargs)
        if generator_objective not in {"coupled_cfg", "decoupled_hybrid"}:
            raise ValueError(f"Unsupported generator_objective: {generator_objective}")
        if not 0.0 < float(ca_timestep_max_ratio) <= 1.0:
            raise ValueError(f"ca_timestep_max_ratio must be in (0, 1], got {ca_timestep_max_ratio}")
        if one_step_conditioning_timestep is not None:
            one_step_conditioning_timestep = float(one_step_conditioning_timestep)
            if not 0.0 < one_step_conditioning_timestep <= 1000.0:
                raise ValueError(
                    "one_step_conditioning_timestep must be in (0, 1000], "
                    f"got {one_step_conditioning_timestep}"
                )
        self.cfg_aux_loss_weight = float(cfg_aux_loss_weight)
        self.ca_loss_weight = float(ca_loss_weight)
        self.ca_timestep_schedule = ca_timestep_schedule
        self.ca_timestep_max_ratio = float(ca_timestep_max_ratio)
        self.generator_objective = generator_objective
        self.grad_clip_value = float(grad_clip_value)
        self.one_step_conditioning_timestep = one_step_conditioning_timestep

    def _get_student_inference_schedule(self, pipe, inputs_shared, device):
        if self.one_step_conditioning_timestep is None:
            return super()._get_student_inference_schedule(pipe, inputs_shared, device)
        timesteps = torch.tensor(
            [self.one_step_conditioning_timestep, 0.0],
            device=device,
            dtype=torch.float32,
        )
        return timesteps, timesteps / 1000.0

    def sample_ca_timestep(self, batch_size, device, scheduler_timestep):
        """Sample tau_CA after the current generation step.

        The paper uses clean-progress tau where 0 is pure noise and 1 is clean.
        This code uses FLUX sigma/timestep where 1000 is pure noise and 0 is
        clean. Therefore tau_CA > t becomes sigma_CA <= sigma_current.
        For 1-step, sigma_current is the first inference sigma, so this reduces
        to almost the full range while preserving the general focused rule.
        `ca_timestep_max_ratio` is an empirical safety valve for 1-step runs
        where full-range CA is too strong.
        """
        _, full_timesteps = self.sample_timestep(batch_size, device)
        if self.ca_timestep_schedule == "full":
            upper = torch.ones((), device=device, dtype=full_timesteps.dtype)
        elif self.ca_timestep_schedule == "focused":
            upper = torch.clamp(
                scheduler_timestep.detach().float().to(device=device) / 1000.0,
                min=0.0,
                max=1.0,
            )
        else:
            raise ValueError(f"Unsupported ca_timestep_schedule: {self.ca_timestep_schedule}")
        upper = torch.minimum(
            upper,
            torch.tensor(self.ca_timestep_max_ratio, device=device, dtype=upper.dtype),
        )
        return full_timesteps * upper

    def _maybe_clip_grad(self, grad):
        if self.grad_clip_value > 0:
            return grad.clamp(-self.grad_clip_value, self.grad_clip_value)
        return grad

    def forward(self, pipe, peft_dit, inputs_shared, inputs_posi, inputs_nega, mode="generator"):
        original_training = peft_dit.training
        latents = inputs_shared["input_latents"]
        batch_size = latents.shape[0]
        device = latents.device

        _, dm_timesteps = self.sample_timestep(batch_size, device)
        student_adapter_name = getattr(peft_dit, "student_adapter_name", "default")
        peft_dit.set_adapter(student_adapter_name)

        original_dit = pipe.dit
        pipe.dit = peft_dit
        generator_with_grad = mode == "generator"
        peft_dit.train(generator_with_grad)
        generated_latents, scheduler_timestep = self.sample_backward(
            pipe,
            latents,
            inputs_shared,
            inputs_posi,
            inputs_nega,
            generator_with_grad=generator_with_grad,
        )
        if generator_with_grad and generated_latents.requires_grad:
            generated_latents.retain_grad()
        pipe.dit = original_dit

        dm_timesteps = dm_timesteps.to(device=device, dtype=torch.float32)
        ca_timesteps = self.sample_ca_timestep(
            batch_size,
            device,
            scheduler_timestep,
        ).to(device=device, dtype=torch.float32)

        self.latest_metrics.update({
            "debug/timestep_dm": float(dm_timesteps.mean().detach().item()),
            "debug/timestep_ca": float(ca_timesteps.mean().detach().item()),
            "debug/scheduler_timestep": float(scheduler_timestep.detach().item()),
        })
        self._update_latent_metrics(latents, generated_latents)

        if mode == "generator":
            loss = self._compute_decoupled_generator_loss(
                pipe,
                peft_dit,
                generated_latents,
                dm_timesteps,
                ca_timesteps,
                inputs_shared,
                inputs_posi,
                inputs_nega,
                student_adapter_name,
            )
            peft_dit.set_adapter(student_adapter_name)
        elif mode == "guidance":
            loss = self._compute_guidance_loss(
                pipe,
                peft_dit,
                generated_latents,
                dm_timesteps,
                inputs_shared,
                inputs_posi,
                inputs_nega,
            )
            peft_dit.set_adapter("guidance")
        else:
            raise ValueError(f"Invalid DMD mode: {mode}")

        peft_dit.train(original_training)
        self.current_step += 1
        return loss

    def _renoise(self, generated_latents, timesteps):
        noise = torch.randn_like(generated_latents)
        sigmas = self._expand_to_latents(timesteps / 1000.0, generated_latents)
        noisy_latents = sigmas * noise + (1 - sigmas) * generated_latents
        return noisy_latents, sigmas

    def _compute_decoupled_generator_loss(
        self,
        pipe,
        peft_dit,
        generated_latents,
        dm_timesteps,
        ca_timesteps,
        inputs_shared,
        inputs_posi,
        inputs_nega,
        student_adapter_name,
    ):
        original_training = peft_dit.training
        teacher_cfg_scale = float(inputs_shared.get("cfg_scale", self.guidance_scale))
        cfg_enabled = self.cfg_distill and teacher_cfg_scale != 1.0
        other_models = {
            name: getattr(pipe, name)
            for name in pipe.in_iteration_models if name != "dit"
        }

        with torch.no_grad():
            peft_dit.eval()
            noisy_dm, sigmas_dm = self._renoise(generated_latents, dm_timesteps)
            noisy_dm = noisy_dm.to(dtype=pipe.torch_dtype)
            timestep_dm = dm_timesteps.to(
                device=generated_latents.device,
                dtype=pipe.torch_dtype,
            )
            shared_dm = dict(inputs_shared)
            shared_dm["latents"] = noisy_dm

            peft_dit.set_adapter("guidance")
            pred_fake_flow_dm = pipe.model_fn(
                dit=peft_dit,
                **other_models,
                **inputs_posi,
                **shared_dm,
                timestep=timestep_dm,
                progress_id=None,
            )

            peft_dit.set_adapter("teacher")
            pred_real_cond_flow_dm = pipe.model_fn(
                dit=peft_dit,
                **other_models,
                **inputs_posi,
                **shared_dm,
                timestep=timestep_dm,
                progress_id=None,
            )
            pred_fake_latents_dm = noisy_dm - sigmas_dm * pred_fake_flow_dm
            pred_real_cond_latents_dm = noisy_dm - sigmas_dm * pred_real_cond_flow_dm

            if self.generator_objective == "coupled_cfg":
                if cfg_enabled:
                    pred_real_uncond_flow_dm = pipe.model_fn(
                        dit=peft_dit,
                        **other_models,
                        **inputs_nega,
                        **shared_dm,
                        timestep=timestep_dm,
                        progress_id=None,
                    )
                    pred_real_cfg_flow_dm = (
                        pred_real_uncond_flow_dm
                        + teacher_cfg_scale
                        * (pred_real_cond_flow_dm - pred_real_uncond_flow_dm)
                    )
                else:
                    pred_real_cfg_flow_dm = pred_real_cond_flow_dm
                pred_real_cfg_latents_dm = noisy_dm - sigmas_dm * pred_real_cfg_flow_dm
                dm_weight_raw = (
                    generated_latents - pred_real_cfg_latents_dm
                ).abs().reshape(generated_latents.size(0), -1).mean(dim=1)
                dm_weight = self._expand_to_latents(dm_weight_raw, generated_latents) + 1e-8
                dm_grad = (pred_fake_latents_dm - pred_real_cfg_latents_dm) / dm_weight
                ca_grad = torch.zeros_like(dm_grad)
                grad = self._maybe_clip_grad(dm_grad)
                aux_noisy = noisy_dm
                aux_timestep = timestep_dm
                aux_target_flow = pred_real_cfg_flow_dm
                ca_flow_abs_mean = (
                    pred_real_cfg_flow_dm - pred_real_cond_flow_dm
                ).abs().mean().detach().item()
                ca_weight_raw = generated_latents.new_zeros((generated_latents.size(0),))
                ca_timestep_metric = dm_timesteps
            else:
                noisy_ca, sigmas_ca = self._renoise(generated_latents, ca_timesteps)
                noisy_ca = noisy_ca.to(dtype=pipe.torch_dtype)
                timestep_ca = ca_timesteps.to(
                    device=generated_latents.device,
                    dtype=pipe.torch_dtype,
                )
                shared_ca = dict(inputs_shared)
                shared_ca["latents"] = noisy_ca

                pred_real_cond_flow_ca = pipe.model_fn(
                    dit=peft_dit,
                    **other_models,
                    **inputs_posi,
                    **shared_ca,
                    timestep=timestep_ca,
                    progress_id=None,
                )
                if cfg_enabled:
                    pred_real_uncond_flow_ca = pipe.model_fn(
                        dit=peft_dit,
                        **other_models,
                        **inputs_nega,
                        **shared_ca,
                        timestep=timestep_ca,
                        progress_id=None,
                    )
                    pred_real_cfg_flow_ca = (
                        pred_real_uncond_flow_ca
                        + teacher_cfg_scale
                        * (pred_real_cond_flow_ca - pred_real_uncond_flow_ca)
                    )
                else:
                    pred_real_cfg_flow_ca = pred_real_cond_flow_ca
                pred_real_cond_latents_ca = noisy_ca - sigmas_ca * pred_real_cond_flow_ca
                pred_real_cfg_latents_ca = noisy_ca - sigmas_ca * pred_real_cfg_flow_ca

                dm_weight_raw = (
                    generated_latents - pred_real_cond_latents_dm
                ).abs().reshape(generated_latents.size(0), -1).mean(dim=1)
                ca_weight_raw = (
                    generated_latents - pred_real_cfg_latents_ca
                ).abs().reshape(generated_latents.size(0), -1).mean(dim=1)
                dm_weight = self._expand_to_latents(dm_weight_raw, generated_latents) + 1e-8
                ca_weight = self._expand_to_latents(ca_weight_raw, generated_latents) + 1e-8

                dm_grad = (pred_fake_latents_dm - pred_real_cond_latents_dm) / dm_weight
                ca_grad = (pred_real_cond_latents_ca - pred_real_cfg_latents_ca) / ca_weight
                grad = self._maybe_clip_grad(dm_grad + self.ca_loss_weight * ca_grad)
                aux_noisy = noisy_ca
                aux_timestep = timestep_ca
                aux_target_flow = pred_real_cfg_flow_ca
                ca_flow_abs_mean = (
                    pred_real_cfg_flow_ca - pred_real_cond_flow_ca
                ).abs().mean().detach().item()
                ca_timestep_metric = ca_timesteps

            dm_grad_abs_mean = dm_grad.abs().mean().detach().item()
            dm_grad_abs_max = dm_grad.abs().amax().detach().item()
            ca_grad_abs_mean = ca_grad.abs().mean().detach().item()
            ca_grad_abs_max = ca_grad.abs().amax().detach().item()

            self.latest_metrics.update({
                "debug/generator_objective_decoupled": float(
                    self.generator_objective == "decoupled_hybrid"
                ),
                "debug/cfg_distill_enabled": float(cfg_enabled),
                "debug/dm_flow_abs_mean": (
                    pred_fake_flow_dm - pred_real_cond_flow_dm
                ).abs().mean().detach().item(),
                "debug/ca_flow_abs_mean": ca_flow_abs_mean,
                "debug/dm_weight_mean": dm_weight_raw.mean().detach().item(),
                "debug/ca_weight_mean": ca_weight_raw.mean().detach().item(),
                "debug/dm_grad_abs_mean": dm_grad_abs_mean,
                "debug/dm_grad_abs_max": dm_grad_abs_max,
                "debug/ca_grad_abs_mean": ca_grad_abs_mean,
                "debug/ca_grad_abs_max": ca_grad_abs_max,
                "debug/grad_abs_mean": grad.abs().mean().detach().item(),
                "debug/grad_abs_max": grad.abs().amax().detach().item(),
                "debug/ca_timestep_max_ratio": self.ca_timestep_max_ratio,
                "debug/grad_clip_value": self.grad_clip_value,
                "debug/timestep_ca_effective": float(
                    ca_timestep_metric.mean().detach().item()
                ),
            })

        loss_proxy = 0.5 * torch.nn.functional.mse_loss(
            generated_latents.float(),
            (generated_latents - grad).float().detach(),
        )

        loss_cfg_aux = generated_latents.new_zeros(())
        if self.cfg_aux_loss_weight > 0 and cfg_enabled:
            peft_dit.train(True)
            peft_dit.set_adapter(student_adapter_name)
            shared_aux = dict(inputs_shared)
            shared_aux["latents"] = aux_noisy.detach()
            pred_student_flow_ca = pipe.model_fn(
                dit=peft_dit,
                **other_models,
                **inputs_posi,
                **shared_aux,
                timestep=aux_timestep,
                progress_id=None,
            )
            loss_cfg_aux = torch.nn.functional.mse_loss(
                pred_student_flow_ca.float(),
                aux_target_flow.float().detach(),
            )

        total_loss = loss_proxy + self.cfg_aux_loss_weight * loss_cfg_aux
        self.latest_metrics.update({
            "train/loss_proxy": loss_proxy.detach().float().item(),
            "train/loss_cfg_aux": loss_cfg_aux.detach().float().item(),
            "debug/cfg_aux_weight": self.cfg_aux_loss_weight,
            "debug/ca_loss_weight": self.ca_loss_weight,
            "debug/generator_objective": 1.0
            if self.generator_objective == "decoupled_hybrid" else 0.0,
        })
        peft_dit.train(original_training)
        return total_loss


class Flux2DMD1StepTrainingModule(Flux2DMD2TrainingModule):
    """Same adapter/checkpoint plumbing as FLUX2 DMD2, with a 1-step loss."""

    def __init__(
        self,
        guidance_scale=4.0,
        embedded_guidance=None,
        timestep_sampling_strategy="logit_normal",
        backward_simulation=True,
        beta_alpha=4.0,
        beta_beta=1.2,
        logit_mean=1.0,
        logit_std=1.0,
        dynamic_rescale_t_steps=500,
        rescale_t_val=1.0,
        student_schedule="linear",
        cfg_distill=True,
        cfg_aux_loss_weight=0.1,
        ca_loss_weight=1.0,
        ca_timestep_schedule="focused",
        ca_timestep_max_ratio=1.0,
        generator_objective="decoupled_hybrid",
        grad_clip_value=0.0,
        one_step_conditioning_timestep=None,
        **kwargs,
    ):
        super().__init__(
            num_denoising_steps=1,
            guidance_scale=guidance_scale,
            embedded_guidance=embedded_guidance,
            timestep_sampling_strategy=timestep_sampling_strategy,
            backward_simulation=backward_simulation,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            logit_mean=logit_mean,
            logit_std=logit_std,
            dynamic_rescale_t_steps=dynamic_rescale_t_steps,
            rescale_t_val=rescale_t_val,
            student_schedule=student_schedule,
            cfg_distill=cfg_distill,
            **kwargs,
        )
        self.dmd2_loss = DMD1StepCFGLoss(
            num_denoising_steps=1,
            guidance_scale=guidance_scale,
            timestep_sampling_strategy=timestep_sampling_strategy,
            backward_simulation=backward_simulation,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            logit_mean=logit_mean,
            logit_std=logit_std,
            dynamic_rescale_t_steps=dynamic_rescale_t_steps,
            rescale_t_val=rescale_t_val,
            student_schedule=student_schedule,
            cfg_distill=cfg_distill,
            cfg_aux_loss_weight=cfg_aux_loss_weight,
            ca_loss_weight=ca_loss_weight,
            ca_timestep_schedule=ca_timestep_schedule,
            ca_timestep_max_ratio=ca_timestep_max_ratio,
            generator_objective=generator_objective,
            grad_clip_value=grad_clip_value,
            one_step_conditioning_timestep=one_step_conditioning_timestep,
        )


def dmd1step_parser() -> argparse.ArgumentParser:
    parser = dmd2_parser()
    parser.set_defaults(
        num_denoising_steps=1,
        generator_update_freq=10,
        learning_rate_gen=3e-6,
        wandb_project=os.environ.get("WANDB_PROJECT", "pbr-material-edit-flux2-dmd-1step"),
    )
    parser.add_argument("--cfg_aux_loss_weight", type=float, default=0.1)
    parser.add_argument("--ca_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--ca_timestep_schedule",
        type=str,
        default="focused",
        choices=["focused", "full"],
    )
    parser.add_argument(
        "--ca_timestep_max_ratio",
        type=float,
        default=1.0,
        help="Upper bound for one-step CA timesteps as a fraction of 1000. "
             "Use <1.0 to reduce full-range CA pressure.",
    )
    parser.add_argument(
        "--dmd2_generator_objective",
        type=str,
        default="decoupled_hybrid",
        choices=["coupled_cfg", "decoupled_hybrid"],
        help="coupled_cfg matches the lower-variance tiling cfg-default run; "
             "decoupled_hybrid uses separate DM and CA timesteps.",
    )
    parser.add_argument(
        "--grad_clip_value",
        type=float,
        default=0.0,
        help="Optional elementwise clamp on the DMD proxy gradient. 0 disables.",
    )
    parser.add_argument(
        "--one_step_conditioning_timestep",
        type=float,
        default=None,
        help="Optional fixed one-step student timestep in [1, 1000]. "
             "Unset keeps --student_schedule behavior; SDXL DMD2 used 399.",
    )
    return parser


if __name__ == "__main__":
    parser = dmd1step_parser()
    args = parser.parse_args()
    if args.num_denoising_steps != 1:
        print("[FLUX2 DMD 1-step] Forcing --num_denoising_steps 1")
        args.num_denoising_steps = 1
    if not args.task or not args.task.startswith("dmd2"):
        args.task = "dmd2:generator"

    accelerator_init_kwargs = dict(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters,
            )
        ],
    )
    if args.deepspeed_config_path is not None:
        accelerator_init_kwargs["deepspeed_plugin"] = accelerate.DeepSpeedPlugin(
            hf_ds_config=args.deepspeed_config_path,
        )
    accelerator = accelerate.Accelerator(**accelerator_init_kwargs)
    accelerate.utils.set_seed(args.seed)

    dataset = build_dataset(args)
    val_dataset = None
    if args.val_dataset_metadata_path:
        if not os.path.isfile(args.val_dataset_metadata_path):
            raise FileNotFoundError(
                f"Validation metadata does not exist: {args.val_dataset_metadata_path}"
            )
        val_dataset = build_dataset_for_metadata(args, args.val_dataset_metadata_path, repeat=1)

    model = Flux2DMD1StepTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
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
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        guidance_scale=args.dmd2_guidance_scale,
        embedded_guidance=args.dmd2_embedded_guidance,
        timestep_sampling_strategy=args.timestep_sampling_strategy,
        backward_simulation=args.backward_simulation,
        beta_alpha=args.beta_alpha,
        beta_beta=args.beta_beta,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        dynamic_rescale_t_steps=args.dynamic_rescale_t_steps,
        rescale_t_val=args.rescale_t_val,
        student_schedule=args.student_schedule,
        cfg_distill=args.cfg_distill,
        lora_checkpoint=args.lora_checkpoint,
        student_lora_checkpoint=args.student_lora_checkpoint,
        cfg_aux_loss_weight=args.cfg_aux_loss_weight,
        ca_loss_weight=args.ca_loss_weight,
        ca_timestep_schedule=args.ca_timestep_schedule,
        ca_timestep_max_ratio=args.ca_timestep_max_ratio,
        generator_objective=args.dmd2_generator_objective,
        grad_clip_value=args.grad_clip_value,
        one_step_conditioning_timestep=args.one_step_conditioning_timestep,
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_wandb=os.environ.get("WANDB_MODE", "online").lower()
        not in {"disabled", "off", "false", "0"},
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name or os.path.basename(args.output_path),
    )
    launch_dmd2_training_task(
        accelerator,
        dataset,
        model,
        model_logger,
        args,
        val_dataset=val_dataset,
    )
