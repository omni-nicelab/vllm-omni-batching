"""FLUX.2 DMD2 LoRA distillation for SMBE diffuse-to-normal data."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Optional

import accelerate
import torch
from accelerate.utils import send_to_device
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from tqdm import tqdm

from diffsynth.core import load_state_dict
from diffsynth.diffusion import ModelLogger

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from train_smbe import Flux2ImageTrainingModule, build_dataset, flux2_parser


def first_item_collate(batch):
    return batch[0]


class DMD2Loss(torch.nn.Module):
    """DMD2 loss adapted for FLUX.2 sequence latents shaped B,HW,C."""

    def __init__(
        self,
        num_denoising_steps: int = 4,
        guidance_scale: float = 4.0,
        timestep_sampling_strategy: str = "logit_normal",
        backward_simulation: bool = False,
        beta_alpha: float = 4.0,
        beta_beta: float = 1.2,
        logit_mean: float = 1.0,
        logit_std: float = 1.0,
        dynamic_rescale_t_steps: int = 500,
        rescale_t_val: float = 1.0,
        student_schedule: str = "linear",
        cfg_distill: bool = True,
    ):
        super().__init__()
        self.num_denoising_steps = num_denoising_steps
        self.guidance_scale = guidance_scale
        self.timestep_sampling_strategy = timestep_sampling_strategy
        self.backward_simulation = backward_simulation
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.dynamic_rescale_t_steps = dynamic_rescale_t_steps
        self.rescale_t_val = rescale_t_val
        if student_schedule not in {"linear", "flux2"}:
            raise ValueError(f"Unsupported student_schedule: {student_schedule}")
        self.student_schedule = student_schedule
        self.cfg_distill = cfg_distill
        self.current_step = 0
        self.latest_metrics = {}

    @staticmethod
    def _expand_to_latents(values: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        return values.reshape((values.shape[0],) + (1,) * (latents.ndim - 1))

    def _update_latent_metrics(self, latents, generated_latents):
        latents_flat = latents.float().reshape(latents.shape[0], -1)
        generated_flat = generated_latents.float().reshape(generated_latents.shape[0], -1)
        gt_std_mean = latents_flat.std(dim=1, unbiased=False).mean().detach().item()
        generated_std_mean = generated_flat.std(dim=1, unbiased=False).mean().detach().item()
        self.latest_metrics.update({
            "latent/std_mean_gt": gt_std_mean,
            "latent/std_mean_generated": generated_std_mean,
            "latent/std_mean_delta": generated_std_mean - gt_std_mean,
        })

    def sample_timestep(self, batch_size, device, current_step=None):
        if current_step is None:
            current_step = self.current_step

        if self.dynamic_rescale_t_steps > 0:
            import math

            progress = min(current_step / self.dynamic_rescale_t_steps, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(progress * math.pi))
        else:
            cosine_decay = 1.0

        if self.timestep_sampling_strategy == "uniform":
            t = torch.rand(batch_size, device=device).float()
        elif self.timestep_sampling_strategy == "beta":
            alpha = 1.0 + (self.beta_alpha - 1.0) * cosine_decay
            beta = 1.0 + (self.beta_beta - 1.0) * cosine_decay
            t = torch.distributions.Beta(alpha, beta).sample((batch_size,)).to(device).float()
        elif self.timestep_sampling_strategy == "logit_normal":
            t = torch.sigmoid(
                torch.randn(batch_size, device=device).float() * self.logit_std
                + self.logit_mean
            )
        else:
            raise ValueError(f"Unsupported timestep_sampling_strategy: {self.timestep_sampling_strategy}")

        current_rescale_t = 1.0 + (self.rescale_t_val - 1.0) * cosine_decay
        if current_rescale_t != 1.0:
            t = current_rescale_t * t / (1 + (current_rescale_t - 1) * t)
        return t, t * 1000.0

    def _get_student_inference_schedule(self, pipe, inputs_shared, device):
        if self.student_schedule == "linear":
            timesteps = torch.linspace(
                1.0,
                0.0,
                self.num_denoising_steps + 1,
                device=device,
                dtype=torch.float32,
            ) * 1000.0
            return timesteps, timesteps / 1000.0

        height = int(inputs_shared["height"])
        width = int(inputs_shared["width"])
        pipe.scheduler.set_timesteps(
            self.num_denoising_steps,
            training=False,
            dynamic_shift_len=height // 16 * width // 16,
        )
        timesteps = pipe.scheduler.timesteps.to(device=device, dtype=torch.float32)
        sigmas = pipe.scheduler.sigmas.to(device=device, dtype=torch.float32)
        return timesteps, sigmas

    def sample_backward(
        self,
        pipe,
        latents,
        inputs_shared,
        inputs_posi,
        inputs_nega,
        generator_with_grad=True,
    ):
        device = latents.device
        timesteps, sigmas = self._get_student_inference_schedule(pipe, inputs_shared, device)
        selected_step = min(self.num_denoising_steps - 1, len(timesteps) - 1)
        return_timesteps = timesteps[selected_step]
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

        def _backward_one_step(current_shared, timestep_index):
            timestep_tensor = timesteps[timestep_index].unsqueeze(0).to(
                dtype=pipe.torch_dtype,
                device=device,
            )
            current_shared["latents"] = current_shared["latents"].to(dtype=pipe.torch_dtype)
            current_latents = current_shared["latents"]
            sigma = sigmas[timestep_index]
            sigma_next = (
                torch.zeros_like(sigma)
                if timestep_index + 1 >= len(sigmas)
                else sigmas[timestep_index + 1]
            )
            pred_flow = pipe.model_fn(
                **inputs_posi,
                **current_shared,
                **models,
                timestep=timestep_tensor,
                progress_id=timestep_index,
            )
            current_shared["latents"] = (
                current_latents + pred_flow * (sigma_next - sigma)
            ).to(dtype=pipe.torch_dtype)
            return current_shared["latents"]

        noisy_latents = torch.randn_like(latents).to(device=device, dtype=pipe.torch_dtype)
        current_shared = dict(inputs_shared)
        current_shared["latents"] = noisy_latents
        if self.backward_simulation:
            with torch.no_grad():
                for step in range(selected_step):
                    _backward_one_step(current_shared, step)
        elif selected_step != 0:
            sigma = sigmas[selected_step]
            current_shared["latents"] = (
                (1 - sigma) * latents.to(dtype=pipe.torch_dtype)
                + sigma * current_shared["latents"]
            )

        with torch.enable_grad() if generator_with_grad else torch.no_grad():
            generated_latents = _backward_one_step(current_shared, selected_step)
        return generated_latents, return_timesteps

    def forward(self, pipe, peft_dit, inputs_shared, inputs_posi, inputs_nega, mode="generator"):
        original_training = peft_dit.training
        latents = inputs_shared["input_latents"]
        batch_size = latents.shape[0]
        device = latents.device

        _, loss_timesteps = self.sample_timestep(batch_size, device)
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

        loss_timesteps = loss_timesteps.to(device=device, dtype=torch.float32)
        self.latest_metrics.update({
            "debug/training_timestep": float(loss_timesteps.mean().detach().item()),
            "debug/scheduler_timestep": float(scheduler_timestep.detach().item()),
        })
        self._update_latent_metrics(latents, generated_latents)

        if mode == "generator":
            loss = self._compute_generator_loss(
                pipe, peft_dit, generated_latents, loss_timesteps,
                inputs_shared, inputs_posi, inputs_nega,
            )
            peft_dit.set_adapter(student_adapter_name)
        elif mode == "guidance":
            loss = self._compute_guidance_loss(
                pipe, peft_dit, generated_latents, loss_timesteps,
                inputs_shared, inputs_posi, inputs_nega,
            )
            peft_dit.set_adapter("guidance")
        else:
            raise ValueError(f"Invalid DMD2 mode: {mode}")

        peft_dit.train(original_training)
        self.current_step += 1
        return loss

    def _compute_generator_loss(
        self,
        pipe,
        peft_dit,
        generated_latents,
        timesteps,
        inputs_shared,
        inputs_posi,
        inputs_nega,
    ):
        original_training = peft_dit.training
        peft_dit.eval()
        teacher_cfg_scale = float(inputs_shared.get("cfg_scale", self.guidance_scale))

        with torch.no_grad():
            noise = torch.randn_like(generated_latents)
            sigmas = self._expand_to_latents(timesteps / 1000.0, generated_latents)
            noisy_latents = (sigmas * noise + (1 - sigmas) * generated_latents).to(
                dtype=pipe.torch_dtype,
            )
            timestep_tensor = timesteps.to(
                device=generated_latents.device,
                dtype=pipe.torch_dtype,
            )
            other_models = {
                name: getattr(pipe, name)
                for name in pipe.in_iteration_models if name != "dit"
            }
            noisy_shared = dict(inputs_shared)
            noisy_shared["latents"] = noisy_latents

            peft_dit.set_adapter("guidance")
            pred_fake_flow = pipe.model_fn(
                dit=peft_dit,
                **other_models,
                **inputs_posi,
                **noisy_shared,
                timestep=timestep_tensor,
                progress_id=None,
            )

            peft_dit.set_adapter("teacher")
            pred_real_flow_posi = pipe.model_fn(
                dit=peft_dit,
                **other_models,
                **inputs_posi,
                **noisy_shared,
                timestep=timestep_tensor,
                progress_id=None,
            )
            if self.cfg_distill and teacher_cfg_scale != 1.0:
                pred_real_flow_nega = pipe.model_fn(
                    dit=peft_dit,
                    **other_models,
                    **inputs_nega,
                    **noisy_shared,
                    timestep=timestep_tensor,
                    progress_id=None,
                )
                pred_real_flow = (
                    pred_real_flow_nega
                    + teacher_cfg_scale * (pred_real_flow_posi - pred_real_flow_nega)
                )
                teacher_cfg_delta = (
                    pred_real_flow_posi - pred_real_flow_nega
                ).abs().mean().detach().item()
            else:
                pred_real_flow = pred_real_flow_posi
                teacher_cfg_delta = 0.0

            pred_real_latents = noisy_latents - sigmas * pred_real_flow
            pred_fake_latents = noisy_latents - sigmas * pred_fake_flow
            p_real = generated_latents - pred_real_latents
            weight_factor_raw = torch.abs(p_real).reshape(p_real.size(0), -1).mean(dim=1)
            weight_factor = self._expand_to_latents(weight_factor_raw, p_real) + 1e-8
            grad = (pred_fake_latents - pred_real_latents) / weight_factor

            self.latest_metrics.update({
                "debug/pred_fake_real_flow_abs_mean": (
                    pred_fake_flow - pred_real_flow
                ).abs().mean().detach().item(),
                "debug/pred_fake_real_latent_abs_mean": (
                    pred_fake_latents - pred_real_latents
                ).abs().mean().detach().item(),
                "debug/weight_factor_mean": weight_factor_raw.mean().detach().item(),
                "debug/weight_factor_min": weight_factor_raw.min().detach().item(),
                "debug/grad_abs_mean": grad.abs().mean().detach().item(),
                "debug/grad_abs_max": grad.abs().amax().detach().item(),
                "debug/cfg_distill_enabled": float(
                    self.cfg_distill and teacher_cfg_scale != 1.0
                ),
                "debug/teacher_cfg_scale": teacher_cfg_scale,
                "debug/teacher_cfg_flow_delta_abs_mean": teacher_cfg_delta,
            })

        loss_dm = 0.5 * torch.nn.functional.mse_loss(
            generated_latents.float(),
            (generated_latents - grad).float().detach(),
        )
        peft_dit.train(original_training)
        return loss_dm

    def _compute_guidance_loss(
        self,
        pipe,
        peft_dit,
        generated_latents,
        timesteps,
        inputs_shared,
        inputs_posi,
        inputs_nega,
    ):
        generated_latents = generated_latents.detach()
        original_training = peft_dit.training
        peft_dit.train()

        noise = torch.randn_like(generated_latents)
        sigmas = self._expand_to_latents(timesteps / 1000.0, generated_latents)
        noisy_latents = (sigmas * noise + (1 - sigmas) * generated_latents).to(
            dtype=pipe.torch_dtype,
        )
        timestep_tensor = timesteps.to(
            device=generated_latents.device,
            dtype=pipe.torch_dtype,
        )
        other_models = {
            name: getattr(pipe, name)
            for name in pipe.in_iteration_models if name != "dit"
        }
        noisy_shared = dict(inputs_shared)
        noisy_shared["latents"] = noisy_latents

        peft_dit.set_adapter("guidance")
        pred_flow_fake = pipe.model_fn(
            dit=peft_dit,
            **other_models,
            **inputs_posi,
            **noisy_shared,
            timestep=timestep_tensor,
            progress_id=None,
        )
        target_flow = noise - generated_latents
        loss_guidance = torch.mean((pred_flow_fake.float() - target_flow.float()) ** 2)
        peft_dit.train(original_training)
        return loss_guidance


class Flux2DMD2TrainingModule(Flux2ImageTrainingModule):
    DEFAULT_STUDENT_ADAPTER_NAME = "default"

    def __init__(
        self,
        num_denoising_steps=4,
        guidance_scale=4.0,
        embedded_guidance: Optional[float] = None,
        timestep_sampling_strategy="logit_normal",
        backward_simulation=False,
        beta_alpha=4.0,
        beta_beta=1.2,
        logit_mean=1.0,
        logit_std=1.0,
        dynamic_rescale_t_steps=500,
        rescale_t_val=1.0,
        student_schedule="linear",
        cfg_distill=True,
        lora_checkpoint=None,
        student_lora_checkpoint=None,
        **kwargs,
    ):
        if not kwargs.get("task", "").startswith("dmd2"):
            raise ValueError(f"Task must start with 'dmd2', got: {kwargs.get('task')}")

        self.lora_checkpoint = lora_checkpoint
        self.student_lora_checkpoint = student_lora_checkpoint
        self.dmd2_embedded_guidance = (
            float(embedded_guidance) if embedded_guidance is not None else float(guidance_scale)
        )

        original_lora_rank = kwargs.get("lora_rank", 32)
        original_lora_target_modules = kwargs.get("lora_target_modules", "")
        original_lora_base_model = kwargs.get("lora_base_model")
        kwargs["lora_base_model"] = None
        super().__init__(**kwargs)

        self.lora_rank = original_lora_rank
        self.lora_target_modules = original_lora_target_modules
        self.lora_base_model = original_lora_base_model
        self.student_adapter_name = self.DEFAULT_STUDENT_ADAPTER_NAME

        self._setup_peft_adapters()
        self.dmd2_loss = DMD2Loss(
            num_denoising_steps=num_denoising_steps,
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
        )
        self._register_dmd2_tasks()

    def _setup_peft_adapters(self):
        print("[FLUX2 DMD2] Converting DiT to PEFT multi-adapter model")

        normalized_teacher_state = None
        lora_rank = self.lora_rank
        lora_target_modules = self.lora_target_modules
        if self.lora_checkpoint:
            lora_rank, lora_target_modules, normalized_teacher_state = (
                self._load_checkpoint_adapter_config(self.lora_checkpoint)
            )
        elif isinstance(lora_target_modules, str):
            lora_target_modules = lora_target_modules.split(",")

        normalized_student_state = normalized_teacher_state
        if self.student_lora_checkpoint:
            _, _, normalized_student_state = self._load_checkpoint_adapter_config(
                self.student_lora_checkpoint,
            )

        base_dit = self.pipe.dit
        if any("lora" in name.lower() for name, _ in base_dit.named_parameters()):
            raise RuntimeError("[FLUX2 DMD2] Base DiT already has LoRA parameters")

        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=lora_target_modules,
        )
        print(f"[FLUX2 DMD2] LoRA rank: {lora_rank}")
        print(f"[FLUX2 DMD2] Target modules: {lora_target_modules}")

        self.peft_dit = get_peft_model(base_dit, peft_config)
        self.peft_dit.student_adapter_name = self.student_adapter_name
        for param in self.peft_dit.parameters():
            if param.requires_grad:
                param.data = param.to(self.pipe.torch_dtype)

        if normalized_student_state is not None:
            print("[FLUX2 DMD2] Loading student/default adapter init")
            load_result = self.peft_dit.load_state_dict(normalized_student_state, strict=False)
            if load_result.unexpected_keys:
                print(
                    "[FLUX2 DMD2] Unexpected keys while loading student adapter: "
                    f"{load_result.unexpected_keys[:5]}"
                )
        else:
            print("[FLUX2 DMD2] No LoRA checkpoint provided; student starts random")

        student_state = get_peft_model_state_dict(
            self.peft_dit,
            adapter_name=self.student_adapter_name,
        )
        for adapter_name in ("teacher", "guidance"):
            self.peft_dit.add_adapter(adapter_name, peft_config)

        if normalized_teacher_state is not None:
            set_peft_model_state_dict(
                self.peft_dit,
                self._strip_adapter_name_for_peft_set(normalized_teacher_state),
                adapter_name="teacher",
            )
        else:
            set_peft_model_state_dict(self.peft_dit, student_state, adapter_name="teacher")
        set_peft_model_state_dict(self.peft_dit, student_state, adapter_name="guidance")

        self._set_adapter_trainable(self.student_adapter_name, trainable=True)
        self._set_adapter_trainable("teacher", trainable=False)
        self._set_adapter_trainable("guidance", trainable=True)
        self.pipe.dit = self.peft_dit
        self._print_adapter_info()

    def _load_checkpoint_adapter_config(self, checkpoint_path):
        lora_state_dict = load_state_dict(checkpoint_path)
        lora_state_dict = self.mapping_lora_state_dict(lora_state_dict)
        normalized_state_dict = self._normalize_student_lora_state_dict(lora_state_dict)
        lora_rank = self._infer_lora_rank(normalized_state_dict)
        lora_target_modules = self._infer_target_modules(normalized_state_dict)
        print(f"[FLUX2 DMD2] Inferred LoRA rank from checkpoint: {lora_rank}")
        return lora_rank, lora_target_modules, normalized_state_dict

    @staticmethod
    def _strip_adapter_name_for_peft_set(normalized_state_dict):
        """PEFT set_peft_model_state_dict inserts the target adapter name itself."""
        stripped_state_dict = {}
        for key, value in normalized_state_dict.items():
            stripped_key = key
            for adapter_name in ("default", "student", "teacher", "guidance"):
                stripped_key = stripped_key.replace(
                    f".lora_A.{adapter_name}.",
                    ".lora_A.",
                )
                stripped_key = stripped_key.replace(
                    f".lora_B.{adapter_name}.",
                    ".lora_B.",
                )
            stripped_state_dict[stripped_key] = value
        return stripped_state_dict

    def _normalize_student_lora_state_dict(self, lora_state_dict):
        normalized_state_dict = {}
        for key, value in lora_state_dict.items():
            if "lora_" not in key:
                continue

            normalized_key = key
            if normalized_key.startswith("pipe.dit."):
                normalized_key = normalized_key[len("pipe.dit."):]
            elif normalized_key.startswith("model."):
                normalized_key = normalized_key[len("model."):]

            if ".lora_A." in normalized_key:
                prefix, suffix = normalized_key.split(".lora_A.", 1)
                if suffix.startswith("student."):
                    suffix = "default." + suffix[len("student."):]
                elif not suffix.startswith("default."):
                    if suffix.startswith("guidance.") or suffix.startswith("teacher."):
                        continue
                    suffix = "default." + suffix
                normalized_key = f"{prefix}.lora_A.{suffix}"
            elif ".lora_B." in normalized_key:
                prefix, suffix = normalized_key.split(".lora_B.", 1)
                if suffix.startswith("student."):
                    suffix = "default." + suffix[len("student."):]
                elif not suffix.startswith("default."):
                    if suffix.startswith("guidance.") or suffix.startswith("teacher."):
                        continue
                    suffix = "default." + suffix
                normalized_key = f"{prefix}.lora_B.{suffix}"

            if not normalized_key.startswith("base_model.model."):
                normalized_key = f"base_model.model.{normalized_key}"
            normalized_state_dict[normalized_key] = value
        return normalized_state_dict

    @staticmethod
    def _infer_lora_rank(normalized_state_dict):
        for key, value in normalized_state_dict.items():
            if ".lora_A.default.weight" in key or ".lora_A.weight" in key:
                return value.shape[0]
        raise ValueError("Cannot infer LoRA rank: no lora_A weights found")

    @staticmethod
    def _infer_target_modules(normalized_state_dict):
        target_modules = set()
        for key in normalized_state_dict:
            if ".lora_A.default.weight" in key:
                module_path = key.split(".lora_A.default.weight", 1)[0]
            elif ".lora_B.default.weight" in key:
                module_path = key.split(".lora_B.default.weight", 1)[0]
            elif ".lora_A.weight" in key:
                module_path = key.split(".lora_A.weight", 1)[0]
            elif ".lora_B.weight" in key:
                module_path = key.split(".lora_B.weight", 1)[0]
            else:
                continue
            if module_path.startswith("base_model.model."):
                module_path = module_path[len("base_model.model."):]
            target_modules.add(module_path)
        return sorted(target_modules)

    def _set_adapter_trainable(self, adapter_name, trainable):
        updated_count = 0
        for name, param in self.peft_dit.named_parameters():
            if self._is_adapter_parameter(name, adapter_name) and "lora" in name.lower():
                param.requires_grad = trainable
                updated_count += 1
        action = "Enabled" if trainable else "Disabled"
        print(f"[FLUX2 DMD2] {action} grad for {updated_count} parameters in {adapter_name!r}")

    def _print_adapter_info(self):
        base_params = sum(
            p.numel() for name, p in self.peft_dit.named_parameters()
            if "lora" not in name.lower()
        )
        print(f"[FLUX2 DMD2] Base params: {base_params:,}")
        for adapter_name in [self.student_adapter_name, "teacher", "guidance"]:
            total = sum(
                p.numel() for name, p in self.peft_dit.named_parameters()
                if self._is_adapter_parameter(name, adapter_name) and "lora" in name.lower()
            )
            trainable = sum(
                p.numel() for name, p in self.peft_dit.named_parameters()
                if self._is_adapter_parameter(name, adapter_name)
                and "lora" in name.lower()
                and p.requires_grad
            )
            label = "student" if adapter_name == self.student_adapter_name else adapter_name
            print(f"[FLUX2 DMD2] {label}: total={total:,} trainable={trainable:,}")

    def _register_dmd2_tasks(self):
        def generator_loss_fn(pipe, inputs_shared, inputs_posi, inputs_nega):
            return self.dmd2_loss(
                pipe=pipe,
                peft_dit=self.peft_dit,
                inputs_shared=inputs_shared,
                inputs_posi=inputs_posi,
                inputs_nega=inputs_nega,
                mode="generator",
            )

        def guidance_loss_fn(pipe, inputs_shared, inputs_posi, inputs_nega):
            return self.dmd2_loss(
                pipe=pipe,
                peft_dit=self.peft_dit,
                inputs_shared=inputs_shared,
                inputs_posi=inputs_posi,
                inputs_nega=inputs_nega,
                mode="guidance",
            )

        self.task_to_loss.update({
            "dmd2:generator": generator_loss_fn,
            "dmd2:guidance": guidance_loss_fn,
        })

    def process_pipeline_inputs(self, inputs):
        if not getattr(self.pipe.scheduler, "training", False):
            self.pipe.scheduler.set_timesteps(1000, training=True)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return inputs

    def prepare_inputs(self, data=None, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs_shared, inputs_posi, inputs_nega = inputs
        inputs_shared = dict(inputs_shared)
        inputs_shared["cfg_scale"] = float(self.dmd2_loss.guidance_scale)
        inputs_shared["embedded_guidance"] = float(self.dmd2_embedded_guidance)
        raw_inputs = (inputs_shared, inputs_posi, inputs_nega)
        processed_inputs = self.process_pipeline_inputs(raw_inputs)
        return raw_inputs, processed_inputs

    def forward(self, data=None, inputs=None, processed_inputs=None):
        if processed_inputs is None:
            _, processed_inputs = self.prepare_inputs(data=data, inputs=inputs)
        return self.task_to_loss[self.task](self.pipe, *processed_inputs)

    def set_adapter_for_training(self, adapter_name):
        self.peft_dit.set_adapter(adapter_name)

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        exported_state_dict = {}
        for name, param in state_dict.items():
            canonical_name = name
            for prefix in ("pipe.dit.", "peft_dit."):
                if canonical_name.startswith(prefix):
                    canonical_name = canonical_name[len(prefix):]
                    break
            if canonical_name.startswith("base_model.model."):
                canonical_name = canonical_name[len("base_model.model."):]
            if canonical_name not in trainable_param_names:
                continue
            if remove_prefix is not None and canonical_name.startswith(remove_prefix):
                canonical_name = canonical_name[len(remove_prefix):]
            exported_state_dict[canonical_name] = param
        return exported_state_dict

    def trainable_param_names(self):
        names = set()
        for name, param in self.peft_dit.named_parameters():
            if not (
                self._is_adapter_parameter(name, self.student_adapter_name)
                and "lora" in name.lower()
                and param.requires_grad
            ):
                continue
            if name.startswith("base_model.model."):
                name = name[len("base_model.model."):]
            names.add(name)
        return names

    def get_guidance_trainable_modules(self):
        return [
            p for name, p in self.peft_dit.named_parameters()
            if self._is_adapter_parameter(name, "guidance") and p.requires_grad
        ]

    def trainable_modules(self):
        return [
            p for name, p in self.peft_dit.named_parameters()
            if self._is_adapter_parameter(name, self.student_adapter_name) and p.requires_grad
        ]

    @staticmethod
    def _is_adapter_parameter(param_name, adapter_name):
        return f".{adapter_name}." in param_name


def dmd2_parser() -> argparse.ArgumentParser:
    parser = flux2_parser()
    parser.add_argument("--num_denoising_steps", type=int, default=4)
    parser.add_argument("--generator_update_freq", type=int, default=5)
    parser.add_argument("--dmd2_guidance_scale", type=float, default=4.0)
    parser.add_argument("--dmd2_embedded_guidance", type=float, default=None)
    parser.add_argument(
        "--disable_cfg_distill",
        dest="cfg_distill",
        action="store_false",
        help="Disable CFG teacher real-score distillation and use conditional teacher score only.",
    )
    parser.set_defaults(cfg_distill=True)
    parser.add_argument(
        "--timestep_sampling_strategy",
        type=str,
        default="logit_normal",
        choices=["uniform", "beta", "logit_normal"],
    )
    parser.add_argument("--backward_simulation", action="store_true")
    parser.add_argument("--beta_alpha", type=float, default=4.0)
    parser.add_argument("--beta_beta", type=float, default=1.2)
    parser.add_argument("--logit_mean", type=float, default=1.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--dynamic_rescale_t_steps", type=int, default=500)
    parser.add_argument("--rescale_t_val", type=float, default=1.0)
    parser.add_argument(
        "--student_schedule",
        type=str,
        default="linear",
        choices=["linear", "flux2"],
        help="Student generation schedule. 'linear' matches the tiling/qwen DMD2 run; 'flux2' uses the native FLUX.2 inference scheduler.",
    )
    parser.add_argument("--learning_rate_gen", type=float, default=None)
    parser.add_argument("--learning_rate_guidance", type=float, default=None)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deepspeed_config_path", type=str, default=None)
    parser.add_argument("--student_lora_checkpoint", type=str, default=None)
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "pbr-material-edit-flux2-dmd2"),
    )
    parser.add_argument("--wandb_run_name", type=str, default=os.environ.get("WANDB_RUN_NAME"))
    return parser


def build_dataset_for_metadata(args, metadata_path, repeat=1):
    dataset_args = copy.copy(args)
    dataset_args.dataset_metadata_path = metadata_path
    dataset_args.dataset_repeat = repeat
    return build_dataset(dataset_args)


def launch_dmd2_training_task(
    accelerator,
    dataset,
    model,
    model_logger,
    args,
    val_dataset=None,
):
    learning_rate_gen = args.learning_rate_gen if args.learning_rate_gen is not None else args.learning_rate
    learning_rate_guidance = (
        args.learning_rate_guidance if args.learning_rate_guidance is not None else args.learning_rate
    )
    generator_params = list(model.trainable_modules())
    guidance_params = list(model.get_guidance_trainable_modules())
    overlap = {id(p) for p in generator_params} & {id(p) for p in guidance_params}
    if overlap:
        raise RuntimeError(f"Generator and guidance parameter groups overlap: {len(overlap)}")

    print(f"[FLUX2 DMD2] Generator trainable params: {sum(p.numel() for p in generator_params):,}")
    print(f"[FLUX2 DMD2] Guidance trainable params: {sum(p.numel() for p in guidance_params):,}")
    print(f"[FLUX2 DMD2] Generator LR: {learning_rate_gen}")
    print(f"[FLUX2 DMD2] Guidance LR: {learning_rate_guidance}")
    print(f"[FLUX2 DMD2] Generator update freq: {args.generator_update_freq}")
    print(f"[FLUX2 DMD2] Student schedule: {model.dmd2_loss.student_schedule}")
    print(f"[FLUX2 DMD2] CFG distill: {model.dmd2_loss.cfg_distill}")

    optimizer = torch.optim.AdamW(
        [
            {"params": generator_params, "lr": learning_rate_gen},
            {"params": guidance_params, "lr": learning_rate_guidance},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    dataloader_kwargs = {
        "collate_fn": first_item_collate,
        "num_workers": args.dataset_num_workers,
        "pin_memory": True,
    }
    if args.dataset_num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 2
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, **dataloader_kwargs)

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            **dataloader_kwargs,
        )

    if val_dataloader is not None:
        model, optimizer, dataloader, val_dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, val_dataloader, scheduler,
        )
    else:
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler,
        )

    def _student_adapter_name():
        return accelerator.unwrap_model(model).student_adapter_name

    def _set_training_state(task_name, adapter_name):
        base_model = accelerator.unwrap_model(model)
        base_model.task = task_name
        base_model.set_adapter_for_training(adapter_name)

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

    def _distributed_mean_scalar(value):
        if value is None:
            return None
        tensor = (
            value.detach().float().reshape(1).to(accelerator.device)
            if torch.is_tensor(value)
            else torch.tensor([float(value)], device=accelerator.device, dtype=torch.float32)
        )
        gathered = accelerator.gather(tensor)
        return gathered.mean().item()

    def _distributed_mean_metrics(metrics):
        averaged = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                averaged[key] = _distributed_mean_scalar(value)
        return averaged

    def _log_metrics(metrics):
        if not accelerator.is_main_process:
            return
        os.makedirs(model_logger.output_path, exist_ok=True)
        with open(os.path.join(model_logger.output_path, "dmd2_metrics.jsonl"), "a", encoding="utf-8") as file:
            file.write(json.dumps(metrics, ensure_ascii=True, sort_keys=True) + "\n")
        model_logger.num_steps = int(metrics.get("train/global_step", 0))
        model_logger.log_metrics(metrics)

    def _restore_training_state(task_name):
        if task_name == "dmd2:guidance":
            _set_training_state("dmd2:guidance", "guidance")
        else:
            _set_training_state("dmd2:generator", _student_adapter_name())

    def _run_validation():
        if val_dataloader is None:
            return None
        base_model = accelerator.unwrap_model(model)
        saved_task = base_model.task
        saved_step = base_model.dmd2_loss.current_step
        was_training = model.training
        _set_training_state("dmd2:generator", _student_adapter_name())
        model.eval()
        total_loss = 0.0
        num_batches = 0
        try:
            for val_idx, data in enumerate(val_dataloader):
                if args.max_val_batches is not None and val_idx >= args.max_val_batches:
                    break
                data = send_to_device(data, accelerator.device, non_blocking=True)
                processed_inputs = accelerator.unwrap_model(model).prepare_inputs(data=data)[1]
                with torch.no_grad():
                    loss = model(processed_inputs=processed_inputs)
                total_loss += _distributed_mean_scalar(loss)
                num_batches += 1
        finally:
            accelerator.unwrap_model(model).dmd2_loss.current_step = saved_step
            _restore_training_state(saved_task)
            if was_training:
                model.train()
        return total_loss / max(num_batches, 1)

    global_step = 0
    log_steps = max(1, args.log_steps)
    generator_update_freq = max(1, args.generator_update_freq)
    val_steps = max(1, args.val_steps)

    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            disable=not accelerator.is_main_process,
        )
        for batch_idx, data in enumerate(progress_bar):
            step_start_time = time.perf_counter()
            data = send_to_device(data, accelerator.device, non_blocking=True)
            processed_inputs = accelerator.unwrap_model(model).prepare_inputs(data=data)[1]
            is_generator_turn = batch_idx % generator_update_freq == 0
            loss_dm = None

            if is_generator_turn:
                _set_training_state("dmd2:generator", _student_adapter_name())
                optimizer.zero_grad()
                with accelerator.accumulate(model):
                    loss_dm = model(processed_inputs=processed_inputs)
                    if not torch.isfinite(loss_dm).all():
                        raise FloatingPointError(
                            f"Non-finite generator loss at step {global_step}: {loss_dm}"
                        )
                    accelerator.backward(loss_dm)
                    optimizer.step()
                    scheduler.step()

            _set_training_state("dmd2:guidance", "guidance")
            optimizer.zero_grad()
            with accelerator.accumulate(model):
                loss_guidance = model(processed_inputs=processed_inputs)
                if not torch.isfinite(loss_guidance).all():
                    raise FloatingPointError(
                        f"Non-finite guidance loss at step {global_step}: {loss_guidance}"
                    )
                accelerator.backward(loss_guidance)
                optimizer.step()
                scheduler.step()

            step_time = time.perf_counter() - step_start_time
            loss_guidance_value = loss_guidance.detach().float().item()
            loss_dm_value = loss_dm.detach().float().item() if loss_dm is not None else None
            postfix = {"step_s": f"{step_time:.1f}", "loss_g": f"{loss_guidance_value:.4f}"}
            if loss_dm_value is not None:
                postfix["loss_dm"] = f"{loss_dm_value:.4f}"
            progress_bar.set_postfix(postfix)

            should_log = is_generator_turn or (global_step % log_steps == 0)
            if should_log:
                metrics = {
                    "train/loss_guidance": _distributed_mean_scalar(loss_guidance),
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                    "perf/step_seconds": step_time,
                }
                if loss_dm is not None:
                    metrics["train/loss_dm"] = _distributed_mean_scalar(loss_dm)
                metrics.update(_distributed_mean_metrics(
                    dict(getattr(accelerator.unwrap_model(model).dmd2_loss, "latest_metrics", {})),
                ))
                _log_metrics(metrics)

            if val_dataloader is not None and (global_step + 1) % val_steps == 0:
                if accelerator.is_main_process:
                    print(f"\n[FLUX2 DMD2] Running validation at step {global_step}")
                val_loss = _run_validation()
                if accelerator.is_main_process:
                    print(f"[FLUX2 DMD2] Validation loss_dm: {val_loss:.4f}")
                _log_metrics({
                    "val/loss": val_loss,
                    "val/loss_dm": val_loss,
                    "train/global_step": global_step,
                })

            if args.save_steps is not None and (global_step + 1) % args.save_steps == 0:
                if accelerator.is_main_process:
                    print(f"\n[FLUX2 DMD2] Saving checkpoint at step {global_step}")
                _save_student_checkpoint(f"step-{global_step}.safetensors")

            global_step += 1
            if args.max_train_steps and global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            print(f"\n[FLUX2 DMD2] Epoch {epoch + 1} complete; saving checkpoint")
        _save_student_checkpoint(f"epoch-{epoch + 1}.safetensors")
        if args.max_train_steps and global_step >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        print("[FLUX2 DMD2] Training completed")


if __name__ == "__main__":
    parser = dmd2_parser()
    args = parser.parse_args()
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

    model = Flux2DMD2TrainingModule(
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
        num_denoising_steps=args.num_denoising_steps,
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
    )

    wandb_enabled = os.environ.get("WANDB_MODE", "online").lower() not in {
        "disabled",
        "off",
        "false",
        "0",
    }
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_wandb=wandb_enabled,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name or os.path.basename(args.output_path),
    )
    if accelerator.is_main_process and wandb_enabled and not getattr(model_logger, "use_wandb", False):
        raise RuntimeError(
            "W&B is enabled but did not initialize. Check that wandb is installed "
            "and logged in, or set WANDB_MODE=disabled to train without upload."
        )
    launch_dmd2_training_task(
        accelerator,
        dataset,
        model,
        model_logger,
        args,
        val_dataset=val_dataset,
    )
