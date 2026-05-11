"""
DMD2 Training Module with PEFT Hotswap

Uses HuggingFace PEFT library to manage multiple LoRA adapters with hotswapping.
This is the most memory-efficient approach:

For 20B Qwen model (bf16):
- Base model: 40GB (loaded once)
- Student LoRA: ~500MB (adapter weights)
- Teacher LoRA: ~500MB (adapter weights)
- Guidance LoRA: ~500MB (adapter weights)
- Total: ~41.5GB (vs 120GB with 3 separate models!)

Hotswapping allows us to switch between adapters in microseconds without
loading/unloading models from GPU memory.
"""

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from diffsynth.diffusion.loss import DMD2Loss

from train import QwenImageTrainingModule


class QwenImageDMD2TrainingModule(QwenImageTrainingModule):
    """
    DMD2 training module using PEFT for multi-adapter hotswapping.

    Architecture:
    - Base Qwen DiT: 40GB (shared)
    - Student adapter: ~500MB (trainable)
    - Teacher adapter: ~500MB (frozen, identical to student at init)
    - Guidance adapter: ~500MB (trainable, identical to student at init)

    All three adapters are initialized from the same lora_checkpoint,
    ensuring identical starting points for DMD2 training.

    Uses PEFT's set_adapter() for fast switching during training.
    """

    DEFAULT_STUDENT_ADAPTER_NAME = "default"

    def __init__(
        self,
        # DMD2-specific parameters
        num_denoising_steps=4,
        guidance_scale=1.0,
        timestep_sampling_strategy="logit_normal",
        backward_simulation=False,
        beta_alpha=4.0,
        beta_beta=1.2,
        logit_mean=1.0,
        logit_std=1.0,
        dynamic_rescale_t_steps=500,
        rescale_t_val=1.0,
        # LoRA checkpoint to initialize all adapters (required for DMD2)
        lora_checkpoint=None,
        # All other parameters from base class
        **kwargs
    ):
        # Verify DMD2 requirements
        if not kwargs.get('task', '').startswith('dmd2'):
            raise ValueError(f"Task must start with 'dmd2' for DMD2 training, got: {kwargs.get('task')}")

        # Store config before parent init
        self.lora_checkpoint = lora_checkpoint

        # CRITICAL: Disable LoRA in base class initialization
        # We'll add PEFT adapters instead
        original_lora_rank = kwargs.get('lora_rank', 32)
        original_lora_target_modules = kwargs.get('lora_target_modules', 'to_q,to_k,to_v,to_out')

        # Temporarily disable base-class LoRA injection.
        # switch_pipe_to_training_mode() only injects when lora_base_model is not None,
        # while PEFT requires a real integer rank later.
        original_lora_base_model = kwargs.get('lora_base_model')
        kwargs['lora_base_model'] = None

        # Initialize base class WITHOUT LoRA
        super().__init__(**kwargs)

        # Restore LoRA config for PEFT
        self.lora_rank = original_lora_rank
        self.lora_target_modules = original_lora_target_modules
        self.lora_base_model = original_lora_base_model
        self.student_adapter_name = self.DEFAULT_STUDENT_ADAPTER_NAME

        # Build PEFT adapters (student/teacher/guidance) on top of the shared base DiT.
        self._setup_peft_adapters()

        # Create DMD2 loss module
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
            rescale_t_val=rescale_t_val
        )

        # Register DMD2 tasks
        self._register_dmd2_tasks()

    def _setup_peft_adapters(self):
        """
        Convert the base DiT model to PEFT with multiple adapters.

        Steps:
        1. Get base DiT (should have NO LoRA since we disabled it)
        2. Create PEFT model with student adapter
        3. Load pre-trained LoRA weights into student
        4. Add teacher adapter, copy from student, freeze
        5. Add guidance adapter, copy from student
        """
        print("[DMD2 PEFT] Converting to multi-adapter PEFT model...")

        # Prefer adapter config inferred from checkpoint so the PEFT structure
        # exactly matches the saved student LoRA.
        normalized_state_dict = None
        lora_rank = self.lora_rank
        lora_target_modules = self.lora_target_modules
        if self.lora_checkpoint:
            lora_rank, lora_target_modules, normalized_state_dict = self._load_checkpoint_adapter_config(self.lora_checkpoint)
        elif isinstance(lora_target_modules, str):
            lora_target_modules = lora_target_modules.split(",")

        # Get the base DiT model (should be clean, no LoRA)
        base_dit = self.pipe.dit

        # Verify no LoRA exists
        has_lora = any('lora' in name.lower() for name, _ in base_dit.named_parameters())
        if has_lora:
            raise RuntimeError(
                "[DMD2 PEFT] ERROR: Base model still has LoRA! "
                "This should not happen if lora_rank was set to None. "
                "Check base class initialization."
            )

        print(f"[DMD2 PEFT] Base DiT is clean (no existing LoRA)")

        # Create PEFT config
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,  # Typically same as rank
            target_modules=lora_target_modules,
        )  # Don't specify task_type to avoid generation-specific attributes

        print(f"[DMD2 PEFT] PEFT Config:")
        print(f"  Rank: {lora_rank}")
        print(f"  Target modules: {lora_target_modules}")

        # Reuse the framework's default-adapter naming/loading convention for student,
        # but keep a real PeftModel so we can add teacher/guidance adapters.
        print("[DMD2 PEFT] Creating PEFT model with default adapter for student...")
        self.peft_dit = get_peft_model(
            base_dit,
            peft_config,
        )
        self.peft_dit.student_adapter_name = self.student_adapter_name
        for param in self.peft_dit.parameters():
            if param.requires_grad:
                param.data = param.to(self.pipe.torch_dtype)

        # Load pre-trained LoRA weights into student adapter
        if self.lora_checkpoint:
            print(f"[DMD2 PEFT] Loading LoRA checkpoint into student: {self.lora_checkpoint}")
            load_result = self.peft_dit.load_state_dict(normalized_state_dict, strict=False)

            loaded_student_state = get_peft_model_state_dict(self.peft_dit, adapter_name=self.student_adapter_name)
            print(
                f"[DMD2 PEFT] Loaded student adapter tensors: "
                f"{len(loaded_student_state)}"
            )
            if len(load_result.unexpected_keys) > 0:
                print(f"[DMD2 PEFT] Unexpected keys while loading student adapter: {len(load_result.unexpected_keys)}")
                print(f"[DMD2 PEFT] Sample unexpected keys: {load_result.unexpected_keys[:5]}")
        else:
            print("[DMD2 PEFT] Warning: No lora_checkpoint provided, student starts with random init")

        student_state = get_peft_model_state_dict(self.peft_dit, adapter_name=self.student_adapter_name)
        for adapter_name in ("teacher", "guidance"):
            print(f"[DMD2 PEFT] Adding {adapter_name} adapter...")
            self.peft_dit.add_adapter(adapter_name, peft_config)
            print(f"[DMD2 PEFT] Copying student weights to {adapter_name}...")
            set_peft_model_state_dict(self.peft_dit, student_state, adapter_name=adapter_name)

        # Teacher stays frozen; guidance/student are trainable.
        self._set_adapter_trainable("teacher", trainable=False)
        self._set_adapter_trainable("guidance", trainable=True)

        # Step 6: Replace original DiT with PEFT DiT
        self.pipe.dit = self.peft_dit

        # Print adapter info
        self._print_adapter_info()

    def _load_checkpoint_adapter_config(self, checkpoint_path):
        from diffsynth.core import load_state_dict

        lora_state_dict = load_state_dict(checkpoint_path)
        lora_state_dict = self.mapping_lora_state_dict(lora_state_dict)
        normalized_state_dict = self._normalize_student_lora_state_dict(lora_state_dict)
        lora_rank = self._infer_lora_rank(normalized_state_dict)
        lora_target_modules = self._infer_target_modules(normalized_state_dict)

        print(f"[DMD2 PEFT] Inferred LoRA rank from checkpoint: {lora_rank}")
        print(f"[DMD2 PEFT] Inferred target modules from checkpoint: {lora_target_modules}")
        return lora_rank, lora_target_modules, normalized_state_dict

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

    def _infer_lora_rank(self, normalized_state_dict):
        for key, value in normalized_state_dict.items():
            if ".lora_A.default.weight" in key:
                return value.shape[0]
            if ".lora_A.weight" in key:
                return value.shape[0]
        raise ValueError("Cannot infer LoRA rank from checkpoint: no lora_A weights found")

    def _infer_target_modules(self, normalized_state_dict):
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

            path_parts = module_path.split(".")
            if len(path_parts) >= 3 and path_parts[0] == "transformer_blocks" and path_parts[1].isdigit():
                module_path = ".".join(path_parts[2:])

            target_modules.add(module_path)

        return sorted(target_modules)

    def _set_adapter_trainable(self, adapter_name, trainable):
        """Set requires_grad for all LoRA params of one adapter."""
        updated_count = 0
        for name, param in self.peft_dit.named_parameters():
            if self._is_adapter_parameter(name, adapter_name) and 'lora' in name.lower():
                param.requires_grad = trainable
                updated_count += 1
        action = "Enabled" if trainable else "Disabled"
        print(f"[DMD2 PEFT] {action} grad for {updated_count} parameters in '{adapter_name}' adapter")

    def _print_adapter_info(self):
        """Print detailed adapter information."""
        print("\n" + "="*70)
        print("PEFT ADAPTER CONFIGURATION")
        print("="*70)

        # Count base parameters
        base_params = sum(
            p.numel() for n, p in self.peft_dit.named_parameters()
            if 'lora' not in n.lower()
        )

        print(f"\nBase Model:")
        print(f"  Parameters: {base_params:,} (~{base_params/1e9:.1f}B)")
        print(f"  Memory: ~{base_params * 2 / 1e9:.1f}GB (bf16)")

        # Count adapter parameters
        for adapter_name in [self.student_adapter_name, "teacher", "guidance"]:
            trainable = sum(
                p.numel() for n, p in self.peft_dit.named_parameters()
                if self._is_adapter_parameter(n, adapter_name) and p.requires_grad
            )
            total = sum(
                p.numel() for n, p in self.peft_dit.named_parameters()
                if self._is_adapter_parameter(n, adapter_name) and 'lora' in n.lower()
            )

            display_name = "student" if adapter_name == self.student_adapter_name else adapter_name
            status = "FROZEN" if adapter_name == "teacher" else "TRAINABLE"

            print(f"\n{display_name.upper()} Adapter [{status}]:")
            print(f"  Total parameters: {total:,} (~{total/1e6:.1f}M)")
            print(f"  Trainable parameters: {trainable:,} (~{trainable/1e6:.1f}M)")
            print(f"  Memory: ~{total * 2 / 1e6:.1f}MB (bf16)")

        total_lora_params = sum(
            p.numel() for n, p in self.peft_dit.named_parameters()
            if 'lora' in n.lower()
        )
        total_memory = (base_params + total_lora_params) * 2 / 1e9

        print(f"\nTotal Memory: ~{total_memory:.1f}GB (base + all adapters)")
        print(f"Memory saving: ~{(base_params * 3 * 2 / 1e9) - total_memory:.1f}GB vs 3 separate models")
        print("="*70 + "\n")

    def _register_dmd2_tasks(self):
        """Register DMD2 tasks with PEFT hotswapping."""
        def generator_loss_fn(pipe, inputs_shared, inputs_posi, inputs_nega):
            return self.dmd2_loss(
                pipe=pipe,
                peft_dit=self.peft_dit,
                inputs_shared=inputs_shared,
                inputs_posi=inputs_posi,
                inputs_nega=inputs_nega,
                mode="generator"
            )

        def guidance_loss_fn(pipe, inputs_shared, inputs_posi, inputs_nega):
            return self.dmd2_loss(
                pipe=pipe,
                peft_dit=self.peft_dit,
                inputs_shared=inputs_shared,
                inputs_posi=inputs_posi,
                inputs_nega=inputs_nega,
                mode="guidance"
            )

        self.task_to_loss.update({
            "dmd2:generator": generator_loss_fn,
            "dmd2:guidance": guidance_loss_fn,
        })
        print("[DMD2 PEFT] Task mappings registered with PEFT hotswapping")

    def process_pipeline_inputs(self, inputs):
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return inputs

    def prepare_inputs(self, data=None, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)

        inputs_shared, inputs_posi, inputs_nega = inputs
        inputs_shared = dict(inputs_shared)

        # DMD2 needs real negative-side preprocessing so CFG teacher inference does
        # not silently reuse positive prompt embeddings from the cfg_scale == 1 path.
        inputs_shared["cfg_scale"] = float(self.dmd2_loss.guidance_scale)

        raw_inputs = (inputs_shared, inputs_posi, inputs_nega)
        processed_inputs = self.process_pipeline_inputs(raw_inputs)
        return raw_inputs, processed_inputs

    def forward(self, data=None, inputs=None, processed_inputs=None):
        if processed_inputs is None:
            _, processed_inputs = self.prepare_inputs(data=data, inputs=inputs)
        loss = self.task_to_loss[self.task](self.pipe, *processed_inputs)
        return loss

    def set_adapter_for_training(self, adapter_name):
        """Set active adapter for training step."""
        self.peft_dit.set_adapter(adapter_name)

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        """Export student/default LoRA weights in the same key format as the teacher LoRA."""
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
        """Return student LoRA parameter names for checkpoint export."""
        trainable_param_names = set()
        for name, param in self.peft_dit.named_parameters():
            if not (self._is_adapter_parameter(name, self.student_adapter_name) and 'lora' in name.lower() and param.requires_grad):
                continue
            if name.startswith("base_model.model."):
                name = name[len("base_model.model."):]
            trainable_param_names.add(name)
        return trainable_param_names

    def get_guidance_trainable_modules(self):
        """Get trainable parameters from guidance adapter."""
        return [
            p for n, p in self.peft_dit.named_parameters()
            if self._is_adapter_parameter(n, "guidance") and p.requires_grad
        ]

    def trainable_modules(self):
        """Get trainable parameters from student adapter (overrides base class)."""
        return [
            p for n, p in self.peft_dit.named_parameters()
            if self._is_adapter_parameter(n, self.student_adapter_name) and p.requires_grad
        ]

    def _is_adapter_parameter(self, param_name, adapter_name):
        return f".{adapter_name}." in param_name
