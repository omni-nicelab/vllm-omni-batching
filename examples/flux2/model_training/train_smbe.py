"""FLUX.2 image-edit training helpers for SMBE tileable data."""

from __future__ import annotations

import argparse
import os
import sys

import accelerate
import torch
from diffsynth.core import UnifiedDataset
from diffsynth.diffusion import *  # noqa: F401,F403
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from augmentation import TileableAugmentationDataset, detect_normal_panels_from_config
from random_circular_crop import RandomCircularCropDataset, make_load_only_operator

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _model_config_from_path_or_id(value: str | None, default_value: ModelConfig) -> ModelConfig:
    if value is None:
        return default_value
    if ":" in value and not os.path.exists(value):
        model_id, origin_file_pattern = value.split(":", 1)
        return ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern)
    return ModelConfig(path=value)


class Flux2ImageTrainingModule(DiffusionTrainingModule):  # noqa: F405
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
    ):
        super().__init__()
        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        tokenizer_config = _model_config_from_path_or_id(
            tokenizer_path,
            default_value=ModelConfig(
                model_id="black-forest-labs/FLUX.2-dev",
                origin_file_pattern="tokenizer/",
            ),
        )
        self.pipe = Flux2ImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )
        self.pipe = self.split_pipeline_units(
            task, self.pipe, trainable_models, lora_base_model,
        )
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(  # noqa: F405
                pipe, **inputs_shared, **inputs_posi,
            ),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(  # noqa: F405
                pipe, **inputs_shared, **inputs_posi,
            ),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(  # noqa: F405
                pipe, **inputs_shared, **inputs_posi,
            ),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(  # noqa: F405
                pipe, **inputs_shared, **inputs_posi,
            ),
        }

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            "embedded_guidance": 1.0,
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return self.task_to_loss[self.task](self.pipe, *inputs)


def flux2_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FLUX.2 SMBE image-edit training.")
    parser = add_general_config(parser)  # noqa: F405
    parser = add_image_size_config(parser)  # noqa: F405
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    parser.add_argument("--random_circular_crop", default=False, action="store_true")
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--augment_tileable", default=False, action="store_true")
    parser.add_argument("--augment_prob_original", type=float, default=0.5)
    return parser


def build_dataset(args) -> torch.utils.data.Dataset:
    if args.random_circular_crop:
        if args.crop_size is None:
            raise SystemExit("--random_circular_crop requires --crop_size.")
        base = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=make_load_only_operator(args.dataset_base_path),
        )
        dataset: torch.utils.data.Dataset = RandomCircularCropDataset(
            base,
            crop_size=args.crop_size,
            image_keys=("image", "edit_image"),
            division_factor=16,
        )
    else:
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_image_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
            ),
        )

    if args.augment_tileable:
        dataset = _wrap_with_tileable_augmentation(dataset, args)
    return dataset


def _wrap_with_tileable_augmentation(
    base_dataset: torch.utils.data.Dataset,
    args,
) -> torch.utils.data.Dataset:
    config_path = os.path.join(args.dataset_base_path, "dataset_config.json")
    normal_panels = {}
    if os.path.exists(config_path):
        normal_panels = detect_normal_panels_from_config(config_path)
        print(
            "[augment] tileable augmentation enabled "
            f"normal_panels={normal_panels} "
            f"prob_original={args.augment_prob_original}"
        )
    else:
        print(
            "[augment] WARN: dataset_config.json not found at "
            f"{config_path}; all panels use spatial transforms only."
        )
    return TileableAugmentationDataset(
        base_dataset,
        normal_panels=normal_panels,
        image_keys=("image", "edit_image"),
        prob_original=args.augment_prob_original,
    )


if __name__ == "__main__":
    parser = flux2_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters,
            )
        ],
    )
    dataset = build_dataset(args)
    model = Flux2ImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
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
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
    )
    model_logger = ModelLogger(  # noqa: F405
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,  # noqa: F405
        "direct_distill:data_process": launch_data_process_task,  # noqa: F405
        "sft": launch_training_task,  # noqa: F405
        "sft:train": launch_training_task,  # noqa: F405
        "direct_distill": launch_training_task,  # noqa: F405
        "direct_distill:train": launch_training_task,  # noqa: F405
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
