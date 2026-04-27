# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.config.stage_config import StageType
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.stage_input_processors.qwen_image import (
    denoise_to_decode,
    encode_to_denoise,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_submodule_stage_type_and_model_stage_survive_config_filtering():
    assert StageType.SUBMODULE.value == "submodule"

    cfg = OmniDiffusionConfig.from_kwargs(
        model="Qwen/Qwen-Image",
        model_class_name="QwenImagePipeline",
        model_stage="encode",
        ignored_unknown_field=True,
    )

    assert cfg.model_stage == "encode"
    assert not hasattr(cfg, "ignored_unknown_field")


def test_qwen_image_diffusion_and_denoise_model_stages_are_distinct():
    all_components = {"scheduler", "text_encoder", "tokenizer", "vae", "transformer"}

    assert QwenImagePipeline._normalize_model_stage(None) == "diffusion"
    assert QwenImagePipeline._STAGE_COMPONENTS["diffusion"] == all_components
    assert QwenImagePipeline._STAGE_COMPONENTS["denoise"] == {"scheduler", "transformer"}


def test_qwen_image_denoise_dummy_run_request_uses_stage_payload():
    od_config = SimpleNamespace(
        model="/path/that/does/not/exist",
        model_stage="denoise",
        dtype=torch.bfloat16,
        tf_model_config={"in_channels": 64, "joint_attention_dim": 3584},
    )

    request = QwenImagePipeline.build_dummy_run_request(
        od_config,
        height=1024,
        width=1024,
        num_inference_steps=1,
    )

    assert request is not None
    assert request.request_ids == ["dummy_req_id"]
    info = request.prompts[0]["additional_information"]
    assert info["context"].shape == (1, 1, 3584)
    assert info["context_mask"].shape == (1, 1)
    assert info["latents"].shape == (1, 4096, 64)
    assert info["latent_shape"] == [1, 1, 16, 128, 128]
    assert info["img_shapes"] == [[(1, 64, 64)]]
    assert info["num_inference_steps"] == 1
    assert request.sampling_params.num_inference_steps == 1


def test_qwen_image_diffusion_dummy_run_uses_default_engine_prompt():
    od_config = SimpleNamespace(model_stage="diffusion")

    assert (
        QwenImagePipeline.build_dummy_run_request(
            od_config,
            height=1024,
            width=1024,
            num_inference_steps=1,
        )
        is None
    )


def test_qwen_image_prepare_encode_dispatches_by_model_stage():
    pipeline = object.__new__(QwenImagePipeline)
    state = object()
    calls = []

    def _prepare_denoise_stage_state(arg):
        calls.append("denoise")
        return arg

    def _prepare_full_diffusion_state(arg, **kwargs):
        calls.append(("diffusion", kwargs))
        return arg

    pipeline._prepare_denoise_stage_state = _prepare_denoise_stage_state
    pipeline._prepare_full_diffusion_state = _prepare_full_diffusion_state

    pipeline.stage = "denoise"
    assert QwenImagePipeline.prepare_encode(pipeline, state) is state
    assert calls == ["denoise"]

    pipeline.stage = "diffusion"
    assert QwenImagePipeline.prepare_encode(pipeline, state, attention_kwargs={"x": 1}) is state
    assert calls[-1] == ("diffusion", {"attention_kwargs": {"x": 1}})


def test_qwen_image_intermediate_output_is_denoise_only():
    pipeline = object.__new__(QwenImagePipeline)

    pipeline.stage = "denoise"
    assert pipeline.produces_intermediate_stage_output is True

    pipeline.stage = "diffusion"
    assert pipeline.produces_intermediate_stage_output is False


def test_qwen_image_decode_postprocess_is_cached(monkeypatch):
    import vllm_omni.diffusion.registry as registry

    calls = 0

    def _fake_post_process_loader(_od_config):
        nonlocal calls
        calls += 1
        return lambda image: image

    monkeypatch.setattr(registry, "get_diffusion_post_process_func", _fake_post_process_loader)
    pipeline = object.__new__(QwenImagePipeline)
    pipeline.od_config = SimpleNamespace(model_class_name="QwenImagePipeline")
    pipeline._decode_post_process_initialized = False
    pipeline._decode_post_process_func = None

    assert pipeline._get_decode_post_process_func() is pipeline._get_decode_post_process_func()
    assert calls == 1


def test_qwen_image_encode_to_denoise_payload():
    latents = torch.zeros((1, 4096, 64), dtype=torch.bfloat16)
    stage_list = [
        SimpleNamespace(
            engine_outputs=[
                SimpleNamespace(
                    multimodal_output={
                        "context": torch.ones((1, 4, 8), dtype=torch.bfloat16),
                        "context_mask": torch.ones((1, 4), dtype=torch.long),
                        "latents": latents,
                        "latent_shape": [1, 1, 16, 64, 64],
                        "timesteps": torch.arange(2),
                        "height": 1024,
                        "width": 1024,
                        "img_shapes": [[(1, 64, 64)]],
                    }
                )
            ]
        )
    ]

    prompts = encode_to_denoise(stage_list, [0])

    assert len(prompts) == 1
    info = prompts[0]["additional_information"]
    assert info["context"].shape == (1, 4, 8)
    assert info["latents"] is latents
    assert info["height"] == 1024
    assert info["width"] == 1024


def test_qwen_image_denoise_to_decode_payload_requires_latents():
    stage_list = [
        SimpleNamespace(
            engine_outputs=[
                SimpleNamespace(
                    multimodal_output={
                        "height": 1024,
                        "width": 1024,
                    }
                )
            ]
        )
    ]

    with pytest.raises(RuntimeError, match="missing required key 'latents'"):
        denoise_to_decode(stage_list, [0])


def test_diffusion_engine_preserves_intermediate_multimodal_output():
    engine = object.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(model_stage="denoise")
    request = OmniDiffusionRequest(
        prompts=[{"prompt": "a cup of coffee"}],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_ids=["req-1"],
    )
    output = DiffusionOutput(
        output=None,
        multimodal_output={
            "latents": torch.ones((1, 4, 8), dtype=torch.float32),
            "height": 1024,
            "width": 1024,
        },
    )

    request_outputs = engine._build_omni_request_outputs(
        request,
        output,
        start_time=0.0,
        preprocess_time=0.0,
        exec_total_time=0.0,
    )

    assert len(request_outputs) == 1
    assert request_outputs[0].request_id == "req-1"
    assert request_outputs[0].final_output_type == "stage_denoise"
    assert torch.equal(request_outputs[0].multimodal_output["latents"], output.multimodal_output["latents"])
