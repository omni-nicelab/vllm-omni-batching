# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from vllm_omni.config.stage_config import StageType
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput, NewRequestData
from vllm_omni.diffusion.stage_submodule_proc import StageSubModuleProc
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.entrypoints.openai.stage_params import (
    build_stage_sampling_params_list,
    resolve_stage_sampling_params,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.stage_input_processors.qwen_image import (
    denoise_to_decode,
    encode_to_denoise,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@contextmanager
def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    yield


class _StepContractPipeline:
    supports_step_execution = True
    stage = "denoise"
    produces_intermediate_stage_output = True
    interrupt = False
    device = torch.device("cpu")

    def __init__(self, num_steps: int):
        self.num_steps = num_steps
        self.post_decode_calls = 0

    def prepare_encode(self, state):
        state.timesteps = torch.arange(self.num_steps)
        state.latents = torch.zeros((1, 1))
        return state

    def denoise_step(self, state):
        return torch.ones_like(state.latents)

    def step_scheduler(self, state, noise_pred):
        state.latents = state.latents + noise_pred
        state.step_index += 1

    def post_decode(self, state):
        del state
        self.post_decode_calls += 1
        return DiffusionOutput(output="decoded")

    def post_intermediate_output(self, state):
        return DiffusionOutput(
            output=None,
            multimodal_output={
                "latents": state.latents.clone(),
                "step_index": state.step_index,
            },
        )


class _OptimizedStagePipeline(_StepContractPipeline):
    def __init__(self, num_steps: int):
        super().__init__(num_steps)
        self.execute_stage_model_calls = 0

    def execute_stage_model(self, req):
        del req
        self.execute_stage_model_calls += 1
        return DiffusionOutput(
            output=None,
            multimodal_output={
                "latents": torch.full((1, 1), 7.0),
                "step_index": 7,
            },
        )


def _make_runner() -> DiffusionModelRunner:
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.cache_backend = None
    runner.offload_backend = None
    runner.od_config = SimpleNamespace(
        cache_backend="none",
        enable_cache_dit_summary=False,
        parallel_config=SimpleNamespace(use_hsdp=False),
    )
    runner.kv_transfer_manager = SimpleNamespace(
        receive_multi_kv_cache_distributed=lambda req, cfg_kv_collect_func=None, target_device=None: None,
    )
    return runner


def _make_diffusion_request(num_steps: int = 1) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[{"prompt": "a prompt", "additional_information": {}}],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=num_steps),
        request_ids=["req-1"],
    )


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
    assert QwenImagePipeline._STAGE_COMPONENTS["encode"] == {"scheduler", "text_encoder", "tokenizer"}
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
    engine.od_config = SimpleNamespace(
        model_stage="denoise",
        enable_cpu_offload=False,
        model_class_name="QwenImagePipeline",
    )
    engine.pre_process_func = None
    engine.post_process_func = None
    engine._post_process_accepts_sampling_params = False
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
    engine.add_req_and_wait_for_response = lambda req: output

    request_outputs = DiffusionEngine.step(engine, request)

    assert len(request_outputs) == 1
    assert request_outputs[0].request_id == "req-1"
    assert request_outputs[0].final_output_type == "stage_denoise"
    assert torch.equal(request_outputs[0].multimodal_output["latents"], output.multimodal_output["latents"])


def test_execute_model_runs_denoise_stage_step_contract_to_intermediate_output(monkeypatch):
    runner = _make_runner()
    runner.pipeline = _StepContractPipeline(num_steps=2)
    runner._record_peak_memory = lambda output: None
    req = _make_diffusion_request(num_steps=2)

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module, "cache_summary", lambda pipeline, details: None)

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.output is None
    assert output.multimodal_output["step_index"] == 2
    assert torch.equal(output.multimodal_output["latents"], torch.full((1, 1), 2.0))
    assert runner.pipeline.post_decode_calls == 0


def test_execute_model_uses_optimized_stage_model_hook(monkeypatch):
    runner = _make_runner()
    runner.pipeline = _OptimizedStagePipeline(num_steps=2)
    runner._record_peak_memory = lambda output: None
    req = _make_diffusion_request(num_steps=2)

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module, "cache_summary", lambda pipeline, details: None)

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.output is None
    assert output.multimodal_output["step_index"] == 7
    assert torch.equal(output.multimodal_output["latents"], torch.full((1, 1), 7.0))
    assert runner.pipeline.execute_stage_model_calls == 1
    assert runner.pipeline.post_decode_calls == 0


def test_execute_stepwise_uses_denoise_stage_intermediate_output(monkeypatch):
    runner = _make_runner()
    runner.pipeline = _StepContractPipeline(num_steps=1)
    runner.state_cache = {}
    req = _make_diffusion_request(num_steps=1)
    scheduler_output = DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=[NewRequestData(sched_req_id="req-1", req=req)],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=1,
        num_waiting_reqs=0,
    )

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

    output = DiffusionModelRunner.execute_stepwise(runner, scheduler_output)

    assert output.req_id == "req-1"
    assert output.finished is True
    assert output.result is not None
    assert output.result.output is None
    assert output.result.multimodal_output["step_index"] == 1
    assert torch.equal(output.result.multimodal_output["latents"], torch.ones((1, 1)))
    assert runner.pipeline.post_decode_calls == 0


def test_submodule_stages_use_request_diffusion_params():
    request_params = OmniDiffusionSamplingParams(height=512, width=512, seed=7)
    submodule_default = OmniDiffusionSamplingParams(height=1024, width=1024)

    resolved = resolve_stage_sampling_params(
        SimpleNamespace(stage_type="submodule"),
        1,
        [],
        diffusion_params=request_params,
    )
    assert resolved is not request_params
    assert resolved.height == 512
    assert resolved.width == 512
    assert resolved.seed == 7

    stages = [
        SimpleNamespace(stage_type="submodule"),
        SimpleNamespace(stage_type="diffusion"),
        SimpleNamespace(stage_type="submodule"),
    ]
    resolved_list = build_stage_sampling_params_list(
        stages,
        [submodule_default],
        diffusion_params=request_params,
        replace_diffusion_params=True,
    )

    assert [params.height for params in resolved_list] == [512, 512, 512]
    assert all(params is not request_params for params in resolved_list)


def test_stage_submodule_proc_uses_submodule_worker(monkeypatch):
    calls: list[tuple[str, object]] = []

    class FakeSubModuleWorker:
        def __init__(self, local_rank, rank, od_config):
            calls.append(("init", (local_rank, rank, od_config)))

        def load_model(self, load_format):
            calls.append(("load_model", load_format))

        def execute_submodule(self, request):
            calls.append(("execute_submodule", request))
            return DiffusionOutput(output=None, multimodal_output={"ok": True})

        def shutdown(self):
            calls.append(("shutdown", None))

    import vllm_omni.diffusion.stage_submodule_proc as proc_mod

    monkeypatch.setattr(proc_mod, "SubModuleWorker", FakeSubModuleWorker)
    od_config = SimpleNamespace(
        model="Qwen/Qwen-Image",
        model_class_name="QwenImagePipeline",
        model_stage="encode",
        tf_model_config=SimpleNamespace(params={"in_channels": 64}),
        diffusion_load_format="dummy",
        enrich_config=lambda: None,
        update_multimodal_support=lambda: None,
    )
    proc = StageSubModuleProc("Qwen/Qwen-Image", od_config)

    proc.initialize()
    pending = proc._submit_stage_request(
        "req-1",
        [{"prompt": "test"}],
        {"num_inference_steps": 1},
        batch_mode=False,
    )
    output = pending.future.result(timeout=1)
    proc.close()

    assert output.multimodal_output == {"ok": True}
    assert [name for name, _ in calls] == [
        "init",
        "load_model",
        "execute_submodule",
        "shutdown",
    ]
