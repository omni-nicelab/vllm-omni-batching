# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import janus
import pytest
import torch

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput, NewRequestData
from vllm_omni.diffusion.stage_submodule_proc import StageSubModuleProc
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.engine.messages import OutputMessage, ShutdownRequestMessage, StageSubmissionMessage
from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.stage_input_processors.qwen_image import (
    denoise_to_decode,
    encode_to_denoise,
)
from vllm_omni.outputs import OmniRequestOutput

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
        state.prompt_embeds = torch.zeros((1, 1, 1))
        state.prompt_embeds_mask = torch.ones((1, 1), dtype=torch.long)
        state.img_shapes = [[(1, 1, 1)]]
        state.txt_seq_lens = [1]
        return state

    def denoise_step(self, input_batch):
        return torch.ones_like(input_batch.latents)

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
        request_id="req-1",
    )


class _E2EDiffusionStageClient:
    stage_type = "diffusion"
    default_sampling_params = OmniDiffusionSamplingParams()
    requires_multimodal_data = False
    engine_input_source = [0]
    is_comprehension = False

    def __init__(
        self,
        *,
        stage_id: int,
        model_stage: str,
        final_output: bool = False,
        final_output_type: str = "stage",
        custom_process_input_func=None,
    ) -> None:
        self.stage_id = stage_id
        self.replica_id = 0
        self.model_stage = model_stage
        self.final_output = final_output
        self.final_output_type = final_output_type
        self.custom_process_input_func = custom_process_input_func
        self.add_request_calls: list[tuple[str, Any, OmniDiffusionSamplingParams, dict[str, Any] | None]] = []
        self.add_batch_request_calls: list[
            tuple[str, list[Any], OmniDiffusionSamplingParams, dict[str, Any] | None]
        ] = []
        self.outputs: queue.Queue[OmniRequestOutput] = queue.Queue()
        self.shutdown_calls = 0

    async def add_request_async(
        self,
        request_id: str,
        prompt,
        sampling_params: OmniDiffusionSamplingParams,
        kv_sender_info=None,
    ) -> None:
        self.add_request_calls.append((request_id, prompt, sampling_params, kv_sender_info))

    async def add_batch_request_async(
        self,
        request_id: str,
        prompts: list[Any],
        sampling_params: OmniDiffusionSamplingParams,
        kv_sender_info=None,
    ) -> None:
        self.add_batch_request_calls.append((request_id, prompts, sampling_params, kv_sender_info))

    def get_diffusion_output_nowait(self) -> OmniRequestOutput | None:
        try:
            return self.outputs.get_nowait()
        except queue.Empty:
            return None

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        del request_ids

    async def collective_rpc_async(self, **kwargs) -> dict[str, Any]:
        del kwargs
        return {"supported": False}

    def check_health(self) -> None:
        return None

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    def push_output(self, output: OmniRequestOutput) -> None:
        self.outputs.put_nowait(output)


@dataclass
class _E2EOrchestratorHarness:
    orchestrator: Orchestrator
    request_sync_q: Any
    output_sync_q: Any
    queues: tuple[janus.Queue, ...]
    thread: threading.Thread
    result_future: concurrent.futures.Future[None]


def _build_e2e_orchestrator_harness(stage_pools: list[StagePool]) -> _E2EOrchestratorHarness:
    ready_future: concurrent.futures.Future[tuple[Orchestrator, janus.Queue, janus.Queue, janus.Queue]] = (
        concurrent.futures.Future()
    )
    result_future: concurrent.futures.Future[None] = concurrent.futures.Future()

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _run() -> None:
            request_queue = janus.Queue()
            output_queue = janus.Queue()
            rpc_queue = janus.Queue()
            orchestrator = Orchestrator(
                request_async_queue=request_queue.async_q,
                output_async_queue=output_queue.async_q,
                rpc_async_queue=rpc_queue.async_q,
                stage_pools=stage_pools,
            )
            ready_future.set_result((orchestrator, request_queue, output_queue, rpc_queue))
            await orchestrator.run()

        try:
            loop.run_until_complete(_run())
            result_future.set_result(None)
        except Exception as exc:
            if not ready_future.done():
                ready_future.set_exception(exc)
            result_future.set_exception(exc)
        finally:
            try:
                pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    thread = threading.Thread(target=_runner, daemon=True, name="test-diffusion-submodule-orchestrator")
    thread.start()

    orchestrator, request_queue, output_queue, rpc_queue = ready_future.result(timeout=5)
    return _E2EOrchestratorHarness(
        orchestrator=orchestrator,
        request_sync_q=request_queue.sync_q,
        output_sync_q=output_queue.sync_q,
        queues=(request_queue, output_queue, rpc_queue),
        thread=thread,
        result_future=result_future,
    )


def _shutdown_e2e_orchestrator_harness(harness: _E2EOrchestratorHarness) -> None:
    if harness.thread.is_alive():
        harness.request_sync_q.put_nowait(ShutdownRequestMessage())
        harness.thread.join(timeout=5)
    if harness.thread.is_alive():
        raise AssertionError("Timed out waiting for orchestrator thread shutdown")
    try:
        harness.result_future.result(timeout=0)
    finally:
        for item_queue in harness.queues:
            item_queue.close()


def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for predicate")
        time.sleep(0.01)


def _next_output_message(output_queue, *, timeout: float = 2.0) -> OutputMessage:
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for orchestrator output")
        try:
            msg = output_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.01)
            continue
        if isinstance(msg, OutputMessage):
            return msg


def test_submodule_model_stage_survives_config_filtering():
    cfg = OmniDiffusionConfig.from_kwargs(
        model="Qwen/Qwen-Image",
        model_class_name="QwenImagePipeline",
        model_stage="encode",
        ignored_unknown_field=True,
    )

    assert cfg.model_stage == "encode"
    assert not hasattr(cfg, "ignored_unknown_field")


def test_qwen_image_stage_components_are_model_specific():
    all_components = {"scheduler", "text_encoder", "tokenizer", "vae", "transformer"}

    assert QwenImagePipeline._STAGE_COMPONENTS["diffusion"] == all_components
    assert QwenImagePipeline._STAGE_COMPONENTS["encode"] == {"scheduler", "text_encoder", "tokenizer"}
    assert QwenImagePipeline._STAGE_COMPONENTS["denoise"] == {"scheduler", "transformer"}


def test_qwen_image_denoise_dummy_run_request_uses_intermediate_payload():
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
    assert request.request_id == "dummy_req_id"
    info = request.prompts[0]["additional_information"]
    assert info["context"].shape == (1, 1, 3584)
    assert info["context_mask"].shape == (1, 1)
    assert info["latents"].shape == (1, 4096, 64)
    assert info["img_shapes"] == [[(1, 64, 64)]]
    assert info["num_inference_steps"] == 1
    assert request.sampling_params.num_inference_steps == 1


def test_qwen_image_diffusion_dummy_run_uses_safe_engine_prompt():
    od_config = SimpleNamespace(model_stage="diffusion")

    request = QwenImagePipeline.build_dummy_run_request(
        od_config,
        height=1024,
        width=1024,
        num_inference_steps=1,
    )

    assert request is not None
    assert request.prompts == [{"prompt": "dummy run"}]
    assert request.request_id == "dummy_req_id"
    assert request.sampling_params.num_inference_steps == 2
    assert request.sampling_params.true_cfg_scale == 1.0


def test_qwen_image_encode_to_denoise_payload():
    latents = torch.zeros((1, 4096, 64), dtype=torch.bfloat16)
    source_outputs = [
        SimpleNamespace(
            multimodal_output={
                "context": torch.ones((1, 4, 8), dtype=torch.bfloat16),
                "context_mask": torch.ones((1, 4), dtype=torch.long),
                "latents": latents,
                "timesteps": torch.arange(2),
                "height": 1024,
                "width": 1024,
                "img_shapes": [[(1, 64, 64)]],
            }
        )
    ]

    prompts = encode_to_denoise(source_outputs)

    assert len(prompts) == 1
    info = prompts[0]["additional_information"]
    assert info["context"].shape == (1, 4, 8)
    assert info["latents"] is latents
    assert info["height"] == 1024
    assert info["width"] == 1024


def test_qwen_image_denoise_to_decode_payload_requires_latents():
    source_outputs = [
        SimpleNamespace(
            multimodal_output={
                "height": 1024,
                "width": 1024,
            }
        )
    ]

    with pytest.raises(RuntimeError, match="missing required key 'latents'"):
        denoise_to_decode(source_outputs)


def test_diffusion_submodule_orchestrator_routes_encode_denoise_decode_e2e():
    encode_payload = {"context": torch.ones((1, 2, 3)), "latents": torch.zeros((1, 4))}
    denoise_payload = {"latents": torch.full((1, 4), 5.0), "height": 64, "width": 64}
    final_payload = {"image": "decoded", "height": 64, "width": 64}

    def encode_output_to_denoise_prompt(source_outputs, prompt, requires_multimodal_data):
        assert prompt == {"prompt": "draw a cat"}
        assert requires_multimodal_data is False
        multimodal_output = source_outputs[0].multimodal_output
        assert torch.equal(multimodal_output["context"], encode_payload["context"])
        assert torch.equal(multimodal_output["latents"], encode_payload["latents"])
        return [{"prompt": "denoise", "additional_information": multimodal_output}]

    def denoise_output_to_decode_prompt(source_outputs, prompt, requires_multimodal_data):
        assert prompt == {"prompt": "draw a cat"}
        assert requires_multimodal_data is False
        assert torch.equal(source_outputs[0].multimodal_output["latents"], denoise_payload["latents"])
        return [{"prompt": "decode", "additional_information": source_outputs[0].multimodal_output}]

    encode_stage = _E2EDiffusionStageClient(stage_id=0, model_stage="encode")
    denoise_stage = _E2EDiffusionStageClient(
        stage_id=1,
        model_stage="denoise",
        custom_process_input_func=encode_output_to_denoise_prompt,
    )
    decode_stage = _E2EDiffusionStageClient(
        stage_id=2,
        model_stage="decode",
        final_output=True,
        final_output_type="image",
        custom_process_input_func=denoise_output_to_decode_prompt,
    )

    harness = _build_e2e_orchestrator_harness(
        [
            StagePool(0, encode_stage),
            StagePool(1, denoise_stage),
            StagePool(2, decode_stage),
        ],
    )

    try:
        params = [OmniDiffusionSamplingParams(num_inference_steps=1) for _ in range(3)]
        harness.request_sync_q.put_nowait(
            StageSubmissionMessage(
                type="add_request",
                request_id="req-e2e",
                prompt={"prompt": "draw a cat"},
                original_prompt={"prompt": "draw a cat"},
                output_prompt_text=None,
                sampling_params_list=params,
                final_stage_id=2,
                preprocess_ms=0.0,
                enqueue_ts=time.perf_counter(),
            )
        )

        _wait_until(lambda: len(encode_stage.add_request_calls) == 1)
        assert encode_stage.add_request_calls[0][1] == {"prompt": "draw a cat"}
        encode_stage.push_output(
            OmniRequestOutput.from_diffusion(
                request_id="req-e2e",
                images=[],
                final_output_type="stage_encode",
                multimodal_output=encode_payload,
            )
        )

        _wait_until(lambda: len(denoise_stage.add_batch_request_calls) == 1)
        denoise_prompt = denoise_stage.add_batch_request_calls[0][1][0]
        assert denoise_prompt["prompt"] == "denoise"
        assert torch.equal(denoise_prompt["additional_information"]["context"], encode_payload["context"])
        assert torch.equal(denoise_prompt["additional_information"]["latents"], encode_payload["latents"])
        denoise_stage.push_output(
            OmniRequestOutput.from_diffusion(
                request_id="req-e2e",
                images=[],
                final_output_type="stage_denoise",
                multimodal_output=denoise_payload,
            )
        )

        _wait_until(lambda: len(decode_stage.add_batch_request_calls) == 1)
        decode_prompt = decode_stage.add_batch_request_calls[0][1][0]
        assert decode_prompt["prompt"] == "decode"
        assert torch.equal(decode_prompt["additional_information"]["latents"], denoise_payload["latents"])
        decode_stage.push_output(
            OmniRequestOutput.from_diffusion(
                request_id="req-e2e",
                images=[],
                final_output_type="image",
                multimodal_output=final_payload,
            )
        )

        output_msg = _next_output_message(harness.output_sync_q)

        assert output_msg.request_id == "req-e2e"
        assert output_msg.stage_id == 2
        assert output_msg.finished is True
        assert output_msg.engine_outputs.final_output_type == "image"
        assert output_msg.engine_outputs.multimodal_output == final_payload
        _wait_until(lambda: "req-e2e" not in harness.orchestrator.request_states)
    finally:
        _shutdown_e2e_orchestrator_harness(harness)


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
        request_id="req-1",
    )
    output = DiffusionOutput(
        output=None,
        multimodal_output={
            "latents": torch.ones((1, 4, 8), dtype=torch.float32),
            "height": 1024,
            "width": 1024,
        },
    )

    async def _async_response(req):
        del req
        return output

    async def _noop_start_loop():
        return None

    engine.async_add_req_and_wait_for_response = _async_response
    engine._check_and_start_background_loop = _noop_start_loop

    request_outputs = asyncio.run(DiffusionEngine.step(engine, request))

    assert len(request_outputs) == 1
    assert request_outputs[0].request_id == "req-1"
    assert request_outputs[0].final_output_type == "stage_denoise"
    assert torch.equal(request_outputs[0].multimodal_output["latents"], output.multimodal_output["latents"])


def test_execute_model_uses_optimized_stage_model_hook(monkeypatch):
    runner = _make_runner()
    runner.pipeline = _OptimizedStagePipeline(num_steps=2)
    runner._record_peak_memory = lambda output: None
    req = _make_diffusion_request(num_steps=2)

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module, "cache_summary", lambda pipeline, details: None)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "reset_peak_memory_stats", lambda: None)

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
        scheduled_new_reqs=[NewRequestData(request_id="req-1", req=req)],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=1,
        num_waiting_reqs=0,
    )

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

    batch_output = DiffusionModelRunner.execute_stepwise(runner, scheduler_output)
    output = batch_output["req-1"]

    assert output is not None
    assert output.request_id == "req-1"
    assert output.finished is True
    assert output.result is not None
    assert output.result.output is None
    assert output.result.multimodal_output["step_index"] == 1
    assert torch.equal(output.result.multimodal_output["latents"], torch.ones((1, 1)))
    assert runner.pipeline.post_decode_calls == 0


def test_diffusion_submodule_proc_uses_submodule_worker(monkeypatch):
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
    output = asyncio.run(
        proc._process_request(
            "req-1",
            {"prompt": "test"},
            {"num_inference_steps": 1},
        )
    )
    proc.close()

    assert output.multimodal_output == {"ok": True}
    assert [name for name, _ in calls] == [
        "init",
        "load_model",
        "execute_submodule",
        "shutdown",
    ]
