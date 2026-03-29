# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch
from PIL import Image

from vllm_omni.diffusion.core.diffusion_core import DiffusionCore
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, StepExecutionDiffusionEngine
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.request import DiffusionRequestState, OmniDiffusionRequest
from vllm_omni.diffusion.worker.gpu_diffusion_model_runner import GPUDiffusionModelRunner
from vllm_omni.diffusion.worker.step_batch import StepBatch, StepOutput, StepRunnerOutput, StepSchedulerOutput


def _normalize_timestep(timestep: torch.Tensor | float | int | None) -> int | None:
    if timestep is None:
        return None
    if isinstance(timestep, torch.Tensor):
        return int(timestep.item())
    return int(timestep)


def _make_test_config(executor_class: type[DiffusionExecutor]) -> OmniDiffusionConfig:
    od_config = OmniDiffusionConfig(
        model="dummy-model",
        step_execution=True,
        num_gpus=1,
    )
    od_config.distributed_executor_backend = executor_class
    od_config.max_batch_size = 2
    od_config.max_model_len = 4096
    od_config.max_num_batched_dit_tokens = 8192
    return od_config


def _make_request(request_id: str, num_inference_steps: int = 2) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        request_id=request_id,
        prompt=f"prompt-{request_id}",
        height=512,
        width=512,
        num_inference_steps=num_inference_steps,
    )


class RecordingStepExecutor(DiffusionExecutor):
    batch_snapshots: list[list[tuple[str, int, int | None]]] = []

    @classmethod
    def reset(cls) -> None:
        cls.batch_snapshots = []

    def _init_executor(self) -> None:
        pass

    def add_req(self, requests: list[OmniDiffusionRequest]):
        raise NotImplementedError

    def execute_step(
        self,
        scheduler_output: StepSchedulerOutput,
        timeout: float | None = None,
    ) -> StepRunnerOutput:
        snapshot = [
            (state.req_id, state.step_index, _normalize_timestep(state.timestep))
            for state in scheduler_output.req_states
        ]
        self.__class__.batch_snapshots.append(snapshot)

        step_outputs = [
            StepOutput(
                req_id=state.req_id,
                step_index=state.step_index,
                timestep=torch.tensor(990 - state.step_index * 10 - idx),
                is_complete=scheduler_output.step_id >= 1,
            )
            for idx, state in enumerate(scheduler_output.req_states)
        ]

        decoded = {}
        if scheduler_output.step_id >= 1:
            decoded = {
                state.req_id: Image.new("RGB", (8, 8), color="blue")
                for state in scheduler_output.req_states
            }

        return StepRunnerOutput(
            step_id=scheduler_output.step_id,
            step_outputs=step_outputs,
            decoded=decoded,
        )

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ):
        return None

    def check_health(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class BlockingBatchingExecutor(DiffusionExecutor):
    batch_sizes: list[int] = []
    batch_snapshots: list[list[tuple[str, int, int | None]]] = []
    first_single_non_dummy_step_seen = threading.Event()
    allow_progress_after_single_step = threading.Event()

    @classmethod
    def reset(cls) -> None:
        cls.batch_sizes = []
        cls.batch_snapshots = []
        cls.first_single_non_dummy_step_seen = threading.Event()
        cls.allow_progress_after_single_step = threading.Event()

    def _init_executor(self) -> None:
        pass

    def add_req(self, requests: list[OmniDiffusionRequest]):
        raise NotImplementedError

    def execute_step(
        self,
        scheduler_output: StepSchedulerOutput,
        timeout: float | None = None,
    ) -> StepRunnerOutput:
        req_ids = [state.req_id for state in scheduler_output.req_states]
        snapshot = [
            (state.req_id, state.step_index, _normalize_timestep(state.timestep))
            for state in scheduler_output.req_states
        ]
        self.__class__.batch_sizes.append(len(req_ids))
        self.__class__.batch_snapshots.append(snapshot)

        step_outputs = [
            StepOutput(
                req_id=state.req_id,
                step_index=state.step_index,
                timestep=torch.tensor(900 - state.step_index * 10 - idx),
                is_complete=len(req_ids) > 1,
            )
            for idx, state in enumerate(scheduler_output.req_states)
        ]

        if req_ids == ["__dummy_warmup__"]:
            return StepRunnerOutput(
                step_id=scheduler_output.step_id,
                step_outputs=step_outputs,
                decoded={"__dummy_warmup__": Image.new("RGB", (8, 8), color="black")},
            )

        if len(req_ids) == 1:
            self.__class__.first_single_non_dummy_step_seen.set()
            assert self.__class__.allow_progress_after_single_step.wait(timeout=5)
            return StepRunnerOutput(
                step_id=scheduler_output.step_id,
                step_outputs=step_outputs,
                decoded={},
            )

        return StepRunnerOutput(
            step_id=scheduler_output.step_id,
            step_outputs=step_outputs,
            decoded={req_id: Image.new("RGB", (8, 8), color="green") for req_id in req_ids},
        )

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ):
        return None

    def check_health(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


def test_make_engine_returns_step_execution_engine(monkeypatch):
    monkeypatch.setattr("vllm_omni.diffusion.data.OmniDiffusionConfig.settle_port", lambda self, port, *_args, **_kwargs: port)
    monkeypatch.setattr("vllm_omni.diffusion.diffusion_engine.get_diffusion_pre_process_func", lambda *_: None)
    monkeypatch.setattr("vllm_omni.diffusion.diffusion_engine.get_diffusion_post_process_func", lambda *_: None)
    monkeypatch.setattr("vllm_omni.diffusion.diffusion_engine.supports_image_input", lambda *_: False)

    BlockingBatchingExecutor.reset()
    engine = DiffusionEngine.make_engine(_make_test_config(BlockingBatchingExecutor))
    try:
        assert isinstance(engine, StepExecutionDiffusionEngine)
    finally:
        engine.close()


def test_diffusion_core_batches_staggered_requests_with_different_timestep_state(monkeypatch):
    monkeypatch.setattr("vllm_omni.diffusion.data.OmniDiffusionConfig.settle_port", lambda self, port, *_args, **_kwargs: port)
    monkeypatch.setattr("vllm_omni.diffusion.core.diffusion_core.get_diffusion_pre_process_func", lambda *_: None)
    monkeypatch.setattr("vllm_omni.diffusion.core.diffusion_core.get_diffusion_post_process_func", lambda *_: None)

    RecordingStepExecutor.reset()
    core = DiffusionCore(_make_test_config(RecordingStepExecutor), RecordingStepExecutor)
    try:
        core.add_request(_make_request("req-1"))
        outputs, executed = core.step()

        assert executed is True
        assert outputs == []

        req1_state = core.step_scheduler.get_request_state("req-1")
        assert req1_state is not None
        assert req1_state.step_index == 1
        assert _normalize_timestep(req1_state.timestep) == 990

        core.add_request(_make_request("req-2"))
        req2_state = core.step_scheduler.get_request_state("req-2")
        assert req2_state is not None
        req2_state.timestep = torch.tensor(1000)

        outputs, executed = core.step()

        assert executed is True
        assert {output.request_id for output in outputs} == {"req-1", "req-2"}
        assert all(len(output.images or []) == 1 for output in outputs)
        assert RecordingStepExecutor.batch_snapshots == [
            [("req-1", 0, None)],
            [("req-1", 1, 990), ("req-2", 0, 1000)],
        ]
    finally:
        core.shutdown()


def test_runner_decodes_request_completed_in_same_step(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.gpu_diffusion_model_runner.set_forward_context",
        lambda **_: nullcontext(),
    )

    class DirectBatchBuilder:
        def build(self, states: list[DiffusionRequestState]) -> StepBatch:
            return StepBatch.from_requests(states)

    class FakePipeline:
        def __init__(self) -> None:
            self.vae = torch.nn.Linear(1, 1)

        def denoise_step(
            self,
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            latents,
            img_shapes,
            txt_seq_lens,
            negative_txt_seq_lens,
            timesteps,
            do_true_cfg,
            guidance,
            true_cfg_scale,
        ):
            return torch.zeros_like(latents)

        def scheduler_step(
            self,
            latents,
            noise_pred,
            timestep,
            do_true_cfg,
            *,
            step_index: int,
            req_id: str | None = None,
            is_complete: bool = False,
            scheduler_override=None,
        ):
            return latents + 1, StepOutput(
                req_id=req_id or "",
                step_index=step_index,
                timestep=timestep,
                latents=latents,
                noise_pred=noise_pred,
                is_complete=is_complete,
            )

        def post_decode(self, latents, height: int, width: int, output_type: str = "pil"):
            return torch.tensor([height, width], dtype=torch.int64)

    runner = GPUDiffusionModelRunner(
        vllm_config=SimpleNamespace(),
        od_config=SimpleNamespace(),
        device=torch.device("cpu"),
        batch_builder=DirectBatchBuilder(),
    )
    runner.pipeline = FakePipeline()

    req = _make_request("req-1", num_inference_steps=1)
    state = DiffusionRequestState(
        req_id="req-1",
        req=req,
        prompt_embeds=torch.randn(1, 4, 8),
        prompt_embeds_mask=torch.tensor([[True, True, True, True]]),
        latents=torch.randn(1, 16, 4, 4),
        timesteps=torch.tensor([999]),
        do_true_cfg=False,
        guidance=None,
        true_cfg_scale=1.0,
        scheduler=object(),
    )
    runner._request_state_cache[state.req_id] = state

    output = runner.execute_step(
        StepSchedulerOutput(
            step_id=0,
            req_states=[state],
            num_running_reqs=1,
            num_waiting_reqs=0,
        )
    )

    assert [step.req_id for step in output.step_outputs] == ["req-1"]
    assert output.step_outputs[0].is_complete is True
    assert output.decoded["req-1"].tolist() == [req.height, req.width]
    assert state.step_index == 1
    assert state.req_id not in runner._request_state_cache


def test_step_execution_engine_batches_concurrent_requests(monkeypatch):
    monkeypatch.setattr("vllm_omni.diffusion.data.OmniDiffusionConfig.settle_port", lambda self, port, *_args, **_kwargs: port)
    monkeypatch.setattr("vllm_omni.diffusion.diffusion_engine.get_diffusion_pre_process_func", lambda *_: None)
    monkeypatch.setattr("vllm_omni.diffusion.diffusion_engine.get_diffusion_post_process_func", lambda *_: None)
    monkeypatch.setattr("vllm_omni.diffusion.diffusion_engine.supports_image_input", lambda *_: False)

    BlockingBatchingExecutor.reset()
    engine = StepExecutionDiffusionEngine(_make_test_config(BlockingBatchingExecutor))
    try:
        BlockingBatchingExecutor.reset()

        req1 = _make_request("req-1")
        req2 = _make_request("req-2")

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut1 = pool.submit(engine.step, [req1])
            assert BlockingBatchingExecutor.first_single_non_dummy_step_seen.wait(timeout=5)

            fut2 = pool.submit(engine.step, [req2])

            deadline = time.time() + 5
            while time.time() < deadline:
                with engine._pending_lock:
                    if "req-2" in engine._pending_outputs:
                        break
                time.sleep(0.01)
            else:
                raise AssertionError("Timed out waiting for req-2 to enter the pending map")

            BlockingBatchingExecutor.allow_progress_after_single_step.set()

            out1 = fut1.result(timeout=5)
            out2 = fut2.result(timeout=5)

        assert out1.request_id == "req-1"
        assert out2.request_id == "req-2"
        assert out1.num_images == 1
        assert out2.num_images == 1
        assert 2 in BlockingBatchingExecutor.batch_sizes
        assert [("req-1", 1, 900), ("req-2", 0, None)] in BlockingBatchingExecutor.batch_snapshots
    finally:
        engine.close()
