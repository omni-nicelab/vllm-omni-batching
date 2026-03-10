# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import RequestScheduler, StepScheduler
from vllm_omni.diffusion.sched.interface import (
    DiffusionRequestState as SchedulerRequestState,
)
from vllm_omni.diffusion.sched.interface import (
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    SchedulerInterface,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.diffusion.worker.utils import RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@contextmanager
def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    yield


class _StepPipeline:
    supports_step_execution = True

    def __init__(self):
        self.prepare_calls = 0
        self.denoise_calls = 0
        self.scheduler_calls = 0
        self.decode_calls = 0

    def prepare_encode(self, state, **kwargs):
        del kwargs
        self.prepare_calls += 1
        state.timesteps = [torch.tensor(10), torch.tensor(5)]
        state.latents = torch.tensor([0.0])
        return state

    def denoise_step(self, state, **kwargs):
        del state, kwargs
        self.denoise_calls += 1
        return torch.tensor([1.0])

    def step_scheduler(self, state, noise_pred, **kwargs):
        del noise_pred, kwargs
        self.scheduler_calls += 1
        state.step_index += 1

    def post_decode(self, state, **kwargs):
        del kwargs
        self.decode_calls += 1
        return DiffusionOutput(output=torch.tensor([state.step_index], dtype=torch.float32))


def _make_step_request():
    return SimpleNamespace(
        prompts=["a prompt"],
        request_ids=["req-1"],
        sampling_params=SimpleNamespace(
            generator=None,
            seed=None,
            generator_device=None,
            num_inference_steps=2,
        ),
    )


def _make_runner():
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.od_config = SimpleNamespace(
        cache_backend=None,
        parallel_config=SimpleNamespace(use_hsdp=False),
    )
    runner.device = torch.device("cpu")
    runner.pipeline = _StepPipeline()
    runner.cache_backend = None
    runner.offload_backend = None
    runner.state_cache = {}
    runner.kv_transfer_manager = SimpleNamespace()
    return runner


def _make_scheduler_output(req, req_id="req-1", step_id=0, finished_req_ids=None):
    return DiffusionSchedulerOutput(
        step_id=step_id,
        req_states=[SchedulerRequestState(sched_req_id=req_id, req=req)],
        finished_req_ids=set() if finished_req_ids is None else set(finished_req_ids),
        num_running_reqs=1,
        num_waiting_reqs=0,
    )


def _make_engine_request(req_id="req-1", num_inference_steps=2):
    return OmniDiffusionRequest(
        prompts=[f"prompt-{req_id}"],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=num_inference_steps),
        request_ids=[req_id],
    )


def _make_engine(
    scheduler: SchedulerInterface,
    execute_fn,
):
    engine = object.__new__(DiffusionEngine)
    engine.scheduler = scheduler
    engine.execute_fn = execute_fn
    engine._rpc_lock = threading.RLock()
    engine.abort_queue = queue.Queue()
    return engine


class _AbortAwareScheduler(SchedulerInterface):
    def __init__(self) -> None:
        self._state = None
        self._finished_req_ids: set[str] = set()
        self._scheduled = False

    def initialize(self, od_config) -> None:
        del od_config

    def add_request(self, request: OmniDiffusionRequest) -> str:
        req_id = request.request_ids[0]
        self._state = SchedulerRequestState(sched_req_id=req_id, req=request)
        self._scheduled = False
        return req_id

    def schedule(self) -> DiffusionSchedulerOutput:
        req_states = []
        if (
            self._state is not None
            and self._state.status < DiffusionRequestStatus.FINISHED_COMPLETED
            and not self._scheduled
        ):
            self._state.status = DiffusionRequestStatus.RUNNING
            req_states = [self._state]
            self._scheduled = True
        return DiffusionSchedulerOutput(
            step_id=0,
            req_states=req_states,
            finished_req_ids=set(self._finished_req_ids),
            num_running_reqs=len(req_states),
            num_waiting_reqs=0,
        )

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: RunnerOutput) -> set[str]:
        del sched_output, output
        if self._state is None:
            return set()
        if self._state.status == DiffusionRequestStatus.FINISHED_ABORTED:
            self._finished_req_ids.add(self._state.sched_req_id)
            return {self._state.sched_req_id}
        return set()

    def abort_request(self, req_id: str) -> bool:
        if self._state is None or self._state.sched_req_id != req_id:
            return False
        self._state.status = DiffusionRequestStatus.FINISHED_ABORTED
        self._finished_req_ids.add(req_id)
        return True

    def has_requests(self) -> bool:
        return self._state is not None and self._state.status < DiffusionRequestStatus.FINISHED_COMPLETED

    def get_request_state(self, req_id: str):
        if self._state is not None and self._state.sched_req_id == req_id:
            return self._state
        return None

    def pop_request_state(self, req_id: str):
        if self._state is not None and self._state.sched_req_id == req_id:
            state, self._state = self._state, None
            return state
        return None

    def preempt_request(self, req_id: str) -> bool:
        del req_id
        return False

    def finish_request(self, req_id: str, status) -> None:
        del req_id, status

    def close(self) -> None:
        self._state = None


def test_runner_execute_stepwise_completes_request_and_clears_state(monkeypatch):
    runner = _make_runner()
    req = _make_step_request()

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

    first = DiffusionModelRunner.execute_stepwise(runner, _make_scheduler_output(req, step_id=0))

    assert isinstance(first, RunnerOutput)
    assert first.req_id == "req-1"
    assert first.step_index == 1
    assert first.finished is False
    assert first.result is None
    assert "req-1" in runner.state_cache

    second = DiffusionModelRunner.execute_stepwise(runner, _make_scheduler_output(req, step_id=1))

    assert isinstance(second, RunnerOutput)
    assert second.req_id == "req-1"
    assert second.step_index == 2
    assert second.finished is True
    assert second.result is not None
    assert second.result.error is None
    assert torch.equal(second.result.output, torch.tensor([2.0]))
    assert "req-1" not in runner.state_cache

    assert runner.pipeline.prepare_calls == 1
    assert runner.pipeline.denoise_calls == 2
    assert runner.pipeline.scheduler_calls == 2
    assert runner.pipeline.decode_calls == 1


def test_worker_execute_stepwise_delegates_to_model_runner():
    worker = object.__new__(DiffusionWorker)
    expected = RunnerOutput(req_id="req-1", step_index=1, finished=False, result=None)
    scheduler_output = SimpleNamespace(
        req_states=[
            SimpleNamespace(
                req=SimpleNamespace(
                    sampling_params=SimpleNamespace(lora_request=None),
                )
            )
        ]
    )
    worker.lora_manager = None
    worker.model_runner = SimpleNamespace(execute_stepwise=lambda arg: expected if arg is scheduler_output else None)

    output = DiffusionWorker.execute_stepwise(worker, scheduler_output)

    assert output is expected


def test_executor_execute_request_wraps_diffusion_output():
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor.od_config = object()
    executor._ensure_open = lambda: None
    executor.collective_rpc = Mock(return_value=DiffusionOutput(output=torch.tensor([1.0])))

    request = _make_engine_request("req-exec", num_inference_steps=1)
    scheduler_output = _make_scheduler_output(request, req_id="req-exec")

    output = MultiprocDiffusionExecutor.execute_request(executor, scheduler_output)

    assert isinstance(output, RunnerOutput)
    assert output.req_id == "req-exec"
    assert output.step_index is None
    assert output.finished is True
    assert output.result is not None
    assert torch.equal(output.result.output, torch.tensor([1.0]))


def test_executor_execute_step_passthroughs_runner_output():
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor._ensure_open = lambda: None
    expected = RunnerOutput(
        req_id="req-step",
        step_index=1,
        finished=False,
        result=None,
    )
    executor.collective_rpc = Mock(return_value=expected)

    request = _make_engine_request("req-step", num_inference_steps=2)
    scheduler_output = _make_scheduler_output(request, req_id="req-step")

    output = MultiprocDiffusionExecutor.execute_step(executor, scheduler_output)

    assert output is expected


def test_engine_step_execution_runs_to_completion():
    scheduler = StepScheduler()
    scheduler.initialize(Mock())
    engine = _make_engine(scheduler, execute_fn=None)
    request = _make_engine_request("req-step-engine", num_inference_steps=2)

    calls = {"count": 0}

    def execute_fn(scheduler_output):
        del scheduler_output
        calls["count"] += 1
        if calls["count"] == 1:
            return RunnerOutput(
                req_id="req-step-engine",
                step_index=1,
                finished=False,
                result=None,
            )
        return RunnerOutput(
            req_id="req-step-engine",
            step_index=2,
            finished=True,
            result=DiffusionOutput(output=torch.tensor([2.0])),
        )

    engine.execute_fn = execute_fn

    output = engine.add_req_and_wait_for_response(request)

    assert calls["count"] == 2
    assert output.error is None
    assert torch.equal(output.output, torch.tensor([2.0]))


def test_engine_returns_diffusion_error_output():
    scheduler = StepScheduler()
    scheduler.initialize(Mock())
    engine = _make_engine(
        scheduler,
        execute_fn=lambda _: RunnerOutput(
            req_id="req-error",
            step_index=1,
            finished=True,
            result=DiffusionOutput(error="boom"),
        ),
    )
    request = _make_engine_request("req-error", num_inference_steps=2)

    output = engine.add_req_and_wait_for_response(request)

    assert output.output is None
    assert output.error == "boom"


def test_engine_request_level_execution_is_compatible():
    scheduler = RequestScheduler()
    scheduler.initialize(Mock())
    engine = _make_engine(
        scheduler,
        execute_fn=lambda _: RunnerOutput(
            req_id="req-request-level",
            step_index=None,
            finished=True,
            result=DiffusionOutput(output=torch.tensor([7.0])),
        ),
    )
    request = _make_engine_request("req-request-level", num_inference_steps=1)

    output = engine.add_req_and_wait_for_response(request)

    assert output.error is None
    assert torch.equal(output.output, torch.tensor([7.0]))


def test_engine_abort_returns_aborted_error():
    scheduler = _AbortAwareScheduler()
    scheduler.initialize(Mock())
    engine = _make_engine(scheduler, execute_fn=None)
    request = _make_engine_request("req-abort", num_inference_steps=2)

    def execute_fn(_):
        engine.abort("req-abort")
        return RunnerOutput(
            req_id="req-abort",
            step_index=1,
            finished=False,
            result=None,
        )

    engine.execute_fn = execute_fn

    output = engine.add_req_and_wait_for_response(request)

    assert output.output is None
    assert output.error == "Request req-abort aborted."
