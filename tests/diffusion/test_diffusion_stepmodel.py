# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.sched.interface import (
    DiffusionRequestState as SchedulerRequestState,
    DiffusionSchedulerOutput,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.diffusion.worker.utils import RunnerOutput

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
        req_states=[SchedulerRequestState(req_id=req_id, req=req)],
        finished_req_ids=set() if finished_req_ids is None else set(finished_req_ids),
        num_running_reqs=1,
        num_waiting_reqs=0,
    )


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
    scheduler_output = object()
    worker.model_runner = SimpleNamespace(
        execute_stepwise=lambda arg: expected if arg is scheduler_output else None
    )

    output = DiffusionWorker.execute_stepwise(worker, scheduler_output)

    assert output is expected
