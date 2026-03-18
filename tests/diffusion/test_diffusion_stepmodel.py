# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for step-level diffusion execution.

Covers the full stack from runner → worker → executor → engine → async
entrypoint, including abort handling at each layer.
"""

import asyncio
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from vllm_omni.diffusion.data import DiffusionOutput, DiffusionRequestAbortedError
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import RequestScheduler, StepScheduler
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    NewRequestData,
    SchedulerInterface,
)
from vllm_omni.diffusion.sched.interface import (
    DiffusionRequestState as SchedulerRequestState,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.diffusion.worker.utils import RunnerOutput
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


def _assert_aborted_output(output: DiffusionOutput, request_id: str) -> None:
    assert output.output is None
    assert output.error is None
    assert output.aborted is True
    assert output.abort_message == f"Request {request_id} aborted."


@contextmanager
def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    yield


class _StepPipeline:
    """Minimal pipeline stub that supports step-wise execution."""

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


class _AbortAwareScheduler(SchedulerInterface):
    """Scheduler stub that supports abort transitions for engine-level tests."""

    def __init__(self) -> None:
        self._state = None
        self._finished_req_ids: set[str] = set()
        self._scheduled = False

    def initialize(self, od_config) -> None:
        del od_config

    def add_request(self, request: OmniDiffusionRequest) -> str:
        sched_req_id = request.request_ids[0]
        self._state = SchedulerRequestState(sched_req_id=sched_req_id, req=request)
        self._scheduled = False
        return sched_req_id

    def schedule(self) -> DiffusionSchedulerOutput:
        scheduled_new_reqs = []
        if (
            self._state is not None
            and self._state.status < DiffusionRequestStatus.FINISHED_COMPLETED
            and not self._scheduled
        ):
            self._state.status = DiffusionRequestStatus.RUNNING
            scheduled_new_reqs = [NewRequestData.from_state(self._state)]
            self._scheduled = True
        return DiffusionSchedulerOutput(
            step_id=0,
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            finished_req_ids=set(self._finished_req_ids),
            num_running_reqs=len(scheduled_new_reqs),
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

    def has_requests(self) -> bool:
        return self._state is not None and self._state.status < DiffusionRequestStatus.FINISHED_COMPLETED

    def get_request_state(self, sched_req_id: str):
        if self._state is not None and self._state.sched_req_id == sched_req_id:
            return self._state
        return None

    def get_sched_req_id(self, request_id: str) -> str | None:
        if self._state is None:
            return None
        if request_id in self._state.req.request_ids:
            return self._state.sched_req_id
        return None

    def pop_request_state(self, sched_req_id: str):
        if self._state is not None and self._state.sched_req_id == sched_req_id:
            state, self._state = self._state, None
            return state
        return None

    def preempt_request(self, sched_req_id: str) -> bool:
        del sched_req_id
        return False

    def finish_requests(self, sched_req_ids, status) -> None:
        if isinstance(sched_req_ids, str):
            sched_req_ids = [sched_req_ids]
        for sched_req_id in sched_req_ids:
            if self._state is not None and self._state.sched_req_id == sched_req_id:
                self._state.status = status
                self._finished_req_ids.add(sched_req_id)

    def close(self) -> None:
        self._state = None


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
    runner.cache_manager = None
    runner.offload_backend = None
    runner.state_cache = {}
    runner.kv_transfer_manager = SimpleNamespace()
    return runner


def _make_scheduler_output(req, sched_req_id="req-1", step_id=0, finished_req_ids=None):
    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=[NewRequestData(sched_req_id=sched_req_id, req=req)],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set() if finished_req_ids is None else set(finished_req_ids),
        num_running_reqs=1,
        num_waiting_reqs=0,
    )


def _make_cached_scheduler_output(sched_req_id="req-1", step_id=0, finished_req_ids=None):
    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(sched_req_ids=[sched_req_id]),
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


def _make_engine(scheduler: SchedulerInterface, execute_fn=None):
    engine = object.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(model_class_name="QwenImagePipeline")
    engine.pre_process_func = None
    engine.post_process_func = None
    engine.scheduler = scheduler
    engine.execute_fn = execute_fn
    engine._rpc_lock = threading.RLock()
    engine.abort_queue = queue.Queue()
    return engine


def _make_async_engine(engine):
    """Create an AsyncOmniDiffusion wrapping *engine*."""
    async_engine = object.__new__(AsyncOmniDiffusion)
    async_engine.engine = engine
    async_engine._executor = ThreadPoolExecutor(max_workers=1)
    async_engine._request_lock = threading.Lock()
    async_engine._request_states = {}
    async_engine._closed = False
    return async_engine


# ---------------------------------------------------------------------------
# Runner / Worker / Executor
# ---------------------------------------------------------------------------


class TestRunner:
    """DiffusionModelRunner.execute_stepwise"""

    def test_completes_request_and_clears_state(self, monkeypatch):
        runner = _make_runner()
        req = _make_step_request()
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        first = DiffusionModelRunner.execute_stepwise(runner, _make_scheduler_output(req, step_id=0))
        assert first.req_id == "req-1"
        assert first.step_index == 1
        assert first.finished is False
        assert first.result is None
        assert "req-1" in runner.state_cache

        second = DiffusionModelRunner.execute_stepwise(runner, _make_cached_scheduler_output(step_id=1))
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

    def test_cached_request_requires_existing_state(self, monkeypatch):
        runner = _make_runner()
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        with pytest.raises(ValueError, match="Missing cached state for request req-1"):
            DiffusionModelRunner.execute_stepwise(runner, _make_cached_scheduler_output(step_id=1))


class TestWorker:
    """DiffusionWorker.execute_stepwise"""

    def test_delegates_to_model_runner(self):
        worker = object.__new__(DiffusionWorker)
        expected = RunnerOutput(req_id="req-1", step_index=1, finished=False, result=None)
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req=SimpleNamespace(
                        sampling_params=SimpleNamespace(lora_request=None),
                    )
                )
            ],
            scheduled_cached_reqs=SimpleNamespace(sched_req_ids=[]),
        )
        worker.lora_manager = None
        worker.model_runner = SimpleNamespace(
            execute_stepwise=lambda arg: expected if arg is scheduler_output else None
        )

        output = DiffusionWorker.execute_stepwise(worker, scheduler_output)

        assert output is expected


class TestExecutor:
    """MultiprocDiffusionExecutor.execute_request / execute_step"""

    def test_execute_request_wraps_diffusion_output(self):
        executor = object.__new__(MultiprocDiffusionExecutor)
        executor.od_config = object()
        executor._ensure_open = lambda: None
        executor.collective_rpc = Mock(return_value=DiffusionOutput(output=torch.tensor([1.0])))

        request = _make_engine_request("req-exec", num_inference_steps=1)
        scheduler_output = _make_scheduler_output(request, sched_req_id="req-exec")

        output = MultiprocDiffusionExecutor.execute_request(executor, scheduler_output)

        assert isinstance(output, RunnerOutput)
        assert output.req_id == "req-exec"
        assert output.step_index is None
        assert output.finished is True
        assert torch.equal(output.result.output, torch.tensor([1.0]))

    def test_execute_step_passes_through_runner_output(self):
        executor = object.__new__(MultiprocDiffusionExecutor)
        executor._ensure_open = lambda: None
        expected = RunnerOutput(req_id="req-step", step_index=1, finished=False, result=None)
        executor.collective_rpc = Mock(return_value=expected)

        request = _make_engine_request("req-step", num_inference_steps=2)
        scheduler_output = _make_scheduler_output(request, sched_req_id="req-step")

        output = MultiprocDiffusionExecutor.execute_step(executor, scheduler_output)

        assert output is expected


# ---------------------------------------------------------------------------
# Engine (synchronous)
# ---------------------------------------------------------------------------


class TestEngine:
    """DiffusionEngine.add_req_and_wait_for_response"""

    def test_step_execution_completes(self):
        """Two-step denoising runs to completion and returns the final output."""
        scheduler = StepScheduler()
        scheduler.initialize(Mock())
        engine = _make_engine(scheduler)
        request = _make_engine_request("req-step", num_inference_steps=2)

        call_count = {"n": 0}

        def execute_fn(_):
            call_count["n"] += 1
            finished = call_count["n"] == 2
            return RunnerOutput(
                req_id="req-step",
                step_index=call_count["n"],
                finished=finished,
                result=(DiffusionOutput(output=torch.tensor([2.0])) if finished else None),
            )

        engine.execute_fn = execute_fn

        output = engine.add_req_and_wait_for_response(request)

        assert call_count["n"] == 2
        assert output.error is None
        assert torch.equal(output.output, torch.tensor([2.0]))

    def test_request_level_execution_compatible(self):
        """RequestScheduler (non-step) path still works through the engine."""
        scheduler = RequestScheduler()
        scheduler.initialize(Mock())
        engine = _make_engine(
            scheduler,
            execute_fn=lambda _: RunnerOutput(
                req_id="req-full",
                step_index=None,
                finished=True,
                result=DiffusionOutput(output=torch.tensor([7.0])),
            ),
        )

        output = engine.add_req_and_wait_for_response(_make_engine_request("req-full", num_inference_steps=1))

        assert output.error is None
        assert torch.equal(output.output, torch.tensor([7.0]))

    def test_error_output_propagates(self):
        """An error result from the runner is propagated unchanged."""
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

        output = engine.add_req_and_wait_for_response(_make_engine_request("req-error", num_inference_steps=2))

        assert output.output is None
        assert output.error == "boom"

    def test_execute_fn_exception_returns_error(self):
        """If execute_fn raises, the engine catches it and returns an error."""
        scheduler = StepScheduler()
        scheduler.initialize(Mock())
        engine = _make_engine(
            scheduler,
            execute_fn=lambda _: (_ for _ in ()).throw(RuntimeError("gpu on fire")),
        )

        output = engine.add_req_and_wait_for_response(_make_engine_request("req-raise", num_inference_steps=2))

        assert output.output is None
        assert "gpu on fire" in output.error


# ---------------------------------------------------------------------------
# Engine abort
# ---------------------------------------------------------------------------


class TestEngineAbort:
    """Abort handling at the engine layer."""

    def test_abort_during_execution(self):
        """Abort issued while execute_fn is running returns an aborted error."""
        scheduler = _AbortAwareScheduler()
        scheduler.initialize(Mock())
        engine = _make_engine(scheduler)
        request = _make_engine_request("req-abort", num_inference_steps=2)

        def execute_fn(_):
            engine.abort("req-abort")
            return RunnerOutput(req_id="req-abort", step_index=1, finished=False, result=None)

        engine.execute_fn = execute_fn

        output = engine.add_req_and_wait_for_response(request)

        _assert_aborted_output(output, "req-abort")

    def test_abort_batched_request_by_secondary_request_id(self):
        scheduler = _AbortAwareScheduler()
        scheduler.initialize(Mock())
        engine = _make_engine(scheduler)
        request = OmniDiffusionRequest(
            prompts=["prompt-a", "prompt-b"],
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2),
            request_ids=["req-batch-a", "req-batch-b"],
        )

        def execute_fn(_):
            engine.abort("req-batch-b")
            return RunnerOutput(
                req_id="req-batch-a",
                step_index=1,
                finished=False,
                result=None,
            )

        engine.execute_fn = execute_fn

        output = engine.add_req_and_wait_for_response(request)

        _assert_aborted_output(output, "req-batch-a")

    def test_duplicate_abort_is_idempotent(self):
        """Calling abort twice for the same request does not raise or corrupt state."""
        scheduler = StepScheduler()
        scheduler.initialize(Mock())
        engine = _make_engine(scheduler)
        request = _make_engine_request("req-dup", num_inference_steps=4)

        step = {"n": 0}

        def execute_fn(_):
            step["n"] += 1
            if step["n"] == 1:
                engine.abort("req-dup")
                engine.abort("req-dup")
            return RunnerOutput(req_id="req-dup", step_index=step["n"], finished=False, result=None)

        engine.execute_fn = execute_fn

        output = engine.add_req_and_wait_for_response(request)

        _assert_aborted_output(output, "req-dup")

    @pytest.mark.parametrize(
        "scenario",
        ["nonexistent", "already_completed"],
        ids=["nonexistent_request", "completed_request"],
    )
    def test_abort_noop_does_not_crash(self, scenario):
        """abort() on a nonexistent or already-completed request is a harmless no-op."""
        scheduler = StepScheduler()
        scheduler.initialize(Mock())
        engine = _make_engine(scheduler)

        if scenario == "already_completed":
            engine.execute_fn = lambda _: RunnerOutput(
                req_id="req-done",
                step_index=1,
                finished=True,
                result=DiffusionOutput(output=torch.tensor([1.0])),
            )
            output = engine.add_req_and_wait_for_response(_make_engine_request("req-done", num_inference_steps=1))
            assert output.error is None
            target_id = "req-done"
        else:
            target_id = "nonexistent-id"

        # Must not raise
        engine.abort(target_id)
        engine._process_aborts_queue()

    def test_abort_mid_step(self):
        """Abort after step 1 of 4 stops execution immediately."""
        scheduler = StepScheduler()
        scheduler.initialize(Mock())
        engine = _make_engine(scheduler)
        request = _make_engine_request("req-mid", num_inference_steps=4)

        step = {"n": 0}

        def execute_fn(_):
            step["n"] += 1
            if step["n"] == 2:
                engine.abort("req-mid")
            return RunnerOutput(req_id="req-mid", step_index=step["n"], finished=False, result=None)

        engine.execute_fn = execute_fn

        output = engine.add_req_and_wait_for_response(request)

        assert step["n"] == 2
        _assert_aborted_output(output, "req-mid")


# ---------------------------------------------------------------------------
# Scheduler abort
# ---------------------------------------------------------------------------


class TestSchedulerAbort:
    """Abort handling at the scheduler layer."""

    def test_step_scheduler_abort_while_waiting(self):
        """Abort a request still in the WAITING queue (not yet scheduled)."""
        scheduler = StepScheduler()
        scheduler.initialize(Mock())

        request = _make_engine_request("req-wait", num_inference_steps=4)
        req_id = scheduler.add_request(request)

        assert scheduler.get_request_state(req_id).status == DiffusionRequestStatus.WAITING

        scheduler.finish_requests(req_id, DiffusionRequestStatus.FINISHED_ABORTED)
        assert scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_ABORTED

        sched_output = scheduler.schedule()
        assert sched_output.num_scheduled_reqs == 0

    def test_request_scheduler_abort(self):
        """Abort path through RequestScheduler returns an aborted error."""
        scheduler = RequestScheduler()
        scheduler.initialize(Mock())
        engine = _make_engine(scheduler)
        request = _make_engine_request("req-rs", num_inference_steps=1)

        def execute_fn(_):
            engine.abort("req-rs")
            return RunnerOutput(
                req_id="req-rs",
                step_index=None,
                finished=True,
                result=DiffusionOutput(output=torch.tensor([1.0])),
            )

        engine.execute_fn = execute_fn

        output = engine.add_req_and_wait_for_response(request)

        _assert_aborted_output(output, "req-rs")


# ---------------------------------------------------------------------------
# Async entrypoint abort
# ---------------------------------------------------------------------------


class TestAsyncAbort:
    """AsyncOmniDiffusion abort scenarios."""

    @pytest.mark.anyio
    async def test_abort_stops_step_execution(self):
        """Aborting a running step-execution request raises RuntimeError."""
        scheduler = StepScheduler()
        scheduler.initialize(Mock())
        engine = _make_engine(scheduler)
        step_started = threading.Event()

        def execute_fn(_):
            step_started.set()
            time.sleep(0.05)
            return RunnerOutput(
                req_id="req-async",
                step_index=1,
                finished=False,
                result=None,
            )

        engine.execute_fn = execute_fn
        async_engine = _make_async_engine(engine)

        try:
            task = asyncio.create_task(
                async_engine.generate(
                    prompt="a prompt",
                    sampling_params=OmniDiffusionSamplingParams(num_inference_steps=4),
                    request_id="req-async",
                )
            )

            for _ in range(50):
                if step_started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert step_started.is_set(), "step execution never started"

            await async_engine.abort("req-async")

            with pytest.raises(DiffusionRequestAbortedError, match="aborted"):
                await task
        finally:
            async_engine.close()

    @pytest.mark.anyio
    async def test_abort_forwards_to_engine(self):
        """Abort of a running request calls engine.abort() and propagates the error."""
        first_started = threading.Event()
        abort_forwarded = threading.Event()

        def step(request):
            request_id = request.request_ids[0]
            first_started.set()
            assert abort_forwarded.wait(timeout=5), "entrypoint abort never reached engine.abort"
            raise DiffusionRequestAbortedError(f"Request {request_id} aborted.")

        engine = Mock()
        engine.step.side_effect = step
        engine.close = Mock()

        def abort_side_effect(request_ids):
            assert request_ids == ["req-running"]
            abort_forwarded.set()

        engine.abort = Mock(side_effect=abort_side_effect)

        async_engine = _make_async_engine(engine)

        try:
            task = asyncio.create_task(
                async_engine.generate(
                    prompt="running prompt",
                    sampling_params=OmniDiffusionSamplingParams(num_inference_steps=4),
                    request_id="req-running",
                )
            )

            assert await asyncio.to_thread(first_started.wait, 5), "running request never started"

            await async_engine.abort("req-running")

            with pytest.raises(DiffusionRequestAbortedError, match="Request req-running aborted"):
                await task

            engine.abort.assert_called_once_with(["req-running"])
        finally:
            abort_forwarded.set()
            async_engine.close()

    @pytest.mark.anyio
    async def test_abort_cancels_queued_request(self):
        """Aborting a queued (not yet running) request cancels its future."""
        first_started = threading.Event()
        release_first = threading.Event()
        started_request_ids: list[str] = []

        def step(request):
            request_id = request.request_ids[0]
            started_request_ids.append(request_id)
            if request_id == "req-running":
                first_started.set()
                assert release_first.wait(timeout=5), "timed out waiting to release first request"
            return [SimpleNamespace(request_id=request_id, images=[])]

        engine = Mock()
        engine.step.side_effect = step
        engine.abort = Mock()
        engine.close = Mock()

        async_engine = _make_async_engine(engine)

        try:
            running_task = asyncio.create_task(
                async_engine.generate(
                    prompt="running prompt",
                    sampling_params=OmniDiffusionSamplingParams(num_inference_steps=4),
                    request_id="req-running",
                )
            )

            assert await asyncio.to_thread(first_started.wait, 5), "first request never started"

            queued_task = asyncio.create_task(
                async_engine.generate(
                    prompt="queued prompt",
                    sampling_params=OmniDiffusionSamplingParams(num_inference_steps=4),
                    request_id="req-queued",
                )
            )

            await asyncio.sleep(0.05)
            await async_engine.abort("req-queued")
            release_first.set()

            running_output = await running_task
            assert running_output.request_id == "req-running"

            with pytest.raises(DiffusionRequestAbortedError, match="req-queued aborted"):
                await queued_task

            assert started_request_ids == ["req-running"]
            engine.abort.assert_not_called()
        finally:
            release_first.set()
            async_engine.close()
