# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import (
    DiffusionRequestStatus,
    RequestScheduler,
    Scheduler,
    SchedulerInterface,
    StepScheduler,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.diffusion]


def _make_request(req_id: str) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[f"prompt_{req_id}"],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_ids=[req_id],
    )


class _StubScheduler(SchedulerInterface):
    def __init__(self, request: OmniDiffusionRequest, output: DiffusionOutput) -> None:
        self._request = request
        self._output = output
        self.initialized_with = None
        self._req_id = request.request_ids[0]
        self._state = None
        self._scheduled = False

    def initialize(self, od_config) -> None:
        self.initialized_with = od_config

    def add_request(self, request: OmniDiffusionRequest) -> str:
        assert request is self._request
        self._state = Mock(req_id=self._req_id, req=request)
        return self._req_id

    def schedule(self):
        if self._scheduled or self._state is None:
            return Mock(req_states=[])
        self._scheduled = True
        return Mock(req_states=[self._state])

    def update_from_output(self, sched_output, output: DiffusionOutput) -> set[str]:
        assert output is self._output
        return {self._req_id}

    def abort_request(self, req_id: str) -> bool:
        return False

    def has_requests(self) -> bool:
        return not self._scheduled

    def get_request_state(self, req_id: str):
        return self._state

    def pop_request_state(self, req_id: str):
        return self._state

    def preempt_request(self, req_id: str) -> bool:
        return False

    def finish_request(self, req_id: str, status) -> None:
        return None

    def close(self) -> None:
        return None


class TestRequestScheduler:
    def setup_method(self) -> None:
        self.scheduler: RequestScheduler = RequestScheduler()
        self.scheduler.initialize(Mock())

    def test_single_request_success_lifecycle(self) -> None:
        req_id = self.scheduler.add_request(_make_request("a"))
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.WAITING

        sched_output = self.scheduler.schedule()
        assert len(sched_output.req_states) == 1
        assert sched_output.req_states[0].req_id == req_id
        assert sched_output.num_running_reqs == 1
        assert sched_output.num_waiting_reqs == 0

        finished = self.scheduler.update_from_output(sched_output, DiffusionOutput(output=None))
        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        assert self.scheduler.has_requests() is False

    def test_error_output_marks_finished_error(self) -> None:
        req_id = self.scheduler.add_request(_make_request("err"))

        sched_output = self.scheduler.schedule()
        finished = self.scheduler.update_from_output(sched_output, DiffusionOutput(error="worker failed"))

        assert finished == {req_id}
        state = self.scheduler.get_request_state(req_id)
        assert state.status == DiffusionRequestStatus.FINISHED_ERROR
        assert state.error == "worker failed"

    def test_empty_output_without_error_marks_completed(self) -> None:
        req_id = self.scheduler.add_request(_make_request("empty"))

        sched_output = self.scheduler.schedule()
        finished = self.scheduler.update_from_output(sched_output, DiffusionOutput(output=None, error=None))

        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED

    def test_fifo_single_request_scheduling(self) -> None:
        req_id_a = self.scheduler.add_request(_make_request("a"))
        req_id_b = self.scheduler.add_request(_make_request("b"))

        first = self.scheduler.schedule()
        assert [s.req_id for s in first.req_states] == [req_id_a]
        assert first.num_running_reqs == 1
        assert first.num_waiting_reqs == 1

        # Request A is still running; scheduling again should not pull B.
        second = self.scheduler.schedule()
        assert [s.req_id for s in second.req_states] == [req_id_a]
        assert second.num_running_reqs == 1
        assert second.num_waiting_reqs == 1

        self.scheduler.update_from_output(first, DiffusionOutput(output=None))

        third = self.scheduler.schedule()
        assert [s.req_id for s in third.req_states] == [req_id_b]
        assert third.num_running_reqs == 1
        assert third.num_waiting_reqs == 0

    def test_abort_request_for_waiting_and_running(self) -> None:
        req_id_a = self.scheduler.add_request(_make_request("a"))
        req_id_b = self.scheduler.add_request(_make_request("b"))

        # Abort waiting request.
        assert self.scheduler.abort_request(req_id_b) is True
        state_b = self.scheduler.get_request_state(req_id_b)
        assert state_b.status == DiffusionRequestStatus.FINISHED_ABORTED

        # A should still run normally.
        output_a = self.scheduler.schedule()
        assert [s.req_id for s in output_a.req_states] == [req_id_a]

        # Abort running request.
        assert self.scheduler.abort_request(req_id_a) is True
        state_a = self.scheduler.get_request_state(req_id_a)
        assert state_a.status == DiffusionRequestStatus.FINISHED_ABORTED

        assert self.scheduler.has_requests() is False
        assert self.scheduler.schedule().req_states == []

    def test_has_requests_state_transition(self) -> None:
        assert self.scheduler.has_requests() is False

        req_id = self.scheduler.add_request(_make_request("has"))
        assert self.scheduler.has_requests() is True

        sched_output = self.scheduler.schedule()
        assert self.scheduler.has_requests() is True

        self.scheduler.update_from_output(sched_output, DiffusionOutput(output=None))
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        assert self.scheduler.has_requests() is False


class TestDiffusionEngine:
    def test_add_req_and_wait_for_response_single_path(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.scheduler = RequestScheduler()
        engine.scheduler.initialize(Mock())
        engine.executor = Mock()
        engine._rpc_lock = threading.Lock()

        request = _make_request("engine")
        expected = DiffusionOutput(output=None)
        engine.executor.add_req.return_value = expected

        output = engine.add_req_and_wait_for_response(request)

        assert output is expected
        engine.executor.add_req.assert_called_once_with(request)

    def test_supports_scheduler_interface_injection(self) -> None:
        request = _make_request("engine_iface")
        expected = DiffusionOutput(output=None)
        scheduler = _StubScheduler(request, expected)

        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.scheduler = scheduler
        engine.executor = Mock()
        engine.executor.add_req.return_value = expected
        engine._rpc_lock = threading.Lock()

        output = engine.add_req_and_wait_for_response(request)

        assert output is expected
        engine.executor.add_req.assert_called_once_with(request)

    def test_initializes_injected_scheduler(self) -> None:
        request = _make_request("init")
        scheduler = _StubScheduler(request, DiffusionOutput(output=None))
        od_config = Mock(model_class_name="mock_model")
        fake_executor_cls = Mock(return_value=Mock())

        with (
            patch("vllm_omni.diffusion.diffusion_engine.get_diffusion_post_process_func", return_value=None),
            patch("vllm_omni.diffusion.diffusion_engine.get_diffusion_pre_process_func", return_value=None),
            patch("vllm_omni.diffusion.diffusion_engine.DiffusionExecutor.get_class", return_value=fake_executor_cls),
            patch.object(DiffusionEngine, "_dummy_run", return_value=None),
        ):
            DiffusionEngine(od_config, scheduler=scheduler)

        assert scheduler.initialized_with is od_config
        fake_executor_cls.assert_called_once_with(od_config)

    def test_scheduler_alias_keeps_default_request_scheduler(self) -> None:
        scheduler = Scheduler()
        scheduler.initialize(Mock())

        req_id = scheduler.add_request(_make_request("alias"))
        sched_output = scheduler.schedule()
        finished = scheduler.update_from_output(sched_output, DiffusionOutput(output=None))

        assert req_id in finished
        assert scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED

    def test_dummy_run_raises_on_output_error(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.od_config = Mock(model_class_name="mock_model")
        engine.pre_process_func = None
        engine.add_req_and_wait_for_response = Mock(return_value=DiffusionOutput(error="boom"))

        with pytest.raises(RuntimeError, match="Dummy run failed: boom"):
            engine._dummy_run()


class TestStepScheduler:
    def setup_method(self) -> None:
        self.scheduler: StepScheduler = StepScheduler()
        self.scheduler.initialize(Mock())
        self.scheduler._max_batch_size = 1

    def test_requires_multiple_success_updates_to_finish(self) -> None:
        request = OmniDiffusionRequest(
            prompts=["prompt_step"],
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=3),
            request_ids=["step"],
        )

        req_id = self.scheduler.add_request(request)

        first = self.scheduler.schedule()
        assert [s.req_id for s in first.req_states] == [req_id]
        assert self.scheduler.update_from_output(first, DiffusionOutput(output=None)) == set()
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.RUNNING
        assert request.sampling_params.step_index == 1
        assert self.scheduler.has_requests() is True

        second = self.scheduler.schedule()
        assert [s.req_id for s in second.req_states] == [req_id]
        assert self.scheduler.update_from_output(second, DiffusionOutput(output=None)) == set()
        assert request.sampling_params.step_index == 2

        third = self.scheduler.schedule()
        assert [s.req_id for s in third.req_states] == [req_id]
        assert self.scheduler.update_from_output(third, DiffusionOutput(output=None)) == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        assert request.sampling_params.step_index == 3
        assert self.scheduler.has_requests() is False

    def test_rotates_requests_round_robin_by_step(self) -> None:
        req_id_a = self.scheduler.add_request(
            OmniDiffusionRequest(
                prompts=["prompt_a"],
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2),
                request_ids=["a"],
            )
        )
        req_id_b = self.scheduler.add_request(
            OmniDiffusionRequest(
                prompts=["prompt_b"],
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2),
                request_ids=["b"],
            )
        )

        first = self.scheduler.schedule()
        assert [s.req_id for s in first.req_states] == [req_id_a]
        assert self.scheduler.update_from_output(first, DiffusionOutput(output=None)) == set()

        second = self.scheduler.schedule()
        assert [s.req_id for s in second.req_states] == [req_id_b]
        assert self.scheduler.update_from_output(second, DiffusionOutput(output=None)) == set()

        third = self.scheduler.schedule()
        assert [s.req_id for s in third.req_states] == [req_id_a]
        assert self.scheduler.update_from_output(third, DiffusionOutput(output=None)) == {req_id_a}

        fourth = self.scheduler.schedule()
        assert [s.req_id for s in fourth.req_states] == [req_id_b]
        assert self.scheduler.update_from_output(fourth, DiffusionOutput(output=None)) == {req_id_b}

    def test_schedule_returns_at_most_one_request_when_multiple_requests_waiting(self) -> None:
        req_id_a = self.scheduler.add_request(
            OmniDiffusionRequest(
                prompts=["prompt_a"],
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2),
                request_ids=["a"],
            )
        )
        self.scheduler.add_request(
            OmniDiffusionRequest(
                prompts=["prompt_b"],
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2),
                request_ids=["b"],
            )
        )
        self.scheduler.add_request(
            OmniDiffusionRequest(
                prompts=["prompt_c"],
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2),
                request_ids=["c"],
            )
        )

        sched_output = self.scheduler.schedule()

        # Current upstream request is already pre-batched, so StepScheduler must
        # not co-batch multiple requests here. Revisit this assertion if the
        # upper-layer request shape is refactored for real continuous batching.
        assert len(sched_output.req_states) <= 1
        assert sched_output.num_running_reqs <= 1
        assert [s.req_id for s in sched_output.req_states] == [req_id_a]
        assert sched_output.num_waiting_reqs == 2

    def test_error_output_marks_finished_error(self) -> None:
        req_id = self.scheduler.add_request(
            OmniDiffusionRequest(
                prompts=["prompt_err"],
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=3),
                request_ids=["err"],
            )
        )

        sched_output = self.scheduler.schedule()
        assert [s.req_id for s in sched_output.req_states] == [req_id]
        finished = self.scheduler.update_from_output(sched_output, DiffusionOutput(error="worker failed"))

        assert finished == {req_id}
        state = self.scheduler.get_request_state(req_id)
        assert state.status == DiffusionRequestStatus.FINISHED_ERROR
        assert state.error == "worker failed"
        assert self.scheduler.has_requests() is False

    def test_abort_request_for_waiting_and_running(self) -> None:
        req_id_a = self.scheduler.add_request(
            OmniDiffusionRequest(
                prompts=["prompt_a"],
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2),
                request_ids=["a"],
            )
        )
        req_id_b = self.scheduler.add_request(
            OmniDiffusionRequest(
                prompts=["prompt_b"],
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2),
                request_ids=["b"],
            )
        )

        assert self.scheduler.abort_request(req_id_b) is True
        assert self.scheduler.get_request_state(req_id_b).status == DiffusionRequestStatus.FINISHED_ABORTED

        running = self.scheduler.schedule()
        assert [s.req_id for s in running.req_states] == [req_id_a]

        assert self.scheduler.abort_request(req_id_a) is True
        assert self.scheduler.get_request_state(req_id_a).status == DiffusionRequestStatus.FINISHED_ABORTED
        assert self.scheduler.has_requests() is False

    def test_preempt_request_preserves_step_index(self) -> None:
        request = OmniDiffusionRequest(
            prompts=["prompt_preempt"],
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=3),
            request_ids=["preempt"],
        )
        req_id = self.scheduler.add_request(request)

        first = self.scheduler.schedule()
        assert self.scheduler.update_from_output(first, DiffusionOutput(output=None)) == set()
        assert request.sampling_params.step_index == 1

        second = self.scheduler.schedule()
        assert [s.req_id for s in second.req_states] == [req_id]
        assert self.scheduler.preempt_request(req_id) is True
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.PREEMPTED
        assert request.sampling_params.step_index == 1

        third = self.scheduler.schedule()
        assert [s.req_id for s in third.req_states] == [req_id]
        assert request.sampling_params.step_index == 1

    @pytest.mark.parametrize(
        ("sampling_params", "expected_steps"),
        [
            (
                OmniDiffusionSamplingParams(
                    timesteps=torch.tensor([1.0, 0.5, 0.0]),
                    sigmas=[1.0, 0.5, 0.25, 0.0],
                    num_inference_steps=5,
                ),
                3,
            ),
            (
                OmniDiffusionSamplingParams(
                    sigmas=[1.0, 0.5],
                    num_inference_steps=5,
                ),
                2,
            ),
            (
                OmniDiffusionSamplingParams(
                    num_inference_steps=4,
                ),
                4,
            ),
        ],
    )
    def test_total_steps_priority(self, sampling_params: OmniDiffusionSamplingParams, expected_steps: int) -> None:
        request = OmniDiffusionRequest(
            prompts=["prompt_priority"],
            sampling_params=sampling_params,
            request_ids=["priority"],
        )
        req_id = self.scheduler.add_request(request)

        for _ in range(expected_steps - 1):
            sched_output = self.scheduler.schedule()
            assert [s.req_id for s in sched_output.req_states] == [req_id]
            assert self.scheduler.update_from_output(sched_output, DiffusionOutput(output=None)) == set()

        final_output = self.scheduler.schedule()
        assert [s.req_id for s in final_output.req_states] == [req_id]
        assert self.scheduler.update_from_output(final_output, DiffusionOutput(output=None)) == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED

    @pytest.mark.parametrize(
        "sampling_params",
        [
            OmniDiffusionSamplingParams(num_inference_steps=0),
            OmniDiffusionSamplingParams(num_inference_steps=3, step_index=3),
            OmniDiffusionSamplingParams(num_inference_steps=3, step_index=-1),
        ],
    )
    def test_rejects_invalid_initial_step_state(self, sampling_params: OmniDiffusionSamplingParams) -> None:
        request = OmniDiffusionRequest(
            prompts=["prompt_invalid"],
            sampling_params=sampling_params,
            request_ids=["invalid"],
        )

        with pytest.raises(ValueError):
            self.scheduler.add_request(request)
