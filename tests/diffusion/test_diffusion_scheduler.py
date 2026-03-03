# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from unittest.mock import Mock

import pytest

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.scheduler import DiffusionRequestStatus, Scheduler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.diffusion]


def _make_scheduler() -> Scheduler:
    scheduler = Scheduler()
    scheduler.initialize(Mock())
    return scheduler


def _make_request(req_id: str) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[f"prompt_{req_id}"],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_ids=[req_id],
    )


def test_single_request_success_lifecycle() -> None:
    scheduler = _make_scheduler()

    req_id = scheduler.add_request(_make_request("a"))
    assert scheduler.get_request_state(req_id).status == DiffusionRequestStatus.WAITING

    sched_output = scheduler.schedule()
    assert len(sched_output.req_states) == 1
    assert sched_output.req_states[0].req_id == req_id
    assert sched_output.num_running_reqs == 1
    assert sched_output.num_waiting_reqs == 0

    finished = scheduler.update_from_output(sched_output, DiffusionOutput(output=None))
    assert finished == {req_id}
    assert scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
    assert scheduler.has_requests() is False


def test_error_output_marks_finished_error() -> None:
    scheduler = _make_scheduler()
    req_id = scheduler.add_request(_make_request("err"))

    sched_output = scheduler.schedule()
    finished = scheduler.update_from_output(sched_output, DiffusionOutput(error="worker failed"))

    assert finished == {req_id}
    state = scheduler.get_request_state(req_id)
    assert state.status == DiffusionRequestStatus.FINISHED_ERROR
    assert state.error == "worker failed"


def test_empty_output_without_error_marks_completed() -> None:
    scheduler = _make_scheduler()
    req_id = scheduler.add_request(_make_request("empty"))

    sched_output = scheduler.schedule()
    finished = scheduler.update_from_output(sched_output, DiffusionOutput(output=None, error=None))

    assert finished == {req_id}
    assert scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED


def test_fifo_single_request_scheduling() -> None:
    scheduler = _make_scheduler()
    req_id_a = scheduler.add_request(_make_request("a"))
    req_id_b = scheduler.add_request(_make_request("b"))

    first = scheduler.schedule()
    assert [s.req_id for s in first.req_states] == [req_id_a]
    assert first.num_running_reqs == 1
    assert first.num_waiting_reqs == 1

    # Request A is still running; scheduling again should not pull B.
    second = scheduler.schedule()
    assert [s.req_id for s in second.req_states] == [req_id_a]
    assert second.num_running_reqs == 1
    assert second.num_waiting_reqs == 1

    scheduler.update_from_output(first, DiffusionOutput(output=None))

    third = scheduler.schedule()
    assert [s.req_id for s in third.req_states] == [req_id_b]
    assert third.num_running_reqs == 1
    assert third.num_waiting_reqs == 0


def test_abort_request_for_waiting_and_running() -> None:
    scheduler = _make_scheduler()
    req_id_a = scheduler.add_request(_make_request("a"))
    req_id_b = scheduler.add_request(_make_request("b"))

    # Abort waiting request.
    assert scheduler.abort_request(req_id_b) is True
    state_b = scheduler.get_request_state(req_id_b)
    assert state_b.status == DiffusionRequestStatus.FINISHED_ABORTED

    # A should still run normally.
    output_a = scheduler.schedule()
    assert [s.req_id for s in output_a.req_states] == [req_id_a]

    # Abort running request.
    assert scheduler.abort_request(req_id_a) is True
    state_a = scheduler.get_request_state(req_id_a)
    assert state_a.status == DiffusionRequestStatus.FINISHED_ABORTED

    assert scheduler.has_requests() is False
    assert scheduler.schedule().req_states == []


def test_has_requests_state_transition() -> None:
    scheduler = _make_scheduler()
    assert scheduler.has_requests() is False

    req_id = scheduler.add_request(_make_request("has"))
    assert scheduler.has_requests() is True

    sched_output = scheduler.schedule()
    assert scheduler.has_requests() is True

    scheduler.update_from_output(sched_output, DiffusionOutput(output=None))
    assert scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
    assert scheduler.has_requests() is False


def test_engine_add_req_and_wait_for_response_single_path() -> None:
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.scheduler = _make_scheduler()
    engine.executor = Mock()
    engine._rpc_lock = threading.Lock()

    request = _make_request("engine")
    expected = DiffusionOutput(output=None)
    engine.executor.add_req.return_value = expected

    output = engine.add_req_and_wait_for_response(request)

    assert output is expected
    engine.executor.add_req.assert_called_once_with(request)


def test_engine_dummy_run_raises_on_output_error() -> None:
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = Mock(model_class_name="mock_model")
    engine.pre_process_func = None
    engine.add_req_and_wait_for_response = Mock(return_value=DiffusionOutput(error="boom"))

    with pytest.raises(RuntimeError, match="Dummy run failed: boom"):
        engine._dummy_run()
