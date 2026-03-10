# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import deque

from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import (
    DiffusionRequestState,
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    SchedulerInterface,
)
from vllm_omni.diffusion.worker.utils import RunnerOutput

logger = init_logger(__name__)


class RequestScheduler(SchedulerInterface):
    """Diffusion scheduler with vLLM-style waiting/running queues."""

    def __init__(self) -> None:
        self.od_config: OmniDiffusionConfig | None = None
        self._request_states: dict[str, DiffusionRequestState] = {}
        self._step_id: int = 0
        self._waiting: deque[str] = deque()
        self._running: list[str] = []
        self._finished_req_ids: set[str] = set()

    def initialize(self, od_config: OmniDiffusionConfig) -> None:
        self.od_config = od_config
        self._request_states.clear()
        self._step_id = 0
        self._waiting.clear()
        self._running.clear()
        self._finished_req_ids.clear()

    def add_request(self, request: OmniDiffusionRequest) -> str:
        req_id = self._make_sched_req_id(request)
        state = DiffusionRequestState(sched_req_id=req_id, req=request)
        self._request_states[req_id] = state
        self._waiting.append(req_id)
        logger.debug("Scheduler add_request: %s (waiting=%d)", req_id, len(self._waiting))
        return req_id

    def schedule(self) -> DiffusionSchedulerOutput:
        if not self._running and self._waiting:
            req_id = self._waiting.popleft()
            state = self._request_states.get(req_id)
            if state is not None:
                state.status = DiffusionRequestStatus.RUNNING
                self._running.append(req_id)

        running_states: list[DiffusionRequestState] = []
        for req_id in self._running:
            state = self._request_states.get(req_id)
            if state is not None:
                running_states.append(state)

        scheduler_output = DiffusionSchedulerOutput(
            step_id=self._step_id,
            req_states=running_states,
            finished_req_ids=set(self._finished_req_ids),
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._waiting),
        )

        self._step_id += 1
        self._finished_req_ids.clear()
        return scheduler_output

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: RunnerOutput) -> set[str]:
        scheduled_req_ids = {state.sched_req_id for state in sched_output.req_states}
        if not scheduled_req_ids:
            return set()

        completed_req_ids: set[str] = set()
        for req_id in scheduled_req_ids:
            state = self._request_states.get(req_id)
            if state is None:
                continue
            if output.result is None:
                state.status = DiffusionRequestStatus.FINISHED_ERROR
                state.error = "No output result"
            elif output.result.error:
                state.status = DiffusionRequestStatus.FINISHED_ERROR
                state.error = output.result.error
            else:
                state.status = DiffusionRequestStatus.FINISHED_COMPLETED
                state.error = None
            completed_req_ids.add(req_id)

        if completed_req_ids:
            self._running = [req_id for req_id in self._running if req_id not in completed_req_ids]
            for req_id in completed_req_ids:
                try:
                    self._waiting.remove(req_id)
                except ValueError:
                    pass
            self._finished_req_ids |= completed_req_ids

        return completed_req_ids

    def abort_request(self, req_id: str) -> bool:
        if req_id not in self._request_states:
            return False
        self.finish_request(req_id, DiffusionRequestStatus.FINISHED_ABORTED)
        self._finished_req_ids.add(req_id)
        return True

    def has_requests(self) -> bool:
        return bool(self._waiting or self._running)

    def get_request_state(self, req_id: str) -> DiffusionRequestState | None:
        return self._request_states.get(req_id)

    def pop_request_state(self, req_id: str) -> DiffusionRequestState | None:
        return self._request_states.pop(req_id, None)

    def preempt_request(self, req_id: str) -> bool:
        if req_id not in self._request_states:
            return False
        if req_id in self._running:
            self._running.remove(req_id)
            self._waiting.appendleft(req_id)
            self._request_states[req_id].status = DiffusionRequestStatus.PREEMPTED
            return True
        return False

    def finish_request(self, req_id: str, status: DiffusionRequestStatus) -> None:
        assert DiffusionRequestStatus.is_finished(status)
        state = self._request_states.get(req_id)
        if state is None:
            return

        state.status = status
        if req_id in self._running:
            self._running.remove(req_id)
        try:
            self._waiting.remove(req_id)
        except ValueError:
            pass

    def close(self) -> None:
        self._request_states.clear()
        self._waiting.clear()
        self._running.clear()
        self._finished_req_ids.clear()
