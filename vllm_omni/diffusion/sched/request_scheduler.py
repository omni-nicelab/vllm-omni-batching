# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.logger import init_logger

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import _BaseScheduler
from vllm_omni.diffusion.sched.interface import (
    DiffusionRequestState,
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
)
from vllm_omni.diffusion.worker.utils import RunnerOutput

logger = init_logger(__name__)


class RequestScheduler(_BaseScheduler):
    """Diffusion scheduler with vLLM-style waiting/running queues."""

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
        scheduled_req_ids = [state.sched_req_id for state in sched_output.req_states]
        if not scheduled_req_ids:
            return set()

        # A scheduled request may be aborted after schedule() but before
        # update_from_output() processes the runner output. It is already
        # marked finished at that point, but we still need to surface its id
        # in this update so the engine can observe the terminal state.
        finished_req_ids = {req_id for req_id in scheduled_req_ids if req_id in self._finished_req_ids}
        terminal_statuses: dict[str, DiffusionRequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}
        for req_id in scheduled_req_ids:
            state = self._request_states.get(req_id)
            if state is None or state.is_finished():
                continue
            if output.result is None:
                terminal_statuses[req_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[req_id] = "No output result"
            elif output.result.error:
                terminal_statuses[req_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[req_id] = output.result.error
            else:
                terminal_statuses[req_id] = DiffusionRequestStatus.FINISHED_COMPLETED
                terminal_errors[req_id] = None

        finished_req_ids |= self._finish_requests(terminal_statuses, terminal_errors)
        return finished_req_ids
