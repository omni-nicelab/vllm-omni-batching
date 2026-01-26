# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from enum import Enum

from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.request import DiffusionRequestState, OmniDiffusionRequest
from vllm_omni.diffusion.worker.step_batch import StepRunnerOutput, StepSchedulerOutput

logger = init_logger(__name__)



class DiffusionScheduleState(str, Enum):
    """Scheduler-only lifecycle state for diffusion requests."""

    WAITING = "waiting"
    RUNNING = "running"
    SUSPENDED = "suspended"
    FINISHED = "finished"


class DiffusionStepScheduler:
    """Step-level scheduler only (no IPC or execution).

    Responsibilities:
    - Manage request state lifecycle (waiting/running/suspended/finished)
    - Select requests per step and build StepSchedulerOutput
    - Update internal state from StepRunnerOutput
    """
    def initialize(self, od_config: OmniDiffusionConfig):
        self.od_config = od_config

        # Step-level scheduling state
        self._request_states: dict[str, DiffusionRequestState] = {}
        self._schedule_states: dict[str, DiffusionScheduleState] = {}
        self._waiting: deque[str] = deque()
        self._running: list[str] = []
        self._finished_req_ids: set[str] = set()
        self._preempted_req_ids: set[str] = set()
        self._step_id: int = 0
        self._max_batch_size: int = int(getattr(od_config, "max_batch_size", 2))

    # ======================================================================
    # Step-level scheduling API
    # ======================================================================

    def add_request(self, request: OmniDiffusionRequest) -> str:
        """Add a request to the scheduler without waiting for completion."""
        req_id = request.request_id
        assert req_id is not None, "Request must have a request_id"
        if req_id in self._request_states:
            raise ValueError(f"Duplicate request_id: {req_id}")

        state = DiffusionRequestState(req_id=req_id, req=request)
        self._request_states[req_id] = state
        self._schedule_states[req_id] = DiffusionScheduleState.WAITING
        self._waiting.append(req_id)
        logger.debug("Scheduler add_request: %s (waiting=%d)", req_id, len(self._waiting))
        return req_id

    def add_requests(self, requests: list[OmniDiffusionRequest]) -> list[str]:
        return [self.add_request(req) for req in requests]

    def schedule(self) -> StepSchedulerOutput:
        """Schedule a single diffusion step.

        Returns a StepSchedulerOutput containing the active request states.
        """
        finished_req_ids = set(self._finished_req_ids)
        self._finished_req_ids.clear()
        preempted_req_ids = set(self._preempted_req_ids)
        self._preempted_req_ids.clear()

        # Fill running slots from waiting queue
        while self._waiting and len(self._running) < self._max_batch_size:
            req_id = self._waiting.popleft()
            if req_id not in self._request_states:
                continue
            self._schedule_states[req_id] = DiffusionScheduleState.RUNNING
            self._running.append(req_id)

        # Filter out completed requests from running
        running_states: list[DiffusionRequestState] = []
        still_running: list[str] = []
        for req_id in self._running:
            state = self._request_states.get(req_id)
            if state is None:
                continue
            if state.is_complete:
                self._mark_finished(req_id)
                finished_req_ids.add(req_id)
                continue
            running_states.append(state)
            still_running.append(req_id)
        self._running = still_running

        scheduler_output = StepSchedulerOutput(
            step_id=self._step_id,
            req_stats=running_states,
            finished_req_ids=finished_req_ids,
            preempted_req_ids=preempted_req_ids,
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._waiting),
        )
        self._step_id += 1
        return scheduler_output

    def update_from_step(self, runner_output: StepRunnerOutput) -> set[str]:
        """Update request states based on the runner output.

        NOTE: We intentionally do NOT store latents in the scheduler's state.
        The latents are kept in the Worker's _request_state_cache to avoid
        expensive IPC serialization of large tensors on every step.
        """
        finished_this_step: set[str] = set()
        for out in runner_output.step_outputs:
            state = self._request_states.get(out.req_id)
            if state is None:
                continue
            # Only update metadata, NOT latents (kept in Worker cache)
            state.step_index = out.step_index + 1
            state.timestep = out.timestep
            if out.is_complete:
                finished_this_step.add(out.req_id)

        # Record decoded outputs
        for req_id, decoded in runner_output.decoded.items():
            state = self._request_states.get(req_id)
            if state is not None:
                state.req.output = decoded
                finished_this_step.add(req_id)

        for req_id in finished_this_step:
            self._mark_finished(req_id)

        if finished_this_step:
            self._finished_req_ids |= finished_this_step
        return finished_this_step

    def abort_request(self, req_id: str) -> bool:
        """Abort a request and mark it finished."""
        if req_id not in self._request_states:
            return False
        self._mark_finished(req_id)
        self._finished_req_ids.add(req_id)
        return True

    def has_requests(self) -> bool:
        """Return True if there are unfinished requests."""
        return bool(self._waiting or self._running)

    def get_request_state(self, req_id: str) -> DiffusionRequestState | None:
        return self._request_states.get(req_id)

    def pop_request_state(self, req_id: str) -> DiffusionRequestState | None:
        state = self._request_states.pop(req_id, None)
        self._schedule_states.pop(req_id, None)
        return state

    def preempt_request(self, req_id: str) -> bool:
        """Preempt a running request and move it back to waiting."""
        if req_id not in self._request_states:
            return False
        if req_id in self._running:
            self._running.remove(req_id)
            self._waiting.appendleft(req_id)
            self._schedule_states[req_id] = DiffusionScheduleState.SUSPENDED
            self._preempted_req_ids.add(req_id)
            return True
        return False

    def _mark_finished(self, req_id: str) -> None:
        self._schedule_states[req_id] = DiffusionScheduleState.FINISHED
        if req_id in self._running:
            self._running.remove(req_id)
        try:
            self._waiting.remove(req_id)
        except ValueError:
            pass
