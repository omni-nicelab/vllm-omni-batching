# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque

from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.request import DiffusionRequestState, DiffusionRequestStatus, OmniDiffusionRequest
from vllm_omni.diffusion.worker.step_batch import StepRunnerOutput, StepSchedulerOutput

logger = init_logger(__name__)


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
        # req_id -> DiffusionRequestState
        self._request_states: dict[str, DiffusionRequestState] = {}
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
        self._waiting.append(req_id)
        logger.debug("Scheduler add_request: %s (waiting=%d)", req_id, len(self._waiting))
        return req_id

    def add_requests(self, requests: list[OmniDiffusionRequest]) -> list[str]:
        return [self.add_request(req) for req in requests]

    def schedule(self) -> StepSchedulerOutput:
        """Schedule a single diffusion step.

        Returns a StepSchedulerOutput containing the active request states.
        """
        # TODO: preempt schedule

        # Fill running slots from waiting queue
        while self._waiting and len(self._running) < self._max_batch_size:
            req_id = self._waiting.popleft()
            if req_id not in self._request_states:
                continue
            self._request_states[req_id].req.status = DiffusionRequestStatus.RUNNING
            self._running.append(req_id)

        # Build output with current running requests
        running_states: list[DiffusionRequestState] = []
        for req_id in self._running:
            state = self._request_states.get(req_id)
            if state is not None:
                running_states.append(state)

        scheduler_output = StepSchedulerOutput(
            step_id=self._step_id,
            req_states=running_states,
            finished_req_ids=self._finished_req_ids,
            preempted_req_ids=self._preempted_req_ids,
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._waiting),
        )

        # update after schedule
        self._step_id += 1
        self._finished_req_ids = set()
        self._preempted_req_ids = set()
        return scheduler_output

    def update_from_step(self, sched_output: StepSchedulerOutput, runner_output: StepRunnerOutput) -> set[str]:
        """Update request states based on the runner output.

        NOTE: We intentionally do NOT store latents in the scheduler's state.
        The latents are kept in the Worker's _request_state_cache to avoid
        expensive IPC serialization of large tensors on every step.
        """
        request_states = self._request_states
        scheduled_req_ids = {s.req_id for s in sched_output.req_states}

        # Abnormal finish ids: runner returned results for non-scheduled req_ids,
        # or the scheduler no longer has state for the req_id.
        unexpected_finished_ids: set[str] = set()

        # StepOutput advances denoising only; "normal completion" is determined by decoded
        # (Currently, the decode stage and the last denoise step are executed in the same scheduling cycle).
        for out in runner_output.step_outputs:
            if out.req_id not in scheduled_req_ids:
                unexpected_finished_ids.add(out.req_id)
            state = request_states.get(out.req_id)
            if state is None:
                unexpected_finished_ids.add(out.req_id)
                continue
            # Only update metadata, NOT latents (kept in Worker cache)
            state.step_index = out.step_index + 1
            state.timestep = out.timestep

        completed_req_ids: set[str] = set()
        for req_id, decoded in runner_output.decoded.items():
            if req_id not in scheduled_req_ids:
                unexpected_finished_ids.add(req_id)
            state = request_states.get(req_id)
            if state is None:
                unexpected_finished_ids.add(req_id)
                continue
            state.req.output = decoded
            state.req.status = DiffusionRequestStatus.FINISHED_COMPLETED
            completed_req_ids.add(req_id)

        # clean _running (drop completed + drop missing).
        if completed_req_ids or unexpected_finished_ids:
            new_running: list[str] = []
            for req_id in self._running:
                if req_id in completed_req_ids:
                    continue
                if req_id not in request_states:
                    unexpected_finished_ids.add(req_id)
                    continue
                new_running.append(req_id)
            self._running = new_running

        # Be defensive: a finished request should not be schedulable again.
        for req_id in completed_req_ids:
            try:
                self._waiting.remove(req_id)
            except ValueError:
                pass

        if unexpected_finished_ids:
            self._finished_req_ids |= unexpected_finished_ids
        return completed_req_ids

    def abort_request(self, req_id: str) -> bool:
        """Abort a request and mark it finished."""
        if req_id not in self._request_states:
            return False
        self.finish_request(req_id, DiffusionRequestStatus.FINISHED_ABORTED)
        self._finished_req_ids.add(req_id)
        return True

    def has_requests(self) -> bool:
        """Return True if there are unfinished requests."""
        return bool(self._waiting or self._running)

    def get_request_state(self, req_id: str) -> DiffusionRequestState | None:
        return self._request_states.get(req_id)

    def pop_request_state(self, req_id: str) -> DiffusionRequestState | None:
        state = self._request_states.pop(req_id, None)
        return state

    def preempt_request(self, req_id: str) -> bool:
        """Preempt a running request and move it back to waiting."""
        if req_id not in self._request_states:
            return False
        if req_id in self._running:
            self._running.remove(req_id)
            self._waiting.appendleft(req_id)
            self._request_states[req_id].req.status = DiffusionRequestStatus.PREEMPTED
            self._preempted_req_ids.add(req_id)
            return True
        return False

    def finish_request(self, req_id: str, status: DiffusionRequestStatus) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert DiffusionRequestStatus.is_finished(status)
        self._request_states[req_id].req.status = status
        if req_id in self._running:
            self._running.remove(req_id)
        try:
            self._waiting.remove(req_id)
        except ValueError:
            pass
