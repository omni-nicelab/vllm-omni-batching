# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import deque

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.sched.interface import (
    DiffusionRequestState,
    DiffusionRequestStatus,
    SchedulerInterface,
)


class _BaseScheduler(SchedulerInterface):
    """Shared queue/state bookkeeping for diffusion schedulers."""

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
        self._reset_scheduler_state()

    def abort_request(self, req_id: str) -> bool:
        if req_id not in self._request_states:
            return False
        self.finish_request(req_id, DiffusionRequestStatus.FINISHED_ABORTED)
        return True

    def has_requests(self) -> bool:
        return bool(self._waiting or self._running)

    def get_request_state(self, req_id: str) -> DiffusionRequestState | None:
        return self._request_states.get(req_id)

    def pop_request_state(self, req_id: str) -> DiffusionRequestState | None:
        self._pop_extra_request_state(req_id)
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
        self._finish_requests({req_id: status})

    def close(self) -> None:
        self._request_states.clear()
        self._waiting.clear()
        self._running.clear()
        self._finished_req_ids.clear()
        self._reset_scheduler_state()

    def _finish_requests(
        self,
        statuses: dict[str, DiffusionRequestStatus],
        errors: dict[str, str | None] | None = None,
    ) -> set[str]:
        if not statuses:
            return set()

        finished_req_ids: set[str] = set()
        running_to_remove: set[str] = set()
        waiting_to_remove: set[str] = set()

        for req_id, status in statuses.items():
            assert DiffusionRequestStatus.is_finished(status)
            state = self._request_states.get(req_id)
            if state is None or state.is_finished():
                continue

            finished_req_ids.add(req_id)
            if req_id in self._running:
                running_to_remove.add(req_id)
            if req_id in self._waiting:
                waiting_to_remove.add(req_id)

        if running_to_remove:
            self._running = [req_id for req_id in self._running if req_id not in running_to_remove]
        if waiting_to_remove:
            self._waiting = deque(req_id for req_id in self._waiting if req_id not in waiting_to_remove)

        for req_id in finished_req_ids:
            state = self._request_states[req_id]
            status = statuses[req_id]
            state.status = status
            if status == DiffusionRequestStatus.FINISHED_ERROR:
                state.error = None if errors is None else errors.get(req_id)
            else:
                state.error = None

        self._finished_req_ids |= finished_req_ids
        return finished_req_ids

    def _reset_scheduler_state(self) -> None:
        """Reset subclass-owned state during initialize()/close()."""

    def _pop_extra_request_state(self, req_id: str) -> None:
        """Remove subclass-owned per-request state before popping request state."""
