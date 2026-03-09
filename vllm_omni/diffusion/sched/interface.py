# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.utils import RunnerOutput


class DiffusionRequestStatus(enum.IntEnum):
    """Request status tracked by diffusion scheduler."""

    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()

    # if any status is after FINISHED_COMPLETED, it is considered finished
    FINISHED_COMPLETED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_ERROR = enum.auto()

    @staticmethod
    def is_finished(status: DiffusionRequestStatus) -> bool:
        return status >= DiffusionRequestStatus.FINISHED_COMPLETED


@dataclass
class DiffusionRequestState:
    """Scheduler-owned state for one queued OmniDiffusionRequest."""

    req_id: str
    req: OmniDiffusionRequest
    status: DiffusionRequestStatus = DiffusionRequestStatus.WAITING
    error: str | None = None


@dataclass
class DiffusionSchedulerOutput:
    """Output of a single scheduling cycle."""

    step_id: int
    req_states: list[DiffusionRequestState]
    finished_req_ids: set[str]
    num_running_reqs: int
    num_waiting_reqs: int


class SchedulerInterface(ABC):
    """Abstract lifecycle contract for diffusion schedulers."""

    @abstractmethod
    def initialize(self, od_config: OmniDiffusionConfig) -> None:
        """Initialize or reset scheduler state."""

    @abstractmethod
    def add_request(self, request: OmniDiffusionRequest) -> str:
        """Add a request and return the scheduler-owned request id."""

    @abstractmethod
    def schedule(self) -> DiffusionSchedulerOutput:
        """Run one scheduling cycle."""

    @abstractmethod
    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: RunnerOutput) -> set[str]:
        """Update scheduler state from executor output."""

    @abstractmethod
    def abort_request(self, req_id: str) -> bool:
        """Abort a queued or running request."""

    @abstractmethod
    def has_requests(self) -> bool:
        """Return whether the scheduler still owns runnable requests."""

    @abstractmethod
    def get_request_state(self, req_id: str) -> DiffusionRequestState | None:
        """Return request state if present."""

    @abstractmethod
    def pop_request_state(self, req_id: str) -> DiffusionRequestState | None:
        """Remove and return request state if present."""

    @abstractmethod
    def preempt_request(self, req_id: str) -> bool:
        """Preempt a running request back to waiting."""

    @abstractmethod
    def finish_request(self, req_id: str, status: DiffusionRequestStatus) -> None:
        """Mark a request finished."""

    @abstractmethod
    def close(self) -> None:
        """Release scheduler-owned state."""
