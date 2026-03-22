# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import _BaseScheduler
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionRequestState,
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    NewRequestData,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import RunnerOutput

logger = init_logger(__name__)


@dataclass
class _StepProgress:
    current_step: int
    total_steps: int


class StepScheduler(_BaseScheduler):
    """Placeholder scheduler that advances a request one denoise step per update."""

    def __init__(self) -> None:
        super().__init__()
        self._request_progress: dict[str, _StepProgress] = {}

    def _reset_scheduler_state(self) -> None:
        self._request_progress.clear()

    def add_request(self, request: OmniDiffusionRequest) -> str:
        sched_req_id = self._make_sched_req_id(request)
        total_steps = self._get_total_steps(request)
        if total_steps <= 0:
            raise ValueError(f"Diffusion request {sched_req_id} must have positive total_steps, got {total_steps}")

        current_step = request.sampling_params.step_index or 0
        if current_step < 0 or current_step >= total_steps:
            raise ValueError(
                f"Diffusion request {sched_req_id} has invalid initial step_index {current_step} "
                f"for total_steps={total_steps}"
            )

        request.sampling_params.step_index = current_step
        state = DiffusionRequestState(sched_req_id=sched_req_id, req=request)
        self._request_states[sched_req_id] = state
        self._register_request_ids(request.request_ids, sched_req_id)
        self._request_progress[sched_req_id] = _StepProgress(current_step=current_step, total_steps=total_steps)
        self._waiting.append(sched_req_id)
        logger.debug(
            "StepScheduler add_request: %s (step=%d/%d, waiting=%d)",
            sched_req_id,
            current_step,
            total_steps,
            len(self._waiting),
        )
        return sched_req_id

    def schedule(self) -> DiffusionSchedulerOutput:
        scheduled_new_reqs: list[NewRequestData] = []
        scheduled_cached_req_ids: list[str] = []

        # First, schedule the RUNNING request(s)
        for sched_req_id in self._running:
            state = self._request_states.get(sched_req_id)
            if state is not None:
                scheduled_cached_req_ids.append(sched_req_id)

        # Second, schedule WAITING requests while capacity remains.
        while self._waiting and len(self._running) < self._max_batch_size:
            sched_req_id = self._waiting.popleft()
            state = self._request_states.get(sched_req_id)
            if state is None:
                continue
            was_new_request = state.status == DiffusionRequestStatus.WAITING
            state.status = DiffusionRequestStatus.RUNNING
            self._running.append(sched_req_id)
            if was_new_request:
                scheduled_new_reqs.append(NewRequestData.from_state(state))
            else:
                scheduled_cached_req_ids.append(sched_req_id)

        scheduler_output = DiffusionSchedulerOutput(
            step_id=self._step_id,
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=CachedRequestData(sched_req_ids=scheduled_cached_req_ids),
            finished_req_ids=set(self._finished_req_ids),
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._waiting),
        )

        # update after schedule
        self._step_id += 1
        self._finished_req_ids.clear()
        return scheduler_output

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: RunnerOutput) -> set[str]:
        scheduled_req_ids = sched_output.scheduled_req_ids
        if not scheduled_req_ids:
            return set()

        # A scheduled request may be aborted after schedule() but before
        # update_from_output() processes the runner output. It is already
        # marked finished at that point, but we still need to surface its id
        # in this update so the engine can observe the terminal state.
        finished_req_ids = {
            sched_req_id for sched_req_id in scheduled_req_ids if sched_req_id in self._finished_req_ids
        }
        terminal_statuses: dict[str, DiffusionRequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}
        output_error = output.result.error if output.result is not None else None
        for sched_req_id in scheduled_req_ids:
            state = self._request_states.get(sched_req_id)
            progress = self._request_progress.get(sched_req_id)
            if state is None or progress is None or state.is_finished():
                continue

            if output_error is not None:
                terminal_statuses[sched_req_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[sched_req_id] = output_error
                continue

            if output.step_index is None:
                logger.warning(
                    "Received RunnerOutput with no step_index for request %s, treating as error",
                    sched_req_id,
                )
                terminal_statuses[sched_req_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[sched_req_id] = "Missing step_index in RunnerOutput"
                continue

            # We assume that the decoding stage is executed immediately after the denoising stage completes.
            progress.current_step = output.step_index
            state.req.sampling_params.step_index = output.step_index
            if output.finished:
                terminal_statuses[sched_req_id] = DiffusionRequestStatus.FINISHED_COMPLETED
                terminal_errors[sched_req_id] = None
            else:
                state.error = None

        finished_req_ids |= self._finish_requests(terminal_statuses, terminal_errors)
        return finished_req_ids

    def _pop_extra_request_state(self, sched_req_id: str) -> None:
        self._request_progress.pop(sched_req_id, None)

    def _get_total_steps(self, request: OmniDiffusionRequest) -> int:
        sampling = request.sampling_params

        if sampling.timesteps is not None:
            return self._sequence_length(sampling.timesteps)
        if sampling.sigmas is not None:
            return len(sampling.sigmas)
        return int(sampling.num_inference_steps)

    @staticmethod
    def _sequence_length(values: Any) -> int:
        ndim = getattr(values, "ndim", None)
        if ndim == 0:
            return 1

        shape = getattr(values, "shape", None)
        if shape is not None:
            return int(shape[0])

        return len(values)
