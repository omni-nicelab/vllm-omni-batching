# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import queue
import time
from logging import DEBUG
from typing import Any

import PIL.Image
from vllm.logger import init_logger

from vllm_omni.diffusion.core.outputs import DiffusionCoreOutput, DiffusionRequestType
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.registry import (
    get_diffusion_post_process_func,
    get_diffusion_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.scheduler import DiffusionStepScheduler
from vllm_omni.diffusion.worker.step_batch import StepRunnerOutput

logger = init_logger(__name__)


class DiffusionCore:
    """Diffusion core that schedules step-level execution."""

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        executor_class: type[DiffusionExecutor],
    ) -> None:
        self.od_config = od_config
        self.executor = executor_class(od_config)

        self.step_scheduler = DiffusionStepScheduler()
        self.step_scheduler.initialize(od_config)

        self.pre_process_func = get_diffusion_pre_process_func(od_config)
        self.post_process_func = get_diffusion_post_process_func(od_config)

        self.input_queue: queue.Queue[tuple[DiffusionRequestType, Any]] = queue.Queue()
        self.output_queue: queue.Queue[DiffusionCoreOutput] = queue.Queue()

    def add_request(self, request: OmniDiffusionRequest) -> None:
        if self.pre_process_func is not None:
            request = self.pre_process_func([request])[0]
        self.step_scheduler.add_request(request)

    def enqueue_request(self, request: OmniDiffusionRequest) -> None:
        self.input_queue.put_nowait((DiffusionRequestType.ADD, request))

    def enqueue_abort(self, request_ids: list[str]) -> None:
        self.input_queue.put_nowait((DiffusionRequestType.ABORT, request_ids))

    def step(self) -> tuple[list[DiffusionCoreOutput], bool]:
        sched_output = self.step_scheduler.schedule()
        if not sched_output.req_states:
            return [], False

        runner_output = self.executor.execute_step(sched_output)
        finished_ids = self.step_scheduler.update_from_step(runner_output)
        outputs = self._build_outputs(finished_ids, runner_output)
        return outputs, True

    def run_busy_loop(self) -> None:
        while True:
            self._process_input_queue()
            self._process_engine_step()

    def has_unfinished_requests(self) -> bool:
        return self.step_scheduler.has_requests()

    def shutdown(self) -> None:
        self.executor.shutdown()

    def _process_input_queue(self) -> None:
        waited = False
        while not self.step_scheduler.has_requests():
            if self.input_queue.empty():
                if logger.isEnabledFor(DEBUG):
                    logger.debug("DiffusionCore waiting for work.")
                    waited = True
            msg = self.input_queue.get()
            self._handle_client_request(*msg)

        if waited:
            logger.debug("DiffusionCore loop active.")

        while not self.input_queue.empty():
            msg = self.input_queue.get_nowait()
            self._handle_client_request(*msg)

    def _process_engine_step(self) -> bool:
        outputs, executed = self.step()
        for output in outputs:
            self.output_queue.put_nowait(output)
        if not executed and self.step_scheduler.has_requests():
            time.sleep(0.001)
        return executed

    def _handle_client_request(self, request_type: DiffusionRequestType, request: Any) -> None:
        if request_type == DiffusionRequestType.ADD:
            self.add_request(request)
        elif request_type == DiffusionRequestType.ABORT:
            for req_id in request:
                self.step_scheduler.abort_request(req_id)
        else:
            logger.error("Unrecognized DiffusionCore request type: %s", request_type)

    def _build_outputs(
        self,
        finished_ids: set[str],
        runner_output: StepRunnerOutput,
    ) -> list[DiffusionCoreOutput]:
        outputs: list[DiffusionCoreOutput] = []
        for req_id in finished_ids:
            state = self.step_scheduler.pop_request_state(req_id)
            if state is None:
                continue

            raw_output = runner_output.decoded.get(req_id)
            if raw_output is None:
                raw_output = state.req.output

            images = self._format_images(raw_output)
            metrics = {}
            if state.req.trajectory_timesteps is not None:
                metrics["trajectory_timesteps"] = state.req.trajectory_timesteps

            outputs.append(
                DiffusionCoreOutput(
                    request_id=req_id,
                    finished=True,
                    images=images,
                    latents=state.latents,
                    metrics=metrics,
                )
            )
        return outputs

    def _format_images(self, raw_output: Any) -> list[PIL.Image.Image] | None:
        if raw_output is None:
            return None

        processed = raw_output
        if self.post_process_func is not None:
            if not isinstance(raw_output, list) and not isinstance(raw_output, PIL.Image.Image):
                processed = self.post_process_func(raw_output)

        if isinstance(processed, list):
            return processed
        return [processed]
