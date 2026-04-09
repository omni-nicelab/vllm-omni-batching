# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations
import queue
import threading
import time
from collections.abc import Iterable
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any

import numpy as np
import PIL.Image
import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import (
    DiffusionOutput,
    DiffusionRequestAbortedError,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.registry import (
    DiffusionModelRegistry,
    get_diffusion_post_process_func,
    get_diffusion_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import RequestScheduler, SchedulerInterface, StepScheduler
from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus
from vllm_omni.diffusion.worker.utils import RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


@dataclass(frozen=True)
class SubmittedDiffusionRequest:
    """Request metadata tracked across enqueue/wait/postprocess."""

    sched_req_id: str
    request: OmniDiffusionRequest
    start_time: float
    preprocess_time: float
    exec_start_time: float

    @property
    def request_handle(self) -> str:
        return self.sched_req_id


@dataclass(frozen=True)
class _QueuedDiffusionRequest:
    request_handle: str
    request: OmniDiffusionRequest


def supports_image_input(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_image_input", False))


def supports_audio_input(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_audio_input", False))


def image_color_format(model_class_name: str) -> str:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    return getattr(model_cls, "color_format", "RGB")


def supports_audio_output(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_audio_output", False))


class DiffusionEngine:
    """The diffusion engine for vLLM-Omni diffusion models."""

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        scheduler: SchedulerInterface | None = None,
    ):
        """Initialize the diffusion engine.

        Args:
            config: The configuration for the diffusion engine.
        """
        self.od_config = od_config

        self.post_process_func = get_diffusion_post_process_func(od_config)
        self.pre_process_func = get_diffusion_pre_process_func(od_config)

        executor_class = DiffusionExecutor.get_class(od_config)
        self.executor = executor_class(od_config)
        self.step_execution = bool(getattr(od_config, "step_execution", False))
        self.scheduler: SchedulerInterface = scheduler or (
            StepScheduler() if self.step_execution else RequestScheduler()
        )
        self.scheduler.initialize(od_config)
        self._state_lock = threading.RLock()
        self._rpc_lock = threading.RLock()
        self._input_queue: queue.Queue[_QueuedDiffusionRequest] = queue.Queue()
        self._output_queue: queue.Queue[str] = queue.Queue()
        self._results_map: dict[str, Future[DiffusionOutput]] = {}
        self._handle_to_sched_req_id: dict[str, str] = {}
        self._sched_req_id_to_handle: dict[str, str] = {}
        self._request_id_to_handle: dict[str, str] = {}
        self._handle_to_request_ids: dict[str, tuple[str, ...]] = {}
        self._cancelled_handles: set[str] = set()
        self.abort_queue: queue.Queue[str] = queue.Queue()
        self._shutdown_event = threading.Event()
        self.execute_fn = self.executor.execute_step if self.step_execution else self.executor.execute_request
        self._configure_scheduler_limits()

        self._loop_thread = threading.Thread(
            target=self._run_background_loop,
            name="DiffusionEngineLoop",
            daemon=True,
        )
        self._loop_thread.start()

        try:
            self._dummy_run()
        except Exception as e:
            logger.error(f"Dummy run failed: {e}")
            self.close()
            raise e

    def step(self, request: OmniDiffusionRequest) -> list[OmniRequestOutput]:
        submitted = self.submit_request(request)
        output = self.wait_for_submitted_request(submitted)

        if output.aborted:
            raise DiffusionRequestAbortedError(output.abort_message or "Diffusion request aborted.")
        if output.error:
            raise RuntimeError(f"{output.error}")
        logger.info("Generation completed successfully.")
        return self.build_outputs_from_submitted_request(submitted, output)

    def prepare_request(self, request: OmniDiffusionRequest) -> tuple[OmniDiffusionRequest, float]:
        preprocess_time = 0.0
        if self.pre_process_func is not None:
            preprocess_start_time = time.perf_counter()
            request = self.pre_process_func(request)
            preprocess_time = time.perf_counter() - preprocess_start_time
            logger.info(f"Pre-processing completed in {preprocess_time:.4f} seconds")
        return request, preprocess_time

    def submit_request(self, request: OmniDiffusionRequest) -> SubmittedDiffusionRequest:
        request_start_time = time.perf_counter()
        request, preprocess_time = self.prepare_request(request)
        exec_start_time = time.perf_counter()
        sched_req_id = self.add_request(request)
        return SubmittedDiffusionRequest(
            sched_req_id=sched_req_id,
            request=request,
            start_time=request_start_time,
            preprocess_time=preprocess_time,
            exec_start_time=exec_start_time,
        )

    def wait_for_submitted_request(
        self,
        submitted: SubmittedDiffusionRequest,
        timeout: float = 30,
    ) -> DiffusionOutput:
        if self._has_background_loop():
            return self.get_result(submitted.sched_req_id, timeout=timeout)
        return self._drive_request_until_complete(submitted.sched_req_id, timeout=timeout)

    def build_outputs_from_submitted_request(
        self,
        submitted: SubmittedDiffusionRequest,
        output: DiffusionOutput,
    ) -> list[OmniRequestOutput]:
        exec_total_time = time.perf_counter() - submitted.exec_start_time
        return self._build_omni_request_outputs(
            submitted.request,
            output,
            start_time=submitted.start_time,
            preprocess_time=submitted.preprocess_time,
            exec_total_time=exec_total_time,
        )

    def _build_omni_request_outputs(
        self,
        request: OmniDiffusionRequest,
        output: DiffusionOutput,
        *,
        start_time: float,
        preprocess_time: float,
        exec_total_time: float,
    ) -> list[OmniRequestOutput]:
        if output.output is None:
            logger.warning("Output is None, returning empty OmniRequestOutput")
            return [
                OmniRequestOutput.from_diffusion(
                    request_id=request.request_ids[i] if i < len(request.request_ids) else "",
                    images=[],
                    prompt=prompt,
                    metrics={},
                    latents=None,
                )
                for i, prompt in enumerate(request.prompts)
            ]

        output_data = output.output
        if (
            self.od_config.enable_cpu_offload
            and isinstance(output_data, torch.Tensor)
            and output_data.device.type != "cpu"
        ):
            output_data = output_data.cpu()

        postprocess_start_time = time.perf_counter()
        outputs = self.post_process_func(output_data) if self.post_process_func is not None else output_data
        audio_payload = None
        model_audio_sample_rate = None
        model_fps = None
        if isinstance(outputs, dict):
            audio_payload = outputs.get("audio")
            model_audio_sample_rate = outputs.get("audio_sample_rate")
            model_fps = outputs.get("fps")
            outputs = outputs.get("video", outputs)
        postprocess_time = time.perf_counter() - postprocess_start_time
        logger.info(f"Post-processing completed in {postprocess_time:.4f} seconds")

        step_total_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "DiffusionEngine.step breakdown: preprocess=%.2f ms, "
            "add_req_and_wait=%.2f ms, postprocess=%.2f ms, total=%.2f ms",
            preprocess_time * 1000,
            exec_total_time * 1000,
            postprocess_time * 1000,
            step_total_ms,
        )

        if not isinstance(outputs, list):
            outputs = [outputs] if outputs is not None else []

        metrics = {
            "preprocess_time_ms": preprocess_time * 1000,
            "diffusion_engine_exec_time_ms": (time.perf_counter() - start_time) * 1000,
            "diffusion_engine_total_time_ms": exec_total_time * 1000,
            "image_num": int(request.sampling_params.num_outputs_per_prompt),
            "resolution": int(request.sampling_params.resolution),
            "postprocess_time_ms": postprocess_time * 1000,
        }
        if self.pre_process_func is not None:
            metrics["preprocessing_time_ms"] = preprocess_time * 1000

        if len(request.prompts) == 1:
            prompt = request.prompts[0]
            request_id = request.request_ids[0] if request.request_ids else ""

            if supports_audio_output(self.od_config.model_class_name):
                request_audio_payload = outputs[0] if len(outputs) == 1 else outputs
                return [
                    OmniRequestOutput.from_diffusion(
                        request_id=request_id,
                        images=[],
                        prompt=prompt,
                        metrics=metrics,
                        latents=output.trajectory_latents,
                        multimodal_output={"audio": request_audio_payload},
                        final_output_type="audio",
                        stage_durations=output.stage_durations,
                        peak_memory_mb=output.peak_memory_mb,
                    ),
                ]

            mm_output = {}
            if audio_payload is not None:
                mm_output["audio"] = audio_payload
            if model_audio_sample_rate is not None:
                mm_output["audio_sample_rate"] = model_audio_sample_rate
            if model_fps is not None:
                mm_output["fps"] = model_fps
            return [
                OmniRequestOutput.from_diffusion(
                    request_id=request_id,
                    images=outputs,
                    prompt=prompt,
                    metrics=metrics,
                    latents=output.trajectory_latents,
                    trajectory_latents=output.trajectory_latents,
                    trajectory_timesteps=output.trajectory_timesteps,
                    trajectory_log_probs=output.trajectory_log_probs,
                    trajectory_decoded=output.trajectory_decoded,
                    custom_output=output.custom_output or {},
                    multimodal_output=mm_output,
                    stage_durations=output.stage_durations,
                    peak_memory_mb=output.peak_memory_mb,
                ),
            ]

        results = []
        output_idx = 0
        for i, prompt in enumerate(request.prompts):
            request_id = request.request_ids[i] if i < len(request.request_ids) else ""
            num_outputs = request.sampling_params.num_outputs_per_prompt
            start_idx = output_idx
            end_idx = start_idx + num_outputs
            request_outputs = outputs[start_idx:end_idx] if output_idx < len(outputs) else []
            output_idx = end_idx

            if supports_audio_output(self.od_config.model_class_name):
                request_audio_payload = request_outputs[0] if len(request_outputs) == 1 else request_outputs
                results.append(
                    OmniRequestOutput.from_diffusion(
                        request_id=request_id,
                        images=[],
                        prompt=prompt,
                        metrics=metrics,
                        latents=output.trajectory_latents,
                        multimodal_output={"audio": request_audio_payload},
                        final_output_type="audio",
                        stage_durations=output.stage_durations,
                        peak_memory_mb=output.peak_memory_mb,
                    ),
                )
                continue

            mm_output = {}
            if audio_payload is not None:
                sliced_audio = audio_payload
                if isinstance(audio_payload, (list, tuple)):
                    sliced_audio = audio_payload[start_idx:end_idx]
                    if len(sliced_audio) == 1:
                        sliced_audio = sliced_audio[0]
                elif hasattr(audio_payload, "shape") and getattr(audio_payload, "shape", None) is not None:
                    if len(audio_payload.shape) > 0 and audio_payload.shape[0] >= end_idx:
                        sliced_audio = audio_payload[start_idx:end_idx]
                        if num_outputs == 1:
                            sliced_audio = sliced_audio[0]
                mm_output["audio"] = sliced_audio
            if model_audio_sample_rate is not None:
                mm_output["audio_sample_rate"] = model_audio_sample_rate
            if model_fps is not None:
                mm_output["fps"] = model_fps
            results.append(
                OmniRequestOutput.from_diffusion(
                    request_id=request_id,
                    images=request_outputs,
                    prompt=prompt,
                    metrics=metrics,
                    latents=output.trajectory_latents,
                    trajectory_latents=output.trajectory_latents,
                    trajectory_timesteps=output.trajectory_timesteps,
                    trajectory_log_probs=output.trajectory_log_probs,
                    trajectory_decoded=output.trajectory_decoded,
                    custom_output=output.custom_output or {},
                    multimodal_output=mm_output,
                    stage_durations=output.stage_durations,
                    peak_memory_mb=output.peak_memory_mb,
                ),
            )
        return results

    def _run_background_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                progressed = self._run_engine_cycle()
            except Exception:
                logger.exception("Diffusion engine background loop failed")
                progressed = False
            if not progressed:
                self._shutdown_event.wait(0.01)

    def _run_engine_cycle(self) -> bool:
        progressed = self._process_aborts_queue()
        progressed = self._process_input_queue() or progressed
        progressed = self._process_aborts_queue() or progressed

        with self._state_lock:
            if not self.scheduler.has_requests():
                return progressed
            sched_output = self.scheduler.schedule()

        if sched_output.is_empty:
            finished_req_ids = self._normalize_finished_ids(getattr(sched_output, "finished_req_ids", set()))
            self._handle_finished_requests(finished_req_ids, None)
            return progressed or bool(finished_req_ids)

        model_output = self._execute_scheduler_output(sched_output)
        self._process_aborts_queue()
        with self._state_lock:
            finished_req_ids = self.scheduler.update_from_output(sched_output, model_output)
        self._handle_finished_requests(finished_req_ids, model_output)
        return True

    def _execute_scheduler_output(self, sched_output) -> RunnerOutput:
        with self._rpc_lock:
            try:
                return self.execute_fn(sched_output)
            except Exception as exc:
                scheduled_req_ids = list(sched_output.scheduled_req_ids)
                sched_req_id = scheduled_req_ids[0] if scheduled_req_ids else ""
                logger.error("Execution failed for diffusion request %s", sched_req_id, exc_info=True)
                if len(scheduled_req_ids) <= 1:
                    return RunnerOutput(
                        req_id=sched_req_id,
                        step_index=None,
                        finished=True,
                        result=DiffusionOutput(error=str(exc)),
                    )
                return RunnerOutput(
                    req_id=scheduled_req_ids,
                    step_index=[None] * len(scheduled_req_ids),
                    finished=[True] * len(scheduled_req_ids),
                    result=[DiffusionOutput(error=str(exc)) for _ in scheduled_req_ids],
                )

    def _handle_finished_requests(self, finished_ids, runner_output: RunnerOutput | None):
        """Resolve finished requests to waiting callers."""
        with self._state_lock:
            for rid in self._normalize_finished_ids(finished_ids):
                state = self.scheduler.get_request_state(rid)
                popped_state = self.scheduler.pop_request_state(rid)
                state = state or popped_state
                request_handle = self._sched_req_id_to_handle.get(rid)
                if state is None:
                    logger.warning("Finished diffusion request %s has no scheduler state", rid)
                    if request_handle is not None:
                        self._clear_request_state_locked(request_handle)
                    continue

                out = self._resolve_finished_output(rid, state, runner_output)
                self._finalize_request_locked(request_handle or rid, out)
                logger.info("Resolved diffusion request %s", rid)

    def _resolve_finished_output(
        self,
        sched_req_id: str,
        state,
        runner_output: RunnerOutput | None,
    ) -> DiffusionOutput:
        if state.status == DiffusionRequestStatus.FINISHED_ABORTED:
            request_id = state.req.request_ids[0] if state.req.request_ids else sched_req_id
            return DiffusionOutput(
                aborted=True,
                abort_message=f"Request {request_id} aborted.",
            )

        if runner_output is not None:
            req_output = runner_output.get_req_output(sched_req_id)
            if req_output is not None:
                if req_output.result is not None:
                    return req_output.result
                if getattr(req_output, "finished", False):
                    return DiffusionOutput(error="Diffusion execution finished without a final output.")

        if state.status == DiffusionRequestStatus.FINISHED_ERROR and state.error:
            return DiffusionOutput(error=state.error)
        return DiffusionOutput(error="No output")

    def _configure_scheduler_limits(self) -> None:
        if not self.step_execution and hasattr(self.scheduler, "max_num_running_reqs"):
            self.scheduler.max_num_running_reqs = 1

    def _has_background_loop(self) -> bool:
        loop_thread = getattr(self, "_loop_thread", None)
        return loop_thread is not None and loop_thread.is_alive()

    @staticmethod
    def _normalize_finished_ids(finished_ids: Any) -> list[str]:
        if not finished_ids:
            return []
        if isinstance(finished_ids, str):
            return [finished_ids]
        if isinstance(finished_ids, Iterable):
            return list(finished_ids)
        return []

    @staticmethod
    def make_engine(
        config: OmniDiffusionConfig,
        scheduler: SchedulerInterface | None = None,
    ) -> DiffusionEngine:
        """Factory method to create a DiffusionEngine instance.

        Args:
            config: The configuration for the diffusion engine.

        Returns:
            An instance of DiffusionEngine.
        """
        return DiffusionEngine(config, scheduler=scheduler)

    def add_request(self, request: OmniDiffusionRequest):
        request_handle = self._register_request_handle(request)
        if self._has_background_loop():
            self._input_queue.put(_QueuedDiffusionRequest(request_handle=request_handle, request=request))
            return request_handle

        try:
            with self._state_lock:
                self._admit_request_locked(request_handle, request)
        except Exception:
            self._discard_request_handle(request_handle)
            raise
        return request_handle

    def get_result(self, request_id: str, timeout: float = 30) -> DiffusionOutput:
        """consumer"""
        fut = self._get_result_future(request_id)
        try:
            result = fut.result(timeout=timeout)
        except FutureTimeoutError as e:
            logger.error("Wait for response timed out for %s", request_id)
            raise TimeoutError(f"Timed out waiting for diffusion request {request_id}.") from e
        with self._state_lock:
            if self._results_map.get(request_id) is fut:
                self._results_map.pop(request_id, None)
        return result

    def get_result_nowait(self, request_id: str) -> DiffusionOutput | None:
        try:
            fut = self._pop_result_future(request_id, require_done=True)
        except RuntimeError:
            return None
        return fut.result()

    def _pop_result_future(self, request_id: str, *, require_done: bool) -> Future[DiffusionOutput]:
        with self._state_lock:
            fut = self._results_map.get(request_id)
            if fut is None:
                raise RuntimeError("Future not found, possibly already processed or race condition.")
            if require_done and not fut.done():
                raise RuntimeError(f"Diffusion request {request_id} is not finished yet.")
            self._results_map.pop(request_id, None)
        return fut

    def _get_result_future(self, request_id: str) -> Future[DiffusionOutput]:
        with self._state_lock:
            fut = self._results_map.get(request_id)
        if fut is None:
            raise RuntimeError("Future not found, possibly already processed or race condition.")
        return fut

    def drain_completed_results_nowait(self) -> list[tuple[str, DiffusionOutput]]:
        completed: list[tuple[str, DiffusionOutput]] = []
        while True:
            try:
                request_handle = self._output_queue.get_nowait()
            except queue.Empty:
                break
            output = self.get_result_nowait(request_handle)
            if output is not None:
                completed.append((request_handle, output))
        return completed

    def add_req_and_wait_for_response(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        exec_start_time = time.perf_counter()
        submitted = SubmittedDiffusionRequest(
            sched_req_id=self.add_request(request),
            request=request,
            start_time=exec_start_time,
            preprocess_time=0.0,
            exec_start_time=exec_start_time,
        )
        return self.wait_for_submitted_request(submitted)

    def _drive_request_until_complete(self, sched_req_id: str, timeout: float = 30) -> DiffusionOutput:
        deadline = time.monotonic() + timeout
        while True:
            result = self.get_result_nowait(sched_req_id)
            if result is not None:
                return result

            progressed = self._run_engine_cycle()
            result = self.get_result_nowait(sched_req_id)
            if result is not None:
                return result

            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for diffusion request {sched_req_id}.")
            if not progressed and not self.scheduler.has_requests():
                raise RuntimeError("Diffusion scheduler has no runnable requests.")

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        """Start or stop torch profiling on all diffusion workers.

        Args:
            is_start: True to start profiling, False to stop.
            profile_prefix: Optional prefix for trace filename (vLLM compat).

        Note:
            Matches vLLM's worker.profile() signature for consistency.
            Traces are saved automatically via on_trace_ready callback.
        """
        if is_start:
            if profile_prefix is None:
                profile_prefix = f"diffusion_{int(time.time())}"
            logger.info(f"Starting diffusion profiling with prefix: {profile_prefix}")
        else:
            logger.info("Stopping diffusion profiling...")

        try:
            self.collective_rpc(method="profile", args=(is_start, profile_prefix))
        except Exception as e:
            action = "start" if is_start else "stop"
            logger.error(f"Failed to {action} profiling on workers", exc_info=True)
            if is_start:
                raise RuntimeError(f"Could not {action} profiler: {e}") from e

    def _dummy_run(self):
        """A dummy run to warm up the model."""
        num_inference_steps = 1
        height = 512
        width = 512
        if supports_image_input(self.od_config.model_class_name):
            # Provide a dummy image input if the model supports it
            color_format = image_color_format(self.od_config.model_class_name)
            dummy_image = PIL.Image.new(color_format, (width, height))
        else:
            dummy_image = None

        if supports_audio_input(self.od_config.model_class_name):
            audio_sr = 16000
            audio_duration_sec = 4
            audio_array = np.random.randn(audio_sr * audio_duration_sec).astype(np.float32)
            dummy_audio = audio_array[audio_sr * 1 : audio_sr * 3]
        else:
            dummy_audio = None

        prompt: OmniTextPrompt = {
            "prompt": "dummy run",
            "multi_modal_data": {"image": dummy_image, "audio": dummy_audio},
        }
        req = OmniDiffusionRequest(
            prompts=[prompt],
            request_ids=["dummy_req_id"],
            sampling_params=OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                # Keep warmup path minimal and robust across text encoders.
                # Some models may fail when warmup implicitly triggers
                # classifier-free guidance with an empty negative prompt.
                guidance_scale=0.0,
                num_outputs_per_prompt=1,
                # Disable CFG for warmup to avoid triggering CFG parallel
                # validation when cfg_parallel_size > 1.
                extra_args={"cfg_text_scale": 1.0, "cfg_img_scale": 1.0},
            ),
        )
        logger.info("dummy run to warm up the model")
        request = self.pre_process_func(req) if self.pre_process_func is not None else req
        output = self.add_req_and_wait_for_response(request)
        if output.error:
            raise RuntimeError(f"Dummy run failed: {output.error}")

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Call a method on worker processes and get results immediately.

        Args:
            method: The method name (str) to execute on workers
            timeout: Optional timeout in seconds
            args: Positional arguments for the method
            kwargs: Keyword arguments for the method
            unique_reply_rank: If set, only get reply from this rank

        Returns:
            Single result if unique_reply_rank is provided, otherwise list of results
        """
        assert isinstance(method, str), "Only string method names are supported for now"

        deadline = None if timeout is None else time.monotonic() + timeout
        acquired = False
        try:
            if deadline is None:
                self._rpc_lock.acquire()
                acquired = True
            else:
                lock_timeout = max(0, deadline - time.monotonic())
                acquired = self._rpc_lock.acquire(timeout=lock_timeout)
            if not acquired:
                raise TimeoutError(f"RPC call to {method} timed out waiting for engine lock.")

            rpc_timeout = None if deadline is None else max(0, deadline - time.monotonic())
            if deadline is not None and rpc_timeout <= 0:
                raise TimeoutError(f"RPC call to {method} timed out.")

            return self.executor.collective_rpc(
                method=method,
                timeout=rpc_timeout,
                args=args,
                kwargs=kwargs,
                unique_reply_rank=unique_reply_rank,
            )
        finally:
            if acquired:
                self._rpc_lock.release()

    def close(self) -> None:
        if hasattr(self, "_shutdown_event"):
            self._shutdown_event.set()
        if hasattr(self, "_loop_thread") and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=1)
        if hasattr(self, "_results_map"):
            with self._state_lock:
                for fut in list(self._results_map.values()):
                    if not fut.done():
                        fut.set_result(DiffusionOutput(error="Diffusion engine closed."))
                self._results_map.clear()
                self._handle_to_sched_req_id.clear()
                self._sched_req_id_to_handle.clear()
                self._request_id_to_handle.clear()
                self._handle_to_request_ids.clear()
                self._cancelled_handles.clear()
        if hasattr(self, "scheduler"):
            self.scheduler.close()
        if hasattr(self, "executor"):
            self.executor.shutdown()

    def abort(self, request_id: str | Iterable[str]) -> None:
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)
        for req_id in request_ids:
            self.abort_queue.put(req_id)

    def _process_aborts_queue(self) -> bool:
        if self.abort_queue.empty():
            return False

        request_ids: list[str] = []
        while not self.abort_queue.empty():
            ids = self.abort_queue.get_nowait()
            request_ids.extend((ids,) if isinstance(ids, str) else ids)

        return self._abort_requests(request_ids)

    def _abort_requests(self, request_ids: str | Iterable[str]) -> bool:
        request_ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)

        pending_handles: list[str] = []
        sched_req_ids: list[str] = []
        with self._state_lock:
            for request_id in dict.fromkeys(request_ids):
                request_handle = self._request_id_to_handle.get(request_id)
                if request_handle is None:
                    sched_req_id = self.scheduler.get_sched_req_id(request_id)
                    if sched_req_id is not None:
                        sched_req_ids.append(sched_req_id)
                    continue
                sched_req_id = self._handle_to_sched_req_id.get(request_handle)
                if sched_req_id is None:
                    pending_handles.append(request_handle)
                else:
                    sched_req_ids.append(sched_req_id)

        progressed = False
        for request_handle in dict.fromkeys(pending_handles):
            progressed = self._abort_pending_request(request_handle) or progressed

        with self._state_lock:
            for sched_req_id in dict.fromkeys(sched_req_ids):
                if self.scheduler.get_request_state(sched_req_id) is not None:
                    self.scheduler.finish_requests(sched_req_id, DiffusionRequestStatus.FINISHED_ABORTED)
                    progressed = True
        return progressed

    def _finalize_finished_request(
        self,
        sched_req_id: str,
        runner_output: RunnerOutput | None = None,
        missing_result_error: str = "Diffusion scheduler finished target request without execution output.",
    ) -> DiffusionOutput:
        state = self.scheduler.get_request_state(sched_req_id)
        popped_state = self.scheduler.pop_request_state(sched_req_id)
        state = state or popped_state

        if state is None:
            raise RuntimeError(f"Diffusion scheduler lost state for request {sched_req_id}.")

        if state.status == DiffusionRequestStatus.FINISHED_ABORTED:
            request_id = state.req.request_ids[0] if state.req.request_ids else sched_req_id
            return DiffusionOutput(
                aborted=True,
                abort_message=f"Request {request_id} aborted.",
            )

        if runner_output is not None:
            req_output = runner_output.get_req_output(sched_req_id)
            if req_output is not None and req_output.result is not None:
                return req_output.result

        return DiffusionOutput(error=missing_result_error)

    def _process_input_queue(self) -> bool:
        progressed = False
        while True:
            try:
                queued_request = self._input_queue.get_nowait()
            except queue.Empty:
                break

            progressed = True
            with self._state_lock:
                if queued_request.request_handle in self._cancelled_handles:
                    self._cancelled_handles.discard(queued_request.request_handle)
                    continue
                if queued_request.request_handle not in self._results_map:
                    continue
                try:
                    self._admit_request_locked(queued_request.request_handle, queued_request.request)
                except Exception as exc:
                    logger.exception(
                        "Failed to admit diffusion request %s",
                        queued_request.request_handle,
                    )
                    self._finalize_request_locked(
                        queued_request.request_handle,
                        DiffusionOutput(error=str(exc)),
                    )
        return progressed

    def _register_request_handle(self, request: OmniDiffusionRequest) -> str:
        with self._state_lock:
            request_handle = self._make_request_handle_locked(request)
            request_ids = tuple(dict.fromkeys(request.request_ids))
            for request_id in request_ids:
                existing = self._request_id_to_handle.get(request_id)
                if existing is not None:
                    raise ValueError(
                        f"request_id {request_id!r} is already mapped to active request handle {existing!r}."
                    )
            self._results_map[request_handle] = Future()
            self._handle_to_request_ids[request_handle] = request_ids
            for request_id in request_ids:
                self._request_id_to_handle[request_id] = request_handle
        return request_handle

    def _discard_request_handle(self, request_handle: str) -> None:
        with self._state_lock:
            self._clear_request_state_locked(request_handle)
            self._results_map.pop(request_handle, None)

    def _make_request_handle_locked(self, request: OmniDiffusionRequest) -> str:
        base = next((request_id for request_id in request.request_ids if request_id), None) or "diffusion_request"
        request_handle = base
        suffix = 1
        while request_handle in self._results_map or request_handle in self._cancelled_handles:
            request_handle = f"{base}#{suffix}"
            suffix += 1
        return request_handle

    def _admit_request_locked(self, request_handle: str, request: OmniDiffusionRequest) -> str:
        sched_req_id = self.scheduler.add_request(request)
        self._handle_to_sched_req_id[request_handle] = sched_req_id
        self._sched_req_id_to_handle[sched_req_id] = request_handle
        return sched_req_id

    def _abort_pending_request(self, request_handle: str) -> bool:
        with self._state_lock:
            if request_handle in self._cancelled_handles:
                return False
            if self._handle_to_sched_req_id.get(request_handle) is not None:
                return False
            request_ids = self._handle_to_request_ids.get(request_handle, ())
            request_id = request_ids[0] if request_ids else request_handle
            self._cancelled_handles.add(request_handle)
            self._finalize_request_locked(
                request_handle,
                DiffusionOutput(
                    aborted=True,
                    abort_message=f"Request {request_id} aborted.",
                ),
            )
        return True

    def _finalize_request_locked(self, request_handle: str, output: DiffusionOutput) -> None:
        fut = self._results_map.get(request_handle)
        if fut is not None and not fut.done():
            fut.set_result(output)
            self._output_queue.put(request_handle)
        self._clear_request_state_locked(request_handle)

    def _clear_request_state_locked(self, request_handle: str) -> None:
        request_ids = self._handle_to_request_ids.pop(request_handle, ())
        for request_id in request_ids:
            if self._request_id_to_handle.get(request_id) == request_handle:
                self._request_id_to_handle.pop(request_id, None)

        sched_req_id = self._handle_to_sched_req_id.pop(request_handle, None)
        if sched_req_id is not None and self._sched_req_id_to_handle.get(sched_req_id) == request_handle:
            self._sched_req_id_to_handle.pop(sched_req_id, None)
