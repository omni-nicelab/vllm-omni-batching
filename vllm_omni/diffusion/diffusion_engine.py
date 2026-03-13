# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import queue
import threading
import time
from collections.abc import Iterable
from typing import Any

import PIL.Image
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, DiffusionRequestAbortedError, OmniDiffusionConfig
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


def supports_image_input(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_image_input", False))


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
        self._rpc_lock = threading.RLock()
        self.abort_queue: queue.Queue[str] = queue.Queue()
        self.execute_fn = self.executor.execute_step if self.step_execution else self.executor.execute_request

        try:
            self._dummy_run()
        except Exception as e:
            logger.error(f"Dummy run failed: {e}")
            self.close()
            raise e

    def step(self, request: OmniDiffusionRequest) -> list[OmniRequestOutput]:
        # Apply pre-processing if available
        if self.pre_process_func is not None:
            preprocess_start_time = time.time()
            request = self.pre_process_func(request)
            preprocess_time = time.time() - preprocess_start_time
            logger.info(f"Pre-processing completed in {preprocess_time:.4f} seconds")

        output = self.add_req_and_wait_for_response(request)
        if output.aborted:
            raise DiffusionRequestAbortedError(output.abort_message or "Diffusion request aborted.")
        if output.error:
            raise Exception(f"{output.error}")
        logger.info("Generation completed successfully.")

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

        postprocess_start_time = time.time()
        outputs = self.post_process_func(output.output) if self.post_process_func is not None else output.output
        postprocess_time = time.time() - postprocess_start_time
        logger.info(f"Post-processing completed in {postprocess_time:.4f} seconds")

        # Convert to OmniRequestOutput format
        # Ensure outputs is a list
        if not isinstance(outputs, list):
            outputs = [outputs] if outputs is not None else []

        metrics = {
            "image_num": int(request.sampling_params.num_outputs_per_prompt),
            "resolution": int(request.sampling_params.resolution),
            "postprocess_time_ms": postprocess_time * 1000,
        }
        if self.pre_process_func is not None:
            metrics["preprocessing_time_ms"] = preprocess_time * 1000

        # Handle single request or multiple requests
        if len(request.prompts) == 1:
            # Single request: return single OmniRequestOutput
            prompt = request.prompts[0]
            request_id = request.request_ids[0] if request.request_ids else ""

            if supports_audio_output(self.od_config.model_class_name):
                audio_payload = outputs[0] if len(outputs) == 1 else outputs
                return [
                    OmniRequestOutput.from_diffusion(
                        request_id=request_id,
                        images=[],
                        prompt=prompt,
                        metrics=metrics,
                        latents=output.trajectory_latents,
                        multimodal_output={"audio": audio_payload},
                        final_output_type="audio",
                    ),
                ]
            else:
                return [
                    OmniRequestOutput.from_diffusion(
                        request_id=request_id,
                        images=outputs,
                        prompt=prompt,
                        metrics=metrics,
                        latents=output.trajectory_latents,
                    ),
                ]
        else:
            # Multiple requests: return list of OmniRequestOutput
            # Split images based on num_outputs_per_prompt for each request
            results = []
            output_idx = 0

            for i, prompt in enumerate(request.prompts):
                request_id = request.request_ids[i] if i < len(request.request_ids) else ""

                # Get images for this request
                num_outputs = request.sampling_params.num_outputs_per_prompt
                request_outputs = outputs[output_idx : output_idx + num_outputs] if output_idx < len(outputs) else []
                output_idx += num_outputs

                if supports_audio_output(self.od_config.model_class_name):
                    audio_payload = request_outputs[0] if len(request_outputs) == 1 else request_outputs
                    results.append(
                        OmniRequestOutput.from_diffusion(
                            request_id=request_id,
                            images=[],
                            prompt=prompt,
                            metrics=metrics,
                            latents=output.trajectory_latents,
                            multimodal_output={"audio": audio_payload},
                            final_output_type="audio",
                        )
                    )
                else:
                    results.append(
                        OmniRequestOutput.from_diffusion(
                            request_id=request_id,
                            images=request_outputs,
                            prompt=prompt,
                            metrics=metrics,
                            latents=output.trajectory_latents,
                        )
                    )

            return results

    @staticmethod
    def make_engine(
        config: OmniDiffusionConfig,
        scheduler: SchedulerInterface | None = None,
    ) -> "DiffusionEngine":
        """Factory method to create a DiffusionEngine instance.

        Args:
            config: The configuration for the diffusion engine.

        Returns:
            An instance of DiffusionEngine.
        """
        return DiffusionEngine(config, scheduler=scheduler)

    def add_req_and_wait_for_response(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        with self._rpc_lock:
            target_sched_req_id = self.scheduler.add_request(request)

            # keep scheduling and executing until the target request is finished
            while True:
                # check abort queue
                self._process_aborts_queue()
                sched_output = self.scheduler.schedule()
                if sched_output.is_empty:
                    if target_sched_req_id in sched_output.finished_req_ids:
                        return self._finalize_finished_request(target_sched_req_id)
                    if not self.scheduler.has_requests():
                        raise RuntimeError("Diffusion scheduler has no runnable requests.")
                    continue

                sched_req_id = sched_output.scheduled_req_ids[0]
                try:
                    model_output = self.execute_fn(sched_output)
                except Exception as exc:
                    logger.error("Execution failed for diffusion request %s", sched_req_id, exc_info=True)
                    model_output = RunnerOutput(
                        req_id=sched_req_id,
                        step_index=None,
                        finished=True,
                        result=DiffusionOutput(error=str(exc)),
                    )

                # check abort queue
                self._process_aborts_queue()

                finished_req_ids = self.scheduler.update_from_output(sched_output, model_output)
                if target_sched_req_id in finished_req_ids:
                    return self._finalize_finished_request(
                        target_sched_req_id,
                        runner_output=model_output,
                        missing_result_error="Diffusion execution finished without a final output.",
                    )

    def start_profile(self, trace_filename: str | None = None) -> None:
        """
        Start torch profiling on all diffusion workers.

        Creates a directory (if needed) and sets up a base filename template
        for per-rank profiler traces (typically saved as <template>_rank<N>.json).

        Args:
            trace_filename: Optional base filename (without extension or rank suffix).
                            If None, generates one using current timestamp.
        """
        if trace_filename is None:
            trace_filename = f"stage_0_diffusion_{int(time.time())}_rank"

        trace_dir = os.environ.get("VLLM_TORCH_PROFILER_DIR", "./profiles")

        # Expand ~ and ~user, then make absolute (robust against cwd changes)
        trace_dir = os.path.expanduser(trace_dir)
        trace_dir = os.path.abspath(trace_dir)

        try:
            os.makedirs(trace_dir, exist_ok=True)
        except OSError as exc:
            logger.error(f"Failed to create profiler directory {trace_dir}: {exc}")
            raise

        # Build final template path (without rank or extension — torch.profiler appends those)
        full_template = os.path.join(trace_dir, trace_filename)

        expected_pattern = f"{full_template}*.json"
        logger.info(f"Starting diffusion profiling → {expected_pattern}")

        # Also log the absolute directory once (useful in multi-node or containers)
        logger.debug(f"Profiler output directory: {trace_dir}")

        # Propagate to all workers
        try:
            self.collective_rpc(method="start_profile", args=(full_template,))
        except Exception as e:
            logger.error("Failed to start profiling on workers", exc_info=True)
            raise RuntimeError(f"Could not start profiler: {e}") from e

    def stop_profile(self) -> dict:
        """
        Stop profiling on all workers and collect the final trace/table paths.

        The worker (torch_profiler.py) now handles trace export, compression to .gz,
        and deletion of the original .json file. This method only collects and
        reports the paths returned by the workers.

        Returns:
            dict with keys:
            - "traces": list of final trace file paths (usually .json.gz)
            - "tables": list of table strings (one per rank)
        """
        logger.info("Stopping diffusion profiling and collecting results...")

        try:
            # Give worker enough time — export + compression + table can be slow
            results = self.collective_rpc(method="stop_profile", timeout=600)
        except Exception:
            logger.error("Failed to stop profiling on workers", exc_info=True)
            return {"traces": [], "tables": []}

        output_files = {"traces": [], "tables": []}
        successful_traces = 0

        if not results:
            logger.warning("No profiling results returned from any rank")
            return output_files

        for rank, res in enumerate(results):
            if not isinstance(res, dict):
                logger.warning(f"Rank {rank}: invalid result format (got {type(res)})")
                continue

            # 1. Trace file — should be .json.gz if compression succeeded
            trace_path = res.get("trace")
            if trace_path:
                # We trust the worker — it created/compressed the file
                logger.info(f"[Rank {rank}] Final trace: {trace_path}")
                output_files["traces"].append(trace_path)
                successful_traces += 1

                # Optional: warn if path looks suspicious (e.g. still .json)
                if not trace_path.endswith((".json.gz", ".json")):
                    logger.warning(f"Rank {rank}: unusual trace path extension: {trace_path}")

            # 2. Table file — plain text
            table = res.get("table")
            if table:
                output_files["tables"].append(table)

        # Final summary logging
        num_ranks = len(results)
        if successful_traces > 0:
            final_paths_str = ", ".join(output_files["traces"][:3])
            if len(output_files["traces"]) > 3:
                final_paths_str += f" ... (+{len(output_files['traces']) - 3} more)"

            logger.info(
                f"Profiling stopped. Collected {successful_traces} trace file(s) "
                f"from {num_ranks} rank(s). "
                f"Final trace paths: {final_paths_str}"
            )
        elif output_files["traces"]:
            logger.info(
                f"Profiling stopped but no traces were successfully collected. "
                f"Reported paths: {', '.join(output_files['traces'][:3])}"
                f"{' ...' if len(output_files['traces']) > 3 else ''}"
            )
        else:
            logger.info("Profiling stopped — no trace files were collected from any rank.")

        if output_files["tables"]:
            logger.debug(f"Collected {len(output_files['tables'])} profiling table(s)")

        return output_files

    def _dummy_run(self):
        """A dummy run to warm up the model."""
        num_inference_steps = 1
        height = 1024
        width = 1024
        if supports_image_input(self.od_config.model_class_name):
            # Provide a dummy image input if the model supports it
            color_format = image_color_format(self.od_config.model_class_name)
            dummy_image = PIL.Image.new(color_format, (width, height))
        else:
            dummy_image = None
        prompt: OmniTextPrompt = {"prompt": "dummy run", "multi_modal_data": {"image": dummy_image}}
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
        if hasattr(self, "scheduler"):
            self.scheduler.close()
        if hasattr(self, "executor"):
            self.executor.shutdown()

    def abort(self, request_id: str | Iterable[str]) -> None:
        if isinstance(request_id, str):
            request_ids = [request_id]
        else:
            request_ids = list(request_id)

        for req_id in request_ids:
            self.abort_queue.put(req_id)

    def _process_aborts_queue(self):
        if not self.abort_queue.empty():
            request_ids = []
            while not self.abort_queue.empty():
                ids = self.abort_queue.get_nowait()
                # Should be a list here, but also handle string just in case.
                request_ids.extend((ids,) if isinstance(ids, str) else ids)
            # More efficient to abort all as a single batch.
            self._abort_requests(request_ids)

    def _abort_requests(self, request_ids: str | Iterable[str]) -> None:
        if isinstance(request_ids, str):
            request_ids = [request_ids]

        sched_req_ids: list[str] = []
        for request_id in dict.fromkeys(request_ids):
            sched_req_id = self.scheduler.get_sched_req_id(request_id)
            if sched_req_id is not None:
                sched_req_ids.append(sched_req_id)

        for sched_req_id in dict.fromkeys(sched_req_ids):
            if self.scheduler.get_request_state(sched_req_id) is not None:
                self.scheduler.finish_requests(sched_req_id, DiffusionRequestStatus.FINISHED_ABORTED)

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

        if runner_output is not None and runner_output.result is not None:
            return runner_output.result

        if runner_output is None:
            raise RuntimeError(missing_result_error)

        return DiffusionOutput(error=missing_result_error)
