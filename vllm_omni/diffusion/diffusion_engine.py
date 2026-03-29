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

from vllm_omni.diffusion.core.diffusion_core import DiffusionCore
from vllm_omni.diffusion.core.outputs import DiffusionCoreOutput
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.registry import (
    DiffusionModelRegistry,
    get_diffusion_post_process_func,
    get_diffusion_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


def supports_image_input(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_image_input", False))


class DiffusionEngine:
    """The diffusion engine for vLLM-Omni diffusion models."""

    def __init__(self, od_config: OmniDiffusionConfig):
        """Initialize the diffusion engine.

        Args:
            config: The configuration for the diffusion engine.
        """
        self.od_config = od_config

        self.post_process_func = get_diffusion_post_process_func(od_config)
        self.pre_process_func = get_diffusion_pre_process_func(od_config)

        executor_class = DiffusionExecutor.get_class(od_config)
        self.executor = executor_class(od_config)

        try:
            self._dummy_run()
        except Exception as e:
            logger.error(f"Dummy run failed: {e}")
            self.close()
            raise e

    def step(self, requests: list[OmniDiffusionRequest]):
        try:
            # Apply pre-processing if available
            if self.pre_process_func is not None:
                preprocess_start_time = time.time()
                requests = self.pre_process_func(requests)
                preprocess_time = time.time() - preprocess_start_time
                logger.info(f"Pre-processing completed in {preprocess_time:.4f} seconds")

            output = self.add_req_and_wait_for_response(requests)
            if output.error:
                raise Exception(f"{output.error}")
            logger.info("Generation completed successfully.")

            if output.output is None:
                logger.warning("Output is None, returning empty OmniRequestOutput")
                # Return empty output for the first request
                if len(requests) > 0:
                    request = requests[0]
                    request_id = request.request_id or ""
                    prompt = request.prompt
                    if isinstance(prompt, list):
                        prompt = prompt[0] if prompt else None
                    return OmniRequestOutput.from_diffusion(
                        request_id=request_id,
                        images=[],
                        prompt=prompt,
                        metrics={},
                        latents=None,
                    )
                return None

            postprocess_start_time = time.time()
            images = self.post_process_func(output.output) if self.post_process_func is not None else output.output
            postprocess_time = time.time() - postprocess_start_time
            logger.info(f"Post-processing completed in {postprocess_time:.4f} seconds")

            # Convert to OmniRequestOutput format
            # Ensure images is a list
            if not isinstance(images, list):
                images = [images] if images is not None else []

            # Handle single request or multiple requests
            if len(requests) == 1:
                # Single request: return single OmniRequestOutput
                request = requests[0]
                request_id = request.request_id or ""
                prompt = request.prompt
                if isinstance(prompt, list):
                    prompt = prompt[0] if prompt else None

                metrics = {}
                if output.trajectory_timesteps is not None:
                    metrics["trajectory_timesteps"] = output.trajectory_timesteps

                return OmniRequestOutput.from_diffusion(
                    request_id=request_id,
                    images=images,
                    prompt=prompt,
                    metrics=metrics,
                    latents=output.trajectory_latents,
                )
            else:
                # Multiple requests: return list of OmniRequestOutput
                # Split images based on num_outputs_per_prompt for each request
                results = []
                image_idx = 0

                for request in requests:
                    request_id = request.request_id or ""
                    prompt = request.prompt
                    if isinstance(prompt, list):
                        prompt = prompt[0] if prompt else None

                    # Get images for this request
                    num_outputs = request.num_outputs_per_prompt
                    request_images = images[image_idx : image_idx + num_outputs] if image_idx < len(images) else []
                    image_idx += num_outputs

                    metrics = {}
                    if output.trajectory_timesteps is not None:
                        metrics["trajectory_timesteps"] = output.trajectory_timesteps

                    results.append(
                        OmniRequestOutput.from_diffusion(
                            request_id=request_id,
                            images=request_images,
                            prompt=prompt,
                            metrics=metrics,
                            latents=output.trajectory_latents,
                        )
                    )

                return results
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None

    @staticmethod
    def make_engine(config: OmniDiffusionConfig) -> "DiffusionEngine":
        """Factory method to create a DiffusionEngine instance.

        Args:
            config: The configuration for the diffusion engine.

        Returns:
            An instance of DiffusionEngine.
        """
        if config.step_execution:
            return StepExecutionDiffusionEngine(config)
        return DiffusionEngine(config)

    def add_req_and_wait_for_response(self, requests: list[OmniDiffusionRequest]):
        return self.executor.add_req(requests)

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
            results = self.collective_rpc(method="stop_profile", timeout=60000)
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
        prompt = "dummy run"
        # note that num_inference_steps=1 will cause timestep and temb None in the pipeline
        num_inference_steps = 1
        height = 1024
        width = 1024
        if supports_image_input(self.od_config.model_class_name):
            # Provide a dummy image input if the model supports it

            dummy_image = PIL.Image.new("RGB", (width, height), color=(0, 0, 0))
        else:
            dummy_image = None
        req = OmniDiffusionRequest(
            prompt=prompt,
            height=height,
            width=width,
            pil_image=dummy_image,
            num_inference_steps=num_inference_steps,
            num_outputs_per_prompt=1,
        )
        logger.info("dummy run to warm up the model")
        requests = self.pre_process_func([req]) if self.pre_process_func is not None else [req]
        self.add_req_and_wait_for_response(requests)

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
        return self.executor.collective_rpc(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
            unique_reply_rank=unique_reply_rank,
        )

    def close(self) -> None:
        if hasattr(self, "executor"):
            self.executor.shutdown()

    def abort(self, request_id: str | Iterable[str]) -> None:
        # TODO implement it
        logger.warning("DiffusionEngine abort is not implemented yet")
        pass


class StepExecutionDiffusionEngine(DiffusionEngine):
    """Diffusion engine backed by DiffusionCore for online step execution."""

    def __init__(self, od_config: OmniDiffusionConfig):
        self.od_config = od_config

        self.post_process_func = get_diffusion_post_process_func(od_config)
        self.pre_process_func = get_diffusion_pre_process_func(od_config)

        executor_class = DiffusionExecutor.get_class(od_config)
        self.core = DiffusionCore(od_config, executor_class)
        self.executor = self.core.executor

        self._closed = False
        self._pending_lock = threading.Lock()
        self._pending_outputs: dict[str, queue.Queue[DiffusionCoreOutput]] = {}
        self._shutdown_event = threading.Event()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            name="DiffusionCoreLoop",
            daemon=True,
        )
        self._loop_thread.start()

        try:
            self._dummy_run()
        except Exception as e:
            logger.error(f"Dummy run failed: {e}")
            self.close()
            raise e

    def step(self, requests: list[OmniDiffusionRequest]):
        if self._closed:
            raise RuntimeError("Diffusion engine is closed")

        request_map: dict[str, OmniDiffusionRequest] = {}
        wait_queues: dict[str, queue.Queue[DiffusionCoreOutput]] = {}

        with self._pending_lock:
            for request in requests:
                request_id = request.request_id
                if request_id is None:
                    raise ValueError("OmniDiffusionRequest.request_id must be set for step_execution")
                if request_id in self._pending_outputs:
                    raise ValueError(f"Duplicate in-flight diffusion request_id: {request_id}")
                waiter: queue.Queue[DiffusionCoreOutput] = queue.Queue(maxsize=1)
                self._pending_outputs[request_id] = waiter
                wait_queues[request_id] = waiter
                request_map[request_id] = request

        try:
            for request in requests:
                self.core.enqueue_request(request)

            outputs: list[OmniRequestOutput] = []
            for request in requests:
                request_id = request.request_id
                assert request_id is not None
                core_output = wait_queues[request_id].get()
                if core_output.error:
                    raise RuntimeError(core_output.error)
                outputs.append(self._build_request_output(request_map[request_id], core_output))

            if len(outputs) == 1:
                return outputs[0]
            return outputs
        finally:
            with self._pending_lock:
                for request in requests:
                    request_id = request.request_id
                    if request_id is not None:
                        self._pending_outputs.pop(request_id, None)

    def add_req_and_wait_for_response(self, requests: list[OmniDiffusionRequest]):
        raise NotImplementedError("step_execution uses DiffusionCore and does not support add_req_and_wait_for_response.")

    def _run_loop(self) -> None:
        try:
            while not self._shutdown_event.is_set():
                if not self.core.has_unfinished_requests():
                    try:
                        request_type, payload = self.core.input_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    self.core._handle_client_request(request_type, payload)

                self._drain_input_queue()

                outputs, executed = self.core.step()
                for output in outputs:
                    self._deliver_output(output)

                if not executed and self.core.has_unfinished_requests():
                    time.sleep(0.001)
        except Exception as e:
            logger.exception("DiffusionCore loop failed")
            self._fail_pending_requests(f"DiffusionCore loop failed: {e}")
            self._shutdown_event.set()

    def _drain_input_queue(self) -> None:
        while True:
            try:
                request_type, payload = self.core.input_queue.get_nowait()
            except queue.Empty:
                return
            self.core._handle_client_request(request_type, payload)

    def _deliver_output(self, output: DiffusionCoreOutput) -> None:
        with self._pending_lock:
            waiter = self._pending_outputs.get(output.request_id)
        if waiter is None:
            return
        try:
            waiter.put_nowait(output)
        except queue.Full:
            logger.debug("Skipping duplicate output for request %s", output.request_id)

    def _build_request_output(
        self,
        request: OmniDiffusionRequest,
        core_output: DiffusionCoreOutput,
    ) -> OmniRequestOutput:
        prompt = request.prompt
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else None
        return OmniRequestOutput.from_diffusion(
            request_id=core_output.request_id,
            images=core_output.images or [],
            prompt=prompt,
            metrics=core_output.metrics,
            latents=core_output.latents,
        )

    def _dummy_run(self):
        prompt = "dummy run"
        num_inference_steps = 1
        height = 1024
        width = 1024
        if supports_image_input(self.od_config.model_class_name):
            dummy_image = PIL.Image.new("RGB", (width, height), color=(0, 0, 0))
        else:
            dummy_image = None
        req = OmniDiffusionRequest(
            request_id="__dummy_warmup__",
            prompt=prompt,
            height=height,
            width=width,
            pil_image=dummy_image,
            num_inference_steps=num_inference_steps,
            num_outputs_per_prompt=1,
        )
        logger.info("dummy run to warm up the model")
        self.step([req])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shutdown_event.set()
        if hasattr(self, "_loop_thread") and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5)
        self._fail_pending_requests("Diffusion engine is shutting down")
        if hasattr(self, "executor"):
            self.executor.shutdown()

    def abort(self, request_id: str | Iterable[str]) -> None:
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)
        for req_id in request_ids:
            self._deliver_output(
                DiffusionCoreOutput(
                    request_id=req_id,
                    finished=True,
                    error="Request aborted",
                )
            )
        self.core.enqueue_abort(request_ids)

    def _fail_pending_requests(self, error: str) -> None:
        with self._pending_lock:
            request_ids = list(self._pending_outputs.keys())
        for req_id in request_ids:
            self._deliver_output(
                DiffusionCoreOutput(
                    request_id=req_id,
                    finished=True,
                    error=error,
                )
            )
