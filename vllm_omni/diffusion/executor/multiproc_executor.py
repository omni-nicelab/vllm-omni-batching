import multiprocessing as mp
import time
import weakref
from dataclasses import dataclass
from typing import Any

import zmq
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger

from vllm_omni.diffusion.data import SHUTDOWN_MESSAGE
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.step_batch import StepRunnerOutput, StepSchedulerOutput
from vllm_omni.utils.platform_utils import get_diffusion_worker_class

logger = init_logger(__name__)


@dataclass
class BackgroundResources:
    """
    Used as a finalizer for clean shutdown.
    """

    broadcast_mq: MessageQueue | None = None
    result_mq: MessageQueue | None = None
    num_workers: int = 0
    processes: list[mp.Process] | None = None

    def __call__(self):
        """Clean up background resources."""
        if self.broadcast_mq is not None:
            try:
                for _ in range(self.num_workers):
                    self.broadcast_mq.enqueue(SHUTDOWN_MESSAGE)
            except Exception as exc:
                logger.warning("Failed to send shutdown signal: %s", exc)
        for queue, label in ((self.broadcast_mq, "broadcast"), (self.result_mq, "result")):
            if queue is None:
                continue
            try:
                close_fn = getattr(queue, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception as exc:
                logger.warning("Failed to close %s queue: %s", label, exc)
        if self.processes:
            for proc in self.processes:
                if not proc.is_alive():
                    continue
                proc.join(30)
                if proc.is_alive():
                    logger.warning("Terminating diffusion worker %s after timeout", proc.name)
                    proc.terminate()
                    proc.join(30)


class MultiprocDiffusionExecutor(DiffusionExecutor):
    uses_multiproc: bool = True

    def _init_executor(self) -> None:
        self._processes: list[mp.Process] = []
        self._closed = False

        num_workers = self.od_config.num_gpus
        self._broadcast_mq = self._init_broadcast_queue(num_workers)
        broadcast_handle = self._broadcast_mq.export_handle()

        # Launch workers
        processes, result_handle = self._launch_workers(broadcast_handle)

        self._result_mq = self._init_result_queue(result_handle)

        self._processes = processes

        self.resources = BackgroundResources(
            broadcast_mq=self._broadcast_mq,
            result_mq=self._result_mq,
            num_workers=num_workers,
            processes=self._processes,
        )
        self._finalizer = weakref.finalize(self, self.resources)

    def _init_broadcast_queue(self, num_workers: int) -> MessageQueue:
        return MessageQueue(
            n_reader=num_workers,
            n_local_reader=num_workers,
            local_reader_ranks=list(range(num_workers)),
        )

    def _init_result_queue(self, result_handle) -> MessageQueue | None:
        if result_handle is None:
            logger.error("Failed to get result queue handle from workers")
            return None
        return MessageQueue.create_from_handle(result_handle, 0)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("DiffusionExecutor is closed.")
        if self._result_mq is None:
            raise RuntimeError("Result queue not initialized")

    def _launch_workers(self, broadcast_handle):
        od_config = self.od_config
        logger.info("Starting server...")

        num_gpus = od_config.num_gpus
        mp.set_start_method("spawn", force=True)
        processes = []

        # Get the appropriate worker class for current device
        worker_proc = get_diffusion_worker_class()

        # Launch all worker processes
        worker_pipe_readers = []
        worker_pipe_writers = []

        for i in range(num_gpus):
            reader, writer = mp.Pipe(duplex=False)
            worker_pipe_writers.append(writer)
            process = mp.Process(
                target=worker_proc.worker_main,
                args=(
                    i,  # rank
                    od_config,
                    writer,
                    broadcast_handle,
                ),
                name=f"DiffusionWorker-{i}",
                daemon=True,
            )
            worker_pipe_readers.append(reader)
            process.start()
            processes.append(process)

        # Wait for all workers to be ready
        worker_infos = []
        result_handle = None
        for writer in worker_pipe_writers:
            writer.close()

        for i, reader in enumerate(worker_pipe_readers):
            try:
                data = reader.recv()
            except EOFError:
                logger.error(f"Rank {i} worker is dead. Please check if there are relevant logs.")
                processes[i].join()
                logger.error(f"Exit code: {processes[i].exitcode}")
                raise

            if data["status"] != "ready":
                raise RuntimeError("Initialization failed. Please see the error messages above.")

            if i == 0:
                result_handle = data.get("result_handle")

            worker_infos.append(data)
            reader.close()

        logger.debug("All workers are ready")

        return processes, result_handle

    def add_req(self, requests: list[OmniDiffusionRequest]):
        raise NotImplementedError("Synchronous generate is deprecated. Use step-level scheduling with execute_step().")

    def execute_step(
        self,
        scheduler_output: StepSchedulerOutput,
        timeout: float | None = None,
    ) -> StepRunnerOutput:
        self._ensure_open()
        return self.collective_rpc(
            method="execute_step",
            timeout=timeout,
            args=(scheduler_output,),
            unique_reply_rank=0,
            exec_all_ranks=True,
        )

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
        exec_all_ranks: bool = False,
    ) -> Any:
        self._ensure_open()

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # Prepare RPC request message
        rpc_request = {
            "type": "rpc",
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "output_rank": unique_reply_rank,
            "exec_all_ranks": exec_all_ranks,
        }

        try:
            # Broadcast RPC request to all workers via unified message queue
            self._broadcast_mq.enqueue(rpc_request)

            # Determine which workers we expect responses from
            num_responses = 1 if unique_reply_rank is not None else self.od_config.num_gpus

            responses = []
            for _ in range(num_responses):
                dequeue_timeout = None if deadline is None else (deadline - time.monotonic())
                try:
                    response = self._result_mq.dequeue(timeout=dequeue_timeout)

                    # Check if response indicates an error
                    if isinstance(response, dict) and response.get("status") == "error":
                        raise RuntimeError(
                            f"Worker failed with error '{response.get('error')}', "
                            "please check the stack trace above for the root cause"
                        )

                    responses.append(response)
                except zmq.error.Again as exc:
                    raise TimeoutError(f"RPC call to {method} timed out.") from exc
                except TimeoutError as e:
                    raise TimeoutError(f"RPC call to {method} timed out.") from e

            return responses[0] if unique_reply_rank is not None else responses

        except Exception as e:
            logger.error(f"RPC call failed: {e}")
            raise

    def check_health(self) -> None:
        # Simple check if processes are alive
        for p in self._processes:
            if not p.is_alive():
                raise RuntimeError(f"Worker process {p.name} is dead")

    def shutdown(self) -> None:
        self._closed = True
        self._finalizer()
