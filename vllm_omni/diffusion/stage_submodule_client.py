# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage client for lightweight submodule stages."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

import zmq
from vllm.logger import init_logger

from vllm_omni.diffusion.stage_submodule_proc import (
    complete_submodule_handshake,
    spawn_submodule_proc,
)
from vllm_omni.distributed.omni_connectors.utils.serialization import (
    OmniMsgpackDecoder,
    OmniMsgpackEncoder,
)
from vllm_omni.engine.stage_init_utils import StageMetadata, terminate_alive_proc
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType

logger = init_logger(__name__)


class StageSubModuleClient:
    """Communicates with StageSubModuleProc via ZMQ."""

    stage_type: str = "submodule"
    _NON_SERIALIZABLE_FIELDS = frozenset({"generator", "modules"})

    def __init__(
        self,
        model: str,
        od_config: OmniDiffusionConfig,
        metadata: StageMetadata,
        batch_size: int = 1,
    ) -> None:
        self.stage_id = metadata.stage_id
        self.final_output = metadata.final_output
        self.final_output_type = metadata.final_output_type
        self.default_sampling_params = metadata.default_sampling_params
        self.custom_process_input_func = metadata.custom_process_input_func
        self.engine_input_source = metadata.engine_input_source
        self.model_stage = metadata.model_stage

        proc, handshake_address, request_address, response_address = spawn_submodule_proc(model, od_config)
        complete_submodule_handshake(proc, handshake_address)
        self._proc = proc

        self._zmq_ctx = zmq.Context()
        self._request_socket = self._zmq_ctx.socket(zmq.PUSH)
        self._request_socket.connect(request_address)
        self._response_socket = self._zmq_ctx.socket(zmq.PULL)
        self._response_socket.connect(response_address)

        self._encoder = OmniMsgpackEncoder()
        self._decoder = OmniMsgpackDecoder()
        self._output_queue: asyncio.Queue[OmniRequestOutput | dict[str, Any]] = asyncio.Queue()
        self._rpc_results: dict[str, Any] = {}
        self._pending_rpcs: set[str] = set()
        self._shutting_down = False
        self.engine_outputs: Any = None

        logger.info(
            "[StageSubModuleClient] Stage-%s initialized (model_stage=%s, batch_size=%d)",
            self.stage_id,
            self.model_stage,
            batch_size,
        )

    @staticmethod
    def _sampling_params_to_dict(sampling_params: Any) -> dict[str, Any]:
        if is_dataclass(sampling_params) and not isinstance(sampling_params, type):
            result = {
                f.name: getattr(sampling_params, f.name)
                for f in fields(sampling_params)
                if f.name not in StageSubModuleClient._NON_SERIALIZABLE_FIELDS
            }
        elif not isinstance(sampling_params, dict):
            raise TypeError(f"sampling_params is not a dict but {sampling_params.__class__.__name__}")
        else:
            result = {
                k: v for k, v in sampling_params.items() if k not in StageSubModuleClient._NON_SERIALIZABLE_FIELDS
            }

        if result.get("seed") is None:
            generator = (
                getattr(sampling_params, "generator", None)
                if not isinstance(sampling_params, dict)
                else sampling_params.get("generator")
            )
            if generator is not None:
                if isinstance(generator, list) and generator:
                    generator = generator[0]
                if hasattr(generator, "initial_seed"):
                    result["seed"] = generator.initial_seed()
        return result

    def _drain_responses(self) -> None:
        while True:
            try:
                raw = self._response_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break

            msg = self._decoder.decode(raw)
            msg_type = msg.get("type")
            if msg_type == "result":
                self._output_queue.put_nowait(msg["output"])
            elif msg_type == "rpc_result":
                self._rpc_results[msg["rpc_id"]] = msg["result"]
            elif msg_type == "error":
                req_id = msg.get("request_id")
                rpc_id = msg.get("rpc_id")
                error_msg = msg.get("error")
                logger.error(
                    "[StageSubModuleClient] Stage-%s subprocess error for %s: %s",
                    self.stage_id,
                    rpc_id or req_id,
                    error_msg,
                )
                if rpc_id is not None and rpc_id in self._pending_rpcs:
                    self._rpc_results[rpc_id] = {"error": True, "reason": error_msg}
                elif req_id is not None:
                    self._output_queue.put_nowait(
                        {"type": "error", "request_id": req_id, "error": error_msg}
                    )

    def set_engine_outputs(self, engine_outputs: Any) -> None:
        self.engine_outputs = engine_outputs

    async def add_request_async(
        self,
        request_id: str,
        prompt: OmniPromptType,
        sampling_params: OmniDiffusionSamplingParams,
    ) -> None:
        self._request_socket.send(
            self._encoder.encode(
                {
                    "type": "add_request",
                    "request_id": request_id,
                    "prompt": prompt,
                    "sampling_params": self._sampling_params_to_dict(sampling_params),
                }
            )
        )

    async def add_batch_request_async(
        self,
        request_id: str,
        prompts: list[OmniPromptType],
        sampling_params: OmniDiffusionSamplingParams,
    ) -> None:
        self._request_socket.send(
            self._encoder.encode(
                {
                    "type": "add_batch_request",
                    "request_id": request_id,
                    "prompts": prompts,
                    "sampling_params": self._sampling_params_to_dict(sampling_params),
                }
            )
        )

    def get_submodule_output_nowait(self) -> OmniRequestOutput | dict[str, Any] | None:
        self._drain_responses()
        try:
            return self._output_queue.get_nowait()
        except asyncio.QueueEmpty:
            if not self._shutting_down and self._proc is not None and not self._proc.is_alive():
                raise RuntimeError(f"StageSubModuleProc died unexpectedly (exit code {self._proc.exitcode})")
            return None

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        self._request_socket.send(
            self._encoder.encode({"type": "abort", "request_ids": list(request_ids)})
        )

    async def collective_rpc_async(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        rpc_id = uuid.uuid4().hex
        self._pending_rpcs.add(rpc_id)
        self._request_socket.send(
            self._encoder.encode(
                {
                    "type": "collective_rpc",
                    "rpc_id": rpc_id,
                    "method": method,
                    "timeout": timeout,
                    "args": list(args),
                    "kwargs": kwargs,
                }
            )
        )
        deadline = time.monotonic() + timeout if timeout else None
        try:
            while True:
                self._drain_responses()
                if rpc_id in self._rpc_results:
                    return self._rpc_results.pop(rpc_id)
                if self._proc is not None and not self._proc.is_alive():
                    raise RuntimeError(
                        f"StageSubModuleProc died while waiting for collective_rpc "
                        f"{method!r} (exit code {self._proc.exitcode})"
                    )
                if deadline and time.monotonic() > deadline:
                    raise TimeoutError(f"collective_rpc_async '{method}' timed out after {timeout}s")
                await asyncio.sleep(0.01)
        finally:
            self._pending_rpcs.discard(rpc_id)

    def shutdown(self) -> None:
        self._shutting_down = True
        try:
            self._request_socket.send(self._encoder.encode({"type": "shutdown"}))
        except Exception:
            pass

        if self._proc is not None and self._proc.is_alive():
            self._proc.join(timeout=10)
            terminate_alive_proc(self._proc)

        self._request_socket.close(linger=0)
        self._response_socket.close(linger=0)
        self._zmq_ctx.term()
