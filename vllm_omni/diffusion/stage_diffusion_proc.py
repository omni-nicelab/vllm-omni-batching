"""Subprocess entry point for the diffusion engine.

StageDiffusionProc runs DiffusionEngine in a child process,
communicating with StageDiffusionClient via ZMQ (PUSH/PULL).
"""

from __future__ import annotations

import asyncio
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Any

import msgspec
import torch
import zmq
import zmq.asyncio
from PIL import Image
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_file_to_dict
from vllm.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx
from vllm.utils.system_utils import get_mp_context
from vllm.v1.utils import shutdown

from vllm_omni.diffusion.data import DiffusionRequestAbortedError, TransformerConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.distributed.omni_connectors.utils.serialization import (
    OmniMsgpackDecoder,
    OmniMsgpackEncoder,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)

_HANDSHAKE_POLL_TIMEOUT_S = 600


@dataclass
class _PendingStageRequest:
    request_id: str
    batch_mode: bool
    submitted: Any
    aborted: bool = False


class StageDiffusionProc:
    """Subprocess entry point for diffusion inference.

    Manages DiffusionEngine lifecycle, async request processing,
    and ZMQ-based communication with StageDiffusionClient.
    """

    def __init__(self, model: str, od_config: OmniDiffusionConfig) -> None:
        self._model = model
        self._od_config = od_config
        self._engine: DiffusionEngine | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Enrich config, create DiffusionEngine and thread pool."""
        self._enrich_config()
        self._engine = DiffusionEngine.make_engine(self._od_config)
        self._executor = ThreadPoolExecutor(max_workers=1)
        logger.info("StageDiffusionProc initialized with model: %s", self._model)

    def _enrich_config(self) -> None:
        """Load model metadata from HuggingFace and populate od_config fields.

        Diffusers-style models expose ``model_index.json`` with ``_class_name``.
        Non-diffusers models (e.g. Bagel, NextStep) only have ``config.json``,
        so we fall back to reading that and mapping model_type manually.
        """
        od_config = self._od_config

        try:
            config_dict = get_hf_file_to_dict("model_index.json", od_config.model)
            if config_dict is not None:
                if od_config.model_class_name is None:
                    od_config.model_class_name = config_dict.get("_class_name", None)
                od_config.update_multimodal_support()

                tf_config_dict = get_hf_file_to_dict("transformer/config.json", od_config.model)
                od_config.tf_model_config = TransformerConfig.from_dict(tf_config_dict)
            else:
                raise FileNotFoundError("model_index.json not found")
        except (AttributeError, OSError, ValueError, FileNotFoundError):
            cfg = get_hf_file_to_dict("config.json", od_config.model)
            if cfg is None:
                raise ValueError(f"Could not find config.json or model_index.json for model {od_config.model}")

            od_config.tf_model_config = TransformerConfig.from_dict(cfg)
            model_type = cfg.get("model_type")
            architectures = cfg.get("architectures") or []

            if model_type == "bagel" or "BagelForConditionalGeneration" in architectures:
                od_config.model_class_name = "BagelPipeline"
                od_config.tf_model_config = TransformerConfig()
                od_config.update_multimodal_support()
            elif model_type == "nextstep":
                if od_config.model_class_name is None:
                    od_config.model_class_name = "NextStep11Pipeline"
                od_config.tf_model_config = TransformerConfig()
                od_config.update_multimodal_support()
            elif architectures and len(architectures) == 1:
                od_config.model_class_name = architectures[0]
            else:
                raise

    # ------------------------------------------------------------------
    # Request processing
    # ------------------------------------------------------------------

    def _reconstruct_sampling_params(self, sampling_params_dict: dict) -> OmniDiffusionSamplingParams:
        """Reconstruct OmniDiffusionSamplingParams from a dict, handling LoRA."""
        lora_req = sampling_params_dict.get("lora_request")
        if lora_req is not None:
            from vllm.lora.request import LoRARequest

            if not isinstance(lora_req, LoRARequest):
                sampling_params_dict["lora_request"] = msgspec.convert(lora_req, LoRARequest)

        return OmniDiffusionSamplingParams(**sampling_params_dict)

    def _submit_stage_request(
        self,
        request_id: str,
        prompts: list[Any],
        sampling_params_dict: dict,
        *,
        batch_mode: bool,
        kv_sender_info: dict[str, Any] | None = None,
    ) -> _PendingStageRequest:
        """Build an engine request and enqueue it into DiffusionEngine."""
        sampling_params = self._reconstruct_sampling_params(sampling_params_dict)

        request = OmniDiffusionRequest(
            prompts=prompts,
            sampling_params=sampling_params,
            request_ids=[request_id] * len(prompts),
            request_id=request_id,
            kv_sender_info=kv_sender_info,
        )
        submitted = self._engine.submit_request(request)
        return _PendingStageRequest(
            request_id=request_id,
            batch_mode=batch_mode,
            submitted=submitted,
        )

    @staticmethod
    def _merge_batch_outputs(request_id: str, outputs: list[OmniRequestOutput]) -> OmniRequestOutput:
        """Merge per-prompt outputs into the batched stage response."""
        all_images: list = []
        merged_mm: dict[str, Any] = {}
        merged_metrics: dict[str, Any] = {}
        merged_durations: dict[str, float] = {}
        merged_custom: dict[str, Any] = {}
        peak_mem = 0.0
        latents = None
        trajectory_latents: list[torch.Tensor] | None = None
        trajectory_timesteps: list[torch.Tensor] | None = None
        trajectory_log_probs: torch.Tensor | None = None
        trajectory_decoded: list[Image.Image] | None = None
        final_output_type = "image"

        for output in outputs:
            all_images.extend(output.images)
            merged_mm.update(output._multimodal_output)
            merged_metrics.update(output.metrics)
            merged_durations.update(output.stage_durations)
            merged_custom.update(output._custom_output)
            peak_mem = max(peak_mem, output.peak_memory_mb)
            if latents is None and output.latents is not None:
                latents = output.latents
            if trajectory_latents is None:
                trajectory_latents = output.trajectory_latents
            if trajectory_timesteps is None:
                trajectory_timesteps = output.trajectory_timesteps
            if trajectory_log_probs is None:
                trajectory_log_probs = output.trajectory_log_probs
            if trajectory_decoded is None:
                trajectory_decoded = output.trajectory_decoded
            if output.final_output_type != "image":
                final_output_type = output.final_output_type

        prompt = outputs[0].prompt if len(outputs) == 1 and outputs else None
        return OmniRequestOutput.from_diffusion(
            request_id=request_id,
            images=all_images,
            prompt=prompt,
            metrics=merged_metrics,
            latents=latents,
            trajectory_latents=trajectory_latents,
            trajectory_timesteps=trajectory_timesteps,
            trajectory_log_probs=trajectory_log_probs,
            trajectory_decoded=trajectory_decoded,
            custom_output=merged_custom or None,
            multimodal_output=merged_mm or None,
            final_output_type=final_output_type,
            stage_durations=merged_durations,
            peak_memory_mb=peak_mem,
        )

    async def _drain_completed_requests(
        self,
        pending_requests: dict[str, _PendingStageRequest],
        pending_requests_by_handle: dict[str, _PendingStageRequest],
        response_socket,
        encoder: OmniMsgpackEncoder,
    ) -> None:
        for request_handle, output in self._engine.drain_completed_results_nowait():
            pending = pending_requests_by_handle.pop(request_handle, None)
            if pending is None:
                continue

            request_id = pending.request_id
            pending_requests.pop(request_id, None)
            if pending.aborted or output.aborted:
                logger.info(
                    "request_id: %s aborted: %s",
                    request_id,
                    output.abort_message or "aborted",
                )
                continue

            if output.error:
                await response_socket.send(
                    encoder.encode(
                        {
                            "type": "error",
                            "request_id": request_id,
                            "error": output.error,
                        }
                    )
                )
                continue

            try:
                request_outputs = self._engine.build_outputs_from_submitted_request(
                    pending.submitted,
                    output,
                )
                if pending.batch_mode:
                    result = self._merge_batch_outputs(request_id, request_outputs)
                else:
                    result = request_outputs[0]
                    if not result.request_id:
                        result.request_id = request_id
                await response_socket.send(encoder.encode({"type": "result", "output": result}))
            except DiffusionRequestAbortedError as e:
                logger.info("request_id: %s aborted during postprocess: %s", request_id, str(e))
            except Exception as e:
                logger.exception("Diffusion request %s failed during finalize: %s", request_id, e)
                await response_socket.send(
                    encoder.encode(
                        {
                            "type": "error",
                            "request_id": request_id,
                            "error": str(e),
                        }
                    )
                )

    # ------------------------------------------------------------------
    # Collective RPC dispatch
    # ------------------------------------------------------------------

    async def _handle_collective_rpc(
        self,
        method: str,
        timeout: float | None,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Dispatch collective RPC calls to DiffusionEngine.

        LoRA methods remap arguments and post-process results to match
        the contract that ``AsyncOmni`` provides.
        """
        loop = asyncio.get_running_loop()

        if method == "profile":
            is_start = args[0] if args else True
            profile_prefix = args[1] if len(args) > 1 else None
            return await loop.run_in_executor(
                self._executor,
                self._engine.profile,
                is_start,
                profile_prefix,
            )

        if method == "add_lora":
            # Reconstruct LoRARequest after IPC if needed.
            lora_request = args[0] if args else kwargs.get("lora_request")
            if lora_request is not None:
                from vllm.lora.request import LoRARequest

                if not isinstance(lora_request, LoRARequest):
                    lora_request = msgspec.convert(lora_request, LoRARequest)
            results = await loop.run_in_executor(
                self._executor,
                self._engine.collective_rpc,
                "add_lora",
                timeout,
                (),
                {"lora_request": lora_request},
                None,
            )
            return all(results) if isinstance(results, list) else results

        if method == "remove_lora":
            results = await loop.run_in_executor(
                self._executor,
                self._engine.collective_rpc,
                "remove_lora",
                timeout,
                args,
                kwargs or {},
                None,
            )
            return all(results) if isinstance(results, list) else results

        if method == "list_loras":
            results = await loop.run_in_executor(
                self._executor,
                self._engine.collective_rpc,
                "list_loras",
                timeout,
                (),
                {},
                None,
            )
            if not isinstance(results, list):
                return results or []
            merged: set[int] = set()
            for part in results:
                merged.update(part or [])
            return sorted(merged)

        if method == "pin_lora":
            lora_id = args[0] if args else kwargs.get("adapter_id")
            results = await loop.run_in_executor(
                self._executor,
                self._engine.collective_rpc,
                "pin_lora",
                timeout,
                (),
                {"adapter_id": lora_id},
                None,
            )
            return all(results) if isinstance(results, list) else results

        # Fall back to DiffusionEngine.collective_rpc for all other methods
        # (e.g. worker extension RPCs like "test_extension_name").
        return await loop.run_in_executor(
            self._executor,
            self._engine.collective_rpc,
            method,
            timeout,
            args,
            kwargs or {},
            None,
        )

    # ------------------------------------------------------------------
    # ZMQ event loop
    # ------------------------------------------------------------------

    async def run_loop(
        self,
        request_address: str,
        response_address: str,
    ) -> None:
        """Async event loop handling ZMQ messages from StageDiffusionClient."""
        ctx = zmq.asyncio.Context()

        request_socket = ctx.socket(zmq.PULL)
        request_socket.bind(request_address)

        response_socket = ctx.socket(zmq.PUSH)
        response_socket.bind(response_address)

        encoder = OmniMsgpackEncoder()
        decoder = OmniMsgpackDecoder()

        pending_requests: dict[str, _PendingStageRequest] = {}
        pending_requests_by_handle: dict[str, _PendingStageRequest] = {}
        poller = zmq.asyncio.Poller()
        poller.register(request_socket, zmq.POLLIN)
        shutdown_requested = False

        try:
            while not shutdown_requested:
                events = dict(await poller.poll(timeout=10))
                if request_socket in events:
                    while True:
                        try:
                            raw = await request_socket.recv(flags=zmq.NOBLOCK)
                        except zmq.Again:
                            break

                        msg = decoder.decode(raw)
                        msg_type = msg.get("type")

                        if msg_type == "add_request":
                            request_id = msg["request_id"]
                            try:
                                pending = self._submit_stage_request(
                                    request_id,
                                    [msg["prompt"]],
                                    msg["sampling_params"],
                                    batch_mode=False,
                                    kv_sender_info=msg.get("kv_sender_info"),
                                )
                                pending_requests[request_id] = pending
                                pending_requests_by_handle[pending.submitted.request_handle] = pending
                            except Exception as e:
                                logger.exception("Diffusion request %s failed to submit: %s", request_id, e)
                                await response_socket.send(
                                    encoder.encode(
                                        {
                                            "type": "error",
                                            "request_id": request_id,
                                            "error": str(e),
                                        }
                                    )
                                )

                        elif msg_type == "add_batch_request":
                            request_id = msg["request_id"]
                            try:
                                pending = self._submit_stage_request(
                                    request_id,
                                    msg["prompts"],
                                    msg["sampling_params"],
                                    batch_mode=True,
                                    kv_sender_info=msg.get("kv_sender_info"),
                                )
                                pending_requests[request_id] = pending
                                pending_requests_by_handle[pending.submitted.request_handle] = pending
                            except Exception as e:
                                logger.exception("Batch diffusion request %s failed to submit: %s", request_id, e)
                                await response_socket.send(
                                    encoder.encode(
                                        {
                                            "type": "error",
                                            "request_id": request_id,
                                            "error": str(e),
                                        }
                                    )
                                )

                        elif msg_type == "abort":
                            for rid in msg.get("request_ids", []):
                                pending = pending_requests.get(rid)
                                if pending is not None:
                                    pending.aborted = True
                                self._engine.abort(rid)

                        elif msg_type == "collective_rpc":
                            rpc_id = msg["rpc_id"]
                            try:
                                result = await self._handle_collective_rpc(
                                    msg["method"],
                                    msg.get("timeout"),
                                    tuple(msg.get("args", ())),
                                    msg.get("kwargs", {}),
                                )
                                await response_socket.send(
                                    encoder.encode(
                                        {
                                            "type": "rpc_result",
                                            "rpc_id": rpc_id,
                                            "result": result,
                                        }
                                    )
                                )
                            except Exception as e:
                                logger.exception("Collective RPC %s failed: %s", msg["method"], e)
                                await response_socket.send(
                                    encoder.encode(
                                        {
                                            "type": "error",
                                            "rpc_id": rpc_id,
                                            "error": str(e),
                                        }
                                    )
                                )

                        elif msg_type == "shutdown":
                            shutdown_requested = True
                            break

                await self._drain_completed_requests(
                    pending_requests,
                    pending_requests_by_handle,
                    response_socket,
                    encoder,
                )

        finally:
            request_socket.close()
            response_socket.close()
            ctx.term()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release engine and thread pool resources."""
        if self._closed:
            return
        self._closed = True

        if self._engine is not None:
            try:
                self._engine.close()
            except Exception as e:
                logger.warning("Error closing diffusion engine: %s", e)

        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False)
            except Exception as e:
                logger.warning("Error shutting down executor: %s", e)

    # ------------------------------------------------------------------
    # Subprocess entry point
    # ------------------------------------------------------------------

    @classmethod
    def run_diffusion_proc(
        cls,
        model: str,
        od_config: OmniDiffusionConfig,
        handshake_address: str,
        request_address: str,
        response_address: str,
    ) -> None:
        """Entry point for the diffusion subprocess."""
        shutdown_requested = False

        def signal_handler(signum: int, frame: Any) -> None:
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        proc = cls(model, od_config)
        try:
            proc.initialize()

            # Send READY via handshake socket
            handshake_ctx = zmq.Context()
            handshake_socket = handshake_ctx.socket(zmq.DEALER)
            handshake_socket.connect(handshake_address)
            handshake_socket.send(msgspec.msgpack.encode({"status": "READY"}))
            handshake_socket.close()
            handshake_ctx.term()

            # Run async event loop
            asyncio.run(proc.run_loop(request_address, response_address))

        except SystemExit:
            logger.debug("StageDiffusionProc exiting.")
            raise
        except Exception:
            logger.exception("StageDiffusionProc encountered a fatal error.")
            raise
        finally:
            proc.close()


# -- Free functions for backward compatibility with StageDiffusionClient ------


def spawn_diffusion_proc(
    model: str,
    od_config: OmniDiffusionConfig,
) -> tuple[BaseProcess, str, str, str]:
    """Spawn a StageDiffusionProc subprocess.

    Returns ``(proc, handshake_address, request_address, response_address)``.
    """
    handshake_address = get_open_zmq_ipc_path()
    request_address = get_open_zmq_ipc_path()
    response_address = get_open_zmq_ipc_path()

    ctx = get_mp_context()
    proc = ctx.Process(
        target=StageDiffusionProc.run_diffusion_proc,
        name="StageDiffusionProc",
        kwargs={
            "model": model,
            "od_config": od_config,
            "handshake_address": handshake_address,
            "request_address": request_address,
            "response_address": response_address,
        },
    )
    proc.start()
    # Wait for the process to become alive before returning.
    deadline = time.monotonic() + 10
    while not proc.is_alive():
        if proc.exitcode is not None:
            raise RuntimeError(f"StageDiffusionProc failed to start (exit code {proc.exitcode})")
        if time.monotonic() > deadline:
            raise TimeoutError("StageDiffusionProc did not become alive within 10s")
        time.sleep(0.01)
    return proc, handshake_address, request_address, response_address


def complete_diffusion_handshake(
    proc: BaseProcess,
    handshake_address: str,
) -> None:
    """Wait for the diffusion subprocess to signal READY.

    On failure the process is terminated before re-raising.
    """
    try:
        _perform_diffusion_handshake(proc, handshake_address)
    except Exception:
        shutdown([proc])
        raise


def _perform_diffusion_handshake(
    proc: BaseProcess,
    handshake_address: str,
) -> None:
    """Run the handshake with the diffusion subprocess."""
    with zmq_socket_ctx(handshake_address, zmq.ROUTER, bind=True) as handshake_socket:
        poller = zmq.Poller()
        poller.register(handshake_socket, zmq.POLLIN)
        poller.register(proc.sentinel, zmq.POLLIN)

        timeout_ms = _HANDSHAKE_POLL_TIMEOUT_S * 1000
        while True:
            events = dict(poller.poll(timeout=timeout_ms))
            if not events:
                raise TimeoutError("Timed out waiting for READY from StageDiffusionProc")
            if handshake_socket in events:
                identity, raw = handshake_socket.recv_multipart()
                msg = msgspec.msgpack.decode(raw)
                if msg.get("status") == "READY":
                    return
                raise RuntimeError(f"Expected READY, got: {msg}")
            if proc.exitcode is not None:
                raise RuntimeError(f"StageDiffusionProc died during handshake (exit code {proc.exitcode})")
