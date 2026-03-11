# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""IPC utilities for transferring large tensors via POSIX shared memory.

Used by Hop1 (GPU worker <-> scheduler) to avoid pickling large video tensors
through the MessageQueue. Tensors above ``_SHM_TENSOR_THRESHOLD`` are copied
into a named shared-memory segment; only a lightweight metadata dict is
serialised through the queue.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.worker.utils import RunnerOutput

_SHM_TENSOR_THRESHOLD = 1_000_000  # 1 MB


def _tensor_to_shm(tensor: torch.Tensor) -> dict[str, Any]:
    """Copy a tensor into POSIX shared memory and return a metadata handle.

    The shared memory segment remains alive after this call (the local fd is
    closed, but the segment persists until ``_tensor_from_shm`` unlinks it).
    """
    from multiprocessing import shared_memory

    import numpy as np

    tensor = tensor.detach().cpu().contiguous()
    arr = tensor.numpy()
    nbytes = arr.nbytes
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf[:nbytes])
    np.copyto(shm_arr, arr)
    handle = {
        "__tensor_shm__": True,
        "name": shm.name,
        "shape": list(tensor.shape),
        "torch_dtype": str(tensor.dtype),
        "numpy_dtype": str(arr.dtype),
        "nbytes": nbytes,
    }
    shm.close()
    return handle


def _tensor_from_shm(handle: dict[str, Any]) -> torch.Tensor:
    """Reconstruct a tensor from a shared-memory handle and free the segment."""
    from multiprocessing import shared_memory

    import numpy as np

    shm = shared_memory.SharedMemory(name=handle["name"])
    try:
        np_dtype = np.dtype(handle["numpy_dtype"])
        arr = np.ndarray(handle["shape"], dtype=np_dtype, buffer=shm.buf[: handle["nbytes"]])
        tensor = torch.from_numpy(arr.copy())
    finally:
        shm.close()
        shm.unlink()
    return tensor


def _resolve_diffusion_output(message: DiffusionOutput | RunnerOutput | Any) -> DiffusionOutput | None:
    if isinstance(message, DiffusionOutput):
        return message
    if isinstance(message, RunnerOutput):
        result = message.result
        return result if isinstance(result, DiffusionOutput) else None
    return None


def pack_diffusion_output_shm(message: DiffusionOutput | RunnerOutput | Any) -> DiffusionOutput | RunnerOutput | Any:
    """Replace large tensors in diffusion results with shared-memory handles.

    Accepts either a raw ``DiffusionOutput`` or a stepwise ``RunnerOutput``.
    Non-diffusion payloads are returned unchanged.
    """
    output = _resolve_diffusion_output(message)
    if output is None:
        return message

    if output.output is not None and isinstance(output.output, torch.Tensor):
        if output.output.nelement() * output.output.element_size() > _SHM_TENSOR_THRESHOLD:
            output.output = _tensor_to_shm(output.output)
    if output.trajectory_latents is not None and isinstance(output.trajectory_latents, torch.Tensor):
        if output.trajectory_latents.nelement() * output.trajectory_latents.element_size() > _SHM_TENSOR_THRESHOLD:
            output.trajectory_latents = _tensor_to_shm(output.trajectory_latents)
    return message


def unpack_diffusion_output_shm(message: DiffusionOutput | RunnerOutput | Any) -> DiffusionOutput | RunnerOutput | Any:
    """Reconstruct tensors from shared-memory handles in diffusion results.

    Accepts either a raw ``DiffusionOutput`` or a stepwise ``RunnerOutput``.
    Non-diffusion payloads are returned unchanged.
    """
    output = _resolve_diffusion_output(message)
    if output is None:
        return message

    if isinstance(output.output, dict) and output.output.get("__tensor_shm__"):
        output.output = _tensor_from_shm(output.output)
    if isinstance(output.trajectory_latents, dict) and output.trajectory_latents.get("__tensor_shm__"):
        output.trajectory_latents = _tensor_from_shm(output.trajectory_latents)
    return message
