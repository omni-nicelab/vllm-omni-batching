# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker for lightweight diffusion submodule stages.

This worker intentionally does not reuse ``DiffusionWorker``.  Submodule stages
execute a single module-level operation such as Qwen-Image encode/decode, so
they have different execution and result semantics from full diffusion/DiT
workers.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext

import torch
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.utils.mem_utils import GiB_bytes
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.vae_model_runner import VAEModelRunner
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.gpu_memory_utils import get_process_gpu_memory

logger = init_logger(__name__)


class SubModuleWorker:
    """Device/process owner for one encode/decode submodule runner."""

    def __init__(
        self,
        local_rank: int,
        rank: int,
        od_config: OmniDiffusionConfig,
    ) -> None:
        self.local_rank = local_rank
        self.rank = rank
        self.od_config = od_config
        self.device: torch.device | None = None
        self.vllm_config: VllmConfig | None = None
        self.model_runner: VAEModelRunner | None = None

        self.init_device()
        self.model_runner = VAEModelRunner(
            vllm_config=self.vllm_config,
            od_config=self.od_config,
            device=self.device,
        )
        logger.info("SubModuleWorker %d initialized.", self.rank)

    def init_device(self) -> None:
        """Initialize device/distributed state for submodule execution."""
        world_size = self.od_config.num_gpus
        rank = self.rank

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.od_config.master_port)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        self.device = current_omni_platform.get_torch_device(rank)
        current_omni_platform.set_device(self.device)

        vllm_config = VllmConfig(compilation_config=CompilationConfig())
        vllm_config.parallel_config.tensor_parallel_size = self.od_config.parallel_config.tensor_parallel_size
        vllm_config.parallel_config.data_parallel_size = self.od_config.parallel_config.data_parallel_size
        vllm_config.parallel_config.enable_expert_parallel = self.od_config.parallel_config.enable_expert_parallel
        self.vllm_config = vllm_config

        with (
            set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config),
            set_current_vllm_config(self.vllm_config),
        ):
            init_distributed_environment(world_size=world_size, rank=rank)
            parallel_config = self.od_config.parallel_config
            initialize_model_parallel(
                data_parallel_size=parallel_config.data_parallel_size,
                cfg_parallel_size=parallel_config.cfg_parallel_size,
                sequence_parallel_size=parallel_config.sequence_parallel_size,
                ulysses_degree=parallel_config.ulysses_degree,
                ring_degree=parallel_config.ring_degree,
                tensor_parallel_size=parallel_config.tensor_parallel_size,
                pipeline_parallel_size=parallel_config.pipeline_parallel_size,
                fully_shard_degree=parallel_config.hsdp_shard_size if parallel_config.use_hsdp else 1,
                hsdp_replicate_size=parallel_config.hsdp_replicate_size if parallel_config.use_hsdp else 1,
                enable_expert_parallel=parallel_config.enable_expert_parallel,
            )
            init_workspace_manager(self.device)

    def load_model(self, load_format: str = "default", custom_pipeline_name: str | None = None) -> None:
        """Load the submodule pipeline through ``VAEModelRunner``."""
        assert self.model_runner is not None, "Model runner not initialized"
        with (
            set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config),
            set_current_vllm_config(self.vllm_config),
        ):
            self.model_runner.load_model(
                memory_pool_context_fn=self._maybe_get_memory_pool_context,
                load_format=load_format,
                custom_pipeline_name=custom_pipeline_name,
            )

        process_memory = get_process_gpu_memory(self.local_rank)
        if process_memory is not None:
            logger.info(
                "SubModuleWorker %d: process-scoped GPU memory after model loading: %.2f GiB.",
                self.rank,
                process_memory / GiB_bytes,
            )

    def execute_submodule(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """Execute one submodule request and return a submodule-shaped output."""
        assert self.model_runner is not None, "Model runner not initialized"
        return self.model_runner.execute_model(req)

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        if self.od_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, "Sleep mode can only be used for one instance per process."
            return allocator.use_memory_pool(tag=tag)
        return nullcontext()

    def shutdown(self) -> None:
        destroy_distributed_env()
