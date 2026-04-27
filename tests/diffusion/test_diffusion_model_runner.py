# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.utils import DiffusionRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@contextmanager
def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    yield


class _DummyPipeline:
    def __init__(self, output):
        self._output = output
        self.forward_calls = 0

    def forward(self, req):
        del req
        self.forward_calls += 1
        return self._output


def _make_request(skip_cache_refresh: bool = True):
    sampling_params = SimpleNamespace(
        generator=None,
        seed=None,
        generator_device=None,
        num_inference_steps=4,
    )
    return SimpleNamespace(
        prompts=["a prompt"],
        sampling_params=sampling_params,
        skip_cache_refresh=skip_cache_refresh,
    )


def _make_runner(cache_backend, cache_backend_name: str, enable_cache_dit_summary: bool = True):
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.pipeline = _DummyPipeline(output=SimpleNamespace(output="ok"))
    runner.cache_backend = cache_backend
    runner.offload_backend = None
    runner.od_config = SimpleNamespace(
        cache_backend=cache_backend_name,
        enable_cache_dit_summary=enable_cache_dit_summary,
        parallel_config=SimpleNamespace(use_hsdp=False),
    )
    runner.kv_transfer_manager = SimpleNamespace(
        receive_kv_cache=lambda req, target_device=None: None,
        receive_multi_kv_cache=lambda req, cfg_kv_collect_func=None, target_device=None: None,
        receive_multi_kv_cache_distributed=lambda req, cfg_kv_collect_func=None, target_device=None: None,
    )
    return runner


def test_execute_model_skips_cache_summary_without_active_cache_backend(monkeypatch):
    """Guard cache diagnostics with runtime backend state to avoid stale-config crashes."""
    runner = _make_runner(cache_backend=None, cache_backend_name="cache_dit")
    req = _make_request(skip_cache_refresh=True)

    cache_summary_calls = []

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(
        model_runner_module,
        "cache_summary",
        lambda pipeline, details: cache_summary_calls.append((pipeline, details)),
    )

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.output == "ok"
    assert cache_summary_calls == []


def test_execute_model_emits_cache_summary_with_active_cache_dit_backend(monkeypatch):
    class _EnabledCacheBackend:
        def is_enabled(self):
            return True

    runner = _make_runner(cache_backend=_EnabledCacheBackend(), cache_backend_name="cache_dit")
    req = _make_request(skip_cache_refresh=True)

    cache_summary_calls = []

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(
        model_runner_module,
        "cache_summary",
        lambda pipeline, details: cache_summary_calls.append((pipeline, details)),
    )

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.output == "ok"
    assert cache_summary_calls == [(runner.pipeline, True)]


def test_update_states_after_uses_pipeline_intermediate_output_hook():
    runner = object.__new__(DiffusionModelRunner)
    state = DiffusionRequestState(
        req_id="req-1",
        sampling=SimpleNamespace(),
        prompts=[],
        latents=torch.ones((1, 2), dtype=torch.float32),
    )
    input_batch = SimpleNamespace(latents=state.latents.clone(), idx_mapping_np=np.array([0], dtype=np.int32))
    runner.pipeline = SimpleNamespace(
        produces_intermediate_stage_output=True,
        post_intermediate_output=lambda request_state: DiffusionOutput(
            output=None,
            multimodal_output={"req_id": request_state.req_id},
        ),
        post_decode=lambda request_state: DiffusionOutput(output=f"decoded:{request_state.req_id}"),
    )
    runner.cache_manager = None
    runner.state_cache = {"req-1": state}
    runner.input_batch = input_batch

    results = DiffusionModelRunner._update_states_after(
        runner,
        [state],
        input_batch,
        [True],
        [None],
    )

    assert results[0].multimodal_output == {"req_id": "req-1"}
    assert state.latents.tolist() == [[1.0, 1.0]]
    assert runner.state_cache == {}
    assert runner.input_batch is None


def test_load_model_clears_cache_backend_for_unsupported_pipeline(monkeypatch):
    class _DummyLoader:
        def __init__(self, load_config, od_config=None):
            del load_config, od_config

        def load_model(self, **kwargs):
            del kwargs
            return SimpleNamespace(transformer=torch.nn.Identity())

    class _DummyMemoryProfiler:
        consumed_memory = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    class _DummyCacheBackend:
        def __init__(self):
            self.enabled = False

        def enable(self, pipeline):
            del pipeline
            self.enabled = True

    dummy_cache_backend = _DummyCacheBackend()

    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.pipeline = None
    runner.cache_backend = None
    runner.offload_backend = None
    runner.od_config = SimpleNamespace(
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        cache_backend="cache_dit",
        cache_config={},
        model_class_name="NextStep11Pipeline",
        enforce_eager=True,
    )

    monkeypatch.setattr(model_runner_module, "LoadConfig", lambda: object())
    monkeypatch.setattr(model_runner_module, "DiffusersPipelineLoader", _DummyLoader)
    monkeypatch.setattr(model_runner_module, "DeviceMemoryProfiler", _DummyMemoryProfiler)
    monkeypatch.setattr(model_runner_module, "get_offload_backend", lambda od_config, device: None)
    monkeypatch.setattr(
        model_runner_module, "get_cache_backend", lambda cache_backend, cache_config: dummy_cache_backend
    )

    DiffusionModelRunner.load_model(runner)

    assert runner.cache_backend is None
    assert runner.od_config.cache_backend is None
    assert dummy_cache_backend.enabled is False
