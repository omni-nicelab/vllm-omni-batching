# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.inputs import parse as vllm_parse

if not hasattr(vllm_parse, "ParsedEmbedsPrompt"):
    class _ParsedEmbedsPrompt(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    vllm_parse.ParsedEmbedsPrompt = _ParsedEmbedsPrompt

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion


class BlockingFakeEngine:
    def __init__(self) -> None:
        self.closed = False

    def step(self, requests):
        raise AssertionError("This smoke test should not call generate().")

    def close(self) -> None:
        self.closed = True

    def abort(self, request_id) -> None:
        pass


def test_async_omni_diffusion_step_execution_allows_concurrent_generate(monkeypatch):
    fake_engine = BlockingFakeEngine()
    monkeypatch.setattr("vllm_omni.diffusion.data.OmniDiffusionConfig.settle_port", lambda self, port, *_args, **_kwargs: port)
    monkeypatch.setattr(
        "vllm_omni.entrypoints.async_omni_diffusion.get_hf_file_to_dict",
        lambda filename, model: {"_class_name": "DummyPipeline"} if filename == "model_index.json" else {},
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.async_omni_diffusion.DiffusionEngine.make_engine",
        lambda od_config: fake_engine,
    )

    od_config = OmniDiffusionConfig(
        model="dummy-model",
        step_execution=True,
        num_gpus=1,
    )

    engine = AsyncOmniDiffusion(model="dummy-model", od_config=od_config)
    try:
        assert engine.engine is fake_engine
        assert engine._executor._max_workers > 1
        request = engine._prepare_request(prompt="prompt-1", request_id="req-1")
        assert request.request_id == "req-1"
        assert request.prompt == "prompt-1"
    finally:
        engine.close()

    assert fake_engine.closed is True
