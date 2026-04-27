# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.stage_submodule_proc import StageSubModuleProc


def test_stage_submodule_proc_uses_submodule_worker(monkeypatch):
    calls: list[tuple[str, object]] = []

    class FakeSubModuleWorker:
        def __init__(self, local_rank, rank, od_config):
            calls.append(("init", (local_rank, rank, od_config)))
            self.od_config = od_config

        def load_model(self, load_format):
            calls.append(("load_model", load_format))

        def execute_submodule(self, request):
            calls.append(("execute_submodule", request))
            return DiffusionOutput(output=None, multimodal_output={"ok": True})

        def shutdown(self):
            calls.append(("shutdown", None))

    import vllm_omni.diffusion.stage_submodule_proc as proc_mod

    monkeypatch.setattr(proc_mod, "SubModuleWorker", FakeSubModuleWorker)
    od_config = SimpleNamespace(
        model="Qwen/Qwen-Image",
        model_class_name="QwenImagePipeline",
        model_stage="encode",
        tf_model_config=SimpleNamespace(params={"in_channels": 64}),
        diffusion_load_format="dummy",
        update_multimodal_support=lambda: None,
    )
    proc = StageSubModuleProc("Qwen/Qwen-Image", od_config)

    proc.initialize()
    pending = proc._submit_stage_request(
        "req-1",
        [{"prompt": "test"}],
        {"num_inference_steps": 1},
        batch_mode=False,
    )
    output = pending.future.result(timeout=1)
    proc.close()

    assert output.multimodal_output == {"ok": True}
    assert [name for name, _ in calls] == [
        "init",
        "load_model",
        "execute_submodule",
        "shutdown",
    ]
