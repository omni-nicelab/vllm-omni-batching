# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionSchedulerOutput,
    NewRequestData,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@contextmanager
def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    yield


class _BatchingPipeline:
    supports_step_execution = True

    def __init__(self):
        self.prepare_calls = 0
        self.denoise_calls = 0
        self.scheduler_calls = 0
        self.decode_calls = 0
        self.denoise_snapshots: list[dict[str, object]] = []

    def prepare_encode(self, state, **kwargs):
        del kwargs
        self.prepare_calls += 1

        sampling = state.sampling
        start_timestep = float(sampling.start_timestep)
        num_steps = int(sampling.num_inference_steps)
        state.timesteps = torch.tensor(
            [start_timestep - float(idx) for idx in range(num_steps)],
            dtype=torch.float32,
        )

        state.latents = torch.full(
            (int(getattr(sampling, "num_latent_rows", 1)), 1),
            float(sampling.initial_latent),
            dtype=torch.float32,
        )
        state.prompt_embeds = torch.zeros((1, 2, 4), dtype=torch.float32)
        state.prompt_embeds_mask = torch.tensor([[True, True]])
        state.guidance = None if sampling.guidance is None else torch.as_tensor(sampling.guidance, dtype=torch.float32)
        state.do_true_cfg = bool(getattr(sampling, "do_true_cfg", False))
        if getattr(sampling, "true_cfg_scale", None) is not None:
            state.sampling.true_cfg_scale = sampling.true_cfg_scale
        if getattr(sampling, "cfg_normalize", False):
            state.sampling.cfg_normalize = True
        if getattr(sampling, "image_latent_value", None) is not None:
            state.sampling.image_latent = torch.full(
                (int(getattr(sampling, "num_latent_rows", 1)), 1),
                float(sampling.image_latent_value),
                dtype=torch.float32,
            )
        state.step_index = 0
        return state

    def denoise_step(self, input_batch, **kwargs):
        del kwargs
        self.denoise_calls += 1
        self.denoise_snapshots.append(
            {
                "req_ids": list(input_batch.req_ids),
                "timesteps": input_batch.timesteps.detach().clone(),
                "latents": input_batch.latents.detach().clone(),
                "guidance": None if input_batch.guidance is None else input_batch.guidance.detach().clone(),
                "do_true_cfg": input_batch.do_true_cfg,
                "true_cfg_scale": input_batch.true_cfg_scale,
                "cfg_normalize": input_batch.cfg_normalize,
                "image_latents": (
                    None if input_batch.image_latents is None else input_batch.image_latents.detach().clone()
                ),
            }
        )
        return torch.ones_like(input_batch.latents)

    def step_scheduler(self, state, noise_pred, **kwargs):
        del kwargs
        self.scheduler_calls += 1
        state.latents = state.latents + noise_pred
        state.step_index += 1

    def post_decode(self, state, **kwargs):
        del kwargs
        self.decode_calls += 1
        return DiffusionOutput(output=state.latents.detach().clone())


def _make_request(
    req_id: str,
    *,
    start_timestep: float,
    num_inference_steps: int = 3,
    initial_latent: float = 0.0,
    num_latent_rows: int = 1,
    guidance: float | list[float] | None = None,
    do_true_cfg: bool = False,
    true_cfg_scale: float | None = None,
    cfg_normalize: bool = False,
    image_latent_value: float | None = None,
):
    return SimpleNamespace(
        prompts=[f"prompt-{req_id}"],
        request_ids=[req_id],
        sampling_params=SimpleNamespace(
            generator=None,
            seed=None,
            generator_device=None,
            num_inference_steps=num_inference_steps,
            start_timestep=start_timestep,
            initial_latent=initial_latent,
            num_latent_rows=num_latent_rows,
            guidance=guidance,
            do_true_cfg=do_true_cfg,
            true_cfg_scale=true_cfg_scale,
            cfg_normalize=cfg_normalize,
            image_latent_value=image_latent_value,
        ),
    )


def _make_runner():
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.od_config = SimpleNamespace(
        cache_backend=None,
        parallel_config=SimpleNamespace(use_hsdp=False),
    )
    runner.device = torch.device("cpu")
    runner.pipeline = _BatchingPipeline()
    runner.cache_backend = None
    runner.cache_manager = None
    runner.offload_backend = None
    runner.state_cache = {}
    runner.input_batch = None
    runner.model_state = SimpleNamespace(prepare_attn=lambda input_batch: {})
    runner.kv_transfer_manager = SimpleNamespace()
    return runner


def _scheduler_output_for_new_reqs(new_reqs: list[tuple[str, object]], step_id: int) -> DiffusionSchedulerOutput:
    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=[NewRequestData(sched_req_id=req_id, req=req) for req_id, req in new_reqs],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=len(new_reqs),
        num_waiting_reqs=0,
    )


def _scheduler_output_mixed(
    *,
    new_reqs: list[tuple[str, object]],
    cached_req_ids: list[str],
    step_id: int,
) -> DiffusionSchedulerOutput:
    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=[NewRequestData(sched_req_id=req_id, req=req) for req_id, req in new_reqs],
        scheduled_cached_reqs=CachedRequestData(sched_req_ids=cached_req_ids),
        finished_req_ids=set(),
        num_running_reqs=len(new_reqs) + len(cached_req_ids),
        num_waiting_reqs=0,
    )


def test_runner_batches_multiple_new_requests_with_distinct_timesteps(monkeypatch):
    runner = _make_runner()
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

    req_a = _make_request("req-a", start_timestep=100.0, initial_latent=1.0)
    req_b = _make_request("req-b", start_timestep=200.0, initial_latent=2.0)

    output = DiffusionModelRunner.execute_stepwise(
        runner,
        _scheduler_output_for_new_reqs(
            [("req-a", req_a), ("req-b", req_b)],
            step_id=0,
        ),
    )

    assert output.req_id == ["req-a", "req-b"]
    assert output.step_index == [1, 1]
    assert output.finished == [False, False]
    assert output.result == [None, None]

    assert runner.pipeline.prepare_calls == 2
    assert runner.pipeline.denoise_calls == 1
    assert runner.pipeline.scheduler_calls == 2
    assert runner.pipeline.decode_calls == 0

    snapshot = runner.pipeline.denoise_snapshots[0]
    assert snapshot["req_ids"] == ["req-a", "req-b"]
    torch.testing.assert_close(
        snapshot["timesteps"],
        torch.tensor([100.0, 200.0]),
    )


def test_runner_batches_cached_and_new_with_different_current_timesteps(monkeypatch):
    runner = _make_runner()
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

    req_a = _make_request("req-a", start_timestep=30.0, num_inference_steps=3, initial_latent=1.0)
    req_b = _make_request("req-b", start_timestep=80.0, num_inference_steps=3, initial_latent=2.0)

    first = DiffusionModelRunner.execute_stepwise(
        runner,
        _scheduler_output_for_new_reqs([("req-a", req_a)], step_id=0),
    )
    assert first.req_id == "req-a"
    assert first.finished is False

    second = DiffusionModelRunner.execute_stepwise(
        runner,
        _scheduler_output_mixed(
            new_reqs=[("req-b", req_b)],
            cached_req_ids=["req-a"],
            step_id=1,
        ),
    )

    assert second.req_id == ["req-b", "req-a"]
    assert second.step_index == [1, 2]
    assert second.finished == [False, False]
    assert second.result == [None, None]

    mixed_snapshot = runner.pipeline.denoise_snapshots[-1]
    req_ids = mixed_snapshot["req_ids"]
    timesteps = mixed_snapshot["timesteps"].tolist()
    timestep_by_req = dict(zip(req_ids, timesteps, strict=True))
    assert timestep_by_req == {"req-b": 80.0, "req-a": 29.0}


def test_runner_passes_cfg_guidance_and_image_latents_to_batched_denoise_step(monkeypatch):
    runner = _make_runner()
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

    req_a = _make_request(
        "req-a",
        start_timestep=100.0,
        initial_latent=1.0,
        num_latent_rows=2,
        guidance=[3.0, 4.0],
        do_true_cfg=True,
        true_cfg_scale=6.5,
        cfg_normalize=True,
        image_latent_value=11.0,
    )
    req_b = _make_request(
        "req-b",
        start_timestep=200.0,
        initial_latent=2.0,
        guidance=5.0,
        do_true_cfg=True,
        true_cfg_scale=6.5,
        cfg_normalize=True,
        image_latent_value=22.0,
    )

    output = DiffusionModelRunner.execute_stepwise(
        runner,
        _scheduler_output_for_new_reqs(
            [("req-a", req_a), ("req-b", req_b)],
            step_id=0,
        ),
    )

    assert output.req_id == ["req-a", "req-b"]
    snapshot = runner.pipeline.denoise_snapshots[0]
    assert snapshot["do_true_cfg"] is True
    assert snapshot["true_cfg_scale"] == pytest.approx(6.5)
    assert snapshot["cfg_normalize"] is True
    assert snapshot["guidance"] is not None
    assert snapshot["image_latents"] is not None
    torch.testing.assert_close(
        snapshot["timesteps"],
        torch.tensor([100.0, 100.0, 200.0]),
    )
    torch.testing.assert_close(
        snapshot["guidance"],
        torch.tensor([3.0, 4.0, 5.0]),
    )
    torch.testing.assert_close(
        snapshot["image_latents"],
        torch.tensor([[11.0], [11.0], [22.0]]),
    )
