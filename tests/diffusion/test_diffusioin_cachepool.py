# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.experimental as diffusion_experimental
import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from vllm_omni.diffusion.cache.cache_manager import CacheManager, CacheStateDriver
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.driver import TeaCacheStateDriver
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput, NewRequestData
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.utils import CacheBackendSlot

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@contextmanager
def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    yield


@pytest.fixture(autouse=True)
def _enable_experiment_cachepool(monkeypatch):
    monkeypatch.setattr(diffusion_experimental, "EXPERIMENT_CACHEPOOL", True)


def _make_request(req_id: str, num_inference_steps: int = 3):
    return SimpleNamespace(
        prompts=[f"prompt-{req_id}"],
        request_ids=[req_id],
        sampling_params=SimpleNamespace(
            generator=None,
            seed=None,
            generator_device=None,
            num_inference_steps=num_inference_steps,
        ),
    )


def _make_new_scheduler_output(req, step_id: int = 0, finished_req_ids=None):
    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=[NewRequestData(sched_req_id=req.request_ids[0], req=req)],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set() if finished_req_ids is None else set(finished_req_ids),
        num_running_reqs=1,
        num_waiting_reqs=0,
    )


def _make_cached_scheduler_output(sched_req_id: str, step_id: int = 0, finished_req_ids=None):
    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(sched_req_ids=[sched_req_id]),
        finished_req_ids=set() if finished_req_ids is None else set(finished_req_ids),
        num_running_reqs=1,
        num_waiting_reqs=0,
    )


@dataclass(frozen=True)
class _CacheDiTSnapshot:
    req_id: str
    step_index: int
    active_req_id: str | None
    waiting_req_ids: tuple[str, ...]
    resident_req_ids: tuple[str, ...]
    slot_id: int
    trace_before: tuple[str, ...]


class _FakeCacheDiTDriver(CacheStateDriver):
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._next_slot_id = 0
        self.initialize_history: list[tuple[int, int]] = []
        self.deactivate_history: list[int] = []
        self.clear_history: list[int] = []

    @property
    def backend_name(self) -> str:
        return "cache_dit"

    def create_empty_slot(self) -> CacheBackendSlot:
        self._next_slot_id += 1
        return CacheBackendSlot(
            backend_name=self.backend_name,
            payload={
                "slot_id": self._next_slot_id,
                "trace": [],
            },
        )

    def install_slot(self, slot: CacheBackendSlot) -> None:
        self.pipeline.live_cache_slot = slot

    def initialize_fresh_slot(self, slot: CacheBackendSlot, num_inference_steps: int) -> None:
        slot.payload["trace"].clear()
        slot.payload["initialized_for_steps"] = num_inference_steps
        self.initialize_history.append((slot.payload["slot_id"], num_inference_steps))

    def is_slot_compatible(self, slot: CacheBackendSlot, num_inference_steps: int) -> bool:
        return slot.metadata.get("num_inference_steps") == num_inference_steps

    def deactivate_slot(self, slot: CacheBackendSlot | None) -> None:
        if slot is not None:
            self.deactivate_history.append(slot.payload["slot_id"])
        self.pipeline.live_cache_slot = None

    def clear_slot(self, slot: CacheBackendSlot) -> None:
        self.clear_history.append(slot.payload["slot_id"])
        slot.payload["trace"].clear()
        slot.payload["cleared"] = True
        slot.metadata.clear()
        slot.resident_bytes = 0
        if self.pipeline.live_cache_slot is slot:
            self.pipeline.live_cache_slot = None

    def estimate_slot_bytes(self, slot: CacheBackendSlot) -> int:
        return 1024 + 64 * len(slot.payload["trace"])


class _CacheDiTPipeline:
    supports_step_execution = True

    def __init__(self):
        self.runner = None
        self.live_cache_slot = None
        self.snapshots: list[_CacheDiTSnapshot] = []
        self.prepare_calls: list[str] = []
        self.decode_calls: list[str] = []
        self.none_req_ids: set[str] = set()
        self.fail_req_ids: set[str] = set()

    def prepare_encode(self, state, **kwargs):
        del kwargs
        self.prepare_calls.append(state.req_id)
        num_steps = state.sampling.num_inference_steps
        state.timesteps = [torch.tensor(num_steps - idx) for idx in range(num_steps)]
        state.latents = torch.tensor([0.0])
        state.prompt_embeds = torch.zeros((1, 2, 4), dtype=torch.float32)
        state.prompt_embeds_mask = torch.tensor([[True, True]])
        return state

    def denoise_step(self, input_batch, **kwargs):
        del input_batch, kwargs
        assert self.runner is not None
        active_req_id = self.runner.cache_manager._active_req_id
        if active_req_id is None:
            raise AssertionError("cache manager must activate one request before denoise_step")
        state = self.runner.state_cache[active_req_id]
        assert self.live_cache_slot is state.cache_slot

        resident_req_ids = tuple(
            sorted(
                req_id
                for req_id, cached_state in self.runner.state_cache.items()
                if cached_state.cache_slot is not None
            )
        )
        waiting_req_ids = tuple(req_id for req_id in resident_req_ids if req_id != active_req_id)
        trace_before = tuple(state.cache_slot.payload["trace"])
        self.snapshots.append(
            _CacheDiTSnapshot(
                req_id=state.req_id,
                step_index=state.step_index,
                active_req_id=active_req_id,
                waiting_req_ids=waiting_req_ids,
                resident_req_ids=resident_req_ids,
                slot_id=state.cache_slot.payload["slot_id"],
                trace_before=trace_before,
            )
        )

        if state.req_id in self.fail_req_ids:
            raise RuntimeError(f"boom-{state.req_id}")
        if state.req_id in self.none_req_ids:
            return None

        state.cache_slot.payload["trace"].append(f"{state.req_id}:{state.step_index}")
        return torch.tensor([float(state.step_index + 1)])

    def step_scheduler(self, state, noise_pred, **kwargs):
        del noise_pred, kwargs
        state.step_index += 1

    def post_decode(self, state, **kwargs):
        del kwargs
        self.decode_calls.append(state.req_id)
        return DiffusionOutput(output=torch.tensor([state.step_index], dtype=torch.float32))


def _make_cache_dit_runner():
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.od_config = SimpleNamespace(
        cache_backend="cache_dit",
        parallel_config=SimpleNamespace(use_hsdp=False),
    )
    runner.device = torch.device("cpu")
    runner.pipeline = _CacheDiTPipeline()
    runner.cache_backend = SimpleNamespace(is_enabled=lambda: True)
    runner.cache_manager = CacheManager(_FakeCacheDiTDriver(runner.pipeline))
    runner.offload_backend = None
    runner.state_cache = {}
    runner.kv_transfer_manager = SimpleNamespace()
    runner.pipeline.runner = runner
    return runner


@dataclass(frozen=True)
class _TeaCacheSnapshot:
    req_id: str
    step_index: int
    active_req_id: str | None
    waiting_req_ids: tuple[str, ...]
    resident_req_ids: tuple[str, ...]
    forward_cnt_before: int
    positive_cnt_before: int
    residual_sum_before: float


class _TeaHookRegistry:
    def __init__(self, hook: TeaCacheHook):
        self._hook = hook

    def get_hook(self, name: str):
        if name == TeaCacheHook._HOOK_NAME:
            return self._hook
        return None


class _TeaCachePipeline:
    supports_step_execution = True

    def __init__(self):
        self.runner = None
        self.snapshots: list[_TeaCacheSnapshot] = []
        self.prepare_calls: list[str] = []
        self.decode_calls: list[str] = []
        self.none_req_ids: set[str] = set()
        self.fail_req_ids: set[str] = set()
        self.hook = TeaCacheHook(
            TeaCacheConfig(
                transformer_type="QwenImageTransformer2DModel",
                coefficients=[0.0, 0.0, 0.0, 0.0, 1.0],
            )
        )
        self.hook.state_manager.set_context("teacache")
        self.transformer = SimpleNamespace(_hook_registry=_TeaHookRegistry(self.hook))

    def prepare_encode(self, state, **kwargs):
        del kwargs
        self.prepare_calls.append(state.req_id)
        num_steps = state.sampling.num_inference_steps
        state.timesteps = [torch.tensor(num_steps - idx) for idx in range(num_steps)]
        state.latents = torch.tensor([0.0])
        state.prompt_embeds = torch.zeros((1, 2, 4), dtype=torch.float32)
        state.prompt_embeds_mask = torch.tensor([[True, True]])
        return state

    def denoise_step(self, input_batch, **kwargs):
        del input_batch, kwargs
        assert self.runner is not None
        active_req_id = self.runner.cache_manager._active_req_id
        if active_req_id is None:
            raise AssertionError("cache manager must activate one request before denoise_step")
        state = self.runner.state_cache[active_req_id]
        resident_req_ids = tuple(
            sorted(
                req_id
                for req_id, cached_state in self.runner.state_cache.items()
                if cached_state.cache_slot is not None
            )
        )
        waiting_req_ids = tuple(req_id for req_id in resident_req_ids if req_id != active_req_id)

        self.hook.state_manager.set_context("teacache_positive")
        positive_state = self.hook.state_manager.get_state()
        residual_sum_before = 0.0
        if positive_state.previous_residual is not None:
            residual_sum_before = float(positive_state.previous_residual.sum().item())

        self.snapshots.append(
            _TeaCacheSnapshot(
                req_id=state.req_id,
                step_index=state.step_index,
                active_req_id=active_req_id,
                waiting_req_ids=waiting_req_ids,
                resident_req_ids=resident_req_ids,
                forward_cnt_before=self.hook._forward_cnt,
                positive_cnt_before=positive_state.cnt,
                residual_sum_before=residual_sum_before,
            )
        )

        if state.req_id in self.fail_req_ids:
            raise RuntimeError(f"boom-{state.req_id}")
        if state.req_id in self.none_req_ids:
            return None

        positive_state.cnt += 1
        positive_state.previous_modulated_input = torch.full((2,), float(state.step_index + 1))
        positive_state.previous_residual = torch.full((2,), float(10 * positive_state.cnt))
        positive_state.previous_residual_encoder = torch.full((1,), float(100 * positive_state.cnt))

        self.hook.state_manager.set_context("teacache_negative")
        negative_state = self.hook.state_manager.get_state()
        negative_state.cnt = positive_state.cnt
        negative_state.previous_modulated_input = torch.full((1,), float(1000 + state.step_index))
        negative_state.previous_residual = torch.full((1,), float(2000 + state.step_index))

        self.hook.state_manager.set_context("teacache_positive")
        self.hook._forward_cnt += 1
        return torch.tensor([float(positive_state.cnt)])

    def step_scheduler(self, state, noise_pred, **kwargs):
        del noise_pred, kwargs
        state.step_index += 1

    def post_decode(self, state, **kwargs):
        del kwargs
        self.decode_calls.append(state.req_id)
        return DiffusionOutput(output=torch.tensor([state.step_index], dtype=torch.float32))


def _make_teacache_runner():
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.od_config = SimpleNamespace(
        cache_backend="tea_cache",
        parallel_config=SimpleNamespace(use_hsdp=False),
    )
    runner.device = torch.device("cpu")
    runner.pipeline = _TeaCachePipeline()
    runner.cache_backend = SimpleNamespace(is_enabled=lambda: True)
    runner.cache_manager = CacheManager(TeaCacheStateDriver(runner.pipeline))
    runner.offload_backend = None
    runner.state_cache = {}
    runner.kv_transfer_manager = SimpleNamespace()
    runner.pipeline.runner = runner
    return runner


class TestExecuteStepwiseCacheDiTCachePool:
    def test_cache_dit_switching_keeps_acceleration_state(self, monkeypatch):
        runner = _make_cache_dit_runner()
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        req_a = _make_request("req-a", num_inference_steps=3)
        req_b = _make_request("req-b", num_inference_steps=3)

        first = DiffusionModelRunner.execute_stepwise(runner, _make_new_scheduler_output(req_a, step_id=0))
        slot_a = runner.state_cache["req-a"].cache_slot

        second = DiffusionModelRunner.execute_stepwise(runner, _make_new_scheduler_output(req_b, step_id=1))
        third = DiffusionModelRunner.execute_stepwise(runner, _make_cached_scheduler_output("req-a", step_id=2))

        assert first.req_id == "req-a"
        assert first.step_index == 1
        assert first.finished is False
        assert second.req_id == "req-b"
        assert second.step_index == 1
        assert second.finished is False
        assert third.req_id == "req-a"
        assert third.step_index == 2
        assert third.finished is False

        assert runner.state_cache["req-a"].cache_slot is slot_a
        assert runner.state_cache["req-a"].cache_slot.payload["trace"] == ["req-a:0", "req-a:1"]
        assert runner.state_cache["req-b"].cache_slot.payload["trace"] == ["req-b:0"]
        assert [snapshot.trace_before for snapshot in runner.pipeline.snapshots if snapshot.req_id == "req-a"] == [
            (),
            ("req-a:0",),
        ]
        assert runner.cache_manager.driver.initialize_history == [(1, 3), (2, 3)]
        assert runner.cache_manager._active_req_id is None
        assert runner.pipeline.live_cache_slot is None

    def test_hbm_can_hold_multiple_waiting_slots_while_one_request_runs(self, monkeypatch):
        runner = _make_cache_dit_runner()
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        for step_id, req_id in enumerate(("req-a", "req-b", "req-c")):
            output = DiffusionModelRunner.execute_stepwise(
                runner,
                _make_new_scheduler_output(_make_request(req_id, num_inference_steps=4), step_id=step_id),
            )
            assert output.finished is False

        snapshot = runner.pipeline.snapshots[-1]
        resident_slots = {req_id: state.cache_slot for req_id, state in runner.state_cache.items()}

        assert snapshot.req_id == "req-c"
        assert snapshot.active_req_id == "req-c"
        assert snapshot.waiting_req_ids == ("req-a", "req-b")
        assert snapshot.resident_req_ids == ("req-a", "req-b", "req-c")
        assert set(resident_slots) == {"req-a", "req-b", "req-c"}
        assert len({slot.payload["slot_id"] for slot in resident_slots.values()}) == 3
        assert all(slot.resident_bytes > 0 for slot in resident_slots.values())
        assert runner.cache_manager._active_req_id is None
        assert runner.pipeline.live_cache_slot is None

    def test_finished_req_ids_free_resident_slot_before_next_run(self, monkeypatch):
        runner = _make_cache_dit_runner()
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        req_a = _make_request("req-a", num_inference_steps=3)
        req_b = _make_request("req-b", num_inference_steps=3)

        DiffusionModelRunner.execute_stepwise(runner, _make_new_scheduler_output(req_a, step_id=0))
        slot_a = runner.state_cache["req-a"].cache_slot

        output = DiffusionModelRunner.execute_stepwise(
            runner,
            _make_new_scheduler_output(req_b, step_id=1, finished_req_ids={"req-a"}),
        )

        assert output.req_id == "req-b"
        assert "req-a" not in runner.state_cache
        assert slot_a.metadata == {}
        assert slot_a.resident_bytes == 0
        assert runner.cache_manager.driver.clear_history == [slot_a.payload["slot_id"]]
        assert runner.state_cache["req-b"].cache_slot.payload["trace"] == ["req-b:0"]

    def test_denoise_none_path_cleans_up_cache_slot(self, monkeypatch):
        runner = _make_cache_dit_runner()
        runner.pipeline.none_req_ids.add("req-a")
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        output = DiffusionModelRunner.execute_stepwise(
            runner,
            _make_new_scheduler_output(_make_request("req-a", num_inference_steps=3), step_id=0),
        )

        assert output.req_id == "req-a"
        assert output.step_index == 0
        assert output.finished is True
        assert output.result is not None
        assert output.result.error == "stepwise denoise returned None"
        assert "req-a" not in runner.state_cache
        assert runner.cache_manager.driver.deactivate_history == [1]
        assert runner.cache_manager.driver.clear_history == [1]
        assert runner.cache_manager._active_req_id is None
        assert runner.pipeline.live_cache_slot is None

    def test_pipeline_error_still_deactivates_active_slot(self, monkeypatch):
        runner = _make_cache_dit_runner()
        runner.pipeline.fail_req_ids.add("req-a")
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        with pytest.raises(RuntimeError, match="boom-req-a"):
            DiffusionModelRunner.execute_stepwise(
                runner,
                _make_new_scheduler_output(_make_request("req-a", num_inference_steps=3), step_id=0),
            )

        assert runner.cache_manager.driver.deactivate_history == [1]
        assert runner.cache_manager._active_req_id is None
        assert runner.pipeline.live_cache_slot is None
        assert runner.state_cache == {}
        assert runner.input_batch is None


    def test_stepwise_cache_backend_requires_experiment_cachepool(self, monkeypatch):
        runner = _make_cache_dit_runner()
        monkeypatch.setattr(diffusion_experimental, "EXPERIMENT_CACHEPOOL", False)
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        with pytest.raises(ValueError, match="global EXPERIMENT_CACHEPOOL=True"):
            DiffusionModelRunner.execute_stepwise(
                runner,
                _make_new_scheduler_output(_make_request("req-a", num_inference_steps=3), step_id=0),
            )


class TestExecuteStepwiseTeaCachePool:
    def test_teacache_switching_restores_hook_state(self, monkeypatch):
        runner = _make_teacache_runner()
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        req_a = _make_request("req-a", num_inference_steps=3)
        req_b = _make_request("req-b", num_inference_steps=3)

        first = DiffusionModelRunner.execute_stepwise(runner, _make_new_scheduler_output(req_a, step_id=0))
        slot_a = runner.state_cache["req-a"].cache_slot
        second = DiffusionModelRunner.execute_stepwise(runner, _make_new_scheduler_output(req_b, step_id=1))
        third = DiffusionModelRunner.execute_stepwise(runner, _make_cached_scheduler_output("req-a", step_id=2))

        assert first.finished is False
        assert second.finished is False
        assert third.finished is False
        assert third.step_index == 2

        payload_a = slot_a.payload
        assert sorted(payload_a["states"]) == ["teacache_negative", "teacache_positive"]
        assert payload_a["forward_cnt"] == 2
        assert payload_a["states"]["teacache_positive"].cnt == 2
        assert payload_a["states"]["teacache_negative"].cnt == 2
        assert runner.state_cache["req-b"].cache_slot.payload["forward_cnt"] == 1
        assert runner.state_cache["req-b"].cache_slot.payload["states"]["teacache_positive"].cnt == 1

        req_a_snapshots = [snapshot for snapshot in runner.pipeline.snapshots if snapshot.req_id == "req-a"]
        assert [snapshot.positive_cnt_before for snapshot in req_a_snapshots] == [0, 1]
        assert [snapshot.forward_cnt_before for snapshot in req_a_snapshots] == [0, 1]
        assert req_a_snapshots[1].residual_sum_before == pytest.approx(20.0)
        assert runner.pipeline.hook._forward_cnt == 0
        assert runner.pipeline.hook.state_manager._states == {}
        assert runner.cache_manager._active_req_id is None

    def test_runner_interleaves_two_teacache_requests_end_to_end(self, monkeypatch):
        runner = _make_teacache_runner()
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        req_a = _make_request("req-a", num_inference_steps=2)
        req_b = _make_request("req-b", num_inference_steps=2)

        first = DiffusionModelRunner.execute_stepwise(runner, _make_new_scheduler_output(req_a, step_id=0))
        slot_a = runner.state_cache["req-a"].cache_slot
        second = DiffusionModelRunner.execute_stepwise(runner, _make_new_scheduler_output(req_b, step_id=1))
        slot_b = runner.state_cache["req-b"].cache_slot
        third = DiffusionModelRunner.execute_stepwise(runner, _make_cached_scheduler_output("req-a", step_id=2))
        fourth = DiffusionModelRunner.execute_stepwise(runner, _make_cached_scheduler_output("req-b", step_id=3))

        assert first.finished is False
        assert second.finished is False
        assert third.finished is True
        assert fourth.finished is True
        assert third.result is not None
        assert fourth.result is not None
        assert torch.equal(third.result.output, torch.tensor([2.0]))
        assert torch.equal(fourth.result.output, torch.tensor([2.0]))

        assert runner.state_cache == {}
        assert runner.pipeline.decode_calls == ["req-a", "req-b"]
        assert slot_a.metadata == {}
        assert slot_b.metadata == {}
        assert slot_a.payload["states"] == {}
        assert slot_b.payload["states"] == {}
        assert slot_a.payload["forward_cnt"] == 0
        assert slot_b.payload["forward_cnt"] == 0
        assert slot_a.resident_bytes == 0
        assert slot_b.resident_bytes == 0
        assert runner.pipeline.hook._forward_cnt == 0
        assert runner.pipeline.hook.state_manager._states == {}
