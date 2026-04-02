# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

<<<<<<< HEAD
import os
import time
=======
>>>>>>> 1ae84936 (Add diffusion model state and input batch tests)
from types import SimpleNamespace

import pytest
import torch

<<<<<<< HEAD
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.worker.input_batch import InputBatch, scatter_latents
from vllm_omni.diffusion.worker.model_states.default import DefaultModelState
=======
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.model_states.default import (
    DefaultModelState,
    DiffusionAttentionMetadata,
)
>>>>>>> 1ae84936 (Add diffusion model state and input batch tests)
from vllm_omni.diffusion.worker.utils import DiffusionRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

<<<<<<< HEAD
RUN_PERF_TESTS = os.getenv("VLLM_OMNI_RUN_PERF_TESTS") == "1"

=======
>>>>>>> 1ae84936 (Add diffusion model state and input batch tests)

def _make_state(
    req_id: str,
    *,
    prompt_seq_len: int,
    negative_prompt_seq_len: int | None,
    latent_value: float,
    timestep_values: list[float],
    img_shape: tuple[int, int, int],
<<<<<<< HEAD
    num_latent_rows: int = 1,
    guidance: float | list[float] | None = None,
    do_true_cfg: bool = False,
    true_cfg_scale: float | None = None,
    cfg_normalize: bool = False,
    image_latent_value: float | None = None,
=======
>>>>>>> 1ae84936 (Add diffusion model state and input batch tests)
) -> DiffusionRequestState:
    state = DiffusionRequestState(
        req_id=req_id,
        sampling=SimpleNamespace(),
        prompts=["prompt"],
    )
    state.prompt_embeds = torch.arange(prompt_seq_len * 2, dtype=torch.float32).reshape(1, prompt_seq_len, 2)
    state.prompt_embeds_mask = torch.ones((1, prompt_seq_len), dtype=torch.int32)
    if negative_prompt_seq_len is not None:
        state.negative_prompt_embeds = torch.arange(
            negative_prompt_seq_len * 2,
            dtype=torch.float32,
        ).reshape(1, negative_prompt_seq_len, 2)
        state.negative_prompt_embeds_mask = torch.ones((1, negative_prompt_seq_len), dtype=torch.int32)
        state.negative_txt_seq_lens = [negative_prompt_seq_len]

<<<<<<< HEAD
    state.latents = torch.full(
        (num_latent_rows, 1, 2),
        latent_value,
        dtype=torch.float32,
    )
    state.timesteps = torch.tensor(timestep_values, dtype=torch.float32)
    state.step_index = 0
    if guidance is not None:
        state.guidance = torch.as_tensor(guidance, dtype=torch.float32)
    state.do_true_cfg = do_true_cfg
    if true_cfg_scale is not None:
        state.sampling.true_cfg_scale = true_cfg_scale
    if cfg_normalize:
        state.sampling.cfg_normalize = True
    if image_latent_value is not None:
        state.sampling.image_latent = torch.full(
            (num_latent_rows, 1, 2),
            image_latent_value,
            dtype=torch.float32,
        )
=======
    state.latents = torch.full((1, 1, 2), latent_value, dtype=torch.float32)
    state.timesteps = torch.tensor(timestep_values, dtype=torch.float32)
    state.step_index = 0
>>>>>>> 1ae84936 (Add diffusion model state and input batch tests)
    state.img_shapes = [[img_shape]]
    state.txt_seq_lens = [prompt_seq_len]
    return state


<<<<<<< HEAD
def _make_perf_state(
    req_id: str,
    *,
    prompt_seq_len: int,
    hidden_size: int,
    latent_hw: int,
    latent_value: float,
) -> DiffusionRequestState:
    state = DiffusionRequestState(
        req_id=req_id,
        sampling=SimpleNamespace(),
        prompts=["prompt"],
    )
    state.prompt_embeds = torch.randn(
        (1, prompt_seq_len, hidden_size),
        dtype=torch.float16,
    )
    state.prompt_embeds_mask = torch.ones((1, prompt_seq_len), dtype=torch.bool)
    state.negative_prompt_embeds = torch.randn(
        (1, prompt_seq_len, hidden_size),
        dtype=torch.float16,
    )
    state.negative_prompt_embeds_mask = torch.ones(
        (1, prompt_seq_len),
        dtype=torch.bool,
    )
    state.latents = torch.full(
        (1, 4, latent_hw, latent_hw),
        latent_value,
        dtype=torch.float32,
    )
    state.timesteps = torch.tensor([1000.0, 999.0], dtype=torch.float32)
    state.step_index = 0
    state.img_shapes = [[(1, 1024, 1024)]]
    state.txt_seq_lens = [prompt_seq_len]
    state.negative_txt_seq_lens = [prompt_seq_len]
    return state


def test_make_batch_uses_idx_mapping_as_source_state_bridge():
=======
def test_from_states_uses_idx_mapping_as_source_state_bridge():
>>>>>>> 1ae84936 (Add diffusion model state and input batch tests)
    state_a = _make_state(
        "req-a",
        prompt_seq_len=2,
        negative_prompt_seq_len=1,
        latent_value=1.0,
        timestep_values=[10.0, 9.0],
        img_shape=(1, 16, 16),
    )
    state_b = _make_state(
        "req-b",
        prompt_seq_len=3,
        negative_prompt_seq_len=2,
        latent_value=2.0,
        timestep_values=[20.0, 19.0],
        img_shape=(1, 32, 32),
    )

<<<<<<< HEAD
    batch = InputBatch.make_batch(
=======
    batch = InputBatch.from_states(
>>>>>>> 1ae84936 (Add diffusion model state and input batch tests)
        [state_a, state_b],
        idx_mapping=torch.tensor([1, 0], dtype=torch.int32),
    )

    assert batch.req_ids == ["req-b", "req-a"]
    assert batch.num_reqs == 2
    assert batch.num_reqs_after_padding == 2
    assert torch.equal(batch.idx_mapping.cpu(), torch.tensor([1, 0], dtype=torch.int32))
    assert batch.idx_mapping_np.tolist() == [1, 0]
    torch.testing.assert_close(batch.timesteps.cpu(), torch.tensor([20.0, 10.0]))
    torch.testing.assert_close(
        batch.latents.cpu(),
        torch.tensor(
            [
                [[2.0, 2.0]],
                [[1.0, 1.0]],
            ]
        ),
    )
    assert tuple(batch.prompt_embeds.shape) == (2, 3, 2)
    assert batch.prompt_embeds_mask is not None
    assert batch.prompt_embeds_mask.tolist() == [[True, True, True], [True, True, False]]
    assert batch.img_shapes == [[(1, 32, 32)], [(1, 16, 16)]]
    assert batch.txt_seq_lens == [3, 2]
    assert batch.negative_txt_seq_lens == [2, 1]


<<<<<<< HEAD
def test_make_batch_reuses_cached_batch_for_same_composition():
=======
def test_from_states_reuses_cached_batch_for_same_composition():
>>>>>>> 1ae84936 (Add diffusion model state and input batch tests)
    state_a = _make_state(
        "req-a",
        prompt_seq_len=2,
        negative_prompt_seq_len=None,
        latent_value=1.0,
        timestep_values=[10.0, 9.0],
        img_shape=(1, 16, 16),
    )
    state_b = _make_state(
        "req-b",
        prompt_seq_len=2,
        negative_prompt_seq_len=None,
        latent_value=2.0,
        timestep_values=[20.0, 19.0],
        img_shape=(1, 16, 16),
    )

<<<<<<< HEAD
    batch = InputBatch.make_batch([state_a, state_b])
=======
    batch = InputBatch.from_states([state_a, state_b])
>>>>>>> 1ae84936 (Add diffusion model state and input batch tests)
    prompt_embeds_before = batch.prompt_embeds.clone()
    prompt_embeds_mask_before = None if batch.prompt_embeds_mask is None else batch.prompt_embeds_mask.clone()
    prompt_embeds_ptr = batch.prompt_embeds.data_ptr()

    state_a.latents = torch.full((1, 1, 2), 3.0, dtype=torch.float32)
    state_b.latents = torch.full((1, 1, 2), 4.0, dtype=torch.float32)
    state_a.step_index = 1
    state_b.step_index = 1

<<<<<<< HEAD
    reused_batch = InputBatch.make_batch([state_a, state_b], cached_batch=batch)
=======
    reused_batch = InputBatch.from_states([state_a, state_b], cached_batch=batch)
>>>>>>> 1ae84936 (Add diffusion model state and input batch tests)

    assert reused_batch is batch
    assert reused_batch.prompt_embeds.data_ptr() == prompt_embeds_ptr
    torch.testing.assert_close(reused_batch.prompt_embeds, prompt_embeds_before)
    if prompt_embeds_mask_before is not None:
        assert reused_batch.prompt_embeds_mask is not None
        torch.testing.assert_close(reused_batch.prompt_embeds_mask, prompt_embeds_mask_before)
    torch.testing.assert_close(
        reused_batch.latents.cpu(),
        torch.tensor(
            [
                [[3.0, 3.0]],
                [[4.0, 4.0]],
            ]
        ),
    )
    torch.testing.assert_close(reused_batch.timesteps.cpu(), torch.tensor([9.0, 19.0]))


<<<<<<< HEAD
def test_make_batch_rebuilds_cached_batch_when_composition_changes():
=======
def test_from_states_rebuilds_cached_batch_when_composition_changes():
>>>>>>> 1ae84936 (Add diffusion model state and input batch tests)
    state_a = _make_state(
        "req-a",
        prompt_seq_len=2,
        negative_prompt_seq_len=None,
        latent_value=1.0,
        timestep_values=[10.0],
        img_shape=(1, 16, 16),
    )
    state_b = _make_state(
        "req-b",
        prompt_seq_len=3,
        negative_prompt_seq_len=None,
        latent_value=2.0,
        timestep_values=[20.0],
        img_shape=(1, 32, 32),
    )
<<<<<<< HEAD
    batch = InputBatch.make_batch([state_a])

    rebuilt_batch = InputBatch.make_batch([state_a, state_b], cached_batch=batch)
=======
    batch = InputBatch.from_states([state_a])

    rebuilt_batch = InputBatch.from_states([state_a, state_b], cached_batch=batch)
>>>>>>> 1ae84936 (Add diffusion model state and input batch tests)

    assert rebuilt_batch is batch
    assert rebuilt_batch.req_ids == ["req-a", "req-b"]
    assert rebuilt_batch.num_reqs == 2
    assert rebuilt_batch.idx_mapping_np.tolist() == [0, 1]
    assert tuple(rebuilt_batch.prompt_embeds.shape) == (2, 3, 2)
    torch.testing.assert_close(rebuilt_batch.timesteps.cpu(), torch.tensor([10.0, 20.0]))


<<<<<<< HEAD
def test_make_batch_rejects_variable_length_prompt_embeds_without_masks():
=======
def test_from_states_rejects_variable_length_prompt_embeds_without_masks():
>>>>>>> 1ae84936 (Add diffusion model state and input batch tests)
    state_a = _make_state(
        "req-a",
        prompt_seq_len=2,
        negative_prompt_seq_len=None,
        latent_value=1.0,
        timestep_values=[10.0],
        img_shape=(1, 16, 16),
    )
    state_b = _make_state(
        "req-b",
        prompt_seq_len=3,
        negative_prompt_seq_len=None,
        latent_value=2.0,
        timestep_values=[20.0],
        img_shape=(1, 16, 16),
    )
    state_a.prompt_embeds_mask = None
    state_b.prompt_embeds_mask = None

    with pytest.raises(ValueError, match="Variable-length prompt_embeds"):
<<<<<<< HEAD
        InputBatch.make_batch([state_a, state_b])


def test_make_batch_pads_variable_length_negative_prompt_embeds():
=======
        InputBatch.from_states([state_a, state_b])


def test_from_states_pads_variable_length_negative_prompt_embeds():
>>>>>>> 1ae84936 (Add diffusion model state and input batch tests)
    state_a = _make_state(
        "req-a",
        prompt_seq_len=2,
        negative_prompt_seq_len=1,
        latent_value=1.0,
        timestep_values=[10.0],
        img_shape=(1, 16, 16),
    )
    state_b = _make_state(
        "req-b",
        prompt_seq_len=2,
        negative_prompt_seq_len=3,
        latent_value=2.0,
        timestep_values=[20.0],
        img_shape=(1, 16, 16),
    )

<<<<<<< HEAD
    batch = InputBatch.make_batch([state_a, state_b])
=======
    batch = InputBatch.from_states([state_a, state_b])
>>>>>>> 1ae84936 (Add diffusion model state and input batch tests)

    assert batch.negative_prompt_embeds is not None
    assert batch.negative_prompt_embeds_mask is not None
    assert tuple(batch.negative_prompt_embeds.shape) == (2, 3, 2)
    assert batch.negative_prompt_embeds_mask.tolist() == [[True, False, False], [True, True, True]]
    assert batch.negative_txt_seq_lens == [1, 3]


<<<<<<< HEAD
def test_make_batch_collects_cfg_guidance_and_image_latents():
    state_a = _make_state(
        "req-a",
        prompt_seq_len=2,
        negative_prompt_seq_len=None,
        latent_value=1.0,
        timestep_values=[10.0],
        img_shape=(1, 16, 16),
        num_latent_rows=2,
        guidance=[3.0, 4.0],
        do_true_cfg=True,
        true_cfg_scale=6.5,
        cfg_normalize=True,
        image_latent_value=11.0,
    )
    state_b = _make_state(
        "req-b",
        prompt_seq_len=2,
        negative_prompt_seq_len=None,
        latent_value=2.0,
        timestep_values=[20.0],
        img_shape=(1, 16, 16),
        guidance=5.0,
        do_true_cfg=True,
        true_cfg_scale=6.5,
        cfg_normalize=True,
        image_latent_value=22.0,
    )
    state_a.timesteps = torch.tensor([[10.0, 11.0]], dtype=torch.float32)

    batch = InputBatch.make_batch([state_a, state_b])

    assert batch.do_true_cfg is True
    assert batch.true_cfg_scale == pytest.approx(6.5)
    assert batch.cfg_normalize is True
    assert batch.guidance is not None
    assert batch.image_latents is not None
    torch.testing.assert_close(
        batch.timesteps.cpu(),
        torch.tensor([10.0, 11.0, 20.0]),
    )
    torch.testing.assert_close(
        batch.guidance.cpu(),
        torch.tensor([3.0, 4.0, 5.0]),
    )
    torch.testing.assert_close(
        batch.image_latents.cpu(),
        torch.tensor(
            [
                [[11.0, 11.0]],
                [[11.0, 11.0]],
                [[22.0, 22.0]],
            ]
        ),
    )


def test_make_batch_rejects_mixed_guidance_presence():
    state_a = _make_state(
        "req-a",
        prompt_seq_len=2,
        negative_prompt_seq_len=None,
        latent_value=1.0,
        timestep_values=[10.0],
        img_shape=(1, 16, 16),
        guidance=3.0,
    )
    state_b = _make_state(
        "req-b",
        prompt_seq_len=2,
        negative_prompt_seq_len=None,
        latent_value=2.0,
        timestep_values=[20.0],
        img_shape=(1, 16, 16),
    )

    with pytest.raises(ValueError, match="Mixed guidance"):
        InputBatch.make_batch([state_a, state_b])


def test_make_batch_rejects_mixed_cfg_settings():
    state_a = _make_state(
        "req-a",
        prompt_seq_len=2,
        negative_prompt_seq_len=None,
        latent_value=1.0,
        timestep_values=[10.0],
        img_shape=(1, 16, 16),
        do_true_cfg=True,
        true_cfg_scale=6.5,
    )
    state_b = _make_state(
        "req-b",
        prompt_seq_len=2,
        negative_prompt_seq_len=None,
        latent_value=2.0,
        timestep_values=[20.0],
        img_shape=(1, 16, 16),
        do_true_cfg=False,
        true_cfg_scale=6.5,
    )

    with pytest.raises(ValueError, match="Mixed CFG settings"):
        InputBatch.make_batch([state_a, state_b])


def test_make_batch_rejects_mixed_image_latent_presence():
    state_a = _make_state(
        "req-a",
        prompt_seq_len=2,
        negative_prompt_seq_len=None,
        latent_value=1.0,
        timestep_values=[10.0],
        img_shape=(1, 16, 16),
        image_latent_value=11.0,
    )
    state_b = _make_state(
        "req-b",
        prompt_seq_len=2,
        negative_prompt_seq_len=None,
        latent_value=2.0,
        timestep_values=[20.0],
        img_shape=(1, 16, 16),
    )

    with pytest.raises(ValueError, match="Mixed image_latent presence"):
        InputBatch.make_batch([state_a, state_b])


def test_make_batch_rejects_mixed_prompt_masks():
=======
def test_from_states_rejects_mixed_prompt_masks():
>>>>>>> 1ae84936 (Add diffusion model state and input batch tests)
    state_a = _make_state(
        "req-a",
        prompt_seq_len=2,
        negative_prompt_seq_len=None,
        latent_value=1.0,
        timestep_values=[10.0],
        img_shape=(1, 16, 16),
    )
    state_b = _make_state(
        "req-b",
        prompt_seq_len=2,
        negative_prompt_seq_len=None,
        latent_value=2.0,
        timestep_values=[20.0],
        img_shape=(1, 16, 16),
    )
    state_b.prompt_embeds_mask = None

    with pytest.raises(ValueError, match="Mixed prompt_embeds_mask"):
<<<<<<< HEAD
        InputBatch.make_batch([state_a, state_b])


def test_make_batch_rejects_mixed_txt_seq_lens():
=======
        InputBatch.from_states([state_a, state_b])


def test_from_states_rejects_mixed_txt_seq_lens():
>>>>>>> 1ae84936 (Add diffusion model state and input batch tests)
    state_a = _make_state(
        "req-a",
        prompt_seq_len=2,
        negative_prompt_seq_len=None,
        latent_value=1.0,
        timestep_values=[10.0],
        img_shape=(1, 16, 16),
    )
    state_b = _make_state(
        "req-b",
        prompt_seq_len=2,
        negative_prompt_seq_len=None,
        latent_value=2.0,
        timestep_values=[20.0],
        img_shape=(1, 16, 16),
    )
    state_b.txt_seq_lens = None

    with pytest.raises(ValueError, match="Mixed txt_seq_lens"):
<<<<<<< HEAD
        InputBatch.make_batch([state_a, state_b])


def test_scatter_latents_writes_back_to_request_states():
    state_a = _make_state(
        "req-a",
        prompt_seq_len=2,
        negative_prompt_seq_len=None,
        latent_value=1.0,
        timestep_values=[10.0],
        img_shape=(1, 16, 16),
    )
    state_b = _make_state(
        "req-b",
        prompt_seq_len=2,
        negative_prompt_seq_len=None,
        latent_value=2.0,
        timestep_values=[20.0],
        img_shape=(1, 16, 16),
    )

    batch = InputBatch.make_batch(
        [state_a, state_b],
        idx_mapping=torch.tensor([1, 0], dtype=torch.int32),
    )
    batch.latents.copy_(
        torch.tensor(
            [
                [[30.0, 30.0]],
                [[40.0, 40.0]],
            ]
        )
    )

    scatter_latents([state_a, state_b], batch)

    torch.testing.assert_close(
        state_a.latents,
        torch.tensor([[[40.0, 40.0]]]),
    )
    torch.testing.assert_close(
        state_b.latents,
        torch.tensor([[[30.0, 30.0]]]),
    )


@pytest.mark.skipif(
    not RUN_PERF_TESTS,
    reason="Set VLLM_OMNI_RUN_PERF_TESTS=1 to run opt-in perf smoke tests.",
)
@pytest.mark.parametrize("num_reqs", [4, 8])
def test_input_batch_perf_smoke(num_reqs: int):
    prompt_seq_len = 500
    hidden_size = 4096
    latent_hw = 128

    torch.manual_seed(0)
    states = [
        _make_perf_state(
            f"req-{i}",
            prompt_seq_len=prompt_seq_len,
            hidden_size=hidden_size,
            latent_hw=latent_hw,
            latent_value=float(i + 1),
        )
        for i in range(num_reqs)
    ]
    idx_mapping = torch.arange(num_reqs - 1, -1, -1, dtype=torch.int32)

    prepare_start = time.perf_counter()
    batch = InputBatch.make_batch(states, idx_mapping=idx_mapping)
    prepare_ms = (time.perf_counter() - prepare_start) * 1000

    for i, state in enumerate(states):
        state.latents = torch.full_like(state.latents, float(100 + i))
        state.step_index = 1

    repack_start = time.perf_counter()
    batch = InputBatch.make_batch(states, idx_mapping=idx_mapping, cached_batch=batch)
    repack_ms = (time.perf_counter() - repack_start) * 1000

    batch.latents.add_(1.0)
    scatter_start = time.perf_counter()
    scatter_latents(states, batch)
    scatter_ms = (time.perf_counter() - scatter_start) * 1000

    print(
        "\n"
        f"[perf] num_reqs={num_reqs} "
        f"prompt_seq_len={prompt_seq_len} hidden_size={hidden_size} "
        f"image=1024x1024 prepare_ms={prepare_ms:.3f} "
        f"repack_ms={repack_ms:.3f} scatter_ms={scatter_ms:.3f}"
    )

    assert batch.num_reqs == num_reqs
    assert states[0].latents is not None
    torch.testing.assert_close(
        states[idx_mapping[-1].item()].latents,
        batch.latents[-1:].clone(),
    )


def test_prepare_attn_generates_positive_and_negative_mask_metadata():
=======
        InputBatch.from_states([state_a, state_b])


def test_make_dummy_matches_v1_style_interface():
    batch = InputBatch.make_dummy(num_reqs=2, num_tokens=8, device=torch.device("cpu"))

    assert batch.num_reqs == 2
    assert batch.num_reqs_after_padding == 2
    assert batch.idx_mapping_np.tolist() == [0, 1]
    assert tuple(batch.latents.shape) == (2, 0)
    assert tuple(batch.prompt_embeds.shape) == (2, 0, 0)


def test_prepare_attn_generates_positive_and_negative_metadata():
>>>>>>> 1ae84936 (Add diffusion model state and input batch tests)
    state_a = _make_state(
        "req-a",
        prompt_seq_len=2,
        negative_prompt_seq_len=1,
        latent_value=1.0,
        timestep_values=[10.0],
        img_shape=(1, 16, 16),
    )
    state_b = _make_state(
        "req-b",
        prompt_seq_len=3,
        negative_prompt_seq_len=2,
        latent_value=2.0,
        timestep_values=[20.0],
        img_shape=(1, 32, 32),
    )
<<<<<<< HEAD
    batch = InputBatch.make_batch(
=======
    batch = InputBatch.from_states(
>>>>>>> 1ae84936 (Add diffusion model state and input batch tests)
        [state_a, state_b],
        idx_mapping=torch.tensor([1, 0], dtype=torch.int32),
    )

    model_state = DefaultModelState(
        od_config=SimpleNamespace(),
        pipeline=SimpleNamespace(),
        device=torch.device("cpu"),
    )
    attn_metadata = model_state.prepare_attn(batch)

    assert set(attn_metadata) == {"positive", "negative"}
    positive = attn_metadata["positive"]
    negative = attn_metadata["negative"]
<<<<<<< HEAD
    assert isinstance(positive, AttentionMetadata)
    assert isinstance(negative, AttentionMetadata)

    assert positive.attn_mask is not None
    assert positive.attn_mask.tolist() == [
        [True, True, True],
        [True, True, False],
    ]

    assert negative.attn_mask is not None
    assert negative.attn_mask.tolist() == [
        [True, True],
        [True, False],
    ]

    assert batch.img_shapes == [[(1, 32, 32)], [(1, 16, 16)]]
    assert batch.txt_seq_lens == [3, 2]
    assert batch.negative_txt_seq_lens == [2, 1]
=======
    assert isinstance(positive, DiffusionAttentionMetadata)
    assert isinstance(negative, DiffusionAttentionMetadata)

    assert positive.txt_seq_lens == [3, 2]
    assert positive.img_shapes == [[(1, 32, 32)], [(1, 16, 16)]]
    assert positive.max_seqlen_q == 3
    assert positive.max_seqlen_k == 3
    assert positive.seq_lens is not None
    assert positive.seq_lens.tolist() == [3, 2]
    assert positive.cu_seqlens_q is not None
    assert positive.cu_seqlens_k is not None
    assert positive.cu_seqlens_q.tolist() == [0, 3, 5]
    assert positive.cu_seqlens_k.tolist() == [0, 3, 5]
    assert positive.attn_mask is not None
    assert positive.attn_mask.tolist() == [[True, True, True], [True, True, False]]

    assert negative.txt_seq_lens == [2, 1]
    assert negative.max_seqlen_q == 2
    assert negative.max_seqlen_k == 2
    assert negative.seq_lens is not None
    assert negative.seq_lens.tolist() == [2, 1]
    assert negative.cu_seqlens_q is not None
    assert negative.cu_seqlens_q.tolist() == [0, 2, 3]
    assert negative.attn_mask is not None
    assert negative.attn_mask.tolist() == [[True, True], [True, False]]


def test_default_model_state_prepare_inputs_returns_input_batch_view():
    state = _make_state(
        "req-a",
        prompt_seq_len=2,
        negative_prompt_seq_len=1,
        latent_value=1.0,
        timestep_values=[10.0],
        img_shape=(1, 16, 16),
    )
    batch = InputBatch.from_states([state])

    model_state = DefaultModelState(
        od_config=SimpleNamespace(),
        pipeline=SimpleNamespace(),
        device=torch.device("cpu"),
    )
    model_inputs = model_state.prepare_inputs(batch)

    assert model_inputs["latents"] is batch.latents
    assert model_inputs["timesteps"] is batch.timesteps
    assert model_inputs["prompt_embeds"] is batch.prompt_embeds
    assert model_inputs["prompt_embeds_mask"] is batch.prompt_embeds_mask
    assert model_inputs["negative_prompt_embeds"] is batch.negative_prompt_embeds
    assert model_inputs["negative_prompt_embeds_mask"] is batch.negative_prompt_embeds_mask
    assert model_inputs["img_shapes"] == batch.img_shapes
    assert model_inputs["txt_seq_lens"] == batch.txt_seq_lens
    assert model_inputs["negative_txt_seq_lens"] == batch.negative_txt_seq_lens
>>>>>>> 1ae84936 (Add diffusion model state and input batch tests)
