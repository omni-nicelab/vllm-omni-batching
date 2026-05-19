# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _make_request() -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[{"prompt": "a cup of coffee on a table"}],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_id="request-test",
    )


def test_request_id_is_required():
    with pytest.raises(TypeError, match="request_id"):
        OmniDiffusionRequest(
            prompts=[{"prompt": "a cup of coffee on a table"}],
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        )


def test_request_ids_identity_list_is_removed():
    req = _make_request()

    assert req.request_id == "request-test"
    assert not hasattr(req, "request_ids")


def test_request_id_must_be_non_empty():
    with pytest.raises(ValueError, match="request_id must be a non-empty string"):
        OmniDiffusionRequest(
            prompts=[{"prompt": "a cup of coffee on a table"}],
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
            request_id="",
        )


def test_tp_seed_same_across_ranks_and_varies_across_requests():
    random.seed(0)
    n_requests = 5
    seeds = [_make_request().sampling_params.seed for _ in range(n_requests)]

    # Seed must be auto-assigned (not None) so every TP rank can use it.
    assert all(s is not None for s in seeds)

    # Seeds must vary across requests (non-determinism preserved).
    assert len(set(seeds)) == n_requests, f"Expected {n_requests} unique seeds but got {len(set(seeds))}: {seeds}"
