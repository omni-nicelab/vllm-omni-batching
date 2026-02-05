# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque

from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.request import DiffusionRequestState, DiffusionRequestStatus, OmniDiffusionRequest
from vllm_omni.diffusion.worker.step_batch import StepRunnerOutput, StepSchedulerOutput

logger = init_logger(__name__)


class DiffusionStepScheduler:
    """Step-level scheduler only (no IPC or execution).

    Responsibilities:
    - Manage request state lifecycle (waiting/running/suspended/finished)
    - Select requests per step and build StepSchedulerOutput
    - Update internal state from StepRunnerOutput
    """

    def initialize(self, od_config: OmniDiffusionConfig):
        self.od_config = od_config

        # Step-level scheduling state
        # req_id -> DiffusionRequestState
        self._request_states: dict[str, DiffusionRequestState] = {}
        self._waiting: deque[str] = deque()
        self._running: list[str] = []
        self._finished_req_ids: set[str] = set()
        self._preempted_req_ids: set[str] = set()
        self._step_id: int = 0

        # scheduling constraints
        # NOTE: for Qwen-Image, axes_dims_rope: [16,56,56]
        # TODO: better defaults values
        self._max_batch_size: int = int(getattr(od_config, "max_batch_size", 2))
        self._max_model_len: int = int(getattr(od_config, "max_model_len", 56 * 56))
        # NOTE: this is an "image token" budget (i.e. tokens), not pixel-space units.
        self._max_num_scheduled_tokens: int = int(getattr(od_config, "max_num_batched_dit_tokens", 4 * 56 * 56))

        # Validate scheduling constraints following vLLM design
        if self._max_num_scheduled_tokens > self._max_batch_size * self._max_model_len:
            logger.warning(
                "max_num_scheduled_tokens (%d) exceeds max_batch_size * max_model_len (%d * %d = %d). "
                "This may cause memory issues.",
                self._max_num_scheduled_tokens,
                self._max_batch_size,
                self._max_model_len,
                self._max_batch_size * self._max_model_len,
            )
            self._max_num_scheduled_tokens = self._max_batch_size * self._max_model_len

    # ======================================================================
    # Step-level scheduling API
    # ======================================================================

    def add_request(self, request: OmniDiffusionRequest) -> str:
        """Add a request to the scheduler without waiting for completion."""
        req_id = request.request_id
        assert req_id is not None, "Request must have a request_id"
        if req_id in self._request_states:
            raise ValueError(f"Duplicate request_id: {req_id}")

        state = DiffusionRequestState(req_id=req_id, req=request)
        self._request_states[req_id] = state
        self._waiting.append(req_id)
        logger.debug("Scheduler add_request: %s (waiting=%d)", req_id, len(self._waiting))
        return req_id

    def add_requests(self, requests: list[OmniDiffusionRequest]) -> list[str]:
        return [self.add_request(req) for req in requests]

    def schedule(self) -> StepSchedulerOutput:
        """Schedule a single diffusion step.

        Returns a StepSchedulerOutput containing the active request states.
        """
        # TODO: preempt schedule
        token_budget = self._max_num_scheduled_tokens

        # Schedule running requests first
        # (Currently, preemption is not supported)
        token_budget -= self._calculate_current_tokens()

        # Schedule waiting requests
        while self._waiting and len(self._running) < self._max_batch_size and token_budget > 0:
            req_id = self._waiting.popleft()
            if req_id not in self._request_states:
                continue

            # Check if adding this request would exceed tokens budget
            req_tokens = self._get_request_tokens(req_id)
            if req_tokens > token_budget:
                # Put request back to front of queue and break
                self._waiting.appendleft(req_id)
                logger.debug(
                    "Cannot schedule request %s: requires tokens=%d but only tokens=%d available",
                    req_id,
                    req_tokens,
                    token_budget,
                )
                break

            self._request_states[req_id].req.status = DiffusionRequestStatus.RUNNING
            self._running.append(req_id)
            token_budget -= req_tokens

        # Build output with current running requests
        running_states: list[DiffusionRequestState] = []
        for req_id in self._running:
            state = self._request_states.get(req_id)
            if state is not None:
                running_states.append(state)

        scheduler_output = StepSchedulerOutput(
            step_id=self._step_id,
            req_states=running_states,
            finished_req_ids=self._finished_req_ids,
            preempted_req_ids=self._preempted_req_ids,
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._waiting),
        )

        # update after schedule
        self._step_id += 1
        self._finished_req_ids = set()
        self._preempted_req_ids = set()
        return scheduler_output

    def update_from_step(self, sched_output: StepSchedulerOutput, runner_output: StepRunnerOutput) -> set[str]:
        """Update request states based on the runner output.

        NOTE: We intentionally do NOT store latents in the scheduler's state.
        The latents are kept in the Worker's _request_state_cache to avoid
        expensive IPC serialization of large tensors on every step.
        """
        request_states = self._request_states
        scheduled_req_ids = {s.req_id for s in sched_output.req_states}

        # Abnormal finish ids: runner returned results for non-scheduled req_ids,
        # or the scheduler no longer has state for the req_id.
        unexpected_finished_ids: set[str] = set()

        # StepOutput advances denoising only; "normal completion" is determined by decoded
        # (Currently, the decode stage and the last denoise step are executed in the same scheduling cycle).
        for out in runner_output.step_outputs:
            if out.req_id not in scheduled_req_ids:
                unexpected_finished_ids.add(out.req_id)
            state = request_states.get(out.req_id)
            if state is None:
                unexpected_finished_ids.add(out.req_id)
                continue
            # Only update metadata, NOT latents (kept in Worker cache)
            state.step_index = out.step_index + 1
            state.timestep = out.timestep

        completed_req_ids: set[str] = set()
        for req_id, decoded in runner_output.decoded.items():
            if req_id not in scheduled_req_ids:
                unexpected_finished_ids.add(req_id)
            state = request_states.get(req_id)
            if state is None:
                unexpected_finished_ids.add(req_id)
                continue
            state.req.output = decoded
            state.req.status = DiffusionRequestStatus.FINISHED_COMPLETED
            completed_req_ids.add(req_id)

        # clean _running (drop completed + drop missing).
        if completed_req_ids or unexpected_finished_ids:
            new_running: list[str] = []
            for req_id in self._running:
                if req_id in completed_req_ids:
                    continue
                if req_id not in request_states:
                    unexpected_finished_ids.add(req_id)
                    continue
                new_running.append(req_id)
            self._running = new_running

        # Be defensive: a finished request should not be schedulable again.
        for req_id in completed_req_ids:
            try:
                self._waiting.remove(req_id)
            except ValueError:
                pass

        if unexpected_finished_ids:
            self._finished_req_ids |= unexpected_finished_ids
        return completed_req_ids

    def abort_request(self, req_id: str) -> bool:
        """Abort a request and mark it finished."""
        if req_id not in self._request_states:
            return False
        self.finish_request(req_id, DiffusionRequestStatus.FINISHED_ABORTED)
        self._finished_req_ids.add(req_id)
        return True

    def has_requests(self) -> bool:
        """Return True if there are unfinished requests."""
        return bool(self._waiting or self._running)

    def get_request_state(self, req_id: str) -> DiffusionRequestState | None:
        return self._request_states.get(req_id)

    def pop_request_state(self, req_id: str) -> DiffusionRequestState | None:
        state = self._request_states.pop(req_id, None)
        return state

    def preempt_request(self, req_id: str) -> bool:
        """Preempt a running request and move it back to waiting."""
        if req_id not in self._request_states:
            return False
        if req_id in self._running:
            self._running.remove(req_id)
            self._waiting.appendleft(req_id)
            self._request_states[req_id].req.status = DiffusionRequestStatus.PREEMPTED
            self._preempted_req_ids.add(req_id)
            return True
        return False

    def finish_request(self, req_id: str, status: DiffusionRequestStatus) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert DiffusionRequestStatus.is_finished(status)
        self._request_states[req_id].req.status = status
        if req_id in self._running:
            self._running.remove(req_id)
        try:
            self._waiting.remove(req_id)
        except ValueError:
            pass

    def _calculate_current_tokens(self) -> int:
        """Calculate total tokens (image token count) for all currently running requests."""
        total_tokens = 0
        for req_id in self._running:
            total_tokens += self._get_request_tokens(req_id)
        return total_tokens

    def _get_request_tokens(self, req_id: str) -> int:
        """Get tokens (image token count) for a specific request."""
        state = self._request_states.get(req_id)
        if state is None:
            return 0
        req = state.req

        # NOTE: tokens here refers to the *image token* budget consumed by a request.
        # For Qwen-Image, the effective pixel-space token size is:
        #   token_px = vae_scale_factor * 2
        # because latents are 8x downsampled by VAE (typically) and then packed into 2x2 tokens.
        # In the pipeline this corresponds to `latents.shape[1]`.
        # TODO: save token size in OmniDiffusionRequest/DiffusionRequestState?

        req_batch_size = int(getattr(req, "batch_size", 1) or 1)
        num_frames = self._to_int(getattr(req, "num_frames", 1), default=1) or 1
        num_frames = max(1, num_frames)

        # Prefer latent-space dimensions if the request already carries them.
        height_latents = self._to_int(getattr(req, "height_latents", None))
        width_latents = self._to_int(getattr(req, "width_latents", None))
        if height_latents is not None and width_latents is not None and height_latents > 0 and width_latents > 0:
            # Latents are packed in 2x2 blocks -> tokens is (H_lat/2)*(W_lat/2) per frame.
            tokens_per_frame = (height_latents // 2) * (width_latents // 2)
            tokens_per_sample = max(1, tokens_per_frame) * num_frames
            return tokens_per_sample * req_batch_size

        # Fallback to pixel-space dimensions.
        height = self._to_int(getattr(req, "height", None))
        width = self._to_int(getattr(req, "width", None))
        if height is None or width is None:
            # Qwen-image commonly uses square bucket resolutions; use that if provided.
            resolution = self._to_int(getattr(req, "resolution", None))
            if resolution is not None and resolution > 0:
                height = height or resolution
                width = width or resolution

        if height is None or width is None:
            # Last-resort fallback to configured defaults.
            vae_scale_factor = int(getattr(self.od_config, "vae_scale_factor", 8) or 8)
            default_sample_size = int(getattr(self.od_config, "default_sample_size", 128) or 128)
            height = height or default_sample_size * vae_scale_factor
            width = width or default_sample_size * vae_scale_factor

        # Determine effective token size in pixel space.
        token_size_px = self._get_effective_token_size_px()
        token_size_px = max(1, int(token_size_px))

        tokens_h = max(1, int(height) // token_size_px)
        tokens_w = max(1, int(width) // token_size_px)
        tokens_per_sample = (tokens_h * tokens_w) * num_frames

        return tokens_per_sample * req_batch_size

    @staticmethod
    def _to_int(value, default: int | None = None) -> int | None:
        if value is None:
            return default
        if isinstance(value, list):
            if not value:
                return default
            value = value[0]
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _get_effective_token_size_px(self) -> int:
        """Return pixel-space token size used for scheduler token budgeting.

        For Qwen-Image, effective token size is `vae_scale_factor * 2`.
        This method allows overriding via config without loading the model.
        """
        # Explicit override if user knows exact tokenization.
        override = self._to_int(getattr(self.od_config, "scheduler_token_size_px", None))
        if override is not None and override > 0:
            return override

        model_class = (getattr(self.od_config, "model_class_name", None) or "").lower()
        model_name = (getattr(self.od_config, "model", None) or "").lower()
        is_qwen_image = ("qwen" in model_class and "image" in model_class) or (
            "qwen" in model_name and "image" in model_name
        )

        if is_qwen_image:
            vae_scale_factor = int(getattr(self.od_config, "vae_scale_factor", 8) or 8)
            return vae_scale_factor * 2

        # Generic default: treat one visual token as 16px for image models unless configured.
        default_token_size_px = getattr(self.od_config, "default_token_size_px", 16)
        return int(default_token_size_px or 16)
