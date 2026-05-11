"""MultiDiffusion + circular wraparound for tileable PBR LoRA inference.

What this does
--------------
We have LoRAs fine-tuned at a "native" resolution (FLUX.2-klein-base @ 2048,
Qwen-Image-Edit-2511 / FireRed-Image-Edit-1.1 @ 1024) on tileable PBR data, and
we want inference at 2048×2048 even for the 1024 backbones, with seamless tile
boundaries.

Strategy: classic MultiDiffusion (Bar-Tal et al. 2023) with circular-wrap tiles.

  - Set the pipeline up at the *full* output resolution (e.g. 2048): full prompt
    embeds, full edit_latents, full noise.
  - Replace the standard denoise loop. At each step:
      For each window (offset y0, x0; size = backbone-native, e.g. 1024):
        - Slice the current global latent window (with circular wrap).
        - Slice each per-ref edit_latents at the SAME (y0, x0) window.
        - Build window-local positional IDs / RoPE shapes (the LoRA was trained
          at this resolution so the model sees a familiar input).
        - Run pipe.cfg_guided_model_fn → noise_pred for this window.
        - Add into a global noise_pred buffer with raised-cosine weights.
      Normalize the global noise_pred by accumulated weights → fused noise_pred.
      scheduler.step(noise_pred_global, latents_global) → next latents.
  - VAE decode the final global latent → output PIL image.

Why circular
------------
For seamlessly tileable textures, we want the model's output to also tile. By
sliding tile windows that wrap across the image edges (modulo H, W), every
output pixel is "predicted from a window that crosses its tile partner on the
opposite side", so the implicit constraint becomes "left edge ↔ right edge,
top ↔ bottom". This matches the random-circular-crop training augmentation in
`code/tiling_generation/training/random_circular_crop.py`.

Backbones
---------
Three adapters share the same driver:
  - `_Flux2Adapter`:   FLUX.2-klein-base-9B (flat latent layout, explicit image_ids)
  - `_QwenAdapter`:    Qwen-Image-Edit-2511 / FireRed-Image-Edit-1.1
                       (grid latent layout, RoPE auto-derived from latent shape)

Notes / limitations
-------------------
- Always B=1 (single-sample inference, matching SMBE training).
- `input_image` (img2img) is not supported; only edit_image conditioning. Our
  SMBE pipeline doesn't use input_image.
- Cost ≈ N_windows × N_steps × DiT(window). For 2048→2048, stride=512 →
  16 windows; 30 steps × 16 × 2 (CFG) = 960 forwards at native 1024 size.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm


# =============================================================== #
#                       Common math helpers                        #
# =============================================================== #


def _tile_offsets_axis(L: int, win: int, stride: int, circular: bool) -> List[int]:
    """Generate tile origins along one axis."""
    if win >= L:
        return [0]
    if circular:
        # Cover the full L with a uniform stride; wrap-around gives extra coverage
        # of the boundary. n = ceil(L / stride) slots evenly spaced (mod L).
        n = max(1, math.ceil(L / stride))
        return [(i * stride) % L for i in range(n)]
    offs = list(range(0, L - win + 1, stride))
    if offs[-1] != L - win:
        offs.append(L - win)
    return offs


def _tile_offsets(H: int, W: int, win_h: int, win_w: int,
                  stride_h: int, stride_w: int, circular: bool) -> List[Tuple[int, int]]:
    ys = _tile_offsets_axis(H, win_h, stride_h, circular)
    xs = _tile_offsets_axis(W, win_w, stride_w, circular)
    return [(y, x) for y in ys for x in xs]


def _hann_2d(h: int, w: int, device, dtype) -> torch.Tensor:
    """Separable raised-cosine weight, shape (h, w), strictly positive (>= eps)."""
    yy = 0.5 * (1 - torch.cos(2 * math.pi * (torch.arange(h, device=device) + 0.5) / h))
    xx = 0.5 * (1 - torch.cos(2 * math.pi * (torch.arange(w, device=device) + 0.5) / w))
    yy = yy.clamp_min(1e-3)
    xx = xx.clamp_min(1e-3)
    return (yy[:, None] * xx[None, :]).to(dtype=dtype)


def _circular_indices(start: int, win: int, total: int, device) -> torch.LongTensor:
    return (torch.arange(win, device=device) + start) % total


def _slice_grid_circular(x: torch.Tensor, y0: int, x0: int,
                         win_h: int, win_w: int, circular: bool) -> torch.Tensor:
    """Slice last 2 dims of x at (y0, x0) with size (win_h, win_w).

    circular=True: wrap-around via index_select (modulo last 2 dims).
    circular=False: assume y0+win_h <= H and x0+win_w <= W and use a contiguous slice.
    """
    H, W = x.shape[-2], x.shape[-1]
    if not circular:
        return x[..., y0:y0 + win_h, x0:x0 + win_w]
    ys = _circular_indices(y0, win_h, H, x.device)
    xs = _circular_indices(x0, win_w, W, x.device)
    return x.index_select(-2, ys).index_select(-1, xs)


def _add_window(buf: torch.Tensor, weight_buf: torch.Tensor,
                window: torch.Tensor, weight2d: torch.Tensor,
                y0: int, x0: int, circular: bool) -> None:
    """In-place: buf[..., yy, xx] += window * weight; weight_buf[yy, xx] += weight.

    buf/weight_buf are contiguous tensors with H, W as their last 2 dims.
    weight2d is a (win_h, win_w) tensor; broadcasts across leading dims of buf.
    """
    H, W = buf.shape[-2], buf.shape[-1]
    win_h, win_w = window.shape[-2], window.shape[-1]
    if circular:
        ys = (torch.arange(win_h, device=buf.device) + y0) % H
        xs = (torch.arange(win_w, device=buf.device) + x0) % W
    else:
        ys = torch.arange(y0, y0 + win_h, device=buf.device)
        xs = torch.arange(x0, x0 + win_w, device=buf.device)

    # Flatten last two dims to 1D so we can use index_add_.
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")           # (win_h, win_w)
    flat_idx = (yy * W + xx).reshape(-1)                     # (win_h*win_w,)

    weighted = (window * weight2d).reshape(*window.shape[:-2], win_h * win_w)
    # buf is contiguous; the reshape is a view that shares storage.
    buf_flat = buf.reshape(*buf.shape[:-2], H * W)
    buf_flat.index_add_(-1, flat_idx, weighted.to(buf_flat.dtype))

    weight_buf_flat = weight_buf.reshape(H * W)
    weight_buf_flat.index_add_(-1, flat_idx, weight2d.reshape(-1).to(weight_buf_flat.dtype))


# =============================================================== #
#                       FLUX.2 adapter                             #
# =============================================================== #


class _Flux2Adapter:
    """Backbone-specific bits for FLUX.2-klein-base-9B."""

    name = "flux2"
    # FLUX.2 latent: VAE downsamples 16× directly, channels=128, no DiT patchify.
    LATENT_SCALE = 16
    LATENT_CHANNELS = 128
    # t coordinate "scale" between output (t=0) and reference images (t=10, 20, ...).
    EDIT_T_SCALE = 10

    def __init__(self, pipe):
        self.pipe = pipe

    # ---- preprocessing ---- #

    def encode_edit_images_per_ref(self, edit_images: List[Image.Image]) -> List[torch.Tensor]:
        """VAE-encode each reference image separately, return list of (1, 128, H_lat, W_lat)."""
        pipe = self.pipe
        out = []
        for img in edit_images:
            x = pipe.preprocess_image(img)              # (1, 3, H, W) in [-1, 1]
            lat = pipe.vae.encode(x).detach()           # (1, 128, H/16, W/16)
            out.append(lat)
        return out

    def setup_global_state(self, *, prompt: str, negative_prompt: str, height: int, width: int,
                           seed: int, embedded_guidance: float, cfg_scale: float,
                           edit_images: List[Image.Image], num_inference_steps: int):
        """Run the cheap units (prompt embed, noise init), encode edits ourselves, return state dict.

        We skip Flux2Unit_EditImageEmbedder + Flux2Unit_ImageIDs (we'll regenerate
        per-window) but reuse the prompt/text-encoder units which are expensive
        and don't depend on spatial tiling.

        `cfg_scale` is recorded into inputs_shared so the seperate_cfg PromptEmbedder
        runs its negative-side pass; the actual per-window CFG strength is applied
        later via the explicit `cfg_scale` arg of `pipe.cfg_guided_model_fn`.
        """
        pipe = self.pipe
        pipe.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=1.0,
            dynamic_shift_len=(height // self.LATENT_SCALE) * (width // self.LATENT_SCALE),
        )

        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,          # ensures the nega prompt pass actually runs
            "embedded_guidance": embedded_guidance,
            "input_image": None,
            "denoising_strength": 1.0,
            "edit_image": None,              # FLUX prompt embedders don't read edit_image
            "edit_image_auto_resize": False,
            "height": height, "width": width,
            "seed": seed, "rand_device": "cpu",
            "initial_noise": None,
            "num_inference_steps": num_inference_steps,
        }

        # Run only the units we need: shape check, prompt embed (×2), noise init,
        # input image embedder (no-op since input_image=None).
        from diffsynth.pipelines.flux2_image import (  # local import to avoid heavy import on module load
            Flux2Unit_ShapeChecker, Flux2Unit_PromptEmbedder, Flux2Unit_Qwen3PromptEmbedder,
            Flux2Unit_NoiseInitializer, Flux2Unit_InputImageEmbedder,
        )
        for unit in pipe.units:
            if isinstance(unit, (
                Flux2Unit_ShapeChecker,
                Flux2Unit_PromptEmbedder, Flux2Unit_Qwen3PromptEmbedder,
                Flux2Unit_NoiseInitializer, Flux2Unit_InputImageEmbedder,
            )):
                inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
                    unit, pipe, inputs_shared, inputs_posi, inputs_nega
                )

        # Encode each reference image into its own latent grid.
        edit_latents_per_ref = self.encode_edit_images_per_ref(edit_images)

        # Reshape global latents (flat) to (1, 128, H_lat, W_lat) grid for windowing.
        H_lat, W_lat = height // self.LATENT_SCALE, width // self.LATENT_SCALE
        latents_flat = inputs_shared["latents"]                       # (1, H_lat*W_lat, 128)
        latents_grid = latents_flat.reshape(1, H_lat, W_lat, self.LATENT_CHANNELS).permute(0, 3, 1, 2).contiguous()

        return {
            "inputs_shared": inputs_shared,
            "inputs_posi": inputs_posi,
            "inputs_nega": inputs_nega,
            "edit_latents_per_ref": edit_latents_per_ref,             # list of (1, 128, EH, EW)
            "latents_grid": latents_grid,                             # (1, 128, H_lat, W_lat)
            "H_lat": H_lat, "W_lat": W_lat,
        }

    # ---- per-window forward ---- #

    def _window_image_ids(self, win_h: int, win_w: int, device) -> torch.Tensor:
        t = torch.arange(1)
        h = torch.arange(win_h)
        w = torch.arange(win_w)
        l = torch.arange(1)
        ids = torch.cartesian_prod(t, h, w, l)
        return ids.unsqueeze(0).to(device)                           # (1, win_h*win_w, 4)

    def _window_edit_image_ids(self, win_h: int, win_w: int, n_refs: int, device) -> torch.Tensor:
        """Window-local edit image IDs for n_refs refs concatenated."""
        ids_list = []
        for i in range(n_refs):
            t_val = self.EDIT_T_SCALE + self.EDIT_T_SCALE * i        # match upstream: 10, 20, 30, ...
            t = torch.tensor([t_val])
            h = torch.arange(win_h)
            w = torch.arange(win_w)
            l = torch.arange(1)
            ids_list.append(torch.cartesian_prod(t, h, w, l))
        ids = torch.cat(ids_list, dim=0).unsqueeze(0).to(device)     # (1, n_refs*win_h*win_w, 4)
        return ids

    def window_forward(self, *, state, latents_grid: torch.Tensor, y0: int, x0: int,
                       win_h_lat: int, win_w_lat: int, circular: bool,
                       timestep: torch.Tensor, cfg_scale: float, progress_id: int) -> torch.Tensor:
        """Run pipe.cfg_guided_model_fn on a single window.

        Returns: noise_pred grid for the window, shape (1, 128, win_h_lat, win_w_lat).
        """
        pipe = self.pipe
        device = latents_grid.device

        # 1) Slice output latent window (1, 128, win, win)
        latents_w_grid = _slice_grid_circular(latents_grid, y0, x0, win_h_lat, win_w_lat, circular)
        # FLUX model_fn wants flat layout (1, seq, 128).
        latents_w_flat = rearrange(latents_w_grid, "B C H W -> B (H W) C").contiguous()

        # 2) Slice each ref's edit_latents at the same (y0, x0). All refs assumed
        #    to share the same spatial size as the output (we cropped the same
        #    way at training time). Then flatten + concat along seq.
        edit_latents_per_ref: List[torch.Tensor] = state["edit_latents_per_ref"]
        n_refs = len(edit_latents_per_ref)
        edit_w_flat_list = []
        for ref in edit_latents_per_ref:
            ref_w = _slice_grid_circular(ref, y0, x0, win_h_lat, win_w_lat, circular)
            edit_w_flat_list.append(rearrange(ref_w, "B C H W -> B (H W) C"))
        edit_latents_w = torch.cat(edit_w_flat_list, dim=1)         # (1, n_refs*seq, 128)

        # 3) Window-local positional IDs.
        image_ids = self._window_image_ids(win_h_lat, win_w_lat, device)
        edit_image_ids = self._window_edit_image_ids(win_h_lat, win_w_lat, n_refs, device)

        # 4) Pack inputs and run cfg-guided forward (handles posi/nega pass internally).
        inputs_shared = dict(state["inputs_shared"])
        inputs_shared.update({
            "latents": latents_w_flat,
            "edit_latents": edit_latents_w,
            "image_ids": image_ids,
            "edit_image_ids": edit_image_ids,
            "embedded_guidance": state["inputs_shared"]["embedded_guidance"],
        })
        models = {"dit": pipe.dit}
        noise_pred_flat = pipe.cfg_guided_model_fn(
            pipe.model_fn, cfg_scale,
            inputs_shared, state["inputs_posi"], state["inputs_nega"],
            **models, timestep=timestep, progress_id=progress_id,
        )
        # Back to grid layout.
        noise_pred_grid = noise_pred_flat.reshape(1, win_h_lat, win_w_lat, self.LATENT_CHANNELS) \
                                          .permute(0, 3, 1, 2).contiguous()
        return noise_pred_grid

    # ---- after the loop ---- #

    def vae_decode(self, latents_grid: torch.Tensor) -> Image.Image:
        pipe = self.pipe
        image = pipe.vae.decode(latents_grid)
        return pipe.vae_output_to_image(image)


# =============================================================== #
#                Qwen / FireRed shared adapter                     #
# =============================================================== #


class _QwenAdapter:
    """Backbone-specific bits for Qwen-Image-Edit-2511 / FireRed-Image-Edit-1.1.

    Both share `QwenImagePipeline` upstream — the only difference is `zero_cond_t`
    (True for Qwen-Image-Edit-2511, False for FireRed-Image-Edit-1.1) and which
    weights are loaded.
    """

    name = "qwen"
    # Qwen latent: VAE 8× downsample, then DiT patchifies 2× → effective 16×.
    # We work in VAE-latent units (H/8) for slicing, but RoPE shape is H/16.
    VAE_SCALE = 8
    PATCH = 2          # DiT patch size used inside model_fn rearrange
    EFFECTIVE_SCALE = VAE_SCALE * PATCH  # 16

    def __init__(self, pipe, *, zero_cond_t: bool):
        self.pipe = pipe
        self.zero_cond_t = zero_cond_t

    def encode_edit_images_per_ref(self, edit_images: List[Image.Image]) -> List[torch.Tensor]:
        """VAE-encode each reference image separately."""
        pipe = self.pipe
        out = []
        for img in edit_images:
            x = pipe.preprocess_image(img).to(device=pipe.device, dtype=pipe.torch_dtype)
            lat = pipe.vae.encode(x).detach()             # (1, 16, H/8, W/8)
            out.append(lat)
        return out

    def setup_global_state(self, *, prompt: str, negative_prompt: str, height: int, width: int,
                           seed: int, cfg_scale: float, edit_images: List[Image.Image],
                           num_inference_steps: int):
        """Cheap setup units, edit encoding, scheduler. Returns state dict for the loop.

        Important: Qwen-Image-Edit's PromptEmbedder is conditioned on `edit_image`
        (it bakes visual tokens into the text embeds). So we must pass the actual
        reference(s) into inputs_shared['edit_image'] before the prompt embedder
        runs. After that we null it out — we skip QwenImageUnit_EditImageEmbedder
        because we encode each ref ourselves to keep per-ref shapes for slicing.

        `cfg_scale` is recorded into inputs_shared so the seperate_cfg PromptEmbedder
        actually runs the nega-side pass too.
        """
        pipe = self.pipe
        pipe.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=1.0,
            dynamic_shift_len=(height // self.EFFECTIVE_SCALE) * (width // self.EFFECTIVE_SCALE),
            exponential_shift_mu=None,
        )
        # Match upstream: edit_image is single PIL when there's only one ref,
        # else a list — affects which template the prompt embedder uses.
        edit_image_arg = edit_images[0] if len(edit_images) == 1 else edit_images

        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": None, "denoising_strength": 1.0,
            "inpaint_mask": None, "inpaint_blur_size": None, "inpaint_blur_sigma": None,
            "height": height, "width": width,
            "seed": seed, "rand_device": "cpu",
            "num_inference_steps": num_inference_steps,
            "blockwise_controlnet_inputs": None,
            "tiled": False, "tile_size": 128, "tile_stride": 64,
            "eligen_entity_prompts": None, "eligen_entity_masks": None, "eligen_enable_on_negative": False,
            "edit_image": edit_image_arg, "edit_image_auto_resize": False, "edit_rope_interpolation": False,
            "context_image": None,
            "zero_cond_t": self.zero_cond_t,
            "layer_input_image": None,
            "layer_num": None,
            "image2lora_images": None,
        }
        # Run only cheap setup units (skip EditImageEmbedder, controlnet/eligen/etc.).
        # PromptEmbedder is the only spatial-independent expensive one; the rest
        # short-circuit when their inputs are None.
        from diffsynth.pipelines.qwen_image import (
            QwenImageUnit_ShapeChecker, QwenImageUnit_NoiseInitializer,
            QwenImageUnit_InputImageEmbedder, QwenImageUnit_PromptEmbedder,
        )
        for unit in pipe.units:
            if isinstance(unit, (
                QwenImageUnit_ShapeChecker, QwenImageUnit_NoiseInitializer,
                QwenImageUnit_InputImageEmbedder, QwenImageUnit_PromptEmbedder,
            )):
                inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
                    unit, pipe, inputs_shared, inputs_posi, inputs_nega
                )

        edit_latents_per_ref = self.encode_edit_images_per_ref(edit_images)

        latents_grid = inputs_shared["latents"]                       # (1, 16, H/8, W/8)

        return {
            "inputs_shared": inputs_shared,
            "inputs_posi": inputs_posi,
            "inputs_nega": inputs_nega,
            "edit_latents_per_ref": edit_latents_per_ref,             # list of (1, 16, EH/8, EW/8)
            "latents_grid": latents_grid,
            "H_lat": height // self.EFFECTIVE_SCALE,
            "W_lat": width // self.EFFECTIVE_SCALE,
        }

    def window_forward(self, *, state, latents_grid: torch.Tensor, y0: int, x0: int,
                       win_h_lat: int, win_w_lat: int, circular: bool,
                       timestep: torch.Tensor, cfg_scale: float, progress_id: int) -> torch.Tensor:
        """Run pipe.cfg_guided_model_fn on a single window.

        Coordinates (y0, x0, win_h_lat, win_w_lat) are in DiT-effective grid units
        (i.e. H/16). We translate to VAE-latent units (×PATCH=2) for slicing.
        """
        pipe = self.pipe
        # Translate to VAE-latent slicing coordinates.
        s = self.PATCH
        y0v, x0v = y0 * s, x0 * s
        winv_h, winv_w = win_h_lat * s, win_w_lat * s

        latents_w = _slice_grid_circular(latents_grid, y0v, x0v, winv_h, winv_w, circular).contiguous()

        edit_latents_per_ref: List[torch.Tensor] = state["edit_latents_per_ref"]
        edit_latents_w = [
            _slice_grid_circular(ref, y0v, x0v, winv_h, winv_w, circular).contiguous()
            for ref in edit_latents_per_ref
        ]
        # Qwen model_fn handles edit_latents as either tensor or list[tensor].
        edit_latents_arg: Union[torch.Tensor, List[torch.Tensor]] = (
            edit_latents_w if len(edit_latents_w) > 1 else edit_latents_w[0]
        )

        # Window-size in image pixels (used by model_fn for the rearrange shape).
        win_image_h = win_h_lat * self.EFFECTIVE_SCALE
        win_image_w = win_w_lat * self.EFFECTIVE_SCALE

        inputs_shared = dict(state["inputs_shared"])
        inputs_shared.update({
            "latents": latents_w,
            "edit_latents": edit_latents_arg,
            "height": win_image_h, "width": win_image_w,
            "zero_cond_t": self.zero_cond_t,
        })
        models = {"dit": pipe.dit}
        noise_pred = pipe.cfg_guided_model_fn(
            pipe.model_fn, cfg_scale,
            inputs_shared, state["inputs_posi"], state["inputs_nega"],
            **models, timestep=timestep, progress_id=progress_id,
        )
        return noise_pred                                              # (1, 16, winv_h, winv_w)

    def vae_decode(self, latents_grid: torch.Tensor) -> Image.Image:
        pipe = self.pipe
        image = pipe.vae.decode(latents_grid, device=pipe.device, tiled=False)
        return pipe.vae_output_to_image(image)


# =============================================================== #
#                       Driver                                     #
# =============================================================== #


def make_adapter(pipe, backbone: str):
    """Instantiate the right adapter for the loaded pipeline."""
    backbone = backbone.lower()
    if backbone in ("flux2", "flux2_2048"):
        return _Flux2Adapter(pipe)
    if backbone in ("qwen_edit_2511", "qwen-image-edit-2511"):
        return _QwenAdapter(pipe, zero_cond_t=True)
    if backbone in ("firered_edit_1_1", "firered-image-edit-1.1"):
        return _QwenAdapter(pipe, zero_cond_t=False)
    raise ValueError(f"Unsupported backbone: {backbone!r}. Must be one of "
                     f"{{flux2, qwen_edit_2511, firered_edit_1_1}}.")


@torch.no_grad()
def multidiffusion_infer(
    pipe,
    backbone: str,
    *,
    edit_image: Union[Image.Image, Sequence[Image.Image]],
    prompt: str,
    negative_prompt: str = "",
    height: int = 2048,
    width: int = 2048,
    window_size: int = 1024,
    stride: Optional[int] = None,
    circular: bool = True,
    num_inference_steps: int = 30,
    cfg_scale: float = 4.0,
    embedded_guidance: float = 4.0,
    seed: int = 0,
    progress_bar_cmd=tqdm,
    step_callback=None,
) -> Image.Image:
    """High-level entry point.

    Parameters
    ----------
    pipe
        A loaded `Flux2ImagePipeline` (backbone='flux2') or `QwenImagePipeline`
        (backbone in {'qwen_edit_2511', 'firered_edit_1_1'}). LoRA must already
        be loaded into `pipe.dit`.
    backbone
        Selects the adapter / per-backbone behavior (e.g. zero_cond_t for
        Qwen-Image-Edit-2511, embedded_guidance for FLUX.2).
    edit_image
        Reference image(s). Single PIL or list of PIL. All refs must share the
        same (H, W) and ideally that (H, W) equals the output (height, width)
        — otherwise the per-window slicing won't align spatially.
    height, width
        Final output size (e.g. 2048×2048). Both must be multiples of
        backbone-effective scale (16 for both).
    window_size
        Per-window edge length in *image pixels* (i.e. the LoRA's training
        resolution: 2048 for FLUX.2, 1024 for Qwen/FireRed). Must be a multiple
        of 16 and <= min(height, width). Set window_size == max(height, width)
        to disable tiling and fall back to a single full-resolution pass.
    stride
        Tile stride in image pixels. Default = window_size // 2 (50% overlap),
        which is the standard MultiDiffusion choice.
    circular
        Wrap tile windows across image boundaries (recommended for tileable
        textures). When False, tiles are clamped to lie inside [0, H/W].
    cfg_scale, embedded_guidance, num_inference_steps, seed
        Same semantics as the upstream `pipe(**...)` interface.
    step_callback
        Optional callable(adapter, step_id, latents_grid) called after each scheduler step.
        Useful for saving intermediate latent states or preview images.
        The adapter argument lets you call adapter.vae_decode(latents_grid).
    """
    if isinstance(edit_image, Image.Image):
        edit_image = [edit_image]
    edit_image = list(edit_image)
    if not edit_image:
        raise ValueError("multidiffusion_infer needs at least one edit_image reference.")

    # Sanity-check sizes
    ref_size = edit_image[0].size
    for i, im in enumerate(edit_image):
        if im.size != ref_size:
            raise ValueError(f"All edit_image refs must share size; got {ref_size} vs {im.size} at index {i}.")
    if ref_size != (width, height):
        # Soft warn: per-window slicing assumes co-located output and refs.
        # Caller typically pre-resizes edit_image to (height, width).
        print(f"[multidiffusion][warn] edit_image size {ref_size} != output (W, H) ({width}, {height}). "
              f"Spatial alignment between window and reference may be off; consider resizing the refs first.")

    if stride is None:
        stride = window_size // 2

    adapter = make_adapter(pipe, backbone)
    eff = adapter.EFFECTIVE_SCALE if hasattr(adapter, "EFFECTIVE_SCALE") else adapter.LATENT_SCALE
    for name, val in [("height", height), ("width", width), ("window_size", window_size), ("stride", stride)]:
        if val % eff != 0:
            raise ValueError(f"{name}={val} must be a multiple of {eff} for backbone {backbone}.")
    if window_size > min(height, width):
        raise ValueError(f"window_size {window_size} must be <= min(height, width) ({min(height, width)}).")

    # ----- 1. Setup full-resolution global state -----
    if backbone.lower() == "flux2":
        state = adapter.setup_global_state(
            prompt=prompt, negative_prompt=negative_prompt,
            height=height, width=width, seed=seed,
            embedded_guidance=embedded_guidance, cfg_scale=cfg_scale,
            edit_images=edit_image, num_inference_steps=num_inference_steps,
        )
    else:
        state = adapter.setup_global_state(
            prompt=prompt, negative_prompt=negative_prompt,
            height=height, width=width, seed=seed, cfg_scale=cfg_scale,
            edit_images=edit_image, num_inference_steps=num_inference_steps,
        )

    latents_grid = state["latents_grid"]                              # (1, C, *grid*)
    H_lat, W_lat = state["H_lat"], state["W_lat"]                     # DiT-effective grid (H/16, W/16)
    win_lat = window_size // eff
    stride_lat = stride // eff

    offsets = _tile_offsets(H_lat, W_lat, win_lat, win_lat, stride_lat, stride_lat, circular)

    # Translation between DiT-grid coords (H_lat = H/16) and the actual
    # `latents_grid` spatial dims (H/16 for FLUX, H/8 for Qwen because Qwen
    # patchifies inside model_fn).
    coord_factor = latents_grid.shape[-2] // H_lat
    win_acc = win_lat * coord_factor
    weight2d_acc = _hann_2d(win_acc, win_acc, latents_grid.device, dtype=torch.float32)

    print(f"[multidiffusion] backbone={backbone}  HxW={height}x{width}  "
          f"window={window_size}  stride={stride}  circular={circular}  "
          f"latent_grid={H_lat}x{W_lat}  win_lat={win_lat}  tiles/step={len(offsets)}  "
          f"cfg_scale={cfg_scale}  steps={num_inference_steps}")

    # ----- 2. Tiled denoise loop -----
    timesteps = pipe.scheduler.timesteps
    pbar = progress_bar_cmd(timesteps) if progress_bar_cmd is not None else timesteps

    for progress_id, timestep in enumerate(pbar):
        timestep_t = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

        noise_pred_global = torch.zeros_like(latents_grid, dtype=torch.float32)
        weight_global = torch.zeros(latents_grid.shape[-2:], device=latents_grid.device, dtype=torch.float32)

        for (y0, x0) in tqdm(offsets, desc=f"  windows", leave=False, total=len(offsets)):
            noise_pred_w = adapter.window_forward(
                state=state, latents_grid=latents_grid,
                y0=y0, x0=x0, win_h_lat=win_lat, win_w_lat=win_lat,
                circular=circular, timestep=timestep_t,
                cfg_scale=cfg_scale, progress_id=progress_id,
            )                                                          # (1, C, win_acc, win_acc)
            _add_window(
                noise_pred_global, weight_global,
                noise_pred_w.to(torch.float32),
                weight2d_acc,
                y0 * coord_factor, x0 * coord_factor, circular,
            )

        noise_pred_fused = (noise_pred_global / weight_global.clamp_min(1e-6)).to(latents_grid.dtype)
        latents_grid = pipe.scheduler.step(noise_pred_fused, timesteps[progress_id], latents_grid)

        if step_callback is not None:
            step_callback(adapter, progress_id, latents_grid)

    # ----- 3. VAE decode -----
    image = adapter.vae_decode(latents_grid)
    pipe.load_models_to_device([])
    return image
