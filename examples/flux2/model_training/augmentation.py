"""Tileable-aware flips and rotations for PBR image-edit datasets."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


def _invert(channel: np.ndarray) -> np.ndarray:
    return (255 - channel.astype(np.int16)).clip(0, 255).astype(np.uint8)


def normal_hflip(arr: np.ndarray) -> np.ndarray:
    out = np.fliplr(arr).copy()
    out[..., 0] = _invert(out[..., 0])
    return out


def normal_vflip(arr: np.ndarray) -> np.ndarray:
    out = np.flipud(arr).copy()
    out[..., 1] = _invert(out[..., 1])
    return out


def normal_rot90cw(arr: np.ndarray) -> np.ndarray:
    out = np.rot90(arr, k=-1).copy()
    red, green = out[..., 0].copy(), out[..., 1].copy()
    out[..., 0] = green
    out[..., 1] = _invert(red)
    return out


def normal_rot180(arr: np.ndarray) -> np.ndarray:
    out = np.rot90(arr, k=2).copy()
    out[..., 0] = _invert(out[..., 0])
    out[..., 1] = _invert(out[..., 1])
    return out


def normal_rot270cw(arr: np.ndarray) -> np.ndarray:
    out = np.rot90(arr, k=1).copy()
    red, green = out[..., 0].copy(), out[..., 1].copy()
    out[..., 0] = _invert(green)
    out[..., 1] = red
    return out


def naive_hflip(arr: np.ndarray) -> np.ndarray:
    return np.fliplr(arr).copy()


def naive_vflip(arr: np.ndarray) -> np.ndarray:
    return np.flipud(arr).copy()


def naive_rot90cw(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, k=-1).copy()


def naive_rot180(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, k=2).copy()


def naive_rot270cw(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, k=1).copy()


_TRANSFORM_PAIRS: Dict[str, Tuple[callable, callable]] = {
    "HFlip": (naive_hflip, normal_hflip),
    "VFlip": (naive_vflip, normal_vflip),
    "Rot90": (naive_rot90cw, normal_rot90cw),
    "Rot180": (naive_rot180, normal_rot180),
    "Rot270": (naive_rot270cw, normal_rot270cw),
}


def detect_normal_panels_from_config(dataset_config_path: str) -> Dict[str, List[int]]:
    """Return panel indices that contain only a Normal map."""
    with open(dataset_config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    def _detect(panels: List[dict]) -> List[int]:
        return [
            index
            for index, panel in enumerate(panels)
            if panel.get("keys", []) == ["Normal"]
        ]

    return {
        "image": _detect(config.get("output_panels", [])),
        "edit_image": _detect(config.get("input_panels", [])),
    }


class TileableAugmentationDataset(torch.utils.data.Dataset):
    """Apply one synchronized transform to all image panels in a sample."""

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        normal_panels: Optional[Dict[str, List[int]]] = None,
        image_keys: Iterable[str] = ("image", "edit_image"),
        prob_original: float = 0.5,
        deterministic: bool = False,
        seed: int = 0,
    ):
        if not 0.0 <= prob_original <= 1.0:
            raise ValueError(f"prob_original must be in [0, 1], got {prob_original}")
        self.base = base_dataset
        self.image_keys = tuple(image_keys)
        self.normal_panels = {key: list(value) for key, value in (normal_panels or {}).items()}
        self.prob_original = float(prob_original)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self._transform_names = list(_TRANSFORM_PAIRS.keys())
        self._choices = ["Original"] + self._transform_names
        remaining = 1.0 - self.prob_original
        per_transform = remaining / len(self._transform_names)
        self._probs = np.array(
            [self.prob_original] + [per_transform] * len(self._transform_names),
            dtype=np.float64,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __getitem__(self, idx: int):
        data = self.base[idx]
        transform_name = self._pick_transform(idx)
        if transform_name == "Original":
            return data

        naive_fn, normal_fn = _TRANSFORM_PAIRS[transform_name]
        for field in self.image_keys:
            if field not in data:
                continue
            value = data[field]
            normal_indices = set(self.normal_panels.get(field, []))
            if isinstance(value, list):
                data[field] = [
                    self._apply_one(image, index in normal_indices, naive_fn, normal_fn)
                    if isinstance(image, Image.Image) else image
                    for index, image in enumerate(value)
                ]
            elif isinstance(value, Image.Image):
                data[field] = self._apply_one(value, 0 in normal_indices, naive_fn, normal_fn)
        return data

    def _pick_transform(self, idx: int) -> str:
        rng = (
            np.random.default_rng((self.seed + idx) % (2**31 - 1))
            if self.deterministic else np.random.default_rng()
        )
        return str(rng.choice(self._choices, p=self._probs))

    @staticmethod
    def _apply_one(
        image: Image.Image,
        is_normal: bool,
        naive_fn: callable,
        normal_fn: callable,
    ) -> Image.Image:
        arr = np.asarray(image)
        if arr.ndim == 2:
            out = naive_fn(arr)
        elif arr.shape[-1] >= 3 and is_normal:
            out = normal_fn(arr[..., :3])
        else:
            out = naive_fn(arr)
        return Image.fromarray(out)
