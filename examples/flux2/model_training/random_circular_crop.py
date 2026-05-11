"""Synchronized circular crop wrapper for tileable image-edit datasets."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image

from diffsynth.core.data.operators import (
    LoadImage,
    RouteByType,
    SequencialProcess,
    ToAbsolutePath,
)


def make_load_only_operator(base_path: str) -> RouteByType:
    """Load image paths as PIL images without DiffSynth resize/crop."""
    return RouteByType(operator_map=[
        (str, ToAbsolutePath(base_path) >> LoadImage()),
        (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage())),
    ])


class RandomCircularCropDataset(torch.utils.data.Dataset):
    """Apply one wraparound crop offset to all image panels in a sample."""

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        crop_size: int,
        image_keys: Iterable[str] = ("image", "edit_image"),
        division_factor: int = 16,
        deterministic: bool = False,
        seed: int = 0,
    ):
        self.base = base_dataset
        self.image_keys = list(image_keys)
        self.division_factor = int(division_factor)
        self.crop_size = (int(crop_size) // self.division_factor) * self.division_factor
        if self.crop_size <= 0:
            raise ValueError(
                f"crop_size {crop_size} rounded to {self.crop_size}; "
                f"it must be >= division_factor ({division_factor})."
            )
        self.deterministic = bool(deterministic)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __getitem__(self, idx: int):
        data = self.base[idx]
        ref_hw = self._collect_and_validate_dims(data)
        if ref_hw is None:
            return data

        height, width = ref_hw
        crop_size = self.crop_size
        if height < crop_size or width < crop_size:
            return data

        rng = (
            np.random.default_rng((self.seed + idx) % (2**31 - 1))
            if self.deterministic else np.random.default_rng()
        )
        y0 = int(rng.integers(0, height))
        x0 = int(rng.integers(0, width))

        for key in self.image_keys:
            if key not in data:
                continue
            value = data[key]
            if isinstance(value, list):
                data[key] = [
                    self._crop_one(image, y0, x0, crop_size)
                    if isinstance(image, Image.Image) else image
                    for image in value
                ]
            elif isinstance(value, Image.Image):
                data[key] = self._crop_one(value, y0, x0, crop_size)
        return data

    def _collect_and_validate_dims(self, data) -> Optional[tuple[int, int]]:
        ref_hw: Optional[tuple[int, int]] = None
        for key in self.image_keys:
            if key not in data:
                continue
            value = data[key]
            images = value if isinstance(value, list) else [value]
            for image in images:
                if not isinstance(image, Image.Image):
                    continue
                hw = (image.size[1], image.size[0])
                if ref_hw is None:
                    ref_hw = hw
                elif hw != ref_hw:
                    raise ValueError(
                        "RandomCircularCropDataset requires all image panels "
                        f"to share one size. Got {hw} for key {key!r}, "
                        f"expected {ref_hw}."
                    )
        return ref_hw

    @staticmethod
    def _crop_one(image: Image.Image, y0: int, x0: int, crop_size: int) -> Image.Image:
        arr = np.asarray(image)
        height, width = arr.shape[:2]
        ys = (np.arange(crop_size) + y0) % height
        xs = (np.arange(crop_size) + x0) % width
        out = arr[ys[:, None], xs[None, :]]
        return Image.fromarray(out)
