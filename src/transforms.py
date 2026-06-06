"""Per-phase clip transforms.

Augmentation policy follows the brief:
  * Train: Albumentations geometric + photometric noise.
  * Val/Test: deterministic resize + per-channel normalization only.

Frame-coherent augmentation
---------------------------
For action recognition we want the same spatial transform applied to
every frame of a clip, otherwise the network sees jitter that has no
temporal meaning. We use Albumentations' `additional_targets` feature so
HorizontalFlip / RandomResizedCrop / RandomBrightnessContrast / etc. all
get the same parameters across the T frames.

Domain notes
------------
ETRI is human-centric, so:
  * Horizontal flip is fine (mirroring a person is still that person).
  * Vertical flip is disabled (people are not upside-down in the wild).
  * CoarseDropout is light — too aggressive can erase the body of
    interest in short actions.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np
import torch

try:
    import albumentations as A
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "albumentations is required. Install with `pip install albumentations`."
    ) from exc


def _to_tensor(frames: np.ndarray) -> torch.Tensor:
    """(T,H,W,3) uint8 -> (T,3,H,W) float32 in [0,1]."""
    t = torch.from_numpy(frames).float().div_(255.0)
    return t.permute(0, 3, 1, 2).contiguous()


def _normalize(clip: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    m = torch.tensor(mean, dtype=clip.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=clip.dtype).view(1, 3, 1, 1)
    return (clip - m) / s


class ClipTransform:
    """Wraps an Albumentations pipeline with frame-coherent semantics."""

    def __init__(
        self,
        pipeline: A.Compose,
        mean: Sequence[float],
        std: Sequence[float],
        clip_length: int,
    ) -> None:
        self.pipeline = pipeline
        self.mean = tuple(float(x) for x in mean)
        self.std = tuple(float(x) for x in std)
        self.clip_length = int(clip_length)

    def __call__(self, frames: np.ndarray, label: int) -> Tuple[torch.Tensor, int]:
        # Albumentations expects `image` + `image1..imageN-1`. We feed the
        # first frame as `image` and the rest as additional_targets so the
        # *same* parameters apply across the whole clip.
        kwargs = {"image": frames[0]}
        for i in range(1, frames.shape[0]):
            kwargs[f"image{i}"] = frames[i]
        out = self.pipeline(**kwargs)
        aug = [out["image"]] + [out[f"image{i}"] for i in range(1, frames.shape[0])]
        clip = _to_tensor(np.stack(aug, axis=0))
        clip = _normalize(clip, self.mean, self.std)
        return clip, int(label)


def _additional_targets(clip_length: int) -> dict:
    return {f"image{i}": "image" for i in range(1, clip_length)}


def build_train_transform(
    img_size: int,
    clip_length: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> ClipTransform:
    pipeline = A.Compose(
        [
            A.SmallestMaxSize(max_size=int(img_size * 1.15)),
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.7, 1.0),
                ratio=(0.85, 1.15),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.3
            ),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(8, 24),
                hole_width_range=(8, 24),
                fill=0,
                p=0.25,
            ),
        ],
        additional_targets=_additional_targets(clip_length),
    )
    return ClipTransform(pipeline, mean=mean, std=std, clip_length=clip_length)


def build_eval_transform(
    img_size: int,
    clip_length: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> ClipTransform:
    pipeline = A.Compose(
        [
            A.SmallestMaxSize(max_size=img_size),
            A.CenterCrop(height=img_size, width=img_size),
        ],
        additional_targets=_additional_targets(clip_length),
    )
    return ClipTransform(pipeline, mean=mean, std=std, clip_length=clip_length)
