"""Baseline model: R3D-18 with a fresh classification head.

Kept deliberately simple per the brief:
  * Off-the-shelf 3D ResNet-18 from torchvision.
  * Optional Kinetics-400 weights for warm-start.
  * One linear head sized to the dataset's num_classes.

This is the *floor*, not the ceiling — every change later (deeper
backbone, two-stream, joint+RGB fusion, etc.) gets compared against
this model under the same data splits and metrics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.video import R3D_18_Weights, r3d_18


def build_baseline(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
    model = r3d_18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def model_input_layout() -> str:
    """torchvision video models expect (B, C, T, H, W).

    Our dataset emits (T, C, H, W) per clip; the training loop transposes
    once with `.permute(0, 2, 1, 3, 4)`.
    """
    return "BCTHW"


def to_model_input(clip: torch.Tensor) -> torch.Tensor:
    """(B, T, C, H, W) -> (B, C, T, H, W)."""
    if clip.dim() != 5:
        raise ValueError(f"Expected 5D clip tensor, got shape {tuple(clip.shape)}")
    return clip.permute(0, 2, 1, 3, 4).contiguous()
