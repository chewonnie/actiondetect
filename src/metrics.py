"""Evaluation metrics + class-imbalance helpers.

The brief calls this out explicitly: accuracy alone hides imbalance.
We log accuracy *and* macro-F1 *and* macro AUC-PR every epoch so the
imbalance story is visible without re-running anything.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
)


def class_weights_from_counts(
    counts: np.ndarray,
    scheme: str = "inv_freq",
    smoothing: float = 1.0,
) -> torch.Tensor:
    """Compute per-class weights for `nn.CrossEntropyLoss(weight=...)`.

    Schemes:
      * `inv_freq`  : w_c = N / (K * n_c)   (sklearn 'balanced')
      * `inv_sqrt`  : w_c = 1 / sqrt(n_c)   (gentler, common for long-tail)
      * `effective` : Cui et al. 2019, beta=0.999

    `smoothing` is added to counts so unseen classes don't blow up.
    """
    counts = counts.astype(np.float64) + float(smoothing)
    K = counts.shape[0]
    if scheme == "inv_freq":
        w = counts.sum() / (K * counts)
    elif scheme == "inv_sqrt":
        w = 1.0 / np.sqrt(counts)
    elif scheme == "effective":
        beta = 0.999
        eff = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / np.maximum(eff, 1e-8)
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme!r}")
    # Normalize so the mean weight is 1 (keeps the loss scale comparable).
    w = w * (K / w.sum())
    return torch.tensor(w, dtype=torch.float32)


def class_counts(samples, num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    for s in samples:
        if 0 <= s.action_idx < num_classes:
            counts[s.action_idx] += 1
    return counts


@torch.no_grad()
def evaluate(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    """Compute metrics on stacked logits/labels for an entire split."""
    probs = F.softmax(logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    y = labels.cpu().numpy()

    acc = float((preds == y).mean()) if y.size else 0.0
    f1_macro = (
        float(f1_score(y, preds, average="macro", zero_division=0))
        if y.size
        else 0.0
    )

    # AUC-PR per class needs at least one positive in y. Compute over
    # the classes that actually appear in this split, then macro-average.
    aps = []
    present = np.unique(y)
    for c in present:
        y_bin = (y == c).astype(np.int32)
        if y_bin.sum() == 0:
            continue
        aps.append(float(average_precision_score(y_bin, probs[:, int(c)])))
    auc_pr_macro = float(np.mean(aps)) if aps else 0.0

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "auc_pr_macro": auc_pr_macro,
        "n": int(y.size),
        "n_classes_present": int(len(present)),
    }


@torch.no_grad()
def confusion(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    preds = logits.argmax(dim=1).cpu().numpy()
    y = labels.cpu().numpy()
    return confusion_matrix(y, preds, labels=list(range(num_classes)))
