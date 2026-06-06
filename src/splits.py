"""Leakage-safe split utilities.

The ETRI dataset is grouped by participant (`P01`..`P20`, `P201`..`P230`).
A naive random split would put the same participant in both train and
validation, which is the textbook example of group leakage. We split
*participants*, then assign all of a participant's clips to that side.

Reproducibility: splits are seeded; calling with the same seed and the
same participant list yields the same partition.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from dataset import ClipSample


@dataclass
class Split:
    train: List[ClipSample]
    val: List[ClipSample]
    test: List[ClipSample]
    train_participants: List[str]
    val_participants: List[str]
    test_participants: List[str]


def group_split(
    samples: Sequence[ClipSample],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Split:
    """Partition `samples` by participant.

    The split is approximate by ratio of *participants*, not clips. The
    actual clip-level ratio may drift slightly if some participants have
    far more clips than others — that is intentional, since the goal is
    to keep groups intact, not to hit an exact percentage.
    """
    if not (0.0 < val_ratio < 1.0) or not (0.0 < test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be in (0,1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    by_pid: Dict[str, List[ClipSample]] = defaultdict(list)
    for s in samples:
        by_pid[s.participant].append(s)
    pids = sorted(by_pid.keys())

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pids))
    shuffled = [pids[i] for i in order]

    n = len(shuffled)
    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, int(round(n * test_ratio)))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"Not enough participants ({n}) for the requested ratios "
            f"(val={val_ratio}, test={test_ratio})."
        )

    train_pids = sorted(shuffled[:n_train])
    val_pids = sorted(shuffled[n_train : n_train + n_val])
    test_pids = sorted(shuffled[n_train + n_val :])

    train = [s for p in train_pids for s in by_pid[p]]
    val = [s for p in val_pids for s in by_pid[p]]
    test = [s for p in test_pids for s in by_pid[p]]

    return Split(
        train=train,
        val=val,
        test=test,
        train_participants=train_pids,
        val_participants=val_pids,
        test_participants=test_pids,
    )


def describe_split(split: Split) -> str:
    def stat(name: str, clips: List[ClipSample], pids: List[str]) -> str:
        classes = sorted({s.action_idx for s in clips})
        return (
            f"{name:>5}: {len(pids):>3} participants, "
            f"{len(clips):>5} clips, {len(classes):>3} classes "
            f"({pids[:3]}{'...' if len(pids) > 3 else ''})"
        )

    return "\n".join(
        [
            stat("train", split.train, split.train_participants),
            stat("val", split.val, split.val_participants),
            stat("test", split.test, split.test_participants),
        ]
    )


def assert_no_leakage(split: Split) -> None:
    """Hard guarantee: a participant cannot appear in more than one split."""
    t = set(split.train_participants)
    v = set(split.val_participants)
    e = set(split.test_participants)
    if t & v or t & e or v & e:
        raise RuntimeError(
            f"Participant leakage detected: "
            f"train∩val={sorted(t & v)}, "
            f"train∩test={sorted(t & e)}, "
            f"val∩test={sorted(v & e)}"
        )
