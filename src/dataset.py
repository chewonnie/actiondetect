"""ETRI action clip dataset for the baseline.

Loads RGB video clips and emits (clip, label) for action classification.
Action class is parsed from the leading `A###` token in the filename
(e.g. `A001_P001_C0li_C0li.mp4` -> class index 0).

Design choices that follow the baseline brief:

- Group-safe: this class does NOT split data itself; pass a list of
  allowed participant IDs (e.g. `['P01','P02',...]`). Train/val/test
  splitting is done in `splits.py` so the same code path serves all
  phases and there is no chance of mixing participants between splits.
- Reproducible: clip frame indices are deterministic given (sample, key).
- Train-only augmentation: this class only loads frames. Augmentation
  is applied by `transforms.py` which is built per-phase.
- Train-only normalization stats: mean/std are not hard-coded here.
  `compute_train_stats` walks a Subset/Dataset and returns per-channel
  mean/std from the training partition only.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "opencv-python is required to read ETRI mp4 clips."
    ) from exc


_ACTION_RE = re.compile(r"^A(\d{3})", re.IGNORECASE)


def parse_action_index(filename: str) -> Optional[int]:
    """Map a filename starting with `A###...` to a 0-based class index."""
    m = _ACTION_RE.match(os.path.basename(filename))
    if not m:
        return None
    n = int(m.group(1))
    return n - 1 if n >= 1 else None


# --- Optional 55->12 class remap (PLAN.md §3.2) -----------------------------
# Active ONLY when the env var ETRI_CLASS_MAP points to a CSV with columns
# `action_idx,core_idx` (e.g. pipeline/etri_actions.csv). When the env var is
# unset/missing the map is empty and scan_etri_root reproduces the original
# 55-class baseline bit-for-bit. This is the single localized src/ edit; no
# other src/ file (train.py/eval.py/splits.py/metrics.py) is touched.
_CLASS_MAP_CACHE: Optional[Dict[int, int]] = None


def _load_class_map() -> Dict[int, int]:
    global _CLASS_MAP_CACHE
    if _CLASS_MAP_CACHE is not None:
        return _CLASS_MAP_CACHE
    path = os.environ.get("ETRI_CLASS_MAP")
    m: Dict[int, int] = {}
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            rows = (ln for ln in fh if not ln.lstrip().startswith("#"))
            for row in csv.DictReader(rows):
                m[int(row["action_idx"])] = int(row["core_idx"])
    _CLASS_MAP_CACHE = m
    return m


@dataclass
class ClipSample:
    rgb_path: str
    csv_path: Optional[str]
    participant: str
    session: str
    base: str
    action_idx: int
    n_frames: int


def _count_csv_rows(csv_path: str) -> int:
    n = 0
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return max(0, n - 1)  # minus header


def _probe_video_length(path: str) -> int:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        # Some decoders mis-report; fall back to a single readable frame check
        ok, _ = cap.read()
        n = 1 if ok else 0
    cap.release()
    return n


def scan_etri_root(
    root: str,
    participants: Optional[Sequence[str]] = None,
) -> List[ClipSample]:
    """Walk an ETRI tree and return the list of usable clips.

    Expected layout (matches the bundled zips after extraction):

        <root>/RGB/P01/<session>/A001_P001_*.mp4
        <root>/JointCSV/P01/<session>/A001_P001_*.csv  (optional)
    """
    rgb_root = os.path.join(root, "RGB")
    csv_root = os.path.join(root, "JointCSV")
    if not os.path.isdir(rgb_root):
        raise FileNotFoundError(f"RGB folder not found under {root}")

    allow = set(participants) if participants else None
    samples: List[ClipSample] = []

    for p in sorted(os.listdir(rgb_root)):
        if not p.startswith("P"):
            continue
        if allow is not None and p not in allow:
            continue
        p_dir = os.path.join(rgb_root, p)
        if not os.path.isdir(p_dir):
            continue
        for session in sorted(os.listdir(p_dir)):
            s_dir = os.path.join(p_dir, session)
            if not os.path.isdir(s_dir):
                continue
            for f in sorted(os.listdir(s_dir)):
                if not f.lower().endswith(".mp4"):
                    continue
                base = f[:-4]
                action_idx = parse_action_index(f)
                if action_idx is None:
                    continue
                _cm = _load_class_map()
                if _cm:
                    if action_idx not in _cm:
                        continue
                    action_idx = _cm[action_idx]
                rgb_path = os.path.join(s_dir, f)
                try:
                    if os.path.getsize(rgb_path) < 1024:
                        continue
                except OSError:
                    continue

                csv_path = os.path.join(csv_root, p, session, base + ".csv")
                csv_path = csv_path if os.path.isfile(csv_path) else None

                if csv_path is not None:
                    n_frames = _count_csv_rows(csv_path)
                else:
                    n_frames = _probe_video_length(rgb_path)
                if n_frames <= 0:
                    continue

                samples.append(
                    ClipSample(
                        rgb_path=rgb_path,
                        csv_path=csv_path,
                        participant=p,
                        session=session,
                        base=base,
                        action_idx=action_idx,
                        n_frames=n_frames,
                    )
                )
    return samples


class ETRIClipDataset(Dataset):
    """Per-clip action classification dataset.

    Each item is one video clip sampled with a fixed length and stride.

    Parameters
    ----------
    samples : list[ClipSample]
        Pre-scanned clips. Use `scan_etri_root` + `splits.group_split` to
        produce one list per phase.
    clip_length : int
        Number of frames per clip handed to the model.
    sampling_rate : int
        Temporal stride between sampled frames.
    img_size : int
        Output spatial size (square). Resize happens inside `transform`.
    transform : callable, optional
        Phase-specific transform. Receives a list of HxWx3 uint8 RGB
        frames and the integer label. Must return (tensor[T,C,H,W], label).
    phase : str
        'train' | 'val' | 'test'. Only changes temporal sampling: train
        picks a random center frame, val/test scans deterministic centers.
    centers_per_sample : int
        How many fixed centers each video contributes during val/test
        (uniform across the clip). Ignored when phase='train'.
    num_classes : int
        Total number of classes. Defaults to one more than the largest
        observed index — pass explicitly when you want to lock the label
        space (e.g. always 55) regardless of which classes appear in
        a given split.
    """

    def __init__(
        self,
        samples: List[ClipSample],
        clip_length: int = 16,
        sampling_rate: int = 2,
        img_size: int = 112,
        transform: Optional[Callable] = None,
        phase: str = "train",
        centers_per_sample: int = 1,
        num_classes: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        if phase not in {"train", "val", "test"}:
            raise ValueError(f"phase must be train|val|test, got {phase!r}")

        self.samples = samples
        self.clip_length = int(clip_length)
        self.sampling_rate = int(sampling_rate)
        self.img_size = int(img_size)
        self.transform = transform
        self.phase = phase
        self.centers_per_sample = max(1, int(centers_per_sample))
        self.seed = int(seed)

        if num_classes is None:
            num_classes = max((s.action_idx for s in samples), default=-1) + 1
        self.num_classes = int(num_classes)

        # Build flat index: (sample_idx, center_frame_1based).
        # Train uses a single random center per __getitem__; we still
        # register one entry per sample so an epoch sees every clip once.
        self.index: List[Tuple[int, int]] = []
        if phase == "train":
            for i in range(len(samples)):
                self.index.append((i, -1))  # -1 means "pick randomly"
        else:
            for i, s in enumerate(samples):
                span = self.clip_length * self.sampling_rate
                if s.n_frames <= span:
                    centers = [max(1, s.n_frames // 2)]
                else:
                    # Evenly spaced centers, all valid for a full clip.
                    lo = span // 2 + 1
                    hi = s.n_frames - span // 2
                    if self.centers_per_sample == 1:
                        centers = [(lo + hi) // 2]
                    else:
                        centers = list(
                            np.linspace(lo, hi, self.centers_per_sample, dtype=int)
                        )
                for c in centers:
                    self.index.append((i, int(c)))

        # Per-worker video capture cache (paths -> cv2.VideoCapture).
        # Kept lazily to play nicely with DataLoader workers.
        self._cap_path: Optional[str] = None
        self._cap = None

    # ------------------------------------------------------------------ #
    # Frame IO
    # ------------------------------------------------------------------ #
    def _open(self, path: str):
        if self._cap_path != path:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
            self._cap = cv2.VideoCapture(path)
            self._cap_path = path
        return self._cap

    def _read_frames(self, path: str, indices_1based: Sequence[int]) -> np.ndarray:
        """Read frames in one forward pass; output is (T, H, W, 3) uint8 RGB."""
        cap = self._open(path)
        ordered = sorted(set(indices_1based))
        start = max(1, ordered[0])
        cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
        cur = start
        collected: Dict[int, np.ndarray] = {}
        last = ordered[-1]
        wanted = set(ordered)
        while cur <= last:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if cur in wanted:
                collected[cur] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cur += 1

        # Fallback for any missing index (e.g. truncated decode).
        if len(collected) < len(wanted):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            fallback = (
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if ok and frame is not None
                else np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            )
            for i in ordered:
                collected.setdefault(i, fallback)

        # Materialize in caller-requested order (allow duplicates / edge clamp).
        frames = np.stack([collected[i] for i in indices_1based], axis=0)
        return frames

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def _frame_indices(self, sample: ClipSample, center: int) -> List[int]:
        """Return clip_length 1-based frame indices around `center`.

        Edge frames are clamped to the available range so short clips
        still produce a valid tensor (training will see repeated edge
        frames, which the model treats as a still segment).
        """
        span = self.clip_length * self.sampling_rate
        half = span // 2
        start = center - half
        idx = [start + t * self.sampling_rate for t in range(self.clip_length)]
        return [int(np.clip(i, 1, sample.n_frames)) for i in idx]

    def _pick_center(self, sample: ClipSample, item_idx: int) -> int:
        span = self.clip_length * self.sampling_rate
        if sample.n_frames <= span:
            return max(1, sample.n_frames // 2)
        lo = span // 2 + 1
        hi = sample.n_frames - span // 2
        # Deterministic-but-shuffled center per (epoch_seed, item_idx).
        rng = np.random.default_rng(self.seed + item_idx * 2654435761 % (2**32))
        return int(rng.integers(lo, hi + 1))

    # ------------------------------------------------------------------ #
    # Dataset protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        sample_idx, center = self.index[idx]
        sample = self.samples[sample_idx]
        if center < 0:
            center = self._pick_center(sample, idx)

        frame_idx = self._frame_indices(sample, center)
        frames = self._read_frames(sample.rgb_path, frame_idx)  # (T,H,W,3)
        label = int(sample.action_idx)

        if self.transform is not None:
            clip, label = self.transform(frames, label)
        else:
            # Minimal default: uint8 -> float tensor in [0,1], CHW per frame.
            clip = torch.from_numpy(frames).float().div_(255.0).permute(0, 3, 1, 2)

        return clip, label

    def __del__(self):
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass


def compute_train_stats(
    dataset: ETRIClipDataset,
    max_clips: int = 256,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate per-channel mean/std from the training partition only.

    Sampling a subset keeps the cost manageable on long datasets;
    256 clips * 16 frames = ~4k frames which is plenty for stable stats.
    Returns mean and std as float32 arrays of shape (3,).
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)
    if n == 0:
        return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)
    take = min(max_clips, n)
    order = rng.choice(n, size=take, replace=False)

    # Use raw frames (skip transform) so stats reflect the underlying data.
    saved = dataset.transform
    dataset.transform = None
    try:
        s = np.zeros(3, dtype=np.float64)
        ss = np.zeros(3, dtype=np.float64)
        count = 0
        for i in order:
            clip, _ = dataset[int(i)]
            arr = clip.numpy() if isinstance(clip, torch.Tensor) else clip
            if arr.ndim == 4 and arr.shape[1] == 3:  # (T,C,H,W)
                arr = arr.transpose(0, 2, 3, 1)
            arr = arr.reshape(-1, 3).astype(np.float64)
            s += arr.sum(axis=0)
            ss += (arr ** 2).sum(axis=0)
            count += arr.shape[0]
    finally:
        dataset.transform = saved

    mean = s / max(1, count)
    var = ss / max(1, count) - mean ** 2
    std = np.sqrt(np.clip(var, 1e-8, None))
    return mean.astype(np.float32), std.astype(np.float32)
