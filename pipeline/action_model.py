"""pipeline/action_model.py — R3D-18 action recognition with a native-fps rolling buffer.

TEMPORAL CONTRACT (PLAN §3.6):
  buffer_span = clip_length * sampling_rate  (default: 16 * 2 = 32 native frames)
  This must equal src/config.yaml clip_length * sampling_rate.
  Every clip fed to the model is formed by taking every sampling_rate-th frame
  from the buffer, mirroring src/dataset.py _frame_indices exactly.

DUAL-RATE ARCHITECTURE:
  This buffer runs at native fps (e.g. 30 fps) and is completely separate from
  the 1-fps YOLO overlay stream in pipeline/frames.py.

hold-last semantics:
  last_label holds the most recent completed (class_idx, prob) inference.
  Callers should call push() for every native frame and infer() every N pushes.
  infer() returns None until the buffer is full for the first time.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from typing import Optional

import numpy as np
import torch

# Allow importing from src/ regardless of CWD (repo root or pipeline/).
_src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from model import build_baseline, to_model_input  # noqa: E402  (after sys.path insert)

# Fallback mean/std (torchvision video defaults) if a checkpoint lacks stats.
_FALLBACK_MEAN = (0.43216, 0.394666, 0.37645)
_FALLBACK_STD = (0.22803, 0.22145, 0.216989)


def _preprocess_frame(
    frame_rgb: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    size: int = 112,
) -> torch.Tensor:
    """uint8 HxWx3 RGB -> normalised (3, size, size) float32, matching
    src/transforms.py (/255 then (x-mean)/std with the train-split stats
    saved in the checkpoint)."""
    import cv2  # local import; cv2 is always available in this project
    frame = cv2.resize(frame_rgb, (size, size))
    t = torch.from_numpy(frame).float() / 255.0  # (H, W, 3) in [0,1]
    t = (t - mean) / std                          # per-channel normalise
    t = t.permute(2, 0, 1)                        # (3, H, W)
    return t


class ActionModel:
    """Loads a trained R3D-18 checkpoint and runs sliding-window inference.

    Usage::

        model = ActionModel("runs/baseline/best.pt", num_classes=12)
        for frame in native_fps_stream:
            model.push(frame)
            result = model.infer()   # call every N pushes; returns None until buffer full
            if result is not None:
                class_idx, prob = result
    """

    # Default clip parameters — must match src/config.yaml.
    _DEFAULT_CLIP_LENGTH = 16
    _DEFAULT_SAMPLING_RATE = 2

    def __init__(
        self,
        ckpt_path: str,
        num_classes: int = 12,
        clip_length: int = _DEFAULT_CLIP_LENGTH,
        sampling_rate: int = _DEFAULT_SAMPLING_RATE,
        device: Optional[str] = None,
    ) -> None:
        self.clip_length = clip_length
        self.sampling_rate = sampling_rate
        self.num_classes = num_classes

        # Temporal contract: assert before loading weights so the check is
        # independent of checkpoint availability (unit-testable).
        assert self.buffer_span == clip_length * sampling_rate, (
            f"buffer_span {self.buffer_span} != clip_length*sampling_rate "
            f"{clip_length}*{sampling_rate}={clip_length * sampling_rate}"
        )

        # Rolling buffer holds exactly buffer_span native frames.
        self._buffer: deque[np.ndarray] = deque(maxlen=self.buffer_span)

        # hold-last: None until first successful inference.
        self.last_label: Optional[tuple[int, float]] = None

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        state = torch.load(ckpt_path, map_location=self.device)
        # train.py saves a checkpoint dict: {epoch, model, config, mean, std,
        # num_classes, val_metrics}. Mirror src/eval.py's loading exactly.
        if isinstance(state, dict) and "model" in state:
            weights = state["model"]
            self.num_classes = int(state.get("num_classes", num_classes))
            mean = state.get("mean")
            std = state.get("std")
        elif isinstance(state, dict) and "model_state_dict" in state:
            weights = state["model_state_dict"]
            mean = std = None
        else:  # raw state_dict
            weights = state
            mean = std = None

        self._mean = torch.tensor(
            list(mean) if mean is not None else _FALLBACK_MEAN, dtype=torch.float32
        )
        self._std = torch.tensor(
            list(std) if std is not None else _FALLBACK_STD, dtype=torch.float32
        )

        self._model = build_baseline(num_classes=self.num_classes, pretrained=False)
        self._model.load_state_dict(weights)
        self._model.to(self.device)
        self._model.eval()

    @property
    def buffer_span(self) -> int:
        """Total native frames the buffer holds: clip_length * sampling_rate."""
        return self.clip_length * self.sampling_rate

    def push(self, frame_rgb: np.ndarray) -> None:
        """Add one native-fps frame (HxWx3 uint8 RGB) to the rolling buffer."""
        self._buffer.append(frame_rgb)

    def infer(self) -> Optional[tuple[int, float]]:
        """Run inference if the buffer is full; otherwise return None.

        Samples every sampling_rate-th frame from the buffer, mirroring
        src/dataset.py _frame_indices (span = clip_length * sampling_rate,
        indices = start + t * sampling_rate for t in range(clip_length)).

        Returns:
            (class_idx, probability) of the top-1 class, or None if buffer not full.
            Updates self.last_label on success.
        """
        if len(self._buffer) < self.buffer_span:
            return None

        # Sample every sampling_rate-th frame: indices 0, sr, 2*sr, ...
        buf = list(self._buffer)
        frames = [buf[t * self.sampling_rate] for t in range(self.clip_length)]

        # Preprocess and stack: (T, 3, H, W) -> add batch -> (1, T, 3, H, W)
        tensors = [_preprocess_frame(f, self._mean, self._std) for f in frames]
        clip = torch.stack(tensors, dim=0).unsqueeze(0)  # (1, T, 3, H, W)

        # to_model_input: (B, T, C, H, W) -> (B, C, T, H, W)
        clip = to_model_input(clip).to(self.device)

        with torch.no_grad():
            logits = self._model(clip)          # (1, num_classes)
            probs = torch.softmax(logits, dim=1)
            prob, class_idx = probs[0].max(dim=0)

        result = (int(class_idx.item()), float(prob.item()))
        self.last_label = result
        return result

    def infer_probs(self) -> Optional[np.ndarray]:
        """Same as infer() but returns the full softmax vector (np.float32).

        Independent of `last_label` mutation — callers needing per-class
        probabilities (e.g. binary p_fall) use this directly. Returns None
        until the buffer is full for the first time.
        """
        if len(self._buffer) < self.buffer_span:
            return None
        buf = list(self._buffer)
        frames = [buf[t * self.sampling_rate] for t in range(self.clip_length)]
        tensors = [_preprocess_frame(f, self._mean, self._std) for f in frames]
        clip = torch.stack(tensors, dim=0).unsqueeze(0)
        clip = to_model_input(clip).to(self.device)
        with torch.no_grad():
            logits = self._model(clip)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float32)
        return probs
