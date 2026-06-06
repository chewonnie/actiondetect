"""URFD R3D-18 fall recognizer adapter.

Wraps pipeline.action_model.ActionModel for the 2-class URFD fall model
(experiments/urfd_fall/train_urfd.py; class 1 = FALL, class 0 = ADL).

Public API mirrors the CNN+LSTM URFD adapter used by dashboard.py.
update(ts) returns True exactly on the frame a new event fires, and
suppresses any further True for `cooldown_s` seconds.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from pipeline.action_model import ActionModel


class UrfdFallRecognizer:
    """R3D-18 (URFD-trained binary) fall detector adapter.

    Args:
        ckpt_path:  Path to URFD best.pt (saved by train_urfd.py).
        prob_thr:   P(fall) threshold for firing an event (default 0.5).
        cooldown_s: Minimum seconds between consecutive fired events.
        device:     'cuda' | 'cpu' | None (auto via ActionModel default).

    Streaming usage::

        rec = UrfdFallRecognizer("runs/urfd_fall/best.pt", prob_thr=0.7)
        for frame_rgb in stream:
            rec.push(frame_rgb)
            if frame_idx % infer_every_n == 0:
                rec.infer()              # updates rec.p_fall
            if rec.update(ts_sec):       # cooldown-gated edge event
                fire_alert(ts_sec)
    """

    FALL_CLASS_IDX = 1  # train_urfd.py URFD_LABELS = {"fall": 1, "adl": 0}

    def __init__(
        self,
        ckpt_path: str,
        prob_thr: float = 0.5,
        cooldown_s: float = 3.0,
        device: Optional[str] = None,
    ) -> None:
        self._m = ActionModel(ckpt_path, num_classes=2, device=device)
        if self._m.num_classes != 2:
            raise ValueError(
                f"URFD model must be 2-class, got {self._m.num_classes}"
            )
        self._prob_thr = float(prob_thr)
        self._cooldown = float(cooldown_s)
        self._last_event_ts: Optional[float] = None
        self.p_fall: float = 0.0
        self.last_probs: Optional[np.ndarray] = None

    def push(self, frame_rgb: np.ndarray) -> None:
        self._m.push(frame_rgb)

    def infer(self) -> Optional[float]:
        """Run inference if buffer full. Updates p_fall, returns it (or None)."""
        probs = self._m.infer_probs()
        if probs is None:
            return None
        self.last_probs = probs
        self.p_fall = float(probs[self.FALL_CLASS_IDX])
        return self.p_fall

    def update(self, ts: float) -> bool:
        """Cooldown-gated edge detector: True exactly on a new fall event."""
        if self.p_fall < self._prob_thr:
            return False
        if (
            self._last_event_ts is not None
            and (ts - self._last_event_ts) < self._cooldown
        ):
            return False
        self._last_event_ts = ts
        return True

    def reset(self) -> None:
        """ActionModel ring buffer + cooldown + p_fall 초기화. 데모/배치
        처리에서 영상 간 상태 격리에 사용."""
        if self._m is not None:
            self._m._buffer.clear()
            self._m.last_label = None
        self._last_event_ts = None
        self.p_fall = 0.0
        self.last_probs = None
