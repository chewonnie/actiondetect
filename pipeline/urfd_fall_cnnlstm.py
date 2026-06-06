"""URFD CNN+LSTM fall recognizer adapter.

Wraps the URFD-trained CNN+LSTM (MobileNetV3-small frozen 576-d →
2-layer LSTM hidden=128 → 2-class) saved by
experiments/urfd_fall/train_urfd_cnnlstm.py. Test metrics on URFD split:
acc=0.90, macro-F1=0.899, aucpr_fall=1.0 (vs R3D-18 sibling 1.0/1.0/1.0).

Public API mirrors pipeline.urfd_fall_model.UrfdFallRecognizer exactly so
dashboard.py can swap it in transparently. update(ts) is the cooldown-gated
edge detector: True exactly on the frame a new event fires.

URFD frames are 640×240 [Depth(320) | RGB(320)] — the model was trained on
the RIGHT half only. push() auto-detects URFD shape and applies the crop;
non-URFD frames (e.g. live webcam 640×480) are resized as-is. Live-webcam
accuracy is unreliable due to domain shift — same caveat as the R3D-18
sibling (PLAN §3.8a).
"""

from __future__ import annotations

import os
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision

T_FRAMES = 16
IMG = 112
FEAT_DIM = 576
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class _CNNLSTM(nn.Module):
    """Architecture matches experiments/urfd_fall/train_urfd_cnnlstm.py."""

    def __init__(self, in_dim: int = FEAT_DIM, hid: int = 128,
                 layers: int = 2, n_cls: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, layers, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(hid, n_cls))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


class UrfdFallCnnLstmRecognizer:
    """CNN+LSTM (URFD-trained binary) fall detector adapter.

    Args:
        ckpt_path:  Path to URFD cnn_lstm.pt (saved by train_urfd_cnnlstm.py).
        prob_thr:   P(fall) threshold for firing an event (default 0.5).
        cooldown_s: Minimum seconds between consecutive fired events.
        device:     'cuda' | 'cpu' | None (auto).

    Streaming usage::

        rec = UrfdFallCnnLstmRecognizer("runs/urfd_fall/cnn_lstm.pt",
                                         prob_thr=0.7)
        for frame_rgb in stream:
            rec.push(frame_rgb)
            if frame_idx % infer_every_n == 0:
                rec.infer()              # updates rec.p_fall
            if rec.update(ts_sec):       # cooldown-gated edge event
                fire_alert(ts_sec)
    """

    FALL_CLASS_IDX = 1   # train_urfd_cnnlstm.py URFD_LABELS = {"fall": 1, "adl": 0}

    def __init__(
        self,
        ckpt_path: str,
        prob_thr: float = 0.5,
        cooldown_s: float = 3.0,
        device: Optional[str] = None,
    ) -> None:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"URFD cnn_lstm ckpt 없음: {ckpt_path} "
                f"(pipeline/config.yaml paths.urfd_fall_ckpt 확인)"
            )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Frozen MobileNetV3-small feature extractor (576-d).
        w = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        cnn = torchvision.models.mobilenet_v3_small(weights=w)
        cnn.classifier = nn.Identity()
        self._cnn = cnn.eval().to(self.device)
        for p in self._cnn.parameters():
            p.requires_grad_(False)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        a = ckpt.get("arch", {})
        self._model = _CNNLSTM(
            in_dim=a.get("in_dim", FEAT_DIM),
            hid=a.get("hid", 128),
            layers=a.get("layers", 2),
            n_cls=a.get("n_cls", 2),
        )
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval().to(self.device)
        self.num_classes = int(a.get("n_cls", 2))
        if self.num_classes != 2:
            raise ValueError(
                f"URFD CNN+LSTM 은 2-class 여야 함 (n_cls={self.num_classes})"
            )

        self._buf: deque = deque(maxlen=T_FRAMES)
        self._prob_thr = float(prob_thr)
        self._cooldown = float(cooldown_s)
        self._last_event_ts: Optional[float] = None
        self.p_fall: float = 0.0
        self.last_probs: Optional[np.ndarray] = None

    @torch.no_grad()
    def push(self, frame_rgb: np.ndarray) -> None:
        """프레임을 push → MobileNetV3 576-d 특징 추출 후 T=16 ring 누적."""
        import cv2
        # URFD signature: 640×240 with Depth|RGB → 오른쪽 절반만 사용
        h, w = frame_rgb.shape[:2]
        if w == 640 and h == 240:
            frame_rgb = frame_rgb[:, 320:, :]
        f = cv2.resize(frame_rgb, (IMG, IMG)).astype(np.float32) / 255.0
        f = (f - _MEAN) / _STD
        x = torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0).float()
        feat = self._cnn(x.to(self.device)).squeeze(0).cpu()   # (576,)
        self._buf.append(feat)

    @torch.no_grad()
    def infer(self) -> Optional[float]:
        """버퍼 가득 차면 추론. p_fall 갱신, 반환(미충분 시 None)."""
        if len(self._buf) < T_FRAMES:
            return None
        seq = torch.stack(list(self._buf)).unsqueeze(0).to(self.device)
        probs = torch.softmax(self._model(seq), dim=1)[0].cpu().numpy().astype(np.float32)
        self.last_probs = probs
        self.p_fall = float(probs[self.FALL_CLASS_IDX])
        return self.p_fall

    def infer_probs(self) -> Optional[np.ndarray]:
        """전체 softmax 벡터 반환(미충분 시 None). UrfdFallRecognizer 와 호환."""
        if self.infer() is None:
            return None
        return self.last_probs

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
        """ring buffer + cooldown + p_fall 초기화. 데모/배치 처리 시 영상 간
        상태 격리에 사용 (T=16 잔류 프레임이 다음 영상 첫 추론을 오염시키는 것 방지).
        """
        self._buf.clear()
        self._last_event_ts = None
        self.p_fall = 0.0
        self.last_probs = None
