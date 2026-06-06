"""pipeline/cnn_lstm_infer.py — 스트리밍 CNN-LSTM 행동 인식기.

experiments/cnn_lstm.py 와 동일 아키텍처(MobileNetV3-small 동결 특징 →
2층 LSTM, T=16)로 학습된 cnn_lstm.pt 를 로드해, 프레임을 밀어 넣으면
T=16 롤링 버퍼에서 12 core class 를 추론한다. 가중치는 저장소에 포함하지
않음 — 사용자가 ckpt 경로(pipeline/config.yaml paths.cnn_lstm_ckpt)에 배치.
"""
from __future__ import annotations

import os
import sys
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.class_map import CORE_NAMES  # noqa: E402

T, IMG = 16, 112
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class CNNLSTM(nn.Module):
    """experiments/cnn_lstm.py 의 CNNLSTM 과 동일 (state_dict 호환)."""

    def __init__(self, in_dim=576, hid=128, layers=2, n_cls=12):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, layers, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(hid, n_cls))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


class CnnLstmRecognizer:
    """프레임 스트림 → (core_idx, prob). T=16 롤링 버퍼, hold-last."""

    def __init__(self, ckpt_path: str, device: str | None = None):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"cnn_lstm ckpt 없음: {ckpt_path} "
                f"(pipeline/config.yaml paths.cnn_lstm_ckpt 에 배치)"
            )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        w = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        cnn = torchvision.models.mobilenet_v3_small(weights=w)
        cnn.classifier = nn.Identity()                 # -> 576-d
        self.cnn = cnn.eval().to(self.device)
        for p in self.cnn.parameters():
            p.requires_grad_(False)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        a = ckpt.get("arch", {})
        self.model = CNNLSTM(
            in_dim=a.get("in_dim", 576), hid=a.get("hid", 128),
            layers=a.get("layers", 2), n_cls=a.get("n_cls", 12),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(self.device)
        self.buf: deque = deque(maxlen=T)
        self.last_label: Optional[int] = None
        self.last_prob: float = 0.0

    @torch.no_grad()
    def push(self, frame_rgb: np.ndarray) -> None:
        import cv2

        f = cv2.resize(frame_rgb, (IMG, IMG)).astype(np.float32) / 255.0
        f = (f - _MEAN) / _STD
        x = torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0).float()
        feat = self.cnn(x.to(self.device)).squeeze(0).cpu()   # (576,)
        self.buf.append(feat)

    @torch.no_grad()
    def infer(self) -> Optional[tuple[int, float]]:
        if len(self.buf) < T:
            return None
        seq = torch.stack(list(self.buf)).unsqueeze(0).to(self.device)  # (1,T,576)
        prob = torch.softmax(self.model(seq), dim=1)[0]
        idx = int(prob.argmax())
        self.last_label, self.last_prob = idx, float(prob[idx])
        return self.last_label, self.last_prob

    def label_name(self, idx: Optional[int]) -> str:
        if idx is None or not (0 <= idx < len(CORE_NAMES)):
            return "?"
        return CORE_NAMES[idx]
