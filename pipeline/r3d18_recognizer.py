"""R3D-18 행동 인식기 어댑터.

pipeline.action_model.ActionModel (R3D-18, ETRI test acc 0.693 / macroF1 0.603,
runs/baseline12/test_metrics.json) 을 CnnLstmRecognizer 와 **동일한 공개
인터페이스**(push / infer / last_label:int / last_prob:float / label_name)로
감싼다. run.py · dashboard.py 가 recognizer 객체만 바꿔 그대로 사용.

주의(temporal contract): ActionModel 은 native-fps 버퍼(clip_length×
sampling_rate=32 프레임)를 기대한다. 호출부는 **매 native 프레임마다 push**
하고 infer 는 N 프레임마다 호출해야 학습 분포와 일치한다.
"""
from __future__ import annotations

from typing import Optional

from pipeline.action_model import ActionModel
from pipeline.class_map import CORE_NAMES


class R3d18Recognizer:
    def __init__(self, ckpt_path: str, device: str | None = None,
                 num_classes: int = 12):
        # 파일 없으면 torch.load 가 FileNotFoundError → 호출부 try/except 가 처리
        self._m = ActionModel(ckpt_path, num_classes=num_classes,
                              device=device)
        self.last_label: Optional[int] = None
        self.last_prob: float = 0.0

    def push(self, frame_rgb) -> None:
        self._m.push(frame_rgb)

    def infer(self):
        r = self._m.infer()
        if r is not None:
            self.last_label, self.last_prob = int(r[0]), float(r[1])
        return r

    def label_name(self, idx: Optional[int]) -> str:
        if idx is None or not (0 <= idx < len(CORE_NAMES)):
            return "?"
        return CORE_NAMES[idx]
