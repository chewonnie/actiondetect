"""이미지특징 + YOLO 검출벡터 → LSTM 행동 분류 (멀티모달 베이스라인).

프레임별 입력 = MobileNetV3-small(동결) 576-d 이미지특징
              ⊕ YOLO 검출벡터 4-d [person, bed, chair, tv 각 max conf, Option A]
            = 580-d  →  2층 LSTM  →  12 core class.

정답(행동 라벨)은 입력에 넣지 않음(누수 없음). YOLO가 내는 '검출 라벨'을
보조 특징으로만 사용. R3D-18 / skeleton-LSTM / image-CNN-LSTM 과 동일
cross-subject split(seed42, val/test .15, test=P02/P09/P14)로 비교.

실행: PYTHONPATH=. python -m experiments.cnn_yolo_lstm
"""
from __future__ import annotations

import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import parse_action_index, scan_etri_root   # noqa: E402
from splits import group_split                           # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.class_map import remap                      # noqa: E402
from pipeline.detector import YoloDetector                # noqa: E402

import torchvision                                        # noqa: E402

ETRI = os.path.join(os.path.dirname(__file__), "..", "etri")
T, NUM_CLASSES, IMG = 16, 12, 112
YOLO_CLASSES = ["person", "bed", "chair", "tv"]            # 검출벡터 순서 (Option A)
FEAT_DIM = 576 + len(YOLO_CLASSES)                         # 580
CACHE = os.path.join(os.path.dirname(__file__), "..", "runs", "baseline12",
                     "cnn_yolo_lstm_feat_cache.pt")
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class _S:
    __slots__ = ("participant", "action_idx", "rgb_path")
    def __init__(self, p, a, path):
        self.participant, self.action_idx, self.rgb_path = p, a, path


def _scan():
    out = []
    for s in scan_etri_root(ETRI):
        core = remap(s.action_idx)
        if core is None:
            continue
        out.append(_S(s.participant, core, s.rgb_path))
    return out


def _clip_frames(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = set(np.linspace(0, max(n - 1, 0), T).astype(int).tolist())
    grabbed, f = {}, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if f in idx:
            grabbed[f] = fr                                # BGR (YOLO/cv2용)
        f += 1
    cap.release()
    if not grabbed:
        return [np.zeros((IMG, IMG, 3), np.uint8)] * T
    order = sorted(grabbed)
    return [grabbed[order[i]]
            for i in np.linspace(0, len(order) - 1, T).astype(int)]


def _yolo_vec(detector, frame_bgr):
    """프레임 → [person, bed, chair, tv] 각 최대 confidence (검출 없으면 0)."""
    best = {c: 0.0 for c in YOLO_CLASSES}
    for cname, _, cf in detector.predict(frame_bgr):
        if cname in best and cf > best[cname]:
            best[cname] = cf
    return np.array([best[c] for c in YOLO_CLASSES], np.float32)


@torch.no_grad()
def build_cache(samples, dev):
    if os.path.exists(CACHE):
        return torch.load(CACHE, map_location="cpu", weights_only=False)
    w = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    cnn = torchvision.models.mobilenet_v3_small(weights=w)
    cnn.classifier = nn.Identity()
    cnn.eval().to(dev)
    cfg_d = {}
    try:
        import yaml
        cfg_d = (yaml.safe_load(open(os.path.join(os.path.dirname(__file__),
                 "..", "pipeline", "config.yaml"))) or {}).get("detector", {})
    except Exception:
        pass
    det = YoloDetector(
        "yolov8s.pt",
        person_conf=cfg_d.get("person_conf", 0.40),
        object_conf=cfg_d.get("object_conf", 0.30),
        object_classes={"bed", "chair", "tv"},
    )
    cache = {}
    for i, s in enumerate(samples):
        frames = _clip_frames(s.rgb_path)
        rgb = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB),
                                   (IMG, IMG)) for f in frames]).astype(np.float32)
        rgb = (rgb / 255.0 - _MEAN) / _STD
        x = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().to(dev)
        img_feat = cnn(x).cpu().numpy()                       # (T,576)
        yv = np.stack([_yolo_vec(det, f) for f in frames])    # (T,4)
        cache[s.rgb_path] = np.concatenate([img_feat, yv], 1).astype(np.float16)
        if (i + 1) % 200 == 0:
            print("  feat %d/%d" % (i + 1, len(samples)), flush=True)
    torch.save(cache, CACHE)
    return cache


class DS(Dataset):
    def __init__(self, s, cache):
        self.s, self.c = s, cache
    def __len__(self):
        return len(self.s)
    def __getitem__(self, i):
        return torch.from_numpy(self.c[self.s[i].rgb_path]).float(), \
            self.s[i].action_idx


class CNNYoloLSTM(nn.Module):
    def __init__(self, in_dim=FEAT_DIM, hid=128, layers=2, n_cls=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, layers, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(hid, n_cls))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


def main():
    torch.manual_seed(42); np.random.seed(42)
    sp = group_split(_scan(), val_ratio=0.15, test_ratio=0.15, seed=42)
    print("test participants:", sp.test_participants,
          "| train/val/test:", len(sp.train), len(sp.val), len(sp.test),
          flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = build_cache(sp.train + sp.val + sp.test, dev)

    tr = DataLoader(DS(sp.train, cache), batch_size=64, shuffle=True)
    va = DataLoader(DS(sp.val, cache), batch_size=128)
    te = DataLoader(DS(sp.test, cache), batch_size=128)
    cnt = np.bincount([s.action_idx for s in sp.train],
                      minlength=NUM_CLASSES) + 1
    w = torch.tensor((cnt.sum() / cnt) / (cnt.sum() / cnt).mean(),
                     dtype=torch.float32, device=dev)
    model = CNNYoloLSTM().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)

    def evaluate(dl):
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for x, y in dl:
                P += model(x.to(dev)).argmax(1).cpu().tolist(); Y += y.tolist()
        from sklearn.metrics import accuracy_score, f1_score
        return accuracy_score(Y, P), f1_score(Y, P, average="macro")

    best_vf, best = -1.0, None
    for ep in range(20):
        model.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); lossf(model(x), y).backward(); opt.step()
        va_a, va_f = evaluate(va)
        if va_f > best_vf:
            best_vf, best = va_f, {k: v.cpu().clone()
                                   for k, v in model.state_dict().items()}
        print("epoch %2d  val_acc %.4f  val_macroF1 %.4f" % (ep + 1, va_a, va_f),
              flush=True)

    model.load_state_dict(best)
    ta, tf = evaluate(te)
    base = os.path.join(os.path.dirname(__file__), "..", "runs", "baseline12")
    torch.save({"state_dict": best,
                "arch": {"in_dim": FEAT_DIM, "hid": 128, "layers": 2,
                         "n_cls": NUM_CLASSES, "T": T},
                "feature": "MobileNetV3-small(576) + YOLO[person,bed,chair,tv] conf(4)",
                "test_accuracy": round(ta, 4), "test_macro_f1": round(tf, 4)},
               os.path.join(base, "cnn_yolo_lstm.pt"))
    res = {
        "model": "CNN(MobileNetV3-small,frozen)+YOLOdet(4) → LSTM, image-only, T=16",
        "input": "image feat 576 ⊕ YOLO[person,bed,chair,tv] max-conf 4 (정답라벨 미사용=누수없음)",
        "selection": "best epoch by val macro-F1 (same as others)",
        "split": "cross-subject group_split seed42 val/test .15 (R3D-18 등과 동일)",
        "test_participants": sp.test_participants,
        "test_clips": len(sp.test),
        "test_accuracy": round(ta, 4),
        "test_macro_f1": round(tf, 4),
        "weights": "runs/baseline12/cnn_yolo_lstm.pt",
        "compare": {"R3D18": {"acc": 0.6928, "f1": 0.6029},
                    "skeleton_LSTM": {"acc": 0.3686, "f1": 0.3372},
                    "image_CNN_LSTM": {"acc": 0.2526, "f1": 0.1902}},
    }
    json.dump(res, open(os.path.join(base, "cnn_yolo_lstm_result.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
