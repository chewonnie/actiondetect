"""이미지+자세(pose) 행동 분류 (CNN+LSTM, T=5). early-fusion concat.
RGB 클립 -> 프레임 T=5개 균일샘플 -> [동결 MobileNetV3 576-d ⊕ YOLOv8-pose 51-d]
프레임당 627-d -> LSTM -> 12클래스. image-only cnn_lstm.py 와 동일 split 비교. 실행:
  PYTHONPATH=. python -m experiments.cnn_lstm_pose
pose 키포인트: 17 COCO keypoints × [x_norm, y_norm, conf] (최고신뢰 person, 미검출 0).
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import scan_etri_root                          # noqa: E402
from splits import group_split                              # noqa: E402  leakage-safe

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.class_map import remap                        # noqa: E402  55 -> 12

import torchvision                                          # noqa: E402
from ultralytics import YOLO                                # noqa: E402

ETRI = os.path.join(os.path.dirname(__file__), "..", "etri")
T, NUM_CLASSES, IMG = 5, 12, 112
N_KPT = 17                              # COCO keypoints
POSE_DIM = N_KPT * 3                    # x_norm, y_norm, conf -> 51
FEAT_DIM = 576 + POSE_DIM              # MobileNetV3(576) ⊕ pose(51) = 627
CACHE = os.path.join(os.path.dirname(__file__), "..", "runs", "baseline12",
                     "cnn_lstm_pose_feat_cache.pt")
POSE_W = os.path.join(os.path.dirname(__file__), "..", "yolov8s-pose.pt")
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _scan():
    """RGB 클립을 참가자/12라벨로. ETRI_CLASS_MAP 미사용(스크립트 자체 remap)."""
    out = []
    for s in scan_etri_root(ETRI):                 # 55-class action_idx
        core = remap(s.action_idx)
        if core is None:
            continue
        s.action_idx = core                        # 12-class로 덮어씀
        out.append(s)
    return out


def _clip_frames(path):
    """mp4 -> T프레임 균일샘플. 원본 BGR(pose용)과 112 RGB(cnn용) 동시 반환.
    반환: (rgb112 (T,112,112,3) uint8, bgr_orig list[T] of (H,W,3) uint8)."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = set(np.linspace(0, max(n - 1, 0), T).astype(int).tolist())
    grabbed, f = {}, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if f in idx:
            grabbed[f] = fr                        # BGR 원본 보관
        f += 1
    cap.release()
    if not grabbed:
        z = np.zeros((IMG, IMG, 3), np.uint8)
        return (np.zeros((T, IMG, IMG, 3), np.uint8), [z] * T)
    order = sorted(grabbed)
    pick = np.linspace(0, len(order) - 1, T).astype(int)
    bgr = [grabbed[order[i]] for i in pick]
    rgb112 = np.stack([cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2RGB), (IMG, IMG))
                       for b in bgr])
    return rgb112, bgr


@torch.no_grad()
def _pose_vec(pose_model, bgr_frame, dev):
    """단일 BGR 프레임 -> 51-d pose 벡터(최고신뢰 person 1명, xyn+conf). 미검출 0."""
    res = pose_model(bgr_frame, verbose=False, device=dev)[0]
    kp = res.keypoints
    if kp is None or kp.xyn is None or kp.xyn.shape[0] == 0:
        return np.zeros(POSE_DIM, np.float32)
    # 사람 다수면 box 신뢰도 최고 1명 선택
    if res.boxes is not None and res.boxes.conf is not None \
            and res.boxes.conf.shape[0] == kp.xyn.shape[0]:
        j = int(res.boxes.conf.argmax())
    else:
        j = 0
    xyn = kp.xyn[j].cpu().numpy()                  # (17,2) [0,1] 정규화
    conf = (kp.conf[j].cpu().numpy() if kp.conf is not None
            else np.ones(N_KPT, np.float32))       # (17,)
    return np.concatenate([xyn, conf[:, None]], axis=1).reshape(-1).astype(np.float32)


@torch.no_grad()
def build_feature_cache(samples, dev):
    """클립당 (T,627) = MobileNetV3(576) ⊕ YOLOv8-pose(51) 1회 추출·캐시."""
    if os.path.exists(CACHE):
        return torch.load(CACHE, map_location="cpu", weights_only=False)
    w = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    cnn = torchvision.models.mobilenet_v3_small(weights=w)
    cnn.classifier = nn.Identity()                 # -> 576-d feature
    cnn.eval().to(dev)
    pose = YOLO(POSE_W)
    cache = {}
    for i, s in enumerate(samples):
        rgb112, bgr = _clip_frames(s.rgb_path)
        fr = rgb112.astype(np.float32) / 255.0
        fr = (fr - _MEAN) / _STD
        x = torch.from_numpy(fr).permute(0, 3, 1, 2).float().to(dev)  # (T,3,H,W)
        img_feat = cnn(x).cpu().numpy()                              # (T,576)
        pose_feat = np.stack([_pose_vec(pose, b, dev) for b in bgr])  # (T,51)
        feat = np.concatenate([img_feat, pose_feat], axis=1)         # (T,627)
        cache[s.rgb_path] = torch.from_numpy(feat).half()
        if (i + 1) % 100 == 0:
            print("  feat %d/%d" % (i + 1, len(samples)), flush=True)
    torch.save(cache, CACHE)
    return cache


class DS(Dataset):
    def __init__(self, s, cache):
        self.s, self.c = s, cache
    def __len__(self):
        return len(self.s)
    def __getitem__(self, i):
        return self.c[self.s[i].rgb_path].float(), self.s[i].action_idx


class CNNLSTM(nn.Module):
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
          "| train/val/test:", len(sp.train), len(sp.val), len(sp.test), flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = build_feature_cache(sp.train + sp.val + sp.test, dev)

    tr = DataLoader(DS(sp.train, cache), batch_size=64, shuffle=True)
    va = DataLoader(DS(sp.val, cache), batch_size=128)
    te = DataLoader(DS(sp.test, cache), batch_size=128)

    cnt = np.bincount([s.action_idx for s in sp.train], minlength=NUM_CLASSES) + 1
    w = torch.tensor((cnt.sum() / cnt) / (cnt.sum() / cnt).mean(),
                     dtype=torch.float32, device=dev)
    model = CNNLSTM().to(dev)
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
                         "n_cls": NUM_CLASSES, "T": T, "pose_dim": POSE_DIM},
                "backbone": "mobilenet_v3_small (ImageNet, frozen) + yolov8s-pose",
                "test_accuracy": round(ta, 4), "test_macro_f1": round(tf, 4)},
               os.path.join(base, "cnn_lstm_pose.pt"))
    res = {
        "model": "CNN(MobileNetV3-small, frozen) ⊕ YOLOv8-pose(51-d) + LSTM, "
                 "image+pose early-fusion, T=5",
        "selection": "best epoch by val macro-F1 (same criterion as R3D-18)",
        "split": "cross-subject group_split seed42 val/test .15 (same as R3D-18 & image-only)",
        "test_participants": sp.test_participants,
        "test_clips": len(sp.test),
        "feat_dim": FEAT_DIM,
        "pose": "yolov8s-pose, 17 COCO kpts × [x_norm, y_norm, conf], top-conf person",
        "test_accuracy": round(ta, 4),
        "test_macro_f1": round(tf, 4),
        "weights": "runs/baseline12/cnn_lstm_pose.pt",
        "compare": {"R3D18": {"acc": 0.6928, "f1": 0.6029},
                    "cnn_lstm_image_only_T16": {"acc": 0.2526, "f1": 0.1902},
                    "skeleton_LSTM": {"acc": 0.3686, "f1": 0.3372}},
        "note": "Image+pose early fusion. Frozen MobileNetV3 backbone + YOLOv8-pose "
                "keypoints; only LSTM trained. RGB-cam deployable (no Kinect).",
    }
    json.dump(res, open(os.path.join(base, "cnn_lstm_pose_result.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
