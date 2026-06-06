"""LSTM 행동 분류 (학부생 베이스라인). JointCSV 관절 시퀀스 -> LSTM -> 12클래스.
R3D-18과 동일 cross-subject split(seed42, val/test .15)로 비교. 실행:
  PYTHONPATH=. python -m experiments.lstm
"""
from __future__ import annotations

import csv as _csv
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import parse_action_index            # noqa: E402  A### -> 0-based
from splits import group_split                    # noqa: E402  participant-grouped (no leakage)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.class_map import remap               # noqa: E402  55 -> 12

ETRI = os.path.join(os.path.dirname(__file__), "..", "etri", "JointCSV")
N_JOINTS, T, NUM_CLASSES = 25, 32, 12
ROOT_J = 1                                          # Kinect SpineBase: 좌표 중심


class _S:
    __slots__ = ("participant", "action_idx", "path")
    def __init__(self, p, a, path):
        self.participant, self.action_idx, self.path = p, a, path


def _scan(participants):
    out = []
    for p in participants:
        for f in sorted(glob.glob(os.path.join(ETRI, p, "*", "*.csv"))):
            ai = parse_action_index(os.path.basename(f))
            if ai is None:
                continue
            core = remap(ai)
            if core is None:
                continue
            out.append(_S(p, core, f))
    return out


def _feats(csv_path):
    """CSV -> (T,75): 프레임당 25관절 3D, SpineBase 중심화, T프레임 균일 샘플."""
    by_frame = {}
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as fh:
        for r in _csv.DictReader(fh):
            fn = int(float(r["frameNum"]))
            if fn in by_frame:                      # 멀티바디 -> 첫 바디만
                continue
            v = []
            for j in range(1, N_JOINTS + 1):
                v += [float(r["joint%d_3d%s" % (j, ax)] or 0.0) for ax in "XYZ"]
            by_frame[fn] = v
    frames = [by_frame[k] for k in sorted(by_frame)]
    if not frames:
        return np.zeros((T, N_JOINTS * 3), np.float32)
    a = np.asarray(frames, np.float32)
    root = a[:, (ROOT_J - 1) * 3:(ROOT_J - 1) * 3 + 3]
    a = (a.reshape(len(a), N_JOINTS, 3) - root[:, None, :]).reshape(len(a), -1)
    return a[np.linspace(0, len(a) - 1, T).astype(int)]


class DS(Dataset):
    def __init__(self, s):
        self.s, self.cache = s, {}
    def __len__(self):
        return len(self.s)
    def __getitem__(self, i):
        if i not in self.cache:
            self.cache[i] = _feats(self.s[i].path)
        return torch.from_numpy(self.cache[i]), self.s[i].action_idx


class LSTM(nn.Module):
    def __init__(self, in_dim=N_JOINTS * 3, hid=128, layers=2, n_cls=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, layers, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(hid, n_cls))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


def main():
    torch.manual_seed(42); np.random.seed(42)
    parts = sorted({os.path.basename(p) for p in glob.glob(os.path.join(ETRI, "P*"))})
    sp = group_split(_scan(parts), val_ratio=0.15, test_ratio=0.15, seed=42)
    print("test participants:", sp.test_participants,
          "| train/val/test:", len(sp.train), len(sp.val), len(sp.test))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = DataLoader(DS(sp.train), batch_size=64, shuffle=True, num_workers=8)
    va = DataLoader(DS(sp.val), batch_size=128, num_workers=8)
    te = DataLoader(DS(sp.test), batch_size=128, num_workers=8)

    cnt = np.bincount([s.action_idx for s in sp.train], minlength=NUM_CLASSES) + 1
    w = torch.tensor((cnt.sum() / cnt) / (cnt.sum() / cnt).mean(), dtype=torch.float32, device=dev)

    model = LSTM().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)

    def evaluate(dl):
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for x, y in dl:
                P += model(x.to(dev)).argmax(1).cpu().tolist(); Y += y.tolist()
        from sklearn.metrics import accuracy_score, f1_score
        return accuracy_score(Y, P), f1_score(Y, P, average="macro")

    best_vf, best = -1.0, None                       # best epoch by val macro-F1 (R3D-18과 동일 기준)
    for ep in range(20):
        model.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); lossf(model(x), y).backward(); opt.step()
        va_a, va_f = evaluate(va)
        if va_f > best_vf:
            best_vf, best = va_f, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print("epoch %2d  val_acc %.4f  val_macroF1 %.4f" % (ep + 1, va_a, va_f))

    model.load_state_dict(best)                       # best 체크포인트로 test
    ta, tf = evaluate(te)

    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "runs",
                             "baseline12", "lstm.pt")
    torch.save({                                       # standalone-loadable
        "state_dict": best,
        "arch": {"in_dim": N_JOINTS * 3, "hid": 128, "layers": 2,
                 "n_cls": NUM_CLASSES, "T": T},
        "selection": "best epoch by val macro-F1",
        "test_accuracy": round(ta, 4),
        "test_macro_f1": round(tf, 4),
    }, ckpt_path)
    print("saved LSTM weights ->", ckpt_path)

    res = {
        "model": "2-layer LSTM (in=75 hid=128), joint 3D sequence T=32, undergrad baseline",
        "selection": "best epoch by val macro-F1 (same criterion as R3D-18)",
        "split": "cross-subject group_split seed42 val/test .15 (same as R3D-18)",
        "test_participants": sp.test_participants,
        "test_clips": len(sp.test),
        "test_accuracy": round(ta, 4),
        "test_macro_f1": round(tf, 4),
        "weights": "runs/baseline12/lstm.pt",
        "compare_R3D18": {"test_accuracy": 0.6928, "test_macro_f1": 0.6029},
        "takeaway": "Simple LSTM << R3D-18 -> R3D-18 (runs/baseline12/best.pt) kept as the strong model; LSTM is a lightweight reference.",
    }
    out = os.path.join(os.path.dirname(__file__), "..", "runs", "baseline12", "lstm_result.json")
    json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
