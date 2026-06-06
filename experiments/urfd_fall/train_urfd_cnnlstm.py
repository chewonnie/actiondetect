"""URFD binary fall-vs-ADL trainer — CNN+LSTM variant.

Same backbone choice as experiments/cnn_lstm.py:
  * MobileNetV3-small (ImageNet, frozen) — per-frame 576-d feature
  * 2-layer LSTM hidden=128 dropout=0.3
  * Linear head sized to n_cls (here: 2)

Differences vs R3D-18 sibling (train_urfd.py):
  * Frame loader uniform-samples T=16 frames per clip (no rolling buffer)
  * URFD right-half crop applied at frame-load time
  * Builds a one-shot feature cache so training itself is sub-minute on CPU

Reuses scan_urfd_root + stratified_split from train_urfd.py — the
train/val/test composition matches the R3D-18 URFD baseline exactly.

Run from repo root:
    python experiments/urfd_fall/train_urfd_cnnlstm.py \
        --config experiments/urfd_fall/config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from experiments.urfd_fall.train_urfd import (  # noqa: E402
    scan_urfd_root,
    stratified_split,
)

T_FRAMES = 16
IMG = 112
FEAT_DIM = 576
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _clip_frames_right_rgb(path: str, t: int = T_FRAMES, img: int = IMG) -> np.ndarray:
    """URFD mp4 → (t,img,img,3) uint8 RGB from RIGHT half only."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = set(np.linspace(0, max(n - 1, 0), t).astype(int).tolist())
    grabbed: dict = {}
    f = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if f in idx:
            rgb = cv2.cvtColor(fr[:, 320:, :], cv2.COLOR_BGR2RGB)
            grabbed[f] = cv2.resize(rgb, (img, img))
        f += 1
    cap.release()
    if not grabbed:
        return np.zeros((t, img, img, 3), np.uint8)
    order = sorted(grabbed)
    pick = np.linspace(0, len(order) - 1, t).astype(int)
    return np.stack([grabbed[order[i]] for i in pick])


@torch.no_grad()
def build_feature_cache(samples, cache_path: str, dev: torch.device):
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    w = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    cnn = torchvision.models.mobilenet_v3_small(weights=w)
    cnn.classifier = nn.Identity()  # → 576-d
    cnn.eval().to(dev)
    cache: dict = {}
    for i, s in enumerate(samples):
        fr = _clip_frames_right_rgb(s.rgb_path).astype(np.float32) / 255.0
        fr = (fr - _MEAN) / _STD
        x = torch.from_numpy(fr).permute(0, 3, 1, 2).float().to(dev)  # (T,3,H,W)
        cache[s.rgb_path] = cnn(x).cpu().half()                       # (T,576)
        if (i + 1) % 20 == 0:
            print(f"  feat {i + 1}/{len(samples)}", flush=True)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(cache, cache_path)
    return cache


class DS(Dataset):
    def __init__(self, samples, cache):
        self.s, self.c = samples, cache

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return self.c[self.s[i].rgb_path].float(), self.s[i].action_idx


class CNNLSTM(nn.Module):
    """Same architecture as experiments/cnn_lstm.py (state_dict-compatible)."""

    def __init__(self, in_dim: int = FEAT_DIM, hid: int = 128,
                 layers: int = 2, n_cls: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, layers, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(hid, n_cls))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


def main(config_path: str, epochs_override: int | None = None) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    n_epochs = epochs_override if epochs_override is not None else 30
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = REPO / cfg.get("output_dir", "runs/urfd_fall")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = cfg["data_root"]
    if not os.path.isabs(data_root):
        data_root = str(REPO / data_root)
    samples = scan_urfd_root(data_root)
    print(f"[scan] {len(samples)} clips "
          f"(fall={sum(s.action_idx == 1 for s in samples)}, "
          f"adl={sum(s.action_idx == 0 for s in samples)})")

    tr, va, te = stratified_split(
        samples,
        val_ratio=float(cfg.get("val_ratio", 0.15)),
        test_ratio=float(cfg.get("test_ratio", 0.15)),
        seed=int(cfg.get("split_seed", 42)),
    )
    print(f"[split] train={len(tr)} val={len(va)} test={len(te)}")

    dev_req = str(cfg.get("device", "auto")).lower()
    if dev_req == "cuda" or (dev_req == "auto" and torch.cuda.is_available()):
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    print(f"[device] {dev}")

    cache_path = str(out_dir / "cnn_lstm_feat_cache.pt")
    cache = build_feature_cache(tr + va + te, cache_path, dev)

    dl_tr = DataLoader(DS(tr, cache), batch_size=16, shuffle=True)
    dl_va = DataLoader(DS(va, cache), batch_size=32)
    dl_te = DataLoader(DS(te, cache), batch_size=32)

    cnt = np.bincount([s.action_idx for s in tr], minlength=2) + 1
    w = torch.tensor(
        (cnt.sum() / cnt) / (cnt.sum() / cnt).mean(),
        dtype=torch.float32, device=dev,
    )
    print(f"[imbalance] train counts={cnt.tolist()}  weights={w.tolist()}")

    model = CNNLSTM(n_cls=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(
        weight=w, label_smoothing=float(cfg.get("label_smoothing", 0.0))
    )

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
    )

    def evaluate(dl):
        model.eval()
        P, Y, S = [], [], []
        with torch.no_grad():
            for x, y in dl:
                logit = model(x.to(dev))
                P += logit.argmax(1).cpu().tolist()
                S += torch.softmax(logit, 1)[:, 1].cpu().tolist()
                Y += y.tolist()
        ap = (float(average_precision_score(Y, S))
              if len(set(Y)) > 1 else float("nan"))
        return (float(accuracy_score(Y, P)),
                float(f1_score(Y, P, average="macro")),
                ap)

    best_vf, best, log = -1.0, None, []
    for ep in range(1, n_epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss, n_seen = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * y.size(0)
            n_seen += y.size(0)
        tr_loss /= max(1, n_seen)
        va_a, va_f, va_ap = evaluate(dl_va)
        dt = time.time() - t0
        print(f"epoch {ep:02d}/{n_epochs} | {dt:5.1f}s | train_loss={tr_loss:.4f} "
              f"| val acc={va_a:.3f} f1={va_f:.3f} aucpr={va_ap:.3f}",
              flush=True)
        log.append({"epoch": ep, "dt": dt, "train_loss": tr_loss,
                    "val_acc": va_a, "val_f1": va_f, "val_aucpr": va_ap})
        if va_f > best_vf:
            best_vf = va_f
            best = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best)
    te_a, te_f, te_ap = evaluate(dl_te)
    print(f"[test] acc={te_a:.3f} f1={te_f:.3f} aucpr={te_ap:.3f}")

    torch.save({
        "state_dict": best,
        "arch": {"in_dim": FEAT_DIM, "hid": 128, "layers": 2,
                 "n_cls": 2, "T": T_FRAMES},
        "backbone": "mobilenet_v3_small (ImageNet, frozen)",
        "test_accuracy": round(te_a, 4),
        "test_macro_f1": round(te_f, 4),
        "test_aucpr_fall": round(te_ap, 4),
        "input_crop": "right_half_320:640_x_240",
        "labels": {"0": "ADL", "1": "FALL"},
    }, out_dir / "cnn_lstm.pt")

    res = {
        "model": "CNN(MobileNetV3-small, frozen) + LSTM, URFD binary fall vs ADL",
        "split": "stratified per-clip (same as R3D-18 URFD baseline)",
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "test_accuracy": round(te_a, 4),
        "test_macro_f1": round(te_f, 4),
        "test_aucpr_fall": round(te_ap, 4),
        "weights": "runs/urfd_fall/cnn_lstm.pt",
        "compare_R3D18_URFD": {"acc": 1.0, "f1": 1.0, "aucpr": 1.0},
        "epochs": n_epochs,
        "epoch_log": log,
    }
    with open(out_dir / "cnn_lstm_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "epoch_log"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(HERE / "config.yaml"))
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()
    main(args.config, args.epochs)
