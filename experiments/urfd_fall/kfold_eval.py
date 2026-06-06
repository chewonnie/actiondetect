"""K-fold cross-validation runner for URFD fall classification.

Compares R3D-18 (Kinetics-pretrained, full fine-tune) vs CNN+LSTM
(MobileNetV3-small frozen → 2-layer LSTM) on a stratified 5-fold split
of all 70 URFD clips. Writes per-fold + aggregated metrics as both
JSON and a Markdown table.

Run from repo root:
    python experiments/urfd_fall/kfold_eval.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from dataset import ETRIClipDataset, compute_train_stats  # noqa: E402
from metrics import class_counts, class_weights_from_counts, evaluate  # noqa: E402
from model import build_baseline, to_model_input  # noqa: E402
from transforms import ClipTransform  # noqa: E402

from experiments.urfd_fall.train_urfd import (  # noqa: E402
    _additional_targets,
    build_urfd_transform,
    scan_urfd_root,
    stratified_split,
)
from experiments.urfd_fall.train_urfd_cnnlstm import (  # noqa: E402
    CNNLSTM,
    DS,
    FEAT_DIM,
    build_feature_cache,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_folds(samples, k: int, val_ratio: float, seed: int):
    """Stratified K-fold. Each fold yields (train, val, test). val is a
    stratified slice (val_ratio) of the non-test remainder; the rest is train.
    """
    labels = np.array([s.action_idx for s in samples])
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for fi, (rest_idx, test_idx) in enumerate(
        skf.split(np.zeros(len(samples)), labels)
    ):
        test = [samples[i] for i in test_idx]
        rest = [samples[i] for i in rest_idx]
        # stratified_split returns (train, val, test); we feed val_ratio twice
        # and merge the dummy 'test' chunk back into train.
        tr_r, va_r, te_r = stratified_split(
            rest, val_ratio=val_ratio, test_ratio=val_ratio, seed=seed + fi,
        )
        train = tr_r + te_r
        folds.append((train, va_r, test))
    return folds


# --- R3D-18 single-split training ------------------------------------------
def run_r3d(tr, va, te, cfg, dev, *, epochs: int = 12) -> dict:
    _set_seed(int(cfg.get("seed", 42)))
    num_classes = int(cfg.get("num_classes", 2))
    clip_length = int(cfg["clip_length"])
    sampling_rate = int(cfg["sampling_rate"])
    img_size = int(cfg["img_size"])

    # Stats on raw RIGHT-half only (no augmentation)
    crop_only = ClipTransform(
        A.Compose(
            [A.Crop(x_min=320, y_min=0, x_max=640, y_max=240, p=1.0)],
            additional_targets=_additional_targets(clip_length),
        ),
        mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), clip_length=clip_length,
    )
    raw = ETRIClipDataset(
        samples=tr, clip_length=clip_length, sampling_rate=sampling_rate,
        img_size=img_size, transform=crop_only, phase="train",
        num_classes=num_classes, seed=int(cfg.get("seed", 42)),
    )
    mean, std = compute_train_stats(
        raw, max_clips=int(cfg.get("stats_max_clips", 40))
    )

    train_tf = build_urfd_transform(img_size, clip_length, mean, std, train=True)
    eval_tf = build_urfd_transform(img_size, clip_length, mean, std, train=False)
    ds_tr = ETRIClipDataset(
        tr, clip_length, sampling_rate, img_size,
        transform=train_tf, phase="train", num_classes=num_classes,
        seed=int(cfg.get("seed", 42)),
    )
    ds_va = ETRIClipDataset(
        va, clip_length, sampling_rate, img_size,
        transform=eval_tf, phase="val", num_classes=num_classes,
        centers_per_sample=int(cfg.get("centers_per_sample", 1)),
    )
    ds_te = ETRIClipDataset(
        te, clip_length, sampling_rate, img_size,
        transform=eval_tf, phase="test", num_classes=num_classes,
        centers_per_sample=int(cfg.get("centers_per_sample", 1)),
    )
    nw = int(cfg.get("num_workers", 2))
    dl_tr = DataLoader(ds_tr, batch_size=int(cfg["batch_size"]), shuffle=True,
                       num_workers=nw, pin_memory=False, drop_last=False,
                       persistent_workers=nw > 0)
    dl_va = DataLoader(ds_va,
                       batch_size=int(cfg.get("eval_batch_size", cfg["batch_size"])),
                       shuffle=False, num_workers=nw, pin_memory=False,
                       persistent_workers=nw > 0)
    dl_te = DataLoader(ds_te,
                       batch_size=int(cfg.get("eval_batch_size", cfg["batch_size"])),
                       shuffle=False, num_workers=nw, pin_memory=False,
                       persistent_workers=nw > 0)

    cnt = class_counts(tr, num_classes=num_classes)
    w = None
    if str(cfg.get("class_weight", "none")).lower() != "none":
        w = class_weights_from_counts(cnt, scheme=str(cfg["class_weight"]))

    model = build_baseline(
        num_classes=num_classes,
        pretrained=bool(cfg.get("pretrained", True)),
        dropout=float(cfg.get("dropout", 0.3)),
    ).to(dev)
    criterion = nn.CrossEntropyLoss(
        weight=w.to(dev) if w is not None else None,
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    def _forward_all(dl):
        model.eval()
        Lo, La = [], []
        with torch.no_grad():
            for clip, label in dl:
                clip = clip.to(dev); label = label.to(dev)
                Lo.append(model(to_model_input(clip)).cpu())
                La.append(label.cpu())
        return torch.cat(Lo), torch.cat(La)

    best_vf, best_state = -1.0, None
    for _ in range(1, epochs + 1):
        model.train()
        for clip, label in dl_tr:
            clip = clip.to(dev); label = label.to(dev)
            optimizer.zero_grad(set_to_none=True)
            criterion(model(to_model_input(clip)), label).backward()
            optimizer.step()
        logits, labels = _forward_all(dl_va)
        m_va = evaluate(logits, labels, num_classes=num_classes)
        scheduler.step()
        if m_va["f1_macro"] > best_vf:
            best_vf = m_va["f1_macro"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    logits, labels = _forward_all(dl_te)
    m_te = evaluate(logits, labels, num_classes=num_classes)

    # Free the model + cache before returning (keeps GPU memory flat across folds).
    del model
    torch.cuda.empty_cache() if dev.type == "cuda" else None

    return {
        "test_acc":   float(m_te["accuracy"]),
        "test_f1":    float(m_te["f1_macro"]),
        "test_aucpr": float(m_te["auc_pr_macro"]),
        "test_n":     int(m_te["n"]),
    }


# --- CNN+LSTM single-split training ----------------------------------------
def run_cnnlstm(tr, va, te, cache, cfg, dev, *, epochs: int = 30) -> dict:
    _set_seed(int(cfg.get("seed", 42)))
    dl_tr = DataLoader(DS(tr, cache), batch_size=16, shuffle=True)
    dl_va = DataLoader(DS(va, cache), batch_size=32)
    dl_te = DataLoader(DS(te, cache), batch_size=32)

    cnt = np.bincount([s.action_idx for s in tr], minlength=2) + 1
    w = torch.tensor(
        (cnt.sum() / cnt) / (cnt.sum() / cnt).mean(),
        dtype=torch.float32, device=dev,
    )
    model = CNNLSTM(in_dim=FEAT_DIM, n_cls=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(
        weight=w, label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )

    from sklearn.metrics import (
        accuracy_score, average_precision_score, f1_score,
    )

    def _eval(dl):
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

    best_vf, best_state = -1.0, None
    for _ in range(1, epochs + 1):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); lossf(model(x), y).backward(); opt.step()
        _, va_f, _ = _eval(dl_va)
        if va_f > best_vf:
            best_vf = va_f
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    te_a, te_f, te_ap = _eval(dl_te)
    del model
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    return {"test_acc": te_a, "test_f1": te_f, "test_aucpr": te_ap,
            "test_n": len(te)}


def main():
    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    data_root = cfg["data_root"]
    if not os.path.isabs(data_root):
        data_root = str(REPO / data_root)
    out_dir = REPO / "runs/urfd_fall/kfold"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = scan_urfd_root(data_root)
    K = 5
    folds = make_folds(samples, k=K, val_ratio=0.15, seed=42)
    print(f"[scan] {len(samples)} clips "
          f"(fall={sum(s.action_idx == 1 for s in samples)}, "
          f"adl={sum(s.action_idx == 0 for s in samples)}) | {K}-fold stratified",
          flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {dev}", flush=True)

    # CNN+LSTM feature cache shared across all folds (only computed once).
    cache_path = str(out_dir / "cnn_lstm_feat_cache.pt")
    cnn_cache = build_feature_cache(samples, cache_path, dev)

    results = {"R3D-18": [], "CNN+LSTM": []}
    for fi, (tr, va, te) in enumerate(folds):
        t_fall = sum(s.action_idx == 1 for s in te)
        print(f"\n=== fold {fi+1}/{K}  train={len(tr)} val={len(va)} "
              f"test={len(te)} (test fall={t_fall}, adl={len(te) - t_fall}) ===",
              flush=True)

        t0 = time.time()
        r = run_r3d(tr, va, te, cfg, dev, epochs=int(cfg.get("epochs", 12)))
        print(f"  R3D-18   [{time.time()-t0:5.1f}s]  "
              f"acc={r['test_acc']:.3f} f1={r['test_f1']:.3f} "
              f"aucpr={r['test_aucpr']:.3f}", flush=True)
        results["R3D-18"].append(r)

        t0 = time.time()
        c = run_cnnlstm(tr, va, te, cnn_cache, cfg, dev, epochs=30)
        print(f"  CNN+LSTM [{time.time()-t0:5.1f}s]  "
              f"acc={c['test_acc']:.3f} f1={c['test_f1']:.3f} "
              f"aucpr={c['test_aucpr']:.3f}", flush=True)
        results["CNN+LSTM"].append(c)

    # Aggregate
    summary = {}
    for model_name, runs in results.items():
        s = {}
        for key in ("test_acc", "test_f1", "test_aucpr"):
            vals = np.array([r[key] for r in runs], dtype=float)
            s[key] = {
                "mean": float(np.nanmean(vals)),
                "std":  float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "vals": [None if np.isnan(v) else float(v) for v in vals],
            }
        summary[model_name] = s

    with open(out_dir / "kfold_results.json", "w") as f:
        json.dump({"k": K, "n_clips": len(samples),
                   "per_fold": results, "summary": summary}, f, indent=2)

    # Markdown table
    md = [
        "# URFD 5-fold Cross-Validation — R3D-18 vs CNN+LSTM",
        "",
        f"- Total clips: **{len(samples)}** "
        f"(fall={sum(s.action_idx == 1 for s in samples)}, "
        f"adl={sum(s.action_idx == 0 for s in samples)})",
        f"- Folds: **{K}** stratified (sklearn StratifiedKFold, seed=42)",
        "- Per-fold val: 15% stratified slice of the non-test remainder",
        "- Models:",
        "  - **R3D-18** — Kinetics-400 pretrained, full fine-tune, 12 epoch",
        "  - **CNN+LSTM** — MobileNetV3-small (frozen, ImageNet) → 2-layer LSTM, 30 epoch",
        "",
        "## Per-fold test metrics",
        "",
        "| Fold | n_test | R3D acc | R3D f1 | R3D AUC-PR | CNN+LSTM acc | CNN+LSTM f1 | CNN+LSTM AUC-PR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i in range(K):
        r = results["R3D-18"][i]
        c = results["CNN+LSTM"][i]
        md.append(
            f"| {i+1} | {r['test_n']} "
            f"| {r['test_acc']:.3f} | {r['test_f1']:.3f} | {r['test_aucpr']:.3f} "
            f"| {c['test_acc']:.3f} | {c['test_f1']:.3f} | {c['test_aucpr']:.3f} |"
        )
    md += [
        "",
        "## Aggregated (mean ± std across folds)",
        "",
        "| Model | Accuracy | Macro-F1 | AUC-PR (fall) |",
        "|---|---|---|---|",
    ]
    for m in ("R3D-18", "CNN+LSTM"):
        s = summary[m]
        md.append(
            f"| **{m}** "
            f"| {s['test_acc']['mean']:.3f} ± {s['test_acc']['std']:.3f} "
            f"| {s['test_f1']['mean']:.3f} ± {s['test_f1']['std']:.3f} "
            f"| {s['test_aucpr']['mean']:.3f} ± {s['test_aucpr']['std']:.3f} |"
        )
    md += ["", "_Generated by `experiments/urfd_fall/kfold_eval.py`._"]
    text = "\n".join(md)
    with open(out_dir / "kfold_results.md", "w") as f:
        f.write(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
