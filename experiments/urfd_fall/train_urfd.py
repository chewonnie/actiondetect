"""URFD binary fall-vs-ADL trainer (R3D-18 baseline).

Reuses src/ infrastructure (ETRIClipDataset, transforms.ClipTransform,
model.build_baseline, metrics.evaluate) and only adds:

  * scan_urfd_root      — build ClipSample list from datasets/fall/urfd/
  * right-half crop     — URFD mp4 is [depth | RGB] side-by-side; keep RGB
  * stratified per-clip split — URFD has no documented per-subject IDs

Run from repo root:
    python experiments/urfd_fall/train_urfd.py \
        --config experiments/urfd_fall/config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence, Tuple

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# --- repo paths -------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))

from dataset import (  # noqa: E402
    ClipSample,
    ETRIClipDataset,
    _probe_video_length,
    compute_train_stats,
)
from metrics import class_counts, class_weights_from_counts, evaluate  # noqa: E402
from model import build_baseline, to_model_input  # noqa: E402
from transforms import ClipTransform  # noqa: E402


# --- URFD-specific glue -----------------------------------------------------
URFD_LABELS = {"fall": 1, "adl": 0}


def scan_urfd_root(root: str) -> List[ClipSample]:
    """Walk datasets/fall/urfd/{fall,adl}/*.mp4 → ClipSample list."""
    samples: List[ClipSample] = []
    for sub, label in URFD_LABELS.items():
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"URFD subdir missing: {d}")
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".mp4"):
                continue
            path = os.path.join(d, fn)
            if os.path.getsize(path) < 1024:
                continue
            n = _probe_video_length(path)
            if n <= 0:
                continue
            samples.append(
                ClipSample(
                    rgb_path=path,
                    csv_path=None,
                    participant="URFD",      # no documented subject ids
                    session=sub,
                    base=fn[:-4],
                    action_idx=label,
                    n_frames=n,
                )
            )
    return samples


def stratified_split(
    samples: Sequence[ClipSample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[ClipSample], List[ClipSample], List[ClipSample]]:
    rng = np.random.default_rng(seed)
    by_c: dict[int, List[ClipSample]] = defaultdict(list)
    for s in samples:
        by_c[s.action_idx].append(s)
    train: List[ClipSample] = []
    val: List[ClipSample] = []
    test: List[ClipSample] = []
    for c in sorted(by_c):
        ss = list(by_c[c])
        order = rng.permutation(len(ss))
        ss = [ss[i] for i in order]
        n = len(ss)
        n_val = max(1, int(round(n * val_ratio)))
        n_test = max(1, int(round(n * test_ratio)))
        n_train = n - n_val - n_test
        if n_train <= 0:
            raise ValueError(f"class {c}: not enough samples ({n}) for split")
        train += ss[:n_train]
        val += ss[n_train : n_train + n_val]
        test += ss[n_train + n_val :]
    return train, val, test


def _additional_targets(clip_length: int) -> dict:
    return {f"image{i}": "image" for i in range(1, clip_length)}


def build_urfd_transform(
    img_size: int,
    clip_length: int,
    mean: Sequence[float],
    std: Sequence[float],
    *,
    train: bool,
) -> ClipTransform:
    """URFD frames are [Depth(320) | RGB(320)] @ 240h. Crop right half first."""
    crop_rgb = A.Crop(x_min=320, y_min=0, x_max=640, y_max=240, p=1.0)
    if train:
        pipeline = A.Compose(
            [
                crop_rgb,
                A.SmallestMaxSize(max_size=int(img_size * 1.15)),
                A.RandomResizedCrop(
                    size=(img_size, img_size),
                    scale=(0.7, 1.0),
                    ratio=(0.85, 1.15),
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
            ],
            additional_targets=_additional_targets(clip_length),
        )
    else:
        pipeline = A.Compose(
            [
                crop_rgb,
                A.SmallestMaxSize(max_size=img_size),
                A.CenterCrop(height=img_size, width=img_size),
            ],
            additional_targets=_additional_targets(clip_length),
        )
    return ClipTransform(pipeline, mean=mean, std=std, clip_length=clip_length)


# --- training plumbing (trimmed copy of src/train.py) -----------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total = 0.0
    n = 0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    for clip, label in loader:
        clip = clip.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        x = to_model_input(clip)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, label)
        if train:
            loss.backward()
            optimizer.step()
        bs = label.size(0)
        total += float(loss.item()) * bs
        n += bs
        all_logits.append(logits.detach().cpu())
        all_labels.append(label.detach().cpu())
    return total / max(1, n), torch.cat(all_logits), torch.cat(all_labels)


def main(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))
    out_dir = REPO / cfg.get("output_dir", "runs/urfd_fall")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    data_root = cfg["data_root"]
    if not os.path.isabs(data_root):
        data_root = str(REPO / data_root)
    samples = scan_urfd_root(data_root)
    print(f"[scan] {len(samples)} clips total "
          f"(fall={sum(s.action_idx==1 for s in samples)}, "
          f"adl={sum(s.action_idx==0 for s in samples)})")

    tr, va, te = stratified_split(
        samples,
        val_ratio=float(cfg.get("val_ratio", 0.15)),
        test_ratio=float(cfg.get("test_ratio", 0.15)),
        seed=int(cfg.get("split_seed", 42)),
    )
    print(f"[split] train={len(tr)} val={len(va)} test={len(te)}")
    print(f"        train class counts: 0={sum(s.action_idx==0 for s in tr)} "
          f"1={sum(s.action_idx==1 for s in tr)}")

    split_dump = {
        "train": [asdict(s) | {"csv_path": None} for s in tr],
        "val":   [asdict(s) | {"csv_path": None} for s in va],
        "test":  [asdict(s) | {"csv_path": None} for s in te],
    }
    with open(out_dir / "split.json", "w", encoding="utf-8") as f:
        json.dump(split_dump, f, indent=2)

    num_classes = int(cfg.get("num_classes", 2))
    clip_length = int(cfg["clip_length"])
    sampling_rate = int(cfg["sampling_rate"])
    img_size = int(cfg["img_size"])

    # 1) stats on raw train clips (transform=None pipeline uses default tensor)
    print("[stats] estimating per-channel mean/std on RIGHT-half (no transform)...")
    # Build a temporary transform that does ONLY the right-half crop so stats
    # match the actual training distribution.
    crop_only = ClipTransform(
        A.Compose(
            [A.Crop(x_min=320, y_min=0, x_max=640, y_max=240, p=1.0)],
            additional_targets=_additional_targets(clip_length),
        ),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        clip_length=clip_length,
    )
    raw_train = ETRIClipDataset(
        samples=tr,
        clip_length=clip_length,
        sampling_rate=sampling_rate,
        img_size=img_size,
        transform=crop_only,
        phase="train",
        num_classes=num_classes,
        seed=int(cfg.get("seed", 42)),
    )
    mean, std = compute_train_stats(raw_train, max_clips=int(cfg.get("stats_max_clips", 40)))
    print(f"[stats] mean={mean.tolist()}  std={std.tolist()}")

    # 2) real datasets with phase-specific transforms
    train_tf = build_urfd_transform(img_size, clip_length, mean, std, train=True)
    eval_tf = build_urfd_transform(img_size, clip_length, mean, std, train=False)
    train_ds = ETRIClipDataset(
        tr, clip_length, sampling_rate, img_size,
        transform=train_tf, phase="train", num_classes=num_classes,
        seed=int(cfg.get("seed", 42)),
    )
    val_ds = ETRIClipDataset(
        va, clip_length, sampling_rate, img_size,
        transform=eval_tf, phase="val", num_classes=num_classes,
        centers_per_sample=int(cfg.get("centers_per_sample", 1)),
    )
    test_ds = ETRIClipDataset(
        te, clip_length, sampling_rate, img_size,
        transform=eval_tf, phase="test", num_classes=num_classes,
        centers_per_sample=int(cfg.get("centers_per_sample", 1)),
    )

    nw = int(cfg.get("num_workers", 2))
    dl_train = DataLoader(train_ds, batch_size=int(cfg["batch_size"]),
                          shuffle=True, num_workers=nw, pin_memory=False,
                          drop_last=False, persistent_workers=nw > 0)
    dl_val = DataLoader(val_ds, batch_size=int(cfg.get("eval_batch_size", cfg["batch_size"])),
                        shuffle=False, num_workers=nw, pin_memory=False,
                        persistent_workers=nw > 0)
    dl_test = DataLoader(test_ds, batch_size=int(cfg.get("eval_batch_size", cfg["batch_size"])),
                         shuffle=False, num_workers=nw, pin_memory=False,
                         persistent_workers=nw > 0)

    # 3) class imbalance
    counts = class_counts(tr, num_classes=num_classes)
    weight = None
    if str(cfg.get("class_weight", "none")).lower() != "none":
        weight = class_weights_from_counts(counts, scheme=str(cfg["class_weight"]))
        print(f"[imbalance] counts={counts.tolist()}  weights={weight.tolist()}")

    # 4) model
    dev_req = str(cfg.get("device", "auto")).lower()
    if dev_req == "cuda" or (dev_req == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    model = build_baseline(
        num_classes=num_classes,
        pretrained=bool(cfg.get("pretrained", True)),
        dropout=float(cfg.get("dropout", 0.3)),
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=weight.to(device) if weight is not None else None,
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["epochs"])
    )

    best_val = -1.0
    ckpt = out_dir / "best.pt"
    epoch_log: List[dict] = []

    for epoch in range(1, int(cfg["epochs"]) + 1):
        t0 = time.time()
        tr_loss, tr_logits, tr_labels = run_epoch(
            model, dl_train, criterion, optimizer, device, train=True
        )
        tr_m = evaluate(tr_logits, tr_labels, num_classes=num_classes)
        va_loss, va_logits, va_labels = run_epoch(
            model, dl_val, criterion, None, device, train=False
        )
        va_m = evaluate(va_logits, va_labels, num_classes=num_classes)
        scheduler.step()
        dt = time.time() - t0
        print(f"epoch {epoch:02d}/{cfg['epochs']} | {dt:5.1f}s "
              f"| train loss={tr_loss:.4f} acc={tr_m['accuracy']:.3f} f1={tr_m['f1_macro']:.3f} "
              f"| val loss={va_loss:.4f} acc={va_m['accuracy']:.3f} f1={va_m['f1_macro']:.3f} "
              f"aucpr={va_m['auc_pr_macro']:.3f}")
        epoch_log.append({
            "epoch": epoch, "dt": dt,
            "train_loss": tr_loss, "train": tr_m,
            "val_loss": va_loss, "val": va_m,
            "lr": optimizer.param_groups[0]["lr"],
        })
        if va_m["f1_macro"] > best_val:
            best_val = va_m["f1_macro"]
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "config": cfg, "mean": mean.tolist(), "std": std.tolist(),
                "num_classes": num_classes, "val_metrics": va_m,
            }, ckpt)

    with open(out_dir / "epoch_log.json", "w", encoding="utf-8") as f:
        json.dump(epoch_log, f, indent=2)

    # 5) final test using best checkpoint
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    te_loss, te_logits, te_labels = run_epoch(
        model, dl_test, criterion, None, device, train=False
    )
    te_m = evaluate(te_logits, te_labels, num_classes=num_classes)
    print(f"[test] loss={te_loss:.4f} acc={te_m['accuracy']:.3f} "
          f"f1={te_m['f1_macro']:.3f} aucpr={te_m['auc_pr_macro']:.3f}")
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"loss": te_loss, **te_m}, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(HERE / "config.yaml"))
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epochs (for smoke runs)")
    args = p.parse_args()
    # Allow CLI override of epochs without editing yaml.
    if args.epochs is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["epochs"] = args.epochs
        tmp = Path(args.config).with_suffix(".override.yaml")
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        main(str(tmp))
        tmp.unlink(missing_ok=True)
    else:
        main(args.config)
