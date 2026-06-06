"""Baseline training script.

Usage:
    python train.py --config config.yaml

Everything that affects results is read from the config so the same
command reproduces the same numbers. Hyperparameters live in YAML, code
stays clean, and W&B (or TensorBoard fallback) records every run.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from dataset import ETRIClipDataset, compute_train_stats, scan_etri_root
from metrics import class_counts, class_weights_from_counts, evaluate
from model import build_baseline, to_model_input
from splits import assert_no_leakage, describe_split, group_split
from transforms import build_eval_transform, build_train_transform


# ---------------------------------------------------------------------- #
# Reproducibility
# ---------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism over speed for the baseline; flip if too slow.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------- #
# Logging — W&B if available, TensorBoard otherwise.
# ---------------------------------------------------------------------- #
class RunLogger:
    def __init__(self, cfg: Dict, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.cfg = cfg
        self.wandb = None
        self.tb = None

        log_cfg = cfg.get("logging", {})
        backend = log_cfg.get("backend", "tensorboard").lower()
        if backend == "wandb":
            try:
                import wandb

                wandb.init(
                    project=log_cfg.get("project", "etri-baseline"),
                    name=log_cfg.get("run_name"),
                    config=cfg,
                    dir=str(out_dir),
                )
                self.wandb = wandb
            except Exception as exc:
                print(f"[log] wandb unavailable ({exc}); falling back to tensorboard")
                backend = "tensorboard"
        if self.wandb is None and backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(log_dir=str(out_dir / "tb"))

    def log(self, metrics: Dict, step: int) -> None:
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)
        if self.tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, step)

    def close(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()
        if self.tb is not None:
            self.tb.close()


# ---------------------------------------------------------------------- #
# Train/eval loops
# ---------------------------------------------------------------------- #
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    model.train(train)
    total_loss = 0.0
    n_seen = 0
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
        total_loss += float(loss.item()) * bs
        n_seen += bs
        all_logits.append(logits.detach().cpu())
        all_labels.append(label.detach().cpu())

    avg_loss = total_loss / max(1, n_seen)
    return avg_loss, torch.cat(all_logits), torch.cat(all_labels)


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #
def main(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))

    out_dir = Path(cfg.get("output_dir", "runs/baseline"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # 1. Scan + group split (no leakage)
    samples = scan_etri_root(cfg["data_root"], participants=cfg.get("participants"))
    if not samples:
        raise RuntimeError(f"No clips found under {cfg['data_root']}")
    split = group_split(
        samples,
        val_ratio=float(cfg.get("val_ratio", 0.15)),
        test_ratio=float(cfg.get("test_ratio", 0.15)),
        seed=int(cfg.get("split_seed", 42)),
    )
    assert_no_leakage(split)
    print(describe_split(split))
    with open(out_dir / "split.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train": split.train_participants,
                "val": split.val_participants,
                "test": split.test_participants,
            },
            f,
            indent=2,
        )

    # 2. Lock the label space. Default to the largest seen action ID across
    #    *all* discovered samples — this prevents val/test from referencing
    #    a class index out of range when train happens to miss it.
    num_classes = int(
        cfg.get("num_classes", max(s.action_idx for s in samples) + 1)
    )

    # 3. Train-only stats. We build a temporary "raw" train dataset (no
    #    transform) to estimate mean/std *before* constructing the real
    #    train transform that uses them.
    print("[stats] estimating per-channel mean/std from train clips...")
    raw_train = ETRIClipDataset(
        samples=split.train,
        clip_length=int(cfg["clip_length"]),
        sampling_rate=int(cfg["sampling_rate"]),
        img_size=int(cfg["img_size"]),
        transform=None,
        phase="train",
        num_classes=num_classes,
        seed=int(cfg.get("seed", 42)),
    )
    mean, std = compute_train_stats(
        raw_train, max_clips=int(cfg.get("stats_max_clips", 256))
    )
    print(f"[stats] mean={mean.tolist()}  std={std.tolist()}")

    # 4. Real datasets with phase-specific transforms.
    train_tf = build_train_transform(
        img_size=int(cfg["img_size"]),
        clip_length=int(cfg["clip_length"]),
        mean=mean,
        std=std,
    )
    eval_tf = build_eval_transform(
        img_size=int(cfg["img_size"]),
        clip_length=int(cfg["clip_length"]),
        mean=mean,
        std=std,
    )
    train_ds = ETRIClipDataset(
        split.train,
        clip_length=int(cfg["clip_length"]),
        sampling_rate=int(cfg["sampling_rate"]),
        img_size=int(cfg["img_size"]),
        transform=train_tf,
        phase="train",
        num_classes=num_classes,
        seed=int(cfg.get("seed", 42)),
    )
    val_ds = ETRIClipDataset(
        split.val,
        clip_length=int(cfg["clip_length"]),
        sampling_rate=int(cfg["sampling_rate"]),
        img_size=int(cfg["img_size"]),
        transform=eval_tf,
        phase="val",
        centers_per_sample=int(cfg.get("centers_per_sample", 1)),
        num_classes=num_classes,
    )
    test_ds = ETRIClipDataset(
        split.test,
        clip_length=int(cfg["clip_length"]),
        sampling_rate=int(cfg["sampling_rate"]),
        img_size=int(cfg["img_size"]),
        transform=eval_tf,
        phase="test",
        centers_per_sample=int(cfg.get("centers_per_sample", 1)),
        num_classes=num_classes,
    )

    nw = int(cfg.get("num_workers", 4))
    dl_train = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=nw > 0,
    )
    dl_val = DataLoader(
        val_ds,
        batch_size=int(cfg.get("eval_batch_size", cfg["batch_size"])),
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
    )
    dl_test = DataLoader(
        test_ds,
        batch_size=int(cfg.get("eval_batch_size", cfg["batch_size"])),
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
    )

    # 5. Class imbalance: train-set counts only.
    counts = class_counts(split.train, num_classes=num_classes)
    weight_scheme = cfg.get("class_weight", "inv_sqrt")
    if weight_scheme and str(weight_scheme).lower() != "none":
        weights = class_weights_from_counts(counts, scheme=str(weight_scheme))
        print(
            f"[imbalance] scheme={weight_scheme} | "
            f"min={counts.min()} max={counts.max()} mean={counts.mean():.1f}"
        )
    else:
        weights = None

    # 6. Model + optimizer.
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda" else "cpu"
    )
    model = build_baseline(
        num_classes=num_classes,
        pretrained=bool(cfg.get("pretrained", True)),
        dropout=float(cfg.get("dropout", 0.3)),
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None,
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

    logger = RunLogger(cfg, out_dir)
    best_val = -1.0
    ckpt_path = out_dir / "best.pt"

    for epoch in range(1, int(cfg["epochs"]) + 1):
        t0 = time.time()
        train_loss, tr_logits, tr_labels = run_epoch(
            model, dl_train, criterion, optimizer, device, train=True
        )
        train_metrics = evaluate(tr_logits, tr_labels, num_classes=num_classes)

        val_loss, val_logits, val_labels = run_epoch(
            model, dl_val, criterion, None, device, train=False
        )
        val_metrics = evaluate(val_logits, val_labels, num_classes=num_classes)

        scheduler.step()
        dt = time.time() - t0

        print(
            f"epoch {epoch:03d}/{cfg['epochs']} | {dt:5.1f}s | "
            f"train loss={train_loss:.4f} acc={train_metrics['accuracy']:.4f} "
            f"f1={train_metrics['f1_macro']:.4f} | "
            f"val loss={val_loss:.4f} acc={val_metrics['accuracy']:.4f} "
            f"f1={val_metrics['f1_macro']:.4f} aucpr={val_metrics['auc_pr_macro']:.4f}"
        )

        logger.log(
            {
                "train/loss": train_loss,
                "train/accuracy": train_metrics["accuracy"],
                "train/f1_macro": train_metrics["f1_macro"],
                "val/loss": val_loss,
                "val/accuracy": val_metrics["accuracy"],
                "val/f1_macro": val_metrics["f1_macro"],
                "val/auc_pr_macro": val_metrics["auc_pr_macro"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": dt,
            },
            step=epoch,
        )

        # Track the best by macro-F1 (more honest under imbalance).
        if val_metrics["f1_macro"] > best_val:
            best_val = val_metrics["f1_macro"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "config": cfg,
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "num_classes": num_classes,
                    "val_metrics": val_metrics,
                },
                ckpt_path,
            )

    # 7. Final test on the held-out participants using the best checkpoint.
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    test_loss, test_logits, test_labels = run_epoch(
        model, dl_test, criterion, None, device, train=False
    )
    test_metrics = evaluate(test_logits, test_labels, num_classes=num_classes)
    print(
        f"[test] loss={test_loss:.4f} acc={test_metrics['accuracy']:.4f} "
        f"f1={test_metrics['f1_macro']:.4f} aucpr={test_metrics['auc_pr_macro']:.4f}"
    )
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"loss": test_loss, **test_metrics}, f, indent=2)

    logger.log({f"test/{k}": v for k, v in test_metrics.items()}, step=int(cfg["epochs"]) + 1)
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
