"""Standalone evaluation against a saved checkpoint.

Re-uses the *exact* split + transforms saved in the checkpoint so a
reported number is reproducible from the artifact alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import ETRIClipDataset, scan_etri_root
from metrics import confusion, evaluate
from model import build_baseline, to_model_input
from splits import group_split
from train import run_epoch
from transforms import build_eval_transform


def main(checkpoint: str, split: str = "test", config_override: str | None = None) -> None:
    state = torch.load(checkpoint, map_location="cpu")
    cfg = yaml.safe_load(open(config_override)) if config_override else state["config"]
    num_classes = int(state["num_classes"])
    mean = np.asarray(state["mean"], dtype=np.float32)
    std = np.asarray(state["std"], dtype=np.float32)

    samples = scan_etri_root(cfg["data_root"], participants=cfg.get("participants"))
    sp = group_split(
        samples,
        val_ratio=float(cfg.get("val_ratio", 0.15)),
        test_ratio=float(cfg.get("test_ratio", 0.15)),
        seed=int(cfg.get("split_seed", 42)),
    )
    chosen = {"train": sp.train, "val": sp.val, "test": sp.test}[split]

    tf = build_eval_transform(
        img_size=int(cfg["img_size"]),
        clip_length=int(cfg["clip_length"]),
        mean=mean,
        std=std,
    )
    ds = ETRIClipDataset(
        chosen,
        clip_length=int(cfg["clip_length"]),
        sampling_rate=int(cfg["sampling_rate"]),
        img_size=int(cfg["img_size"]),
        transform=tf,
        phase="val",
        centers_per_sample=int(cfg.get("centers_per_sample", 1)),
        num_classes=num_classes,
    )
    dl = DataLoader(
        ds,
        batch_size=int(cfg.get("eval_batch_size", cfg["batch_size"])),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_baseline(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(state["model"])
    criterion = torch.nn.CrossEntropyLoss()

    loss, logits, labels = run_epoch(model, dl, criterion, None, device, train=False)
    metrics = evaluate(logits, labels, num_classes=num_classes)
    print(json.dumps({"split": split, "loss": loss, **metrics}, indent=2))

    cm = confusion(logits, labels, num_classes=num_classes)
    out_path = Path(checkpoint).with_suffix(f".{split}.confusion.npy")
    np.save(out_path, cm)
    print(f"[eval] confusion matrix -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--config", default=None, help="Optional config override")
    args = parser.parse_args()
    main(args.checkpoint, args.split, args.config)
