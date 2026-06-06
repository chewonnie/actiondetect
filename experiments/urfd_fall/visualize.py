"""URFD inference visualizer.

Runs the trained R3D-18 fall classifier over selected URFD clips with a
sliding window and overlays predicted label + probability on each frame.
Concatenates multiple clips into a single MP4 for quick visual QA.

Run:
    python experiments/urfd_fall/visualize.py \
        --ckpt runs/urfd_fall/best.pt \
        --out  runs/urfd_fall/preview.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))

from model import build_baseline, to_model_input  # noqa: E402


DEFAULT_CLIPS = [
    ("fall", "fall-28-cam0.mp4"),
    ("adl",  "adl-34-cam0.mp4"),
    ("fall", "fall-05-cam0.mp4"),
    ("adl",  "adl-09-cam0.mp4"),
]


def load_model(ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = int(state["num_classes"])
    model = build_baseline(num_classes=num_classes, pretrained=False, dropout=0.0)
    model.load_state_dict(state["model"])
    model.eval().to(device)
    mean = np.asarray(state["mean"], dtype=np.float32)
    std = np.asarray(state["std"], dtype=np.float32)
    return model, mean, std, num_classes


def read_rgb_right_half(path: str) -> np.ndarray:
    """Return (T, H, W, 3) uint8 RGB from URFD mp4 (right half only)."""
    cap = cv2.VideoCapture(path)
    frames: List[np.ndarray] = []
    while True:
        ok, f = cap.read()
        if not ok or f is None:
            break
        rgb = cv2.cvtColor(f[:, 320:, :], cv2.COLOR_BGR2RGB)  # right half = RGB cam
        frames.append(rgb)
    cap.release()
    return np.stack(frames, axis=0) if frames else np.zeros((0, 240, 320, 3), np.uint8)


def make_clip_tensor(
    frames_rgb: np.ndarray,
    center_idx: int,
    clip_length: int,
    sampling_rate: int,
    img_size: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    span = clip_length * sampling_rate
    half = span // 2
    T = frames_rgb.shape[0]
    indices = [int(np.clip(center_idx + (t - clip_length // 2) * sampling_rate, 0, T - 1))
               for t in range(clip_length)]
    sel = frames_rgb[indices]  # (T,H,W,3)
    # Resize + center-crop (matches eval transform without albumentations dep)
    out = np.empty((clip_length, img_size, img_size, 3), dtype=np.float32)
    for i, f in enumerate(sel):
        h, w = f.shape[:2]
        scale = img_size / min(h, w)
        rh, rw = int(round(h * scale)), int(round(w * scale))
        r = cv2.resize(f, (rw, rh), interpolation=cv2.INTER_LINEAR)
        y0 = (rh - img_size) // 2
        x0 = (rw - img_size) // 2
        out[i] = r[y0:y0 + img_size, x0:x0 + img_size].astype(np.float32) / 255.0
    out = (out - mean) / std                       # (T,H,W,3)
    t = torch.from_numpy(out).permute(0, 3, 1, 2)  # (T,3,H,W)
    return t.unsqueeze(0)                           # (1,T,3,H,W)


def overlay(
    bgr_frame: np.ndarray,
    title: str,
    pred_label: str,
    prob: float,
    smoothed: float,
    gt_label: str,
) -> np.ndarray:
    img = bgr_frame.copy()
    h, w = img.shape[:2]
    bar_h = 70
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    # Color: red if predicted FALL, green otherwise
    color = (0, 0, 220) if pred_label == "FALL" else (0, 180, 0)
    correct = pred_label == gt_label
    gt_color = (0, 0, 220) if gt_label == "FALL" else (0, 180, 0)
    cv2.rectangle(bar, (0, 0), (w, bar_h), (32, 32, 32), -1)
    cv2.putText(bar, f"{title}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(bar, f"PRED: {pred_label}  p_fall={prob:.2f}  smoothed={smoothed:.2f}",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(bar, f"GT: {gt_label}",
                (w - 160, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, gt_color, 2, cv2.LINE_AA)
    tag = "OK" if correct else "MISS"
    tag_color = (0, 180, 0) if correct else (0, 0, 220)
    cv2.putText(bar, tag, (w - 60, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, tag_color, 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def annotate_clip(
    path: str,
    gt_label_idx: int,
    model,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    *,
    clip_length: int = 16,
    sampling_rate: int = 2,
    img_size: int = 112,
    stride: int = 4,
    smooth_alpha: float = 0.4,
) -> Tuple[np.ndarray, float]:
    """Return (annotated_frames BGR, mean p_fall) for one URFD clip."""
    frames_rgb = read_rgb_right_half(path)
    T = frames_rgb.shape[0]
    title = os.path.basename(path)
    gt_label = "FALL" if gt_label_idx == 1 else "ADL"

    # Pre-compute predictions at sliding centers
    centers = list(range(0, T, stride)) or [0]
    with torch.no_grad():
        probs: dict[int, float] = {}
        for c in centers:
            tensor = make_clip_tensor(
                frames_rgb, c, clip_length, sampling_rate, img_size, mean, std
            ).to(device)
            x = to_model_input(tensor)
            logits = model(x)
            p = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probs[c] = float(p[1])  # P(fall)

    # Interpolate per-frame p_fall from sparse centers
    cs = np.asarray(sorted(probs.keys()), dtype=np.float32)
    ps = np.asarray([probs[int(c)] for c in cs], dtype=np.float32)
    per_frame = np.interp(np.arange(T), cs, ps).astype(np.float32)

    # EMA smoother
    sm = np.empty_like(per_frame)
    s = per_frame[0]
    for i, p in enumerate(per_frame):
        s = smooth_alpha * p + (1 - smooth_alpha) * s
        sm[i] = s

    # Render
    out: List[np.ndarray] = []
    for i in range(T):
        bgr = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_LINEAR)
        pred_label = "FALL" if sm[i] >= 0.5 else "ADL"
        out.append(overlay(bgr, title, pred_label, float(per_frame[i]),
                           float(sm[i]), gt_label))
    return np.stack(out, axis=0), float(per_frame.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(REPO / "runs/urfd_fall/best.pt"))
    p.add_argument("--data", default=str(REPO / "datasets/fall/urfd"))
    p.add_argument("--out",  default=str(REPO / "runs/urfd_fall/preview.mp4"))
    p.add_argument("--fps", type=int, default=15)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    model, mean, std, num_classes = load_model(args.ckpt, device)
    print(f"[model] num_classes={num_classes}  mean={mean.tolist()}  std={std.tolist()}")

    all_frames: List[np.ndarray] = []
    for session, fn in DEFAULT_CLIPS:
        path = os.path.join(args.data, session, fn)
        if not os.path.isfile(path):
            print(f"[skip] {path} missing")
            continue
        gt = 1 if session == "fall" else 0
        ann, mean_p = annotate_clip(path, gt, model, mean, std, device)
        print(f"[clip] {fn:22s} GT={session:4s} T={len(ann):3d} mean_p_fall={mean_p:.3f}")
        all_frames.append(ann)
        # Insert a 1-second black gap between clips for readability
        if len(all_frames) and ann.size:
            gap = np.zeros_like(ann[:1]).repeat(args.fps, axis=0)
            all_frames.append(gap)

    if not all_frames:
        raise SystemExit("No clips rendered.")
    video = np.concatenate(all_frames, axis=0)
    h, w = video.shape[1], video.shape[2]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h)
    )
    for f in video:
        writer.write(f)
    writer.release()
    print(f"[save] {args.out}  ({len(video)} frames, {len(video)/args.fps:.1f}s, {w}x{h})")


if __name__ == "__main__":
    main()
