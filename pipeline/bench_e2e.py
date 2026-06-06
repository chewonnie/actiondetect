"""pipeline/bench_e2e.py — End-to-end fps benchmark (US-008, PLAN §3.6 §8.3 P3).

Measures throughput of the combined YOLO + R3D-18 pipeline over a fixed
number of frames, using synthetic random frames if no video is supplied.

GPU is shared with training; reported fps reflects that contention.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import yaml


def run_benchmark(
    video_path: str | None,
    n_frames: int = 120,
    cfg_path: str = "pipeline/config.yaml",
) -> dict:
    """Run end-to-end pipeline benchmark and return result dict.

    Args:
        video_path: Path to a video file, or None to use synthetic frames.
        n_frames:   Number of frames to process (ignored if video_path given).
        cfg_path:   Path to pipeline/config.yaml.

    Returns:
        dict with keys: frames, seconds, fps, yolo_only_fps,
                        action_model_loaded, device, note.
    """
    # ── Load config ───────────────────────────────────────────────────────────
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    yolo_weights = cfg["paths"].get("yolo_weights", "yolov8n.pt")
    r3d18_ckpt = cfg["paths"].get("r3d18_ckpt", "")
    infer_every_n = cfg["action_model"]["infer_every_n"]
    clip_length = cfg["action_model"]["clip_length"]
    sampling_rate = cfg["action_model"]["sampling_rate"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    note = "GPU shared with training job" if device == "cuda" else "CPU only"

    # ── Load YOLO detector ────────────────────────────────────────────────────
    from pipeline.detector import YoloDetector
    detector = YoloDetector(yolo_weights, device=device)

    # ── Optionally load ActionModel ───────────────────────────────────────────
    action_model_loaded = False
    action_model = None
    if r3d18_ckpt and os.path.isfile(r3d18_ckpt):
        from pipeline.action_model import ActionModel
        action_model = ActionModel(
            r3d18_ckpt,
            clip_length=clip_length,
            sampling_rate=sampling_rate,
            device=device,
        )
        action_model_loaded = True

    # ── Build frame sequence ──────────────────────────────────────────────────
    if video_path is not None:
        from pipeline.frames import FrameSource
        src_fps = cfg["stream"]["src_fps"]
        frames_iter = FrameSource.from_video(
            video_path,
            src_fps=src_fps,
            target_fps=src_fps,  # native fps so action buffer fills correctly
            size=cfg["stream"]["frame_size"],
        )
        frames = list(frames_iter)[:n_frames]
    else:
        # Synthetic: random uint8 frames, 640×640×3
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)
                  for _ in range(n_frames)]

    # ── Timed loop ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    for i, frame in enumerate(frames):
        detector.predict(frame)
        if action_model is not None:
            action_model.push(frame)
            if (i + 1) % infer_every_n == 0:
                action_model.infer()
    elapsed = time.perf_counter() - t0

    n = len(frames)
    fps = n / elapsed if elapsed > 0 else 0.0

    # YOLO-only baseline: run YOLO again on same frames (after warmup) --------
    t1 = time.perf_counter()
    for frame in frames:
        detector.predict(frame)
    yolo_elapsed = time.perf_counter() - t1
    yolo_only_fps = n / yolo_elapsed if yolo_elapsed > 0 else 0.0

    result = {
        "frames": n,
        "seconds": round(elapsed, 4),
        "fps": round(fps, 2),
        "yolo_only_fps": round(yolo_only_fps, 2),
        "action_model_loaded": action_model_loaded,
        "device": device,
        "note": note,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end pipeline fps benchmark")
    parser.add_argument("--video", default=None, help="Path to video file (omit for synthetic frames)")
    parser.add_argument("--n-frames", type=int, default=120, help="Number of frames to benchmark")
    args = parser.parse_args()
    run_benchmark(video_path=args.video, n_frames=args.n_frames)
