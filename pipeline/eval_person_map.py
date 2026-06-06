"""US-007 fix — person mAP@0.5 from JointCSV-derived pseudo-GT boxes.

Earlier this was wrongly reported "not computable". It IS computable:
ETRI JointCSV gives 25 joints with (depthX,depthY) pixel coords +
trackingState per body. A person box = min/max over tracked joints.

CAVEAT (honestly stated): depthX/Y are in Kinect-v2 DEPTH space (~512x424)
while the RGB mp4 is 1920x1080. Kinect's exact depth->color registration
(CoordinateMapper) is NOT shipped with this data, so we apply a NAIVE
linear scale (depth_res -> color_res) ignoring the depth/color FOV offset.
The resulting mAP is therefore APPROXIMATE / UNCALIBRATED — reported with
that assumption explicit, plus median IoU and mAP@0.3 so detector quality
is visible despite calibration uncertainty. Joints sit inside the body
silhouette, so the skeleton box is padded outward by `PAD` to better
approximate the true person extent (documented assumption, not tuned).

Run: PYTHONPATH=. python -m pipeline.eval_person_map [--participants P02 P09 P14] [--max-clips N] [--stride S]
"""
from __future__ import annotations

import argparse
import csv as _csv
import glob
import json
import os

import cv2
import numpy as np

DEPTH_W, DEPTH_H = 512.0, 424.0          # Kinect v2 depth sensor resolution
PAD = 0.08                                # outward pad: joints are inside the silhouette
N_JOINTS = 25


def _frame_boxes_from_csv(csv_path: str) -> dict[int, list[tuple[float, float, float, float]]]:
    """frameNum -> list of (x1,y1,x2,y2) in DEPTH space, one box per tracked body."""
    out: dict[int, list] = {}
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as fh:
        for r in _csv.DictReader(fh):
            xs, ys = [], []
            for j in range(1, N_JOINTS + 1):
                ts = r.get("joint%d_trackingState" % j) or "0"
                try:
                    if float(ts) <= 0:
                        continue
                    x = float(r["joint%d_depthX" % j]); y = float(r["joint%d_depthY" % j])
                except (TypeError, ValueError):
                    continue
                if x <= 0 and y <= 0:
                    continue
                xs.append(x); ys.append(y)
            if len(xs) < 4:                       # too few joints -> unreliable box
                continue
            x1, x2 = min(xs), max(xs); y1, y2 = min(ys), max(ys)
            pw, ph = (x2 - x1) * PAD, (y2 - y1) * PAD
            fn = int(float(r["frameNum"]))
            out.setdefault(fn, []).append((x1 - pw, y1 - ph, x2 + pw, y2 + ph))
    return out


def _to_color(box, cw, ch):
    sx, sy = cw / DEPTH_W, ch / DEPTH_H        # NAIVE depth->color scale (FOV ignored)
    x1, y1, x2, y2 = box
    return (max(0, x1 * sx), max(0, y1 * sy), min(cw, x2 * sx), min(ch, y2 * sy))


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _ap(dets, n_gt: int) -> float:
    """VOC-style AP from (confidence, is_tp) det list at a fixed IoU threshold."""
    if n_gt == 0:
        return 0.0
    dets = sorted(dets, key=lambda d: -d[0])
    tp = np.array([d[1] for d in dets], dtype=np.float64)
    fp = 1.0 - tp
    tpc, fpc = np.cumsum(tp), np.cumsum(fp)
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def run(participants, etri_root="etri", weights="yolov8s.pt",
        max_clips=120, stride=15, conf=0.25):
    from ultralytics import YOLO
    model = YOLO(weights)
    person_id = next(k for k, v in model.names.items() if v == "person")

    clips = []
    for p in participants:
        clips += sorted(glob.glob(os.path.join(etri_root, "RGB", p, "*", "*.mp4")))
    if max_clips:
        clips = clips[:max_clips]

    iou_thr_main = 0.5
    dets05, dets03, ious = [], [], []
    n_gt = 0
    frames_scored = 0

    for mp4 in clips:
        rel = mp4.replace(os.path.join(etri_root, "RGB"), "").lstrip("/")
        csv_path = os.path.join(etri_root, "JointCSV", rel[:-4] + ".csv")
        if not os.path.isfile(csv_path):
            continue
        gt_by_frame = _frame_boxes_from_csv(csv_path)
        if not gt_by_frame:
            continue
        cap = cv2.VideoCapture(mp4)
        cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        fno = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fno += 1
            if fno % stride != 0 or fno not in gt_by_frame:
                continue
            gts = [_to_color(b, cw, ch) for b in gt_by_frame[fno]]
            n_gt += len(gts)
            frames_scored += 1
            res = model.predict(frame, conf=conf, classes=[person_id], verbose=False)[0]
            preds = []
            for b in res.boxes:
                xy = b.xyxy[0].tolist()
                preds.append((float(b.conf[0]), tuple(xy)))
            preds.sort(key=lambda d: -d[0])
            used05 = [False] * len(gts)
            used03 = [False] * len(gts)
            for c, pb in preds:
                best, bj = 0.0, -1
                for j, g in enumerate(gts):
                    v = _iou(pb, g)
                    if v > best:
                        best, bj = v, j
                if bj >= 0:
                    ious.append(best)
                tp05 = best >= 0.5 and bj >= 0 and not used05[bj]
                tp03 = best >= 0.3 and bj >= 0 and not used03[bj]
                if tp05:
                    used05[bj] = True
                if tp03:
                    used03[bj] = True
                dets05.append((c, 1.0 if tp05 else 0.0))
                dets03.append((c, 1.0 if tp03 else 0.0))
        cap.release()

    out = {
        "story": "US-007 — person mAP@0.5 (CORRECTED: computable from JointCSV)",
        "method": "GT = min/max of tracked-joint (depthX,depthY) per body, padded %.0f%%; depth->color = NAIVE linear scale (Kinect FOV/offset ignored, registration params not shipped)." % (PAD * 100),
        "assumption_caveat": "UNCALIBRATED depth->color mapping -> mAP is APPROXIMATE; median IoU and mAP@0.3 reported alongside so detector localization quality is visible despite calibration uncertainty.",
        "participants": participants,
        "clips_scored": len(clips),
        "frames_scored": frames_scored,
        "gt_boxes": n_gt,
        "person_mAP@0.5": round(_ap(dets05, n_gt), 4),
        "person_mAP@0.3": round(_ap(dets03, n_gt), 4),
        "median_IoU": round(float(np.median(ious)) if ious else 0.0, 4),
        "mean_IoU": round(float(np.mean(ious)) if ious else 0.0, 4),
        "weights": weights,
        "supersedes": "runs/baseline12/detector_eval_note.json claim 'NOT_COMPUTABLE' (that was wrong: JointCSV gives joint pixel coords).",
    }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", nargs="+", default=["P02", "P09", "P14"])
    ap.add_argument("--max-clips", type=int, default=120)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--weights", default="yolov8s.pt")
    a = ap.parse_args()
    r = run(a.participants, max_clips=a.max_clips, stride=a.stride, weights=a.weights)
    print(json.dumps(r, indent=2, ensure_ascii=False))
