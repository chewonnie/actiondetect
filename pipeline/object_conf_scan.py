"""객체 confidence 분포 스캔 — object_conf 임계값을 추측 말고 데이터로 정하기 위함.

object mAP는 ETRI에 객체 bbox GT가 없어 검증 불가. 그래서 최소한
"COCO-pretrained YOLOv8이 ETRI 가정영상에서 타깃 객체를 어떤 confidence로
잡는가 / 임계별로 몇 %가 살아남는가"를 측정해 임계 선택 근거로 삼는다.

실행: PYTHONPATH=. python -m pipeline.object_conf_scan [--participants ...] [--max-clips N] [--stride S]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import cv2
import numpy as np

TARGETS = ["person", "cup", "bottle", "book", "cell phone", "tv",
           "bed", "chair", "dining table", "remote"]
THRESHOLDS = [0.3, 0.5, 0.6, 0.8]


def run(participants, etri_root="etri", weights="yolov8s.pt",
        max_clips=90, stride=15):
    from ultralytics import YOLO
    model = YOLO(weights)
    confs = defaultdict(list)            # class_name -> [conf,...]
    frames = 0

    clips = []
    for p in participants:
        clips += sorted(glob.glob(os.path.join(etri_root, "RGB", p, "*", "*.mp4")))
    clips = clips[:max_clips]

    for mp4 in clips:
        cap = cv2.VideoCapture(mp4)
        fno = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            fno += 1
            if fno % stride:
                continue
            frames += 1
            res = model.predict(fr, conf=0.05, verbose=False)[0]   # 낮게 잡아 분포 확보
            for b in res.boxes:
                name = model.names[int(b.cls[0])]
                if name in TARGETS:
                    confs[name].append(float(b.conf[0]))
        cap.release()

    table = {}
    for name in TARGETS:
        cs = np.asarray(confs.get(name, []), dtype=np.float32)
        if cs.size == 0:
            table[name] = {"detections@0.05": 0, "note": "거의 안 잡힘"}
            continue
        row = {
            "detections@0.05": int(cs.size),
            "median_conf": round(float(np.median(cs)), 3),
            "p90_conf": round(float(np.percentile(cs, 90)), 3),
        }
        for t in THRESHOLDS:
            row["survive@%.1f" % t] = round(float((cs >= t).mean()), 3)
        table[name] = row

    out = {
        "purpose": "object_conf 임계 선택 근거 (object mAP는 GT부재로 검증불가 -> 분포/생존율만)",
        "participants": participants,
        "clips": len(clips), "frames": frames,
        "weights": weights,
        "per_class": table,
        "reading": "survive@0.8 가 낮은 객체(약통 대용 cup/book/cell phone 등)는 0.8 임계 시 거의 사라짐. "
                   "person은 높게 유지(코어 파이프라인). object_conf는 이 표를 보고 객체별로 정할 것.",
    }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", nargs="+", default=["P02", "P09", "P14"])
    ap.add_argument("--max-clips", type=int, default=90)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--weights", default="yolov8s.pt")
    a = ap.parse_args()
    print(json.dumps(run(a.participants, max_clips=a.max_clips,
                         stride=a.stride, weights=a.weights),
                      indent=2, ensure_ascii=False))
