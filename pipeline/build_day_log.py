"""pipeline/build_day_log.py — 데모 비디오 여러 개 → 하루치 logs/<date>.csv.

각 클립을 하루 [day_start, day_end] 구간에 균등 배치하고, R3D-18 행동검출
결과를 그 클립의 벽시계 시각으로 기록 → 기존 dashboard 패널(2~5)과
rhythm(생활리듬)이 '24시간' 데모로 동작하게 한다.

비실시간 검증 흐름(제안서). 모델은 R3D-18(검증된 최강), 객체는 Option A.
GT 없음 → 정확도 미산정. 24h 는 단편 클립을 시각에 배치한 합성 타임라인.

실행:
  PYTHONPATH=. python -m pipeline.build_day_log --date 2026-05-18 \
      --glob 'etri/RGB/P02/*/A0*.mp4' --max 8
"""
from __future__ import annotations

import argparse
import glob as _glob
import os
import sys
import tempfile
from datetime import datetime, timedelta

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.activity_logger import ActivityLogger          # noqa: E402
from pipeline.detector import YoloDetector                   # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg():
    try:
        return yaml.safe_load(
            open(os.path.join(_ROOT, "pipeline", "config.yaml"))) or {}
    except FileNotFoundError:
        return {}


def build_day_log(videos, date, log_dir="logs", day_start="07:00",
                  day_end="23:00", every_n=3, subject_id="P_home") -> dict:
    """videos 를 [day_start,day_end] 에 균등배치, 행동을 그 시각으로 기록."""
    from app.dashboard import process_video_actions             # 재사용

    cfg = _cfg()
    d = cfg.get("detector", {})
    det = YoloDetector(
        cfg.get("paths", {}).get("yolo_weights", "yolov8s.pt"),
        person_conf=d.get("person_conf", 0.40),
        object_conf=d.get("object_conf", 0.30),
        object_classes=set(d.get("object_classes", ["bed", "chair", "tv"])),
    )
    rec = None
    try:
        from pipeline.r3d18_recognizer import R3d18Recognizer
        ck = cfg.get("paths", {}).get("r3d18_ckpt", "runs/baseline12/best.pt")
        rec = R3d18Recognizer(ck if os.path.isabs(ck)
                              else os.path.join(_ROOT, ck))
    except Exception as e:
        print(f"[warn] R3D-18 미로드 → 행동 없이 진행 ({e})", flush=True)
    infer_n = cfg.get("action_model", {}).get("infer_every_n", 16)

    base = datetime.fromisoformat(date + "T00:00:00")
    h0, m0 = map(int, day_start.split(":"))
    h1, m1 = map(int, day_end.split(":"))
    t0 = base + timedelta(hours=h0, minutes=m0)
    span = (timedelta(hours=h1, minutes=m1) - timedelta(hours=h0, minutes=m0))
    n = max(len(videos), 1)
    step = span / n

    logger = ActivityLogger(log_dir, subject_id)
    events = 0
    for i, v in enumerate(videos):
        clip_wall = t0 + step * i
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
            r = process_video_actions(v, tmp.name, det, rec,
                                      every_n=every_n, infer_every_n=infer_n)
        for sec, name in r["actions"]:
            logger.log(clip_wall + timedelta(seconds=sec), name, 1.0,
                       (0, 0, 0, 0))
            events += 1
        print(f"  [{i+1}/{n}] {os.path.basename(v)} @ "
              f"{clip_wall.strftime('%H:%M')} → {len(r['actions'])} events",
              flush=True)

    return {"date": date, "clips": len(videos), "events_logged": events,
            "log_path": os.path.join(log_dir, f"{date}.csv")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--glob", default="etri/RGB/P02/*/A0*.mp4")
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--max", type=int, default=8)
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--every-n", type=int, default=3)
    a = ap.parse_args()
    vids = a.videos or sorted(_glob.glob(a.glob))[: a.max]
    if not vids:
        raise SystemExit(f"비디오 없음: --videos 또는 --glob '{a.glob}'")
    print(f"{len(vids)} clips → {a.date} 24h 로그 생성", flush=True)
    import json
    print(json.dumps(build_day_log(vids, a.date, log_dir=a.log_dir,
                                   every_n=a.every_n), indent=2,
                      ensure_ascii=False))
