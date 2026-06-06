"""app/run.py — 분석 없는 단순 뷰어 (완전 독립).

동영상(mp4)을 업로드하면 YOLOv8로 **사람 + 신뢰 객체** bbox를 검출해
박스를 그린 주석 영상을 출력한다. R3D-18 ckpt(best.pt) 가 있으면 R3D-18
행동 라벨도 오버레이(없으면 박스만 — graceful). 집계/알림/타임라인/로그는
없음. app/dashboard.py 에 의존하지 않음 (독립).

객체 정책은 사전 합의(Option A): COCO 유지 + 큰 가구만 신뢰
(pipeline/config.yaml detector.object_classes = bed/chair/tv).
person_conf / object_conf 도 config 값 사용.

각 박스에 confidence 를 강조 표기하고, 화면 상단에 모델 평가 기준치
(person mAP@0.5, runs/baseline12/person_map.json) 를 캡션으로 보여준다.
※ F1/정밀도/재현율은 업로드 영상에 정답(GT)이 없어 계산 불가 — 표시 안 함.

실행:  streamlit run app/run.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import cv2
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:                       # repo 루트 보장 (pipeline import)
    sys.path.insert(0, _ROOT)

from pipeline.detector import YoloDetector  # noqa: E402

_CFG = os.path.join(_ROOT, "pipeline", "config.yaml")
_PERSON_MAP = os.path.join(_ROOT, "runs", "baseline12", "person_map.json")


def _load_config(path: str = _CFG) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _load_eval_ref(path: str = _PERSON_MAP) -> dict:
    """person 검출 평가 기준치(저장된 실측). 없으면 빈 dict."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return {k: d[k] for k in ("person_mAP@0.5", "person_mAP@0.3",
                                  "median_IoU") if k in d}
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return {}


def _draw(frame, detections):
    """사람=초록/객체=파랑. confidence 를 배경박스 위에 강조 표기 (원본 불변)."""
    out = frame.copy()
    for cls_name, (x1, y1, x2, y2), conf in detections:
        color = (0, 255, 0) if cls_name == "person" else (255, 128, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ly = max(y1 - 8, th + 4)
        cv2.rectangle(out, (x1, ly - th - 4), (x1 + tw + 4, ly + 2),
                      color, -1)                       # confidence 강조 배경
        cv2.putText(out, label, (x1 + 2, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def process_video(in_path: str, out_path: str, detector: YoloDetector,
                   every_n: int = 1, recognizer=None,
                   infer_every_n: int = 8) -> dict:
    """mp4 → 사람+신뢰객체 bbox 그린 mp4.

    recognizer: CnnLstmRecognizer | None. 주면 CNN-LSTM 행동 라벨도 오버레이.
    Returns: {frames, processed, person_boxes, object_boxes,
              person_conf_avg, object_conf_avg, actions, out_path}
    """
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                         fps / max(every_n, 1), (w, h))

    frames = processed = person_boxes = object_boxes = 0
    psum = osum = 0.0
    last_name = None
    actions: list = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        # R3D-18 temporal contract: 매 native 프레임 push, infer 는 N 프레임마다
        if recognizer is not None:
            recognizer.push(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if frames % max(infer_every_n, 1) == 0:
                recognizer.infer()
        if (frames - 1) % every_n != 0:
            continue
        dets = detector.predict(frame)            # person + 신뢰객체 (Option A)
        for cname, _, cf in dets:
            if cname == "person":
                person_boxes += 1
                psum += cf
            else:
                object_boxes += 1
                osum += cf
        out = _draw(frame, dets)
        if recognizer is not None:
            nm = recognizer.label_name(recognizer.last_label)
            cv2.putText(out, f"ACTION: {nm} {recognizer.last_prob:.2f}",
                        (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255), 2, cv2.LINE_AA)
            if nm not in ("?", None) and nm != last_name:
                actions.append((round(frames / fps, 2), nm))
                last_name = nm
        vw.write(out)
        processed += 1
    cap.release()
    vw.release()
    return {"frames": frames, "processed": processed,
            "person_boxes": person_boxes, "object_boxes": object_boxes,
            "person_conf_avg": round(psum / person_boxes, 3) if person_boxes else 0.0,
            "object_conf_avg": round(osum / object_boxes, 3) if object_boxes else 0.0,
            "actions": actions, "out_path": out_path}


def _build_detector(cfg: dict) -> YoloDetector:
    d = cfg.get("detector", {})
    return YoloDetector(
        cfg.get("paths", {}).get("yolo_weights", "yolov8s.pt"),
        person_conf=d.get("person_conf", 0.40),
        object_conf=d.get("object_conf", 0.30),
        object_classes=set(d.get("object_classes", ["bed", "chair", "tv"])),
    )


def main():  # pragma: no cover  (Streamlit UI)
    import streamlit as st

    st.set_page_config(page_title="BBox 뷰어", layout="wide")
    st.title("사람 + 물체 BBox 검출 뷰어")

    cfg = _load_config()
    obj = cfg.get("detector", {}).get("object_classes", ["bed", "chair", "tv"])
    st.caption(f"검출 객체: person + {obj}  ·  박스에 confidence 표기")

    ref = _load_eval_ref()
    if ref:
        st.caption(
            "모델 평가 기준치(ETRI test, joint-GT): "
            f"person mAP@0.5={ref.get('person_mAP@0.5')} · "
            f"mAP@0.3={ref.get('person_mAP@0.3')} · "
            f"median IoU={ref.get('median_IoU')}  "
            "— F1/정밀도/재현율은 업로드 영상에 GT가 없어 계산 불가(표시 안 함)."
        )

    # R3D-18 행동 라벨 (ckpt 있으면 표시, 없으면 박스만 — graceful)
    ckpt = cfg.get("paths", {}).get("r3d18_ckpt",
                                    "runs/baseline12/best.pt")
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(_ROOT, ckpt)
    recognizer = None
    try:
        from pipeline.r3d18_recognizer import R3d18Recognizer
        recognizer = R3d18Recognizer(ckpt)
        st.caption("R3D-18 행동 라벨 ON · 오프라인 ETRI test acc 0.693 "
                   "(eval.py 중심클립 프로토콜). 라이브/영상은 슬라이딩윈도 "
                   "근사라 창별 라벨은 빗나갈 수 있음 — 검증치는 오프라인 기준.")
    except Exception as e:
        st.info(f"R3D-18 미로드 → 박스만 표시 ({e})")
    infer_every_n = cfg.get("action_model", {}).get("infer_every_n", 16)

    up = st.file_uploader("동영상 업로드", type=["mp4", "avi", "mov", "mkv"])
    every_n = st.slider("프레임 간격 (1=모든 프레임)", 1, 10, 1)

    if up is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tin:
            tin.write(up.read())
            in_path = tin.name
        out_path = in_path + ".annotated.mp4"
        with st.spinner("검출 중..."):
            r = process_video(in_path, out_path, _build_detector(cfg),
                               every_n=every_n, recognizer=recognizer,
                               infer_every_n=infer_every_n)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("처리 프레임", r["processed"])
        c2.metric("person 박스", r["person_boxes"],
                  delta=f"avg conf {r['person_conf_avg']}")
        c3.metric("object 박스", r["object_boxes"],
                  delta=f"avg conf {r['object_conf_avg']}")
        c4.metric("행동 전환", len(r["actions"]))
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            st.video(out_path)
            with open(out_path, "rb") as f:
                st.download_button("주석 영상 다운로드", f,
                                   "bbox.mp4", "video/mp4")
        else:
            st.error("출력 영상 생성 실패 (입력 코덱 확인).")
        if r["actions"]:
            import pandas as pd
            st.dataframe(pd.DataFrame(r["actions"], columns=["sec", "action"]))
        os.unlink(in_path)


if __name__ == "__main__":
    main()
