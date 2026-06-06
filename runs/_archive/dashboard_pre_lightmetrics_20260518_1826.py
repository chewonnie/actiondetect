"""
app/dashboard.py — Streamlit dashboard for elderly activity monitoring.

Structure
---------
Each of the 5 panels has a pure data/figure function (testable without a server):

    daily_summary(df)              -> dict   (panel 2: summary cards)
    timeline_figure(df)            -> go.Figure (panel 3: 24-h timeline)
    trend_figure(daily_df, drop_pct) -> go.Figure (panel 4: 7-day trend)
    alerts_table(daily_df, cfg)    -> list[dict] (panel 5: alert list)
    draw_boxes(frame, detections)  -> np.ndarray (panel 1: bbox overlay helper)

main() wires them with st.* calls and is only executed when the script is run
directly (or via `streamlit run`).  webrtc import is lazy inside main() to
prevent camera initialisation on import.

Class → card concept mapping
-----------------------------
CORE_NAMES from pipeline/class_map.py -> dashboard concept:

    eating           (0)  → meal
    drinking         (1)  → meal
    medicine         (2)  → meal       (taking medicine near meal time)
    cooking_kitchen  (3)  → meal
    hygiene_grooming (4)  → (ignored in cards)
    housework        (5)  → (ignored in cards)
    phone            (6)  → sedentary
    sedentary_screen (7)  → sedentary
    exercise         (8)  → walk       (exercise counts as movement)
    mobility         (9)  → walk
    posture_transition (10) -> sleep   (lying down / getting up is sleep proxy)
    other_social     (11) → (ignored in cards)

The four card keys are: "meal", "walk", "sedentary", "sleep".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

# ── Class → card concept mapping ──────────────────────────────────────────────
# Keys are CORE_NAMES strings; values are the 4 dashboard card buckets.
# Classes not listed here are excluded from summary cards.
_CLASS_TO_CARD: dict[str, str] = {
    "eating":             "meal",
    "drinking":           "meal",
    "medicine":           "meal",
    "cooking_kitchen":    "meal",
    "exercise":           "walk",
    "mobility":           "walk",
    "sedentary_screen":   "sedentary",
    "phone":              "sedentary",
    "posture_transition": "sleep",
}

_CARD_KEYS = ["meal", "walk", "sedentary", "sleep"]

# Colour palette for the 12 core classes (used in timeline and trend figures).
_CLASS_COLORS: dict[str, str] = {
    "eating":             "#FF6B6B",
    "drinking":           "#FF8E53",
    "medicine":           "#FFC300",
    "cooking_kitchen":    "#C0C0C0",
    "hygiene_grooming":   "#82E0AA",
    "housework":          "#5DADE2",
    "phone":              "#9B59B6",
    "sedentary_screen":   "#8E44AD",
    "exercise":           "#27AE60",
    "mobility":           "#1ABC9C",
    "posture_transition": "#2E86C1",
    "other_social":       "#F0B27A",
}


# ── Panel 2: daily summary cards ──────────────────────────────────────────────

def daily_summary(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return card data for the four summary concepts.

    Args:
        df: Raw log DataFrame for a single day (columns: timestamp, class, …).
            Each row represents 1 second of activity (1-fps assumption).

    Returns:
        Dict with keys "meal", "walk", "sedentary", "sleep".  Each value is a
        dict with:
            count   — number of detection events
            seconds — cumulative seconds (== count under 1-fps assumption)
    """
    result: dict[str, dict[str, Any]] = {k: {"count": 0, "seconds": 0} for k in _CARD_KEYS}

    if df.empty:
        return result

    for cls, card in _CLASS_TO_CARD.items():
        n = int((df["class"] == cls).sum())
        result[card]["count"] += n
        result[card]["seconds"] += n  # 1 event = 1 second

    return result


# ── Panel 3: 24-h timeline ────────────────────────────────────────────────────

def timeline_figure(df: pd.DataFrame) -> go.Figure:
    """Horizontal coloured band chart over 24 h, one band per detection event.

    Args:
        df: Raw log DataFrame (columns: timestamp [datetime64], class, …).

    Returns:
        Plotly Figure with one bar trace per core class present in the data.
        X-axis spans 0–24 h (hour of day).  Each event becomes a thin band of
        width 1/3600 (one second expressed in hours).
    """
    fig = go.Figure()

    if df.empty:
        fig.update_layout(title="24-h Activity Timeline (no data)", xaxis_title="Hour of day")
        return fig

    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60 + df["timestamp"].dt.second / 3600

    classes_present = df["class"].unique()
    for cls in sorted(classes_present):
        sub = df[df["class"] == cls]
        color = _CLASS_COLORS.get(cls, "#AAAAAA")
        fig.add_trace(go.Bar(
            name=cls,
            x=sub["hour"],
            y=[1] * len(sub),
            width=1 / 60,  # ~1 min wide for visibility
            marker_color=color,
            hovertemplate=f"{cls}<br>%{{x:.2f}}h<extra></extra>",
        ))

    fig.update_layout(
        title="24-h Activity Timeline",
        xaxis_title="Hour of day",
        yaxis=dict(visible=False),
        barmode="overlay",
        xaxis=dict(range=[0, 24], dtick=2),
        legend_title="Class",
    )
    return fig


# ── Panel 4: 7-day trend ──────────────────────────────────────────────────────

def trend_figure(daily_df: pd.DataFrame, drop_pct: float = 0.30) -> go.Figure:
    """Line chart of per-class activity seconds over the last 7 days.

    Args:
        daily_df: DataFrame with columns [date, class, seconds] as produced by
                  pipeline.alerts.daily_class_seconds.
        drop_pct: Alert threshold from config alerts.drop_pct (default 0.30).
                  A horizontal reference line is drawn at (1 - drop_pct) × mean.

    Returns:
        Plotly Figure with one line per class plus a threshold reference line.
    """
    fig = go.Figure()

    if daily_df.empty:
        fig.update_layout(title="7-Day Activity Trend (no data)")
        return fig

    classes_present = daily_df["class"].unique()
    for cls in sorted(classes_present):
        sub = daily_df[daily_df["class"] == cls].sort_values("date")
        color = _CLASS_COLORS.get(cls, "#AAAAAA")
        fig.add_trace(go.Scatter(
            x=sub["date"].astype(str),
            y=sub["seconds"],
            mode="lines+markers",
            name=cls,
            line=dict(color=color),
        ))

    # Threshold reference line: (1 - drop_pct) * overall mean seconds
    overall_mean = daily_df["seconds"].mean()
    threshold = overall_mean * (1.0 - drop_pct)
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Alert threshold (−{int(drop_pct*100)}%)",
        annotation_position="top left",
    )

    fig.update_layout(
        title="7-Day Activity Trend",
        xaxis_title="Date",
        yaxis_title="Seconds",
        legend_title="Class",
    )
    return fig


# ── Panel 5: alert table ──────────────────────────────────────────────────────

def alerts_table(daily_df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Compute and return the list of active alerts.

    Args:
        daily_df: DataFrame with columns [date, class, seconds].
        cfg:      Full pipeline config dict (loaded from pipeline/config.yaml).
                  Uses cfg["alerts"]["drop_pct"], cfg["alerts"]["baseline_days"],
                  cfg["alerts"]["consecutive_days"].

    Returns:
        List of alert dicts from pipeline.alerts.compute_alerts.
        Only dicts with status == "alert" are returned (not "ok"/"down"/etc.).
    """
    from pipeline.alerts import compute_alerts  # lazy import; pipeline may not be installed

    alert_cfg = cfg.get("alerts", {})
    all_results = compute_alerts(
        daily_df,
        drop_pct=alert_cfg.get("drop_pct", 0.30),
        baseline_days=alert_cfg.get("baseline_days", 7),
        consecutive_days=alert_cfg.get("consecutive_days", 2),
    )
    return [r for r in all_results if r.get("status") == "alert"]


# ── Panel 1 helper: bbox overlay ─────────────────────────────────────────────

def draw_boxes(
    frame: np.ndarray,
    detections: list[tuple[str, tuple[int, int, int, int], float]],
) -> np.ndarray:
    """Draw YOLO bounding boxes on a copy of frame.

    Args:
        frame:      HxWx3 numpy array (BGR or RGB).
        detections: List of (class_name, (x1, y1, x2, y2), confidence) tuples
                    as returned by YoloDetector.predict().

    Returns:
        New numpy array with boxes drawn.  Does NOT modify the original frame.
    """
    import cv2  # lazy import so tests on headless CI don't require display

    out = frame.copy()
    for cls_name, (x1, y1, x2, y2), conf in detections:
        color = (0, 255, 0)  # green
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


# ── 동영상 행동검출 (YOLO 박스 + R3D-18 행동라벨) ─────────────────────────────

def process_video_actions(in_path, out_path, detector, recognizer,
                          every_n: int = 1, infer_every_n: int = 8,
                          logger=None) -> dict:
    """mp4 → YOLO 박스 + R3D-18 행동라벨 오버레이 mp4. 라이브와 동일 로직.

    recognizer: R3d18Recognizer (push/infer 스트리밍). None이면 박스만.
    logger: ActivityLogger | None. 주면 행동 바뀔 때마다 CSV 기록(분석패널 연동).
    Returns: {frames, processed, person_boxes, object_boxes, actions, out_path}
    """
    import cv2

    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                         fps / max(every_n, 1), (w, h))

    frames = processed = person_boxes = object_boxes = 0
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
        processed += 1
        dets = detector.predict(frame)
        person_boxes += sum(1 for d in dets if d[0] == "person")
        object_boxes += sum(1 for d in dets if d[0] != "person")
        out = draw_boxes(frame, dets)

        if recognizer is not None:
            name = recognizer.label_name(recognizer.last_label)
            cv2.putText(out, f"ACTION: {name} {recognizer.last_prob:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255), 2, cv2.LINE_AA)
            if name not in ("?", None) and name != last_name:
                sec = frames / fps
                actions.append((round(sec, 2), name))
                if logger is not None:
                    from datetime import datetime, timedelta
                    ts = datetime(2025, 1, 1) + timedelta(seconds=sec)
                    logger.log(ts, name, recognizer.last_prob, (0, 0, 0, 0))
                last_name = name
        vw.write(out)
    cap.release()
    vw.release()
    return {"frames": frames, "processed": processed,
            "person_boxes": person_boxes, "object_boxes": object_boxes,
            "actions": actions, "out_path": out_path}


# ── Load config helper ────────────────────────────────────────────────────────

def _load_config(config_path: str | None = None) -> dict:
    """Load pipeline/config.yaml.  Returns empty dict on failure."""
    if config_path is None:
        # Resolve relative to this file's location (app/ -> detect/ -> pipeline/)
        here = Path(__file__).parent
        config_path = str(here.parent / "pipeline" / "config.yaml")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


# ── main() — only runs under `streamlit run` or direct python invocation ──────

if __name__ == "__main__":
    # streamlit-webrtc and st.* calls live here so importing this module in
    # tests does NOT start a server or open a camera.
    import streamlit as st
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

    from pipeline.aggregate import load_logs
    from pipeline.alerts import daily_class_seconds
    from pipeline.detector import YoloDetector

    cfg = _load_config()
    log_dir = cfg.get("paths", {}).get("log_dir", "./logs")
    yolo_weights = cfg.get("paths", {}).get("yolo_weights", "yolov8n.pt")
    drop_pct = cfg.get("alerts", {}).get("drop_pct", 0.30)

    st.set_page_config(page_title="Activity Monitor", layout="wide")
    st.title("고령자 일상행동 모니터링 대시보드")

    # ── Sidebar: date selector ────────────────────────────────────────────────
    all_logs = sorted(Path(log_dir).glob("*.csv")) if Path(log_dir).exists() else []
    available_dates = [f.stem for f in all_logs]  # "YYYY-MM-DD"
    selected_date = st.sidebar.selectbox("날짜 선택", available_dates or ["(no logs)"])

    # detector: Option A 설정(person/object conf + 신뢰객체) 사용
    _dc = cfg.get("detector", {})
    detector = YoloDetector(
        yolo_weights,
        person_conf=_dc.get("person_conf", 0.40),
        object_conf=_dc.get("object_conf", 0.30),
        object_classes=set(_dc.get("object_classes", ["bed", "chair", "tv"])),
    )
    infer_every_n = cfg.get("action_model", {}).get("infer_every_n", 16)

    # R3D-18 행동검출기 (검증된 최강 모델, ckpt 없으면 graceful: 박스만)
    ckpt = cfg.get("paths", {}).get("r3d18_ckpt", "runs/baseline12/best.pt")
    if not Path(ckpt).is_absolute():
        ckpt = str(Path(__file__).parent.parent / ckpt)
    recognizer = None
    _rec_err = None
    try:
        from pipeline.r3d18_recognizer import R3d18Recognizer
        recognizer = R3d18Recognizer(ckpt)
    except Exception as e:
        _rec_err = str(e)

    # ── Panel 1: 실시간 뷰 (라이브: YOLO 박스 + R3D-18 행동) ──────────────────
    st.header("1. 실시간 뷰 (webrtc + YOLO + R3D-18 행동)")
    st.caption("R3D-18 오프라인 ETRI test acc 0.693 (eval.py 중심클립 기준). "
               "라이브/영상은 슬라이딩윈도 근사 → 창별 라벨은 빗나갈 수 있음.")
    if recognizer is None:
        st.warning(f"R3D-18 미로드 → 박스만 표시. ({_rec_err})")

    class _Processor(VideoProcessorBase):
        def __init__(self):
            self._n = 0

        def recv(self, frame):
            import av
            import cv2
            img = frame.to_ndarray(format="bgr24")
            dets = detector.predict(img)
            img = draw_boxes(img, dets)
            if recognizer is not None:
                self._n += 1
                recognizer.push(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if self._n % max(infer_every_n, 1) == 0:
                    recognizer.infer()
                nm = recognizer.label_name(recognizer.last_label)
                cv2.putText(img, f"ACTION: {nm} {recognizer.last_prob:.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 255), 2, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(key="live", video_processor_factory=_Processor)

    # ── Panel 1b: 동영상 업로드 → 행동검출 (비실시간 검증) ────────────────────
    st.header("1b. 동영상 분석 (업로드 → YOLO + R3D-18 행동검출)")
    up = st.file_uploader("동영상 업로드", type=["mp4", "avi", "mov", "mkv"])
    every_n = st.slider("프레임 간격 (1=모든 프레임)", 1, 10, 2)
    write_log = st.checkbox("결과를 logs/ 에 기록 (아래 분석패널 반영)", value=False)
    if up is not None:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tin:
            tin.write(up.read())
            vin = tin.name
        vout = vin + ".annotated.mp4"
        vlog = None
        if write_log:
            from pipeline.activity_logger import ActivityLogger
            vlog = ActivityLogger(log_dir,
                                  cfg.get("subject_id", "P_home"))
        with st.spinner("YOLO + R3D-18 처리 중..."):
            vr = process_video_actions(vin, vout, detector, recognizer,
                                       every_n=every_n,
                                       infer_every_n=infer_every_n,
                                       logger=vlog)
        cva, cvb, cvc, cvd = st.columns(4)
        cva.metric("처리 프레임", vr["processed"])
        cvb.metric("person 박스", vr["person_boxes"])
        cvc.metric("object 박스", vr["object_boxes"])
        cvd.metric("행동 전환", len(vr["actions"]))
        if Path(vout).exists() and Path(vout).stat().st_size > 0:
            st.video(vout)
        if vr["actions"]:
            st.dataframe(pd.DataFrame(vr["actions"],
                                      columns=["sec", "action"]))
        import os as _os
        _os.unlink(vin)

    # ── Panel 2: 일별 요약 카드 ──────────────────────────────────────────────
    st.header("2. 일별 요약 카드")
    if available_dates and selected_date in available_dates:
        day_df = load_logs(log_dir, dates=[selected_date])
        summary = daily_summary(day_df)
        cols = st.columns(4)
        labels = {"meal": "식사", "walk": "보행/운동", "sedentary": "좌식", "sleep": "수면(자세전환)"}
        for col, key in zip(cols, _CARD_KEYS):
            col.metric(
                label=labels[key],
                value=f"{summary[key]['seconds'] // 60} 분",
                delta=f"{summary[key]['count']} 이벤트",
            )
    else:
        st.info("로그 파일이 없습니다.")

    # ── Panel 3: 24h 타임라인 ────────────────────────────────────────────────
    st.header("3. 24h 활동 타임라인")
    if available_dates and selected_date in available_dates:
        st.plotly_chart(timeline_figure(day_df), use_container_width=True)
    else:
        st.plotly_chart(timeline_figure(pd.DataFrame()), use_container_width=True)

    # ── Panel 4: 7일 추이 ────────────────────────────────────────────────────
    st.header("4. 7일 활동 추이")
    all_df = load_logs(log_dir)
    daily = daily_class_seconds(all_df)
    st.plotly_chart(trend_figure(daily, drop_pct), use_container_width=True)

    # ── Panel 5: 알림 패널 ───────────────────────────────────────────────────
    st.header("5. 알림 패널")
    alerts = alerts_table(daily, cfg)
    if alerts:
        st.dataframe(pd.DataFrame(alerts))
    else:
        st.success("현재 활성 알림 없음")
