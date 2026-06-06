"""
app/dashboard.py — Streamlit dashboard for elderly activity monitoring.

Structure
---------
Pure data/figure functions (testable without a server):

    timeline_figure(df)            -> go.Figure (panel 3: 24-h timeline)
    alerts_table(daily_df, cfg)    -> list[dict] (panel 5: alert list)
    draw_boxes(frame, detections)  -> np.ndarray (panel 1: bbox overlay helper)

main() wires them with st.* calls and is only executed when the script is run
directly (or via `streamlit run`).  webrtc import is lazy inside main() to
prevent camera initialisation on import.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

# Colour palette for the 12 core classes (used in the timeline figure).
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


def _read_json(path: Path) -> dict:
    """Best-effort JSON loader for dashboard metric artifacts."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _metric_value(value, ndigits: int = 3):
    """Format dashboard metric values without inventing unavailable numbers."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return round(value, ndigits)
    return value


def model_metric_tables(repo_root: Path) -> dict[str, pd.DataFrame]:
    """Build required model-performance tables from checked artifacts.

    The project currently has concrete artifacts for person detection, R3D
    action recognition, and URFD fall recognition. Unsupported model families
    are not promoted as demo tabs; unavailable values are shown as N/A instead
    of fabricated numbers.
    """
    base = repo_root / "runs" / "baseline12"
    fall = repo_root / "runs" / "urfd_fall"
    person_map = _read_json(base / "person_map.json")
    detector_note = _read_json(base / "detector_eval_note.json")
    detector_latency = _read_json(base / "detector_latency.json")
    action_metrics = _read_json(base / "test_metrics.json")
    fall_metrics = _read_json(fall / "cnn_lstm_result.json") or _read_json(fall / "test_metrics.json")

    detector_rows = [{
        "Model": "YOLOv8 COCO detector",
        "Scope": "person pseudo-GT (JointCSV); objects N/A(no object bbox GT)",
        "mAP50": _metric_value(
            person_map.get("person_mAP@0.5")
            or detector_note.get("person_mAP@0.5")
        ),
        "mAP50-95": "N/A (GT/COCO-style sweep artifact 없음)",
        "Inference Latency (ms)": _metric_value(detector_latency.get("latency_ms_mean")),
        "Reference": "runs/baseline12/person_map.json",
    }]
    action_rows = [{
        "Model": "R3D-18 action recognizer",
        "Task": "12-class action classification",
        "Accuracy": _metric_value(action_metrics.get("accuracy")),
        "Macro-F1": _metric_value(action_metrics.get("f1_macro")),
        "n": action_metrics.get("n", "N/A"),
        "Reference": "runs/baseline12/test_metrics.json",
    }]
    fall_rows = [{
        "Model": "URFD CNN+LSTM fall recognizer",
        "Task": "fall vs ADL",
        "Accuracy": _metric_value(
            fall_metrics.get("test_accuracy")
            or fall_metrics.get("test_acc")
            or fall_metrics.get("accuracy")
        ),
        "Macro-F1": _metric_value(
            fall_metrics.get("test_macro_f1")
            or fall_metrics.get("test_f1")
            or fall_metrics.get("f1_macro")
        ),
        "AUC-PR": _metric_value(
            fall_metrics.get("test_aucpr_fall")
            or fall_metrics.get("test_aucpr")
            or fall_metrics.get("auc_pr_macro")
        ),
        "Reference": "runs/urfd_fall/cnn_lstm_result.json",
    }]
    return {
        "object_detection": pd.DataFrame(detector_rows),
        "action": pd.DataFrame(action_rows),
        "fall": pd.DataFrame(fall_rows),
    }


def discover_action_demo_dirs(repo_root: Path) -> list[Path]:
    """Return action-demo directories that contain playable mp4 artifacts.

    The dashboard treats ``runs/**`` as generated evidence and only lists
    directories whose path name includes ``action`` so fall demos or raw
    training artifacts do not crowd the action-demo picker. Returned paths are
    repository-relative and sorted for stable UI/tests.
    """
    runs_dir = repo_root / "runs"
    if not runs_dir.exists():
        return []
    dirs: set[Path] = set()
    for mp4 in runs_dir.glob("**/*.mp4"):
        rel_parent = mp4.parent.relative_to(repo_root)
        if "action" in rel_parent.as_posix().lower():
            dirs.add(rel_parent)
    return sorted(dirs, key=lambda p: p.as_posix())


def action_demo_videos(repo_root: Path, rel_dir: Path | str) -> list[Path]:
    """List mp4 files for one repository-local action-demo directory.

    Raises ValueError on absolute/path-traversal input so the Streamlit picker
    cannot be abused to browse outside the repository.
    """
    rel = Path(rel_dir)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"action demo dir must be repository-relative: {rel_dir}")
    target = (repo_root / rel).resolve()
    root = repo_root.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"action demo dir escapes repository: {rel_dir}")
    if not target.exists():
        return []
    return sorted(target.glob("*.mp4"), key=lambda p: p.name)


def action_demo_manifest_rows(demo_dir: Path) -> dict[str, dict]:
    """Map demo output filename to metadata from an optional manifest.json."""
    manifest = _read_json(demo_dir / "manifest.json")
    rows: dict[str, dict] = {}
    for row in manifest.get("samples", []):
        out = row.get("output")
        if out:
            rows[Path(out).name] = row
    return rows


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
    largest_person_override: tuple[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Draw YOLO bounding boxes on a copy of frame.

    Args:
        frame:      HxWx3 numpy array (BGR or RGB).
        detections: List of (class_name, (x1, y1, x2, y2), confidence) tuples
                    as returned by YoloDetector.predict().
        largest_person_override: 옵션 (label_str, color_bgr). 제공되면 가장 면적이
                    큰 'person' 박스의 라벨/색을 대체 — 행동/낙상 라벨을 bbox
                    위에 직접 표시할 때 사용. 다른 박스는 기본 동작.

    Returns:
        New numpy array with boxes drawn.  Does NOT modify the original frame.
    """
    import cv2  # lazy import so tests on headless CI don't require display

    out = frame.copy()
    largest_idx = -1
    if largest_person_override is not None:
        max_area = 0
        for i, (cls, (x1, y1, x2, y2), _c) in enumerate(detections):
            if cls != "person":
                continue
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest_idx = i
    for i, (cls_name, (x1, y1, x2, y2), conf) in enumerate(detections):
        if i == largest_idx:
            label, color = largest_person_override
            thickness = 2
            font_scale = 0.7
            font_thick = 2
        else:
            color = (0, 255, 0)  # green
            label = f"{cls_name} {conf:.2f}"
            thickness = 2
            font_scale = 0.5
            font_thick = 1
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(out, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thick,
                    cv2.LINE_AA)
    return out


# ── 동영상 행동검출 (YOLO 박스 + CNN+LSTM 행동라벨, URFD 낙상) ────────────────

class _AVH264Writer:
    """PyAV 기반 H.264 mp4 writer. cv2.VideoWriter 와 호환 인터페이스
    (isOpened/write/release). PyAV 의 ffmpeg 빌드는 libx264 포함 →
    macOS·Windows·Linux 모두 실제 H.264 (avc1) mp4 생성 → 브라우저 재생 보장.
    """

    def __init__(self, out_path: str, fps: float, size: tuple[int, int]):
        import av
        w, h = size
        # H.264 / yuv420p 는 짝수 폭/높이 필수 → 홀수면 -1 로 조정
        if w % 2:
            w -= 1
        if h % 2:
            h -= 1
        self._w, self._h = w, h
        self._container = av.open(out_path, mode="w")
        self._stream = self._container.add_stream(
            "libx264", rate=max(int(round(fps)), 1)
        )
        self._stream.width = w
        self._stream.height = h
        self._stream.pix_fmt = "yuv420p"
        # 빠른 인코딩(데모/라이브 용)
        self._stream.options = {"preset": "ultrafast", "crf": "23"}
        self._open = True

    def isOpened(self) -> bool:
        return self._open

    def write(self, frame_bgr) -> None:
        import av
        if not self._open:
            return
        # 짝수 크기에 맞춰 자름 (PyAV from_ndarray 가 stride 까다로움)
        if frame_bgr.shape[1] != self._w or frame_bgr.shape[0] != self._h:
            frame_bgr = frame_bgr[: self._h, : self._w]
        # ascontiguous 필요한 경우 보장
        if not frame_bgr.flags["C_CONTIGUOUS"]:
            import numpy as np
            frame_bgr = np.ascontiguousarray(frame_bgr)
        frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def release(self) -> None:
        if not self._open:
            return
        try:
            for packet in self._stream.encode():   # flush
                self._container.mux(packet)
        finally:
            self._container.close()
            self._open = False


def _open_browser_friendly_writer(out_path: str, fps: float,
                                   size: tuple[int, int]):
    """H.264 mp4 writer. PyAV(libx264) 우선 → 실패 시 cv2 avc1/mp4v 폴백.

    PyAV 의 ffmpeg 빌드는 libx264 포함이라 어느 OS 에서든 실제 H.264 출력
    → 브라우저 <video> 재생 보장. PyAV 가 없거나 실패하면 cv2 로 폴백
    (Linux 등 일부 환경은 mp4v 로 떨어져 브라우저 재생 보장 안 됨).
    Returns: writer (isOpened()=True) 또는 None.
    """
    try:
        return _AVH264Writer(out_path, fps, size)
    except Exception:
        pass
    # cv2 폴백
    import cv2
    w, h = size
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass
    try:
        for fourcc in ("avc1", "H264", "mp4v"):
            vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc),
                                 max(fps, 1.0), (w, h))
            if vw.isOpened():
                return vw
            vw.release()
        return None
    finally:
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_WARNING)
        except Exception:
            pass


def _crop_frame_region(frame: np.ndarray, region: str | None) -> np.ndarray:
    """Return a display/processing crop for known composite video layouts.

    URFD demo clips used by Panel 1d are side-by-side composites
    (left=Depth, right=RGB).  Passing ``"right_half"`` removes the depth half
    so the dashboard shows and annotates RGB only.
    """
    if region is None:
        return frame
    if region != "right_half":
        raise ValueError(f"unsupported frame crop region: {region}")
    return frame[:, frame.shape[1] // 2 :]


def process_video_actions(in_path, out_path, detector, recognizer,
                          every_n: int = 1, infer_every_n: int = 8,
                          logger=None, urfd_fall=None,
                          frame_crop: str | None = None) -> dict:
    """mp4 → YOLO 박스 + 행동라벨 + 낙상 오버레이 mp4. 라이브와 동일 로직.

    recognizer: 행동 인식기 (CnnLstmRecognizer 기본, R3d18Recognizer fallback).
        push/infer 스트리밍 인터페이스. None이면 박스만.
    logger: ActivityLogger | None. 주면 행동 바뀔 때마다 CSV 기록(분석패널 연동).
    urfd_fall: UrfdFallCnnLstmRecognizer | UrfdFallRecognizer | None.
        URFD 이진 모델(CNN+LSTM 기본). 낙상 이벤트는 이 모델의 p_fall
        임계값 통과로만 기록된다.
    frame_crop: None | "right_half". URFD 데모 원본처럼 좌=Depth/우=RGB
        결합영상일 때 우측 RGB 영역만 처리/저장한다.
    Returns: dict — fall_events 는 [(sec, "urfd", p_urfd)] 이다.
    """
    import cv2

    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    if frame_crop not in (None, "right_half"):
        cap.release()
        raise ValueError(f"unsupported frame crop region: {frame_crop}")
    out_w = w - (w // 2) if frame_crop == "right_half" else w
    vw = _open_browser_friendly_writer(out_path, fps / max(every_n, 1), (out_w, h))
    if vw is None:
        cap.release()
        raise RuntimeError(
            f"VideoWriter 오픈 실패 (avc1/H264/mp4v 모두 실패) — "
            f"OpenCV/FFmpeg 코덱 미설치 가능. out={out_path}"
        )

    frames = processed = person_boxes = object_boxes = 0
    last_name = None
    actions: list = []
    fall_until_sec = 0.0                       # FALL 라벨 bbox 잔류 시간(2초)
    fall_until_score: float | None = None      # 잔류창 동안 표시할 URFD p_fall
    # ── 경량 활동 지표 (영상 기반, 보정 없음) ──────────────────────────────
    diag = (w * w + h * h) ** 0.5
    dt = max(every_n, 1) / fps                 # 처리 프레임 간 초
    still_thr = 0.01                           # 정규화 이동량 < 이면 '정지'
    presence = 0
    prev_c = None
    disp_sum, n_disp = 0.0, 0
    still_run = absent_run = max_still = max_absent = 0.0
    zone = {"bed": 0.0, "chair": 0.0, "tv": 0.0}
    fall_events: list = []                     # [(sec, "urfd", p_fall)]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = _crop_frame_region(frame, frame_crop)
        frames += 1
        # 행동 인식기 temporal contract: 매 native 프레임 push, infer 는 N 프레임마다
        if recognizer is not None:
            recognizer.push(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if frames % max(infer_every_n, 1) == 0:
                recognizer.infer()
        # URFD 낙상 모델: 동일 temporal contract (CNN+LSTM 은 T=16, R3D-18 은 clip_length×sampling_rate)
        if urfd_fall is not None:
            urfd_fall.push(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if frames % max(infer_every_n, 1) == 0:
                urfd_fall.infer()
        if (frames - 1) % every_n != 0:
            continue
        processed += 1
        tsec = frames / fps
        dets = detector.predict(frame)
        person_boxes += sum(1 for d in dets if d[0] == "person")
        object_boxes += sum(1 for d in dets if d[0] != "person")

        # person bbox 라벨에 행동 라벨을 직접 표시(없으면 기본 "person").
        # 낙상 발동 중에는 아래 굵은 FALL 오버레이가 이 위를 덮음 → priority OK.
        _override = None
        if recognizer is not None and recognizer.last_label is not None:
            _nm = recognizer.label_name(recognizer.last_label)
            if _nm not in ("?", None):
                _override = (f"{_nm} {recognizer.last_prob:.2f}",
                              (0, 165, 255))   # 오렌지
        out = draw_boxes(frame, dets, largest_person_override=_override)

        # 경량 지표: 재실/부재 · 무동작 지속 · 활동량 · 구역 체류
        persons = [d for d in dets if d[0] == "person"]
        px = None
        if persons:
            presence += 1
            absent_run = 0.0
            px = max(persons, key=lambda d: d[2])[1]      # 최고conf person box
            cx, cy = (px[0] + px[2]) / 2, (px[1] + px[3]) / 2
            if prev_c is not None:
                disp = ((cx - prev_c[0]) ** 2 + (cy - prev_c[1]) ** 2) ** 0.5 / diag
                disp_sum += disp
                n_disp += 1
                if disp < still_thr:
                    still_run += dt
                else:
                    still_run = 0.0
                max_still = max(max_still, still_run)
            prev_c = (cx, cy)
            for cn, (ox1, oy1, ox2, oy2), _ in dets:
                if cn in zone and ox1 <= cx <= ox2 and oy1 <= cy <= oy2:
                    zone[cn] += dt
        else:
            absent_run += dt
            max_absent = max(max_absent, absent_run)
            still_run = 0.0
            prev_c = None

        if recognizer is not None:
            # 행동 라벨은 person bbox 위에 (위 draw_boxes 의 override) 표시됨.
            # 여기선 전환 로깅만 처리.
            name = recognizer.label_name(recognizer.last_label)
            if name not in ("?", None) and name != last_name:
                actions.append((round(tsec, 2), name))
                if logger is not None:
                    from datetime import datetime, timedelta
                    # 영상 분석 시점의 오늘 자정 + 영상 내 경과초 → 로그 타임스탬프
                    _today = datetime.now().replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    ts = _today + timedelta(seconds=tsec)
                    logger.log(ts, name, recognizer.last_prob, (0, 0, 0, 0))
                last_name = name

        # ── 낙상: URFD 학습 모델 단독 ─────────────────────────────────
        urfd_fired = False
        if urfd_fall is not None:
            urfd_fired = urfd_fall.update(tsec)
            color = (0, 0, 255) if urfd_fired else (180, 180, 180)
            cv2.putText(out, f"Fall risk {urfd_fall.p_fall:.2f}",
                        (10, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color, 2, cv2.LINE_AA)

        if urfd_fired:
            p_urfd = float(urfd_fall.p_fall)
            fall_events.append((round(tsec, 2), "urfd", p_urfd))
            cv2.putText(out, f"FALL risk {p_urfd:.2f}", (10, 94),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255), 2, cv2.LINE_AA)
            fall_until_sec = tsec + 2.0
            fall_until_score = p_urfd

        # 낙상 잔류창: 가장 큰 person bbox 위에 빨간 박스+"FALL" 라벨
        if tsec < fall_until_sec and persons:
            (fx1, fy1, fx2, fy2) = max(
                persons, key=lambda d: (d[1][2] - d[1][0]) * (d[1][3] - d[1][1])
            )[1]
            cv2.rectangle(out, (fx1, fy1), (fx2, fy2), (0, 0, 255), 3)
            fall_label = f"FALL risk {fall_until_score:.2f}"
            cv2.putText(out, fall_label, (fx1, max(fy1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 3, cv2.LINE_AA)

        vw.write(out)
    cap.release()
    vw.release()
    return {"frames": frames, "processed": processed,
            "person_boxes": person_boxes, "object_boxes": object_boxes,
            "actions": actions, "out_path": out_path,
            "presence_pct": round(presence / processed, 3) if processed else 0.0,
            "max_absence_sec": round(max_absent, 1),
            "max_immobility_sec": round(max_still, 1),
            "activity_index": round(disp_sum / n_disp, 4) if n_disp else 0.0,
            "zone_dwell_sec": {k: round(v, 1) for k, v in zone.items()},
            "fall_events": fall_events}


def process_clip_action_tag_demo(in_path, out_path, detector, action_label: str,
                                 every_n: int = 1) -> dict:
    """Create an RGB mp4 with YOLO bbox + one clip-level action tag.

    This intentionally does NOT run rolling action inference. It is for
    validation clips whose action is evaluated as one center-clip label; the
    same label is displayed above the person bbox on every frame.
    """
    import cv2

    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    vw = _open_browser_friendly_writer(out_path, fps / max(every_n, 1), (w, h))
    if vw is None:
        cap.release()
        raise RuntimeError(f"VideoWriter 오픈 실패: {out_path}")

    frames = processed = person_boxes = object_boxes = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        if (frames - 1) % every_n != 0:
            continue
        processed += 1
        dets = detector.predict(frame)
        person_boxes += sum(1 for d in dets if d[0] == "person")
        object_boxes += sum(1 for d in dets if d[0] != "person")
        out = draw_boxes(
            frame, dets,
            largest_person_override=(f"ACTION: {action_label}", (0, 165, 255)),
        )
        vw.write(out)
    cap.release()
    vw.release()
    return {
        "frames": frames,
        "processed": processed,
        "person_boxes": person_boxes,
        "object_boxes": object_boxes,
        "out_path": out_path,
        "action_label": action_label,
    }


def process_fall_binary_demo(in_path, out_path, detector, urfd_fall,
                             every_n: int = 1, infer_every_n: int = 8,
                             frame_crop: str | None = None) -> dict:
    """Create an RGB fall-vs-ADL demo mp4 with bbox + binary URFD label."""
    import cv2

    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    if frame_crop not in (None, "right_half"):
        cap.release()
        raise ValueError(f"unsupported frame crop region: {frame_crop}")
    out_w = w - (w // 2) if frame_crop == "right_half" else w
    vw = _open_browser_friendly_writer(out_path, fps / max(every_n, 1), (out_w, h))
    if vw is None:
        cap.release()
        raise RuntimeError(f"VideoWriter 오픈 실패: {out_path}")

    frames = processed = fall_frames = adl_frames = 0
    thr = float(getattr(urfd_fall, "_prob_thr", 0.5)) if urfd_fall is not None else 0.5
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = _crop_frame_region(frame, frame_crop)
        frames += 1
        if urfd_fall is not None:
            urfd_fall.push(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if frames % max(infer_every_n, 1) == 0:
                urfd_fall.infer()
        if (frames - 1) % every_n != 0:
            continue
        processed += 1
        dets = detector.predict(frame)
        p_fall = float(getattr(urfd_fall, "p_fall", 0.0)) if urfd_fall is not None else 0.0
        is_fall = p_fall >= thr
        if is_fall:
            fall_frames += 1
        else:
            adl_frames += 1
        label = f"{'FALL' if is_fall else 'ADL'} p_fall={p_fall:.2f}"
        color = (0, 0, 255) if is_fall else (0, 200, 0)
        persons = [d for d in dets if d[0] == "person"]
        out = draw_boxes(frame, dets)
        if persons:
            x1, y1, x2, y2 = max(
                persons, key=lambda d: (d[1][2] - d[1][0]) * (d[1][3] - d[1][1])
            )[1]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            cv2.putText(out, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
        cv2.putText(out, f"URFD fall-vs-ADL: {label}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    color, 2, cv2.LINE_AA)
        vw.write(out)
    cap.release()
    vw.release()
    return {
        "frames": frames,
        "processed": processed,
        "fall_frames": fall_frames,
        "adl_frames": adl_frames,
        "out_path": out_path,
    }


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

def _save_clip_bgr(frames: list, fps: float, out_path: str) -> bool:
    """frames(list[np.ndarray BGR]) → mp4. 빈 리스트면 no-op.

    브라우저 호환 코덱(avc1 > H264 > mp4v) 으로 저장.
    Returns: True 면 정상 작성, False 면 코덱/쓰기 실패.
    """
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    vw = _open_browser_friendly_writer(out_path, fps, (w, h))
    if vw is None:
        return False
    for f in frames:
        vw.write(f)
    vw.release()
    return True


if __name__ == "__main__":
    # streamlit-webrtc and st.* calls live here so importing this module in
    # tests does NOT start a server or open a camera.
    import sys
    import time
    from collections import deque
    # `streamlit run app/dashboard.py` sets sys.path[0] to app/, hiding pipeline/.
    # Prepend the repo root so `pipeline.*` resolves regardless of cwd.
    _repo = str(Path(__file__).resolve().parent.parent)
    if _repo not in sys.path:
        sys.path.insert(0, _repo)
    _repo_root = Path(_repo)

    import streamlit as st
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

    from pipeline.aggregate import load_logs
    from pipeline.alerts import daily_class_seconds
    from pipeline.detector import YoloDetector

    cfg = _load_config()
    log_dir = cfg.get("paths", {}).get("log_dir", "./logs")
    yolo_weights = cfg.get("paths", {}).get("yolo_weights", "yolov8n.pt")
    # 단일 진실원: URFD 발화 임계치 (운영). 라이브/1b 패널/캡션/오버레이 색은
    # 전부 이 한 값을 참조 → cfg 키 1곳만 바꾸면 일관 변경. 1d 데모는 의도된
    # 별도 임계치(_urfd_demo_thr) 사용 — 데이터 분포 차이상 분리 유지.
    _URFD_THR = float(cfg.get("fall", {}).get("urfd_prob_thr", 0.7))
    _URFD_COOLDOWN = float(cfg.get("fall", {}).get("cooldown_s", 3.0))

    st.set_page_config(page_title="Activity Monitor", layout="wide")
    st.title("고령자 일상행동 모니터링 대시보드")
    st.caption(
        "현재 대시보드 행동 인식 기준 모델: **R3D-18** "
        "(`runs/baseline12/best.pt`). 모든 라이브/데모 영상은 RGB 프레임 기준입니다."
    )

    # ── Model performance evidence ────────────────────────────────────────
    st.header("모델 검증 지표(참고)")
    _metric_tabs = model_metric_tables(_repo_root)
    st.caption(
        "데모 화면 해석에 필요한 검증 결과만 정리했습니다. 산출물/GT/모델이 없는 값은 "
        "임의 수치를 만들지 않고 N/A로 표기합니다."
    )
    t_obj, t_act, t_fall = st.tabs(
        ["객체 탐지", "행동 인식", "낙상 인식"]
    )
    with t_obj:
        st.dataframe(_metric_tabs["object_detection"], use_container_width=True)
    with t_act:
        st.dataframe(_metric_tabs["action"], use_container_width=True)
    with t_fall:
        st.dataframe(_metric_tabs["fall"], use_container_width=True)

    with st.expander("시연 운영 원칙", expanded=False):
        st.markdown(
            "- **핵심 흐름**: 입력 영상/웹캠 → 객체 탐지 → 행동 인식 → 낙상 후보 확인을 끊김 없이 보여줍니다.\n"
            "- **검증된 항목만 노출**: 현재 산출물이 없는 CLIP/추적 기능은 데모 전면에서 제외했습니다.\n"
            "- **현장 리스크 관리**: 가중치와 샘플 영상은 로컬 캐시 기준으로 준비하고, 네트워크 의존 설명은 최소화합니다.\n"
            "- **스토리라인**: 문제 정의 → 접근 방식 → 결과 지표 → 실제 데모 → 한계/향후 과제 순서로 설명합니다."
        )

    # detector: Option A 설정(person/object conf + 신뢰객체) 사용.
    # yolov8*.pt 가 없으면 ultralytics 가 첫 호출 시 자동 다운로드(인터넷 필요).
    _dc = cfg.get("detector", {})
    _det_err = None
    try:
        detector = YoloDetector(
            yolo_weights,
            person_conf=_dc.get("person_conf", 0.40),
            object_conf=_dc.get("object_conf", 0.30),
            object_classes=set(_dc.get("object_classes", ["bed", "chair", "tv"])),
        )
    except Exception as e:
        detector = None
        _det_err = str(e)
        st.error(f"YOLO 로드 실패 → 라이브/영상 분석 비활성. ({_det_err})  "
                 "오프라인이면 `yolov8n.pt` 를 프로젝트 루트에 배치하세요.")
    infer_every_n = cfg.get("action_model", {}).get("infer_every_n", 16)

    # 행동 인식기: R3D-18을 대시보드 기준 모델로 사용.
    # 둘 다 동일 인터페이스(push/infer/last_label/last_prob/label_name).
    _cnn_lstm_ckpt = cfg.get("paths", {}).get("cnn_lstm_ckpt",
                                                "runs/baseline12/cnn_lstm.pt")
    _r3d_ckpt = cfg.get("paths", {}).get("r3d18_ckpt", "runs/baseline12/best.pt")
    if not Path(_cnn_lstm_ckpt).is_absolute():
        _cnn_lstm_ckpt = str(Path(__file__).parent.parent / _cnn_lstm_ckpt)
    if not Path(_r3d_ckpt).is_absolute():
        _r3d_ckpt = str(Path(__file__).parent.parent / _r3d_ckpt)
    recognizer = None
    _rec_err = None
    _rec_name = None
    try:
        from pipeline.r3d18_recognizer import R3d18Recognizer
        recognizer = R3d18Recognizer(_r3d_ckpt)
        _rec_name = "R3D-18 (127 MB)"
    except Exception as e1:
        try:
            from pipeline.cnn_lstm_infer import CnnLstmRecognizer
            recognizer = CnnLstmRecognizer(_cnn_lstm_ckpt)
            _rec_name = "CNN+LSTM (fallback)"
        except Exception as e2:
            _rec_err = f"R3D-18: {e1} | CNN+LSTM: {e2}"

    # URFD 낙상 모델 (기본: CNN+LSTM, MobileNetV3-small + 2층 LSTM, 2-class).
    # ckpt 파일명으로 자동 분기 (cnn_lstm.pt → CNN+LSTM, best.pt → R3D-18).
    # ckpt가 없으면 낙상 score/검출은 비활성화된다.
    urfd_ckpt = cfg.get("paths", {}).get("urfd_fall_ckpt",
                                          "runs/urfd_fall/cnn_lstm.pt")
    if not Path(urfd_ckpt).is_absolute():
        urfd_ckpt = str(Path(__file__).parent.parent / urfd_ckpt)
    urfd_fall = None
    _urfd_err = None
    try:
        if Path(urfd_ckpt).name.startswith("cnn_lstm"):
            from pipeline.urfd_fall_cnnlstm import UrfdFallCnnLstmRecognizer
            urfd_fall = UrfdFallCnnLstmRecognizer(
                urfd_ckpt, prob_thr=_URFD_THR, cooldown_s=_URFD_COOLDOWN,
            )
        else:
            from pipeline.urfd_fall_model import UrfdFallRecognizer
            urfd_fall = UrfdFallRecognizer(
                urfd_ckpt, prob_thr=_URFD_THR, cooldown_s=_URFD_COOLDOWN,
            )
    except Exception as e:
        _urfd_err = str(e)

    # ── Panel 1: 실시간 뷰 (라이브: YOLO 박스 + 행동 라벨) ───────────────────
    st.header("1. 실시간 모니터링")
    if recognizer is not None:
        _urfd_live_note = (
            "  좌상단에는 URFD CNN+LSTM의 낙상 가능성이 표시되고, "
            f"임계 {_URFD_THR} 이상이면 "
            "사람 박스 위에 '낙상 가능성 …' 으로 강조됩니다."
            if Path(urfd_ckpt).exists() else
            "  URFD ckpt 미배치 → 낙상 score/검출 비활성."
        )
        st.caption(f"행동 인식기: {_rec_name}. "
                   "라이브/영상은 슬라이딩윈도 근사 → 창별 라벨은 빗나갈 수 있음."
                   + _urfd_live_note)
    else:
        st.caption("행동 인식기 미로드 → YOLO 박스만 표시. "
                   "CNN+LSTM(`runs/baseline12/cnn_lstm.pt`, 1.9 MB) 또는 "
                   "R3D-18(`runs/baseline12/best.pt`, 127 MB) 중 하나를 배치하세요.")
        with st.expander("디버그: 로드 실패 사유"):
            st.code(_rec_err or "(없음)")

    # 라이브 낙상 클립 저장 폴더
    _clip_dir = Path(__file__).parent.parent / "runs" / "fall_clips"
    _clip_dir.mkdir(parents=True, exist_ok=True)

    class _Processor(VideoProcessorBase):
        # 라이브: per-processor 상태 — URFD CNN+LSTM 낙상 + 6초 클립 저장.
        # URFD 는 staged 실험실 분포 학습이라 라이브 정확도는 보장 없음.
        def __init__(self):
            import time as _t
            self._n = 0
            self._t0 = _t.time()
            self._fall_until_ts = 0.0
            self._fall_until_score: float | None = None
            # per-processor URFD 인스턴스 (영상 간 ring buffer/cooldown 상태 격리)
            self._urfd = None
            try:
                from pipeline.urfd_fall_cnnlstm import UrfdFallCnnLstmRecognizer
                if Path(urfd_ckpt).name.startswith("cnn_lstm") and Path(urfd_ckpt).exists():
                    self._urfd = UrfdFallCnnLstmRecognizer(
                        urfd_ckpt, prob_thr=_URFD_THR, cooldown_s=_URFD_COOLDOWN,
                    )
            except Exception:
                self._urfd = None
            # 6초 클립 = 3초 pre + 3초 post (가정 fps 15; 실제 fps 로 보정 후 저장)
            self._pre: deque = deque(maxlen=90)   # (ts, frame_bgr)
            self._post: list = []
            self._post_target = 0

        def recv(self, frame):
            import av
            import cv2
            import time as _t
            img = frame.to_ndarray(format="bgr24")
            ts = _t.time() - self._t0
            dets = detector.predict(img)

            self._n += 1
            # 행동 인식: 매 native 프레임 push, infer 는 N 프레임마다
            if recognizer is not None:
                recognizer.push(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if self._n % max(infer_every_n, 1) == 0:
                    recognizer.infer()
            # URFD 낙상 모델: 동일 contract (T=16). 가벼워 라이브에서도 동작.
            if self._urfd is not None:
                self._urfd.push(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if self._n % max(infer_every_n, 1) == 0:
                    self._urfd.infer()

            persons = [d for d in dets if d[0] == "person"]
            pbox = None
            if persons:
                pbox = max(
                    persons,
                    key=lambda d: (d[1][2] - d[1][0]) * (d[1][3] - d[1][1]),
                )[1]
            urfd_fired = self._urfd.update(ts) if self._urfd is not None else False
            if urfd_fired:
                self._fall_until_ts = ts + 2.0
                self._fall_until_score = float(self._urfd.p_fall)
                # 사후 3초 클립 캡처 시작 (pre 는 이미 ring buffer 에 누적됨)
                if self._post_target == 0:
                    self._post = [(t, f.copy()) for t, f in self._pre]
                    self._post_target = 45   # ≈3s @ 15fps; 실측 fps 로 출력

            # person bbox 라벨 오버라이드(행동 라벨을 bbox 위에 직접 표시).
            # 낙상 발동 중엔 아래 굵은 FALL 오버레이가 위에 덮음 → priority OK.
            _override = None
            if recognizer is not None and recognizer.last_label is not None:
                _nm = recognizer.label_name(recognizer.last_label)
                if _nm not in ("?", None):
                    _override = (f"{_nm} {recognizer.last_prob:.2f}",
                                  (0, 165, 255))    # 오렌지
            out = draw_boxes(img, dets, largest_person_override=_override)

            # URFD 낙상 가능성 상시 표시 (좌상단). 임계 이상이면 빨강, 아니면 회색.
            if self._urfd is not None:
                _color = (0, 0, 255) if self._urfd.p_fall >= _URFD_THR else (180, 180, 180)
                cv2.putText(out, f"Fall risk {self._urfd.p_fall:.2f}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            _color, 2, cv2.LINE_AA)

            # FALL 잔류창: 굵은 빨간 박스 + 낙상 가능성 값을 가장 큰 person bbox 위에
            if pbox is not None and ts < self._fall_until_ts:
                x1, y1, x2, y2 = pbox
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
                _lbl = (f"FALL risk {self._fall_until_score:.2f}"
                        if self._fall_until_score is not None else "FALL")
                cv2.putText(out, _lbl, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255), 3, cv2.LINE_AA)

            # ring buffer (annotated) 누적
            self._pre.append((ts, out.copy()))

            # 사후 캡처 & 완료 시 mp4 저장
            if self._post_target > 0:
                self._post.append((ts, out.copy()))
                self._post_target -= 1
                if self._post_target == 0 and len(self._post) >= 2:
                    t_first, t_last = self._post[0][0], self._post[-1][0]
                    fps_est = (len(self._post) - 1) / max(t_last - t_first, 0.1)
                    fname = _clip_dir / f"fall_{int(_t.time())}.mp4"
                    _save_clip_bgr([f for _, f in self._post],
                                   fps_est, str(fname))
                    self._post = []
            return av.VideoFrame.from_ndarray(out, format="bgr24")

    if detector is not None:
        webrtc_streamer(
            key="live",
            video_processor_factory=_Processor,
            media_stream_constraints={
                "video": {
                    "width":  {"ideal": 1280, "min": 640},
                    "height": {"ideal": 720,  "min": 480},
                    "frameRate": {"ideal": 30, "min": 15},
                },
                "audio": False,
            },
        )
    else:
        st.info("YOLO 미로드 → 라이브 뷰 비활성.")

    # ── Panel 1a: 저장된 낙상 클립 (라이브에서 자동 캡처) ─────────────────────
    st.header("1a. 저장된 낙상 클립 (라이브 자동 캡처)")
    _ca, _cb = st.columns([4, 1])
    _ca.caption(f"URFD 낙상 모델이 발동한 순간 전후 약 6초가 "
                 f"`{_clip_dir.relative_to(Path(__file__).parent.parent)}/`에 "
                 "mp4(H.264 우선) 로 저장됩니다. 라이브 중 새 클립은 "
                 "**새로고침** 후 표시됩니다.")
    if _cb.button("🔄 새로고침"):
        st.rerun()
    _clips = sorted(_clip_dir.glob("fall_*.mp4"), reverse=True)[:6]
    if _clips:
        _cc = st.columns(min(3, len(_clips)))
        for i, cp in enumerate(_clips):
            with _cc[i % len(_cc)]:
                _sz = cp.stat().st_size
                _bad = _sz < 1024     # < 1 KB 면 코덱 실패로 거의 빈 파일
                st.caption(f"{cp.name}  ·  {_sz / 1024:.0f} KB"
                            + ("  ⚠ 너무 작음" if _bad else ""))
                if _bad:
                    st.error("저장된 파일이 비정상적으로 작습니다 — "
                              "OpenCV 코덱 누락 가능 (`pip install opencv-python` "
                              "재설치 또는 ffmpeg 설치).")
                else:
                    st.video(str(cp))
        if st.button("🗑 모든 라이브 클립 삭제"):
            for cp in _clip_dir.glob("fall_*.mp4"):
                cp.unlink()
            st.rerun()
    else:
        st.info("아직 라이브에서 캡처된 낙상 클립이 없습니다.")

    # ── Panel 1b: 동영상 업로드 → 행동검출 (비실시간 검증) ────────────────────
    st.header("1b. 동영상 분석 (업로드 → YOLO + 행동검출 + 낙상)")
    up = st.file_uploader("동영상 업로드", type=["mp4", "avi", "mov", "mkv"])
    every_n = st.slider("프레임 간격 (1=모든 프레임)", 1, 10, 2)
    write_log = st.checkbox("결과를 logs/ 에 기록 (아래 분석패널 반영)", value=False)
    if up is not None and detector is None:
        st.error("YOLO 미로드 → 분석 불가.")
    if up is not None and detector is not None:
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
        if urfd_fall is None:
            st.caption("URFD 낙상 모델 미로드 → 낙상 검출 비활성 "
                       "(`runs/urfd_fall/cnn_lstm.pt`, 1.9 MB 배치 시 활성화)")
        else:
            st.caption(f"낙상 신호: URFD 모델 단독 "
                       f"(thr={_URFD_THR} · cooldown={_URFD_COOLDOWN}s)")
        with st.spinner("YOLO + 행동검출 + 낙상 처리 중..."):
            vr = process_video_actions(vin, vout, detector, recognizer,
                                       every_n=every_n,
                                       infer_every_n=infer_every_n,
                                       logger=vlog,
                                       urfd_fall=urfd_fall)
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

        # ── 1c: 경량 활동 지표 (영상 기반, 보정 없음) ────────────────────
        st.subheader("1c. 경량 활동 지표 (영상 기반·보정 없음·GT 없음→정확도 미산정)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("재실 비율", f"{vr['presence_pct'] * 100:.0f}%")
        m2.metric("최장 부재(초)", vr["max_absence_sec"])
        m3.metric("최장 무동작(초)", vr["max_immobility_sec"])
        m4.metric("활동량 지수", vr["activity_index"])
        st.caption(f"구역 체류(초): {vr['zone_dwell_sec']}  "
                   "— 화면영역 기준 근사(미터 아님)")
        if vr["max_immobility_sec"] >= 60:
            st.warning(f"⚠ 최장 무동작 {vr['max_immobility_sec']}초 — "
                       "장시간 무동작 상태입니다.")
        if vr["max_absence_sec"] >= 120:
            st.warning(f"⚠ 최장 부재 {vr['max_absence_sec']}초 — "
                       "시야이탈/외출 가능")

        # ── 낙상 알림: URFD 모델 단독 ────────────────────────────────
        fe = vr.get("fall_events") or []
        if fe:
            srcs: dict = {}
            for ev in fe:
                # 구버전(2-tuple) 호환: (sec, src) 또는 (sec, src, p_urfd)
                s = ev[1]
                srcs[s] = srcs.get(s, 0) + 1
            breakdown = " · ".join(f"{k}={v}" for k, v in sorted(srcs.items()))
            # 발화 순간의 시각/소스/낙상 가능성을 함께 표시
            head = []
            for ev in fe[:10]:
                t = ev[0]
                p = ev[2] if len(ev) >= 3 else None
                head.append(f"{t}s" + (f" (낙상 가능성 {p:.2f})" if p is not None else ""))
            st.warning(f"⚠ URFD 낙상 후보 {len(fe)}건  ({breakdown}) — "
                       f"@ {', '.join(head)}  "
                       "(URFD 모델 기반; 확정 아님)")
            # 상세 표: 시각·소스·URFD score
            _fe_rows = [
                {"sec": ev[0], "source": ev[1],
                 "낙상 가능성(URFD)": (f"{ev[2]:.3f}" if len(ev) >= 3 and ev[2] is not None else "-")}
                for ev in fe
            ]
            st.dataframe(pd.DataFrame(_fe_rows), use_container_width=True)


        import os as _os
        _os.unlink(vin)

    # ── Panel 1d: 행동 검출 데모 (bbox + clip-level R3D-18 action tag) ───
    st.header("1d. 행동 검출 데모 (RGB · bbox + clip-level R3D-18 행동 태그)")
    _action_demo_src_manifest = (
        _repo_root / "runs" / "baseline12" / "val_r3d_samples" / "manifest.json"
    )
    _default_action_demo_dir = Path("runs/baseline12/val_r3d_bbox_clip_action")
    (_repo_root / _default_action_demo_dir).mkdir(parents=True, exist_ok=True)
    _action_demo_dirs = discover_action_demo_dirs(_repo_root)
    if _default_action_demo_dir not in _action_demo_dirs:
        _default_outs = action_demo_videos(_repo_root, _default_action_demo_dir)
        if _default_outs:
            _action_demo_dirs.insert(0, _default_action_demo_dir)

    if _action_demo_dirs:
        st.caption(
            "`runs/**` 아래 action demo mp4 디렉터리를 자동 탐색해 재생합니다. "
            "Streamlit `st.video`는 로컬 파일 경로/URL/bytes를 받을 수 있으므로 "
            "생성된 mp4를 그대로 대시보드에 붙입니다."
        )
        _labels = [p.as_posix() for p in _action_demo_dirs]
        _default_idx = _labels.index(_default_action_demo_dir.as_posix()) if _default_action_demo_dir.as_posix() in _labels else 0
        _selected_label = st.selectbox(
            "action demo 폴더", _labels, index=_default_idx,
            help="예: runs/baseline12/val_r3d_bbox_clip_action",
        )
        _action_demo_dir = _repo_root / _selected_label
        _action_outs = action_demo_videos(_repo_root, _selected_label)
        _meta_by_name = action_demo_manifest_rows(_action_demo_dir)
        _max_show = st.slider(
            "표시할 영상 수", 1, max(len(_action_outs), 1),
            min(10, max(len(_action_outs), 1)),
        )
        st.caption(f"{_selected_label}: mp4 {_max_show}/{len(_action_outs)}개 표시")
        _cols = st.columns(2)
        for i, out in enumerate(_action_outs[:_max_show]):
            meta = _meta_by_name.get(out.name, {})
            _detail = ""
            if meta:
                _detail = (
                    f" · gt={meta.get('gt_name', '?')}"
                    f" · pred={meta.get('pred_name', meta.get('clip_action_label', '?'))}"
                    f" · 신뢰도={float(meta.get('pred_prob_center_clip', 0.0)):.3f}"
                    if meta.get("pred_prob_center_clip") is not None
                    else f" · pred={meta.get('pred_name', meta.get('clip_action_label', '?'))}"
                )
            with _cols[i % 2]:
                st.caption(f"{out.name}{_detail}")
                st.video(str(out))
        if _selected_label == _default_action_demo_dir.as_posix() and st.button("♻ 기본 행동 태그 데모 재생성"):
            for out in _action_outs:
                out.unlink()
            st.rerun()
    elif not _action_demo_src_manifest.exists():
        st.info("행동 데모 manifest가 없습니다. 먼저 validation R3D 샘플 10개를 생성해야 합니다.")
    elif detector is None or recognizer is None:
        st.info("YOLO 또는 R3D-18 행동 인식기 미로드 → 행동 데모 생성 불가.")
    else:
        if st.button("📹 행동 태그 데모 10개 생성 (bbox + clip-level R3D-18 action)"):
            _manifest = _read_json(_action_demo_src_manifest)
            _samples = _manifest.get("samples", [])[:10]
            with st.spinner("validation RGB 클립 10개에 bbox + clip-level 행동 태그 오버레이 생성 중..."):
                for i, row in enumerate(_samples, 1):
                    src = row.get("source")
                    if not src or not Path(src).exists():
                        continue
                    action_label = row.get("pred_name") or row.get("gt_name") or "unknown"
                    out = _repo_root / _default_action_demo_dir / f"{i:02d}_{Path(src).stem}.bbox_clip_action.mp4"
                    process_clip_action_tag_demo(
                        src, str(out), detector, action_label, every_n=1,
                    )
            st.success("행동 태그 데모 생성 완료.")
            st.rerun()

    # ── Panel 1e: 낙상/ADL 데모 (bbox + URFD fall-vs-ADL) ────────────────
    st.header("1e. 낙상 검출 데모 (RGB · bbox + FALL vs ADL)")
    st.caption("URFD fall/ADL 샘플의 우측 RGB 영역만 사용해 YOLO bbox와 URFD 이진 라벨(FALL/ADL)을 오버레이합니다.")
    _demo_dir = _repo_root / "runs" / "fall_demos"
    _demo_dir.mkdir(parents=True, exist_ok=True)
    _demo_srcs = [
        _repo_root / f"datasets/fall/urfd/fall/fall-{i:02d}-cam0.mp4"
        for i in (1, 2, 3)
    ] + [
        _repo_root / f"datasets/fall/urfd/adl/adl-{i:02d}-cam0.mp4"
        for i in (1, 2, 3)
    ]
    _demo_names = ["fall-01", "fall-02", "fall-03", "adl-01", "adl-02", "adl-03"]
    _demo_outs = [_demo_dir / f"{name}.fall_vs_adl.rgb.mp4" for name in _demo_names]

    # 이전 빌드(mp4v 또는 좌Depth+우RGB 풀프레임) 캐시는 브라우저 호환성/표시
    # 정책이 달라졌으므로 1회 자동 purge. 새 캐시는 우측 RGB만 저장한다.
    _demo_sentinel = _demo_dir / ".rgb_only_v1"
    if not _demo_sentinel.exists():
        for _old in list(_demo_dir.glob("*.annotated.mp4")) + list(_demo_dir.glob("*.fall_vs_adl.rgb.mp4")):
            try:
                _old.unlink()
            except OSError:
                pass
        _demo_sentinel.touch()

    _missing = [s for s in _demo_srcs if not s.exists()]
    # 데모 영상 누락 시: 첫 페이지 로드에서 한 번만 자동 다운로드 시도(~4 MB).
    if _missing and not st.session_state.get("_urfd_demo_dl_tried"):
        st.session_state._urfd_demo_dl_tried = True
        _BASE = "https://fenix.ur.edu.pl/~mkepski/ds/data"
        import urllib.error
        import urllib.request
        _fail = []
        with st.spinner(f"URFD 데모 영상 {len(_missing)}개 자동 다운로드 중 "
                          "(~4 MB · CC BY-NC-SA 4.0)..."):
            for src in _missing:
                try:
                    src.parent.mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve(f"{_BASE}/{src.name}", str(src))
                except (urllib.error.URLError, OSError) as e:
                    _fail.append(f"{src.name}: {e}")
        if _fail:
            st.error("일부 데모 다운로드 실패 — " + "; ".join(_fail)
                      + "  수동: `bash scripts/fetch_fall_demos.sh`")
        else:
            st.rerun()
        _missing = [s for s in _demo_srcs if not s.exists()]
    if _missing:
        st.warning(
            f"URFD 소스 영상 누락: {[m.name for m in _missing]} — "
            "수동: `bash scripts/fetch_fall_demos.sh` (~4 MB, CC BY-NC-SA 4.0)"
        )
    elif all(o.exists() and o.stat().st_size > 0 for o in _demo_outs):
        _dc = st.columns(3)
        for i, (name, out) in enumerate(zip(_demo_names, _demo_outs)):
            with _dc[i % 3]:
                st.caption(name)
                st.video(str(out))
        if st.button("♻ 낙상/ADL 데모 영상 재생성"):
            for o in _demo_outs:
                if o.exists():
                    o.unlink()
            st.rerun()
    elif detector is None:
        st.info("YOLO 미로드 → 데모 생성 불가.")
    else:
        if st.button("📹 낙상/ADL 데모 생성 (bbox + FALL vs ADL)"):
            # 데모는 학습된 URFD CNN+LSTM 모델을 메인 신호로 사용
            # (runs/urfd_fall/cnn_lstm.pt — git 포함, 1.9 MB).
            # URFD 테스트셋 peak p_fall ≈ 0.65 → 데모 한정 임계치 0.5 로 낮춤
            # (운영 config 의 0.7 은 라이브 카메라 보수치).
            _urfd_demo_thr = 0.5
            try:
                from pipeline.urfd_fall_cnnlstm import UrfdFallCnnLstmRecognizer
                _demo_urfd = UrfdFallCnnLstmRecognizer(
                    urfd_ckpt, prob_thr=_urfd_demo_thr, cooldown_s=3.0,
                )
                _demo_urfd_err = None
            except Exception as e:
                _demo_urfd = None
                _demo_urfd_err = str(e)
                st.error(f"URFD CNN+LSTM 로드 실패 → 데모 생성 불가. "
                         f"({_demo_urfd_err})")

            if _demo_urfd is not None:
                with st.spinner("fall/ADL 클립을 YOLO + URFD CNN+LSTM 으로 처리 중..."):
                    for src, out in zip(_demo_srcs, _demo_outs):
                    # 영상 간 ring buffer/cooldown 상태 격리
                        _demo_urfd.reset()
                        process_fall_binary_demo(
                            str(src), str(out), detector, _demo_urfd,
                            every_n=1, infer_every_n=infer_every_n,
                            frame_crop="right_half",
                        )
                st.success("생성 완료.")
                st.rerun()

    # ── Panel 5: 알림 패널 ───────────────────────────────────────────────────
    st.header("5. 알림 패널")
    all_df = load_logs(log_dir)
    daily = daily_class_seconds(all_df)
    alerts = alerts_table(daily, cfg)
    if alerts:
        st.dataframe(pd.DataFrame(alerts))
    else:
        st.success("현재 활성 알림 없음")
