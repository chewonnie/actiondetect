"""
tests/test_dashboard.py — headless smoke tests for app/dashboard.py.

These tests exercise the pure panel functions WITHOUT starting a Streamlit
server or opening a camera.  The webrtc import and main() are guarded under
`if __name__ == "__main__"` in dashboard.py, so importing the module here is safe.

Full visual check: `streamlit run app/dashboard.py`
"""

import numpy as np
import pandas as pd
import pytest

# Import the pure functions from the dashboard module.
# This must NOT start a server — if it does, the guard in dashboard.py is broken.
import app.dashboard as dash


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_log_df(events: list[tuple[str, str]]) -> pd.DataFrame:
    """Build a minimal log DataFrame from (timestamp_str, class) pairs."""
    rows = [
        {
            "timestamp": pd.Timestamp(ts),
            "class": cls,
            "confidence": 0.9,
            "bbox": "0 0 100 100",
            "subject_id": "P_home",
        }
        for ts, cls in events
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def two_day_df():
    """2-day synthetic log covering eating, mobility, sedentary_screen, posture_transition."""
    events = []
    for h in [9, 12, 18]:
        events.append((f"2026-05-15 {h:02d}:00:00", "eating"))
    for m in range(10):
        events.append((f"2026-05-15 10:{m:02d}:00", "mobility"))
    for m in range(5):
        events.append((f"2026-05-15 20:{m:02d}:00", "sedentary_screen"))
    events.append(("2026-05-15 22:00:00", "posture_transition"))

    for h in [8, 13]:
        events.append((f"2026-05-16 {h:02d}:00:00", "eating"))
    for m in range(6):
        events.append((f"2026-05-16 10:{m:02d}:00", "mobility"))
    events.append(("2026-05-16 21:00:00", "posture_transition"))
    return _make_log_df(events)


@pytest.fixture
def daily_df(two_day_df):
    """Per-day per-class seconds DataFrame (mirrors alerts.daily_class_seconds output)."""
    from pipeline.alerts import daily_class_seconds
    return daily_class_seconds(two_day_df)


@pytest.fixture
def minimal_cfg():
    """Minimal config dict matching pipeline/config.yaml structure."""
    return {
        "alerts": {
            "drop_pct": 0.30,
            "baseline_days": 7,
            "consecutive_days": 2,
        }
    }


# ── Panel 3: timeline_figure ──────────────────────────────────────────────────

@pytest.mark.smoke
def test_timeline_figure_returns_figure(two_day_df):
    import plotly.graph_objects as go
    fig = dash.timeline_figure(two_day_df)
    assert isinstance(fig, go.Figure), "timeline_figure must return a go.Figure"


@pytest.mark.smoke
def test_timeline_figure_has_traces(two_day_df):
    fig = dash.timeline_figure(two_day_df)
    assert len(fig.data) >= 1, "timeline_figure must have at least 1 trace"


@pytest.mark.smoke
def test_timeline_figure_empty_df():
    """Empty DataFrame must return a Figure (possibly with 0 traces, no crash)."""
    import plotly.graph_objects as go
    fig = dash.timeline_figure(pd.DataFrame(columns=["timestamp", "class"]))
    assert isinstance(fig, go.Figure)


# ── Panel 5: alerts_table ─────────────────────────────────────────────────────

@pytest.mark.smoke
def test_alerts_table_returns_list(daily_df, minimal_cfg):
    result = dash.alerts_table(daily_df, minimal_cfg)
    assert isinstance(result, list), "alerts_table must return a list"


@pytest.mark.smoke
def test_alerts_table_only_alert_status(daily_df, minimal_cfg):
    """Every returned dict must have status == 'alert'."""
    result = dash.alerts_table(daily_df, minimal_cfg)
    for item in result:
        assert item.get("status") == "alert"


@pytest.mark.smoke
def test_alerts_table_empty_df(minimal_cfg):
    result = dash.alerts_table(pd.DataFrame(columns=["date", "class", "seconds"]), minimal_cfg)
    assert result == []


# ── Panel 1 helper: draw_boxes ────────────────────────────────────────────────

@pytest.mark.smoke
def test_draw_boxes_returns_array():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [("person", (10, 20, 200, 400), 0.91)]
    out = dash.draw_boxes(frame, detections)
    assert isinstance(out, np.ndarray), "draw_boxes must return a numpy array"
    assert out.shape == frame.shape, "output shape must match input shape"


@pytest.mark.smoke
def test_draw_boxes_does_not_modify_original():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [("eating", (0, 0, 100, 100), 0.80)]
    original_sum = frame.sum()
    out = dash.draw_boxes(frame, detections)
    assert frame.sum() == original_sum, "draw_boxes must not modify the original frame"
    # The output should have some non-zero pixels (the drawn box)
    assert out.sum() > 0


@pytest.mark.smoke
def test_draw_boxes_no_detections():
    """Zero detections — output equals input."""
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
    out = dash.draw_boxes(frame, [])
    np.testing.assert_array_equal(out, frame)


@pytest.mark.smoke
def test_crop_frame_region_right_half_keeps_rgb_side():
    """URFD fall demo composites are left=Depth/right=RGB; keep right half."""
    frame = np.zeros((4, 8, 3), dtype=np.uint8)
    frame[:, :4] = 10
    frame[:, 4:] = 200

    out = dash._crop_frame_region(frame, "right_half")

    assert out.shape == (4, 4, 3)
    assert np.all(out == 200)


@pytest.mark.smoke
def test_crop_frame_region_none_is_noop():
    frame = np.ones((4, 8, 3), dtype=np.uint8)

    out = dash._crop_frame_region(frame, None)

    assert out is frame


@pytest.mark.smoke
def test_model_metric_tables_include_required_columns():
    tables = dash.model_metric_tables(dash.Path("."))

    assert {"mAP50", "mAP50-95", "Inference Latency (ms)"} <= set(
        tables["object_detection"].columns
    )
    assert "clip" not in tables
    assert "tracking" not in tables
    assert {"Accuracy", "Macro-F1", "n"} <= set(tables["action"].columns)
    assert {"Accuracy", "Macro-F1", "AUC-PR"} <= set(tables["fall"].columns)


@pytest.mark.smoke
def test_discover_action_demo_dirs_lists_action_mp4_dirs(tmp_path):
    action_dir = tmp_path / "runs" / "baseline12" / "val_r3d_bbox_clip_action"
    action_dir.mkdir(parents=True)
    (action_dir / "demo.mp4").write_bytes(b"mp4")
    fall_dir = tmp_path / "runs" / "fall_demos"
    fall_dir.mkdir(parents=True)
    (fall_dir / "fall.mp4").write_bytes(b"mp4")

    result = dash.discover_action_demo_dirs(tmp_path)

    assert result == [dash.Path("runs/baseline12/val_r3d_bbox_clip_action")]


@pytest.mark.smoke
def test_action_demo_videos_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        dash.action_demo_videos(tmp_path, "../outside")


@pytest.mark.smoke
def test_action_demo_manifest_rows_keys_by_output_filename(tmp_path):
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    (demo_dir / "manifest.json").write_text(
        '{"samples": [{"output": "/abs/path/sample.mp4", "pred_name": "eating"}]}',
        encoding="utf-8",
    )

    rows = dash.action_demo_manifest_rows(demo_dir)

    assert rows["sample.mp4"]["pred_name"] == "eating"
