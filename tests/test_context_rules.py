"""Unit tests for pipeline/context_rules.py."""

import pytest
from pipeline.context_rules import _iou, context_tags


# ---------------------------------------------------------------------------
# _iou sanity
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_iou_identical():
    """Identical boxes have IoU == 1.0."""
    box = (10, 10, 50, 50)
    assert _iou(box, box) == 1.0


@pytest.mark.unit
def test_iou_disjoint():
    """Non-overlapping boxes have IoU == 0.0."""
    a = (0, 0, 10, 10)
    b = (20, 20, 30, 30)
    assert _iou(a, b) == 0.0


# ---------------------------------------------------------------------------
# context_tags — flag OFF (no-op)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_flag_off_returns_empty():
    """With enable=False, context_tags always returns [] regardless of input."""
    detections = [
        ("person", (0, 0, 100, 200), 0.95),
        ("cup",    (10, 10, 60, 80), 0.80),
        ("phone",  (5, 5, 30, 40),   0.70),
    ]
    cfg = {"enable": False, "iou_thr": 0.01, "center_dist_thr": 500}
    result = context_tags(detections, cfg)
    assert result == [], f"Expected [], got {result}"


# ---------------------------------------------------------------------------
# context_tags — flag ON
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_overlapping_cup_tagged():
    """A cup box overlapping the person box produces 'near cup' tag."""
    person_box = (0, 0, 100, 200)
    cup_box    = (50, 50, 150, 150)   # overlaps person box
    detections = [
        ("person", person_box, 0.95),
        ("cup",    cup_box,    0.80),
    ]
    cfg = {"enable": True, "iou_thr": 0.01, "center_dist_thr": 0}
    result = context_tags(detections, cfg)
    assert "near cup" in result, f"Expected 'near cup' in {result}"


@pytest.mark.unit
def test_far_tv_not_tagged():
    """A tv box far from the person box is not tagged."""
    person_box = (0, 0, 100, 100)
    tv_box     = (1000, 1000, 1200, 1200)   # far away, no overlap
    detections = [
        ("person", person_box, 0.95),
        ("tv",     tv_box,     0.85),
    ]
    cfg = {"enable": True, "iou_thr": 0.01, "center_dist_thr": 50}
    result = context_tags(detections, cfg)
    assert "near tv" not in result, f"Did not expect 'near tv' in {result}"


@pytest.mark.unit
def test_no_person_returns_empty():
    """If there is no person detection, no tags are produced."""
    detections = [
        ("cup", (10, 10, 60, 80), 0.80),
    ]
    cfg = {"enable": True, "iou_thr": 0.01, "center_dist_thr": 500}
    result = context_tags(detections, cfg)
    assert result == []


@pytest.mark.unit
def test_close_by_center_distance_tagged():
    """An object close in center-distance (but not overlapping) is still tagged."""
    person_box = (0, 0, 100, 100)     # center: (50, 50)
    phone_box  = (110, 0, 210, 100)   # center: (160, 50), dist = 110 px; no overlap
    detections = [
        ("person",     person_box, 0.95),
        ("cell phone", phone_box,  0.75),
    ]
    cfg = {"enable": True, "iou_thr": 0.0, "center_dist_thr": 120}
    result = context_tags(detections, cfg)
    assert "near cell phone" in result, f"Expected 'near cell phone' in {result}"
