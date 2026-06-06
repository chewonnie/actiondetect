"""Smoke tests for pipeline/detector.py.

Uses ultralytics' bundled sample image (bus.jpg) which contains persons,
so we can assert at least one 'person' detection without relying on a network
call beyond the one-time yolov8n.pt auto-download.
"""

import pytest
from ultralytics.utils import ASSETS

from pipeline.detector import YoloDetector, COCO_TARGETS


@pytest.mark.smoke
def test_predict_returns_list():
    """predict() must return a list (possibly empty)."""
    detector = YoloDetector("yolov8n.pt", conf=0.25)
    import numpy as np
    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    result = detector.predict(blank)
    assert isinstance(result, list)


@pytest.mark.smoke
def test_predict_detection_structure():
    """Each detection must be (str, (int,int,int,int), float)."""
    detector = YoloDetector("yolov8n.pt", conf=0.25)
    import cv2
    img_path = ASSETS / "bus.jpg"
    frame = cv2.imread(str(img_path))
    assert frame is not None, f"Could not load bundled sample image: {img_path}"

    detections = detector.predict(frame)
    assert isinstance(detections, list)

    for cls_name, bbox, conf in detections:
        assert isinstance(cls_name, str)
        assert isinstance(bbox, tuple) and len(bbox) == 4
        assert all(isinstance(v, int) for v in bbox)
        assert isinstance(conf, float)


@pytest.mark.smoke
def test_predict_person_detected_on_bus_image():
    """bus.jpg contains persons — at least one 'person' detection expected."""
    detector = YoloDetector("yolov8n.pt", conf=0.25)
    import cv2
    img_path = ASSETS / "bus.jpg"
    frame = cv2.imread(str(img_path))
    assert frame is not None, f"Could not load bundled sample image: {img_path}"

    detections = detector.predict(frame)
    class_names = [cls for cls, _, _ in detections]
    assert "person" in class_names, (
        f"Expected at least one 'person' detection in bus.jpg. Got: {class_names}"
    )


@pytest.mark.smoke
def test_targets_only_filters_correctly():
    """targets_only=True should exclude classes not in COCO_TARGETS."""
    detector = YoloDetector("yolov8n.pt", conf=0.25)
    import cv2
    img_path = ASSETS / "bus.jpg"
    frame = cv2.imread(str(img_path))
    assert frame is not None

    detections = detector.predict(frame, targets_only=True)
    for cls_name, _, _ in detections:
        assert cls_name in COCO_TARGETS, (
            f"'{cls_name}' is not in COCO_TARGETS but appeared in targets_only output"
        )


@pytest.mark.smoke
def test_coco_targets_contains_expected_classes():
    """COCO_TARGETS must contain the objects specified in PLAN §3.5."""
    required = {"person", "cup", "bottle", "book", "cell phone", "tv",
                "bed", "chair", "dining table", "remote"}
    assert required <= COCO_TARGETS
