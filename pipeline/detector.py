"""pipeline/detector.py — YOLOv8 object detector wrapper.

Usage:
    detector = YoloDetector("yolov8n.pt", conf=0.25)
    detections = detector.predict(frame_bgr)
    # returns list of (class_name, (x1,y1,x2,y2), confidence)
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

# Objects the pipeline cares about (COCO class names).
COCO_TARGETS: set[str] = {
    "person",
    "cup",
    "bottle",
    "book",
    "cell phone",
    "tv",
    "bed",
    "chair",
    "dining table",
    "remote",
}

# Option A (data-driven): COCO->ETRI domain shift makes small objects
# unreliable (cup/book/phone/bottle/remote median conf ~0.10). Only large
# furniture with usable confidence mass is trusted as object context.
# Per runs/baseline12/object_conf_scan.json: bed median 0.49, chair p90 0.93,
# tv p90 0.72 are usable; 'dining table' median 0.12 ~ small-object tier so
# it is EXCLUDED here (override via pipeline/config.yaml detector.object_classes).
RELIABLE_OBJECTS: set[str] = {"bed", "chair", "tv"}


class YoloDetector:
    """Thin wrapper around an Ultralytics YOLO model for single-frame inference."""

    def __init__(
        self,
        weights: str,
        conf: float = 0.25,
        device: str | None = None,
        person_conf: float | None = None,
        object_conf: float | None = None,
        object_classes: "set[str] | None" = None,
    ):
        """Load a YOLO model from *weights* (.pt or .onnx).

        Args:
            weights: Path to model weights or a model name (e.g. 'yolov8n.pt').
            conf:    Default min confidence (used when the per-kind ones are None).
            device:  Inference device, e.g. 'cpu', 'cuda:0'. None = auto.
            person_conf: Min confidence for the 'person' class. None -> conf.
            object_conf: Min confidence for all non-person classes. None -> conf.
            object_classes: Allowed non-person classes (Option A: only trusted
                            large furniture). None -> RELIABLE_OBJECTS.

        Split thresholds exist because COCO->ETRI domain shift makes everyday
        objects (cup/book/phone/remote) low-confidence while 'person' stays
        high (see runs/baseline12/object_conf_scan.json). A single high
        threshold would wipe object context; a single low one would add
        person false positives. Defaults from pipeline/config.yaml `detector`.
        """
        self.conf = conf
        self.person_conf = conf if person_conf is None else person_conf
        self.object_conf = conf if object_conf is None else object_conf
        self.object_classes = (
            RELIABLE_OBJECTS if object_classes is None else set(object_classes)
        )
        self.device = device
        self._model = YOLO(weights)

    def predict(
        self,
        frame: "np.ndarray",
        targets_only: bool = False,
    ) -> list[tuple[str, tuple[int, int, int, int], float]]:
        """Run inference on a single HxWx3 numpy frame.

        Args:
            frame:        BGR or RGB numpy array of shape (H, W, 3).
            targets_only: If True, return only detections whose class name is in
                          COCO_TARGETS.

        Returns:
            List of (class_name, (x1, y1, x2, y2), confidence) tuples,
            filtered to detections >= self.conf.
        """
        # Run the model at the lower of the two thresholds, then post-filter
        # per class so 'person' and objects get their own cutoffs.
        floor = min(self.person_conf, self.object_conf)
        kwargs: dict = {"conf": floor, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device

        results = self._model(frame, **kwargs)

        detections: list[tuple[str, tuple[int, int, int, int], float]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf_val = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name: str = self._model.names[cls_id]
                is_person = cls_name == "person"
                thr = self.person_conf if is_person else self.object_conf
                if conf_val < thr:
                    continue
                # Option A: drop unreliable objects; person always passes.
                if not is_person and cls_name not in self.object_classes:
                    continue
                if targets_only and cls_name not in COCO_TARGETS:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                detections.append((cls_name, (x1, y1, x2, y2), conf_val))

        return detections
