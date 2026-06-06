"""
context_rules.py — optional dashboard-context enrichment.

Given YOLO detections for a frame, produce human-readable context tags
such as "near phone" or "near cup" by checking proximity between the
`person` bounding box and object bounding boxes.

ARCHITECTURAL NOTE:
  This module is enrichment ONLY for dashboard display.
  It MUST NEVER be used as the action label source.
  Action labels come exclusively from the R3D-18 classifier (pipeline/action_model.py).
  See PLAN.md §1 and ADR §6 for the rationale.

API:
  context_tags(detections, cfg) -> list[str]

  detections: list of (cls_name: str, bbox: (x1, y1, x2, y2), conf: float)
  cfg: dict with keys:
    "enable"          bool  — if False, return [] immediately (no-op)
    "iou_thr"         float — IoU threshold; tag if IoU >= this value
    "center_dist_thr" float — pixel distance threshold; tag if center distance <= this value
"""


def _iou(a, b):
    """Compute Intersection over Union for two boxes (x1,y1,x2,y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def _center_dist(a, b):
    """Euclidean distance between centers of two boxes (x1,y1,x2,y2)."""
    ax = (a[0] + a[2]) / 2
    ay = (a[1] + a[3]) / 2
    bx = (b[0] + b[2]) / 2
    by = (b[1] + b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def context_tags(detections, cfg):
    """
    Return a list of context tag strings for a single frame.

    If cfg["enable"] is False, returns [] immediately without examining
    detections — the pipeline result is completely unchanged.

    Args:
        detections: list of (cls_name, bbox, conf) tuples where
                    bbox is (x1, y1, x2, y2) in pixels.
        cfg:        dict with "enable", "iou_thr", "center_dist_thr".

    Returns:
        list[str]: e.g. ["near cup", "near phone"], or [] if disabled.
    """
    if not cfg.get("enable", False):
        return []

    iou_thr = cfg["iou_thr"]
    dist_thr = cfg["center_dist_thr"]

    # Find the first person box (highest-confidence person, or just first).
    person_box = None
    for cls_name, bbox, conf in detections:
        if cls_name == "person":
            person_box = bbox
            break

    if person_box is None:
        return []

    tags = []
    for cls_name, bbox, conf in detections:
        if cls_name == "person":
            continue
        near = (
            _iou(person_box, bbox) >= iou_thr
            or _center_dist(person_box, bbox) <= dist_thr
        )
        if near:
            tags.append(f"near {cls_name}")

    return tags
