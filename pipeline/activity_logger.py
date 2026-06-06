"""pipeline/activity_logger.py — Append activity events to logs/YYYY-MM-DD.csv.

Schema (fixed header): timestamp,class,confidence,bbox,subject_id
  - timestamp  : ISO 8601 string (e.g. "2026-05-17T14:30:00")
  - class      : core class name string
  - confidence : float rounded to 4 decimal places
  - bbox       : space-joined ints "x1 y1 x2 y2" (pixel coords, top-left + bottom-right)
  - subject_id : constant string from config (single-home deployment; no re-ID)

One file per calendar day (local time derived from event timestamp).
Header is written only when the file is newly created.
Subsequent calls on the same day append rows; the file is never truncated.
"""

import csv
import os
from datetime import datetime

HEADER = ["timestamp", "class", "confidence", "bbox", "subject_id"]


class ActivityLogger:
    """Append-only CSV logger, one file per day.

    Args:
        log_dir:    Directory where CSV files are written (created if absent).
        subject_id: Constant identifier for the monitored person, e.g. "P_home".
    """

    def __init__(self, log_dir: str, subject_id: str) -> None:
        self.log_dir = log_dir
        self.subject_id = subject_id
        os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def log(self, ts: datetime, cls: str, conf: float, bbox) -> None:
        """Append one event row to the day's CSV file.

        Args:
            ts:   Event datetime (local time used to pick the day's file).
            cls:  Core class name string.
            conf: Confidence score (float).
            bbox: Bounding box — any four-element iterable of ints [x1, y1, x2, y2].
        """
        date_str = ts.date().isoformat()          # "YYYY-MM-DD"
        csv_path = os.path.join(self.log_dir, f"{date_str}.csv")

        is_new_file = not os.path.exists(csv_path)

        # "a" mode never truncates; creates the file on first write.
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new_file:
                writer.writerow(HEADER)
            bbox_str = " ".join(str(int(v)) for v in bbox)
            writer.writerow([
                ts.isoformat(),
                cls,
                round(float(conf), 4),
                bbox_str,
                self.subject_id,
            ])
