"""tests/test_logger.py — Unit tests for pipeline/activity_logger.py."""

import csv
from datetime import datetime

import pytest

from pipeline.activity_logger import ActivityLogger, HEADER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_rows(path):
    """Return (header_row, data_rows) from a CSV file."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return rows[0], rows[1:]


def _count_header_lines(path):
    """Count how many rows in the file exactly match the header."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return sum(1 for row in reader if row == HEADER)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_three_events_same_day(tmp_path):
    """3 events on the same day → 1 header line + 3 data rows, correct schema."""
    logger = ActivityLogger(log_dir=str(tmp_path), subject_id="P_home")
    day = datetime(2026, 5, 17, 10, 0, 0)

    logger.log(day.replace(hour=10), "walking",  0.91, [10, 20, 100, 200])
    logger.log(day.replace(hour=11), "sitting",  0.85, [15, 25, 110, 210])
    logger.log(day.replace(hour=12), "eating",   0.78, [20, 30, 120, 220])

    csv_path = tmp_path / "2026-05-17.csv"
    assert csv_path.exists(), "Expected 2026-05-17.csv to be created"

    header, data = _read_rows(str(csv_path))

    # Schema columns are exactly right
    assert header == HEADER, f"Header mismatch: {header}"
    # Exactly 3 data rows
    assert len(data) == 3, f"Expected 3 data rows, got {len(data)}"

    # Spot-check first row content
    ts, cls, conf, bbox, subject_id = data[0]
    assert cls == "walking"
    assert float(conf) == pytest.approx(0.91, abs=1e-4)
    assert bbox == "10 20 100 200"
    assert subject_id == "P_home"


@pytest.mark.unit
def test_append_no_header_duplication(tmp_path):
    """Re-instantiate logger same day → row appended, header NOT duplicated, old rows intact."""
    logger1 = ActivityLogger(log_dir=str(tmp_path), subject_id="P_home")
    ts = datetime(2026, 5, 17, 9, 0, 0)
    logger1.log(ts, "walking", 0.90, [0, 0, 50, 50])
    logger1.log(ts.replace(hour=10), "sitting", 0.80, [0, 0, 60, 60])
    logger1.log(ts.replace(hour=11), "eating",  0.70, [0, 0, 70, 70])

    # Re-instantiate (simulates process restart mid-day)
    logger2 = ActivityLogger(log_dir=str(tmp_path), subject_id="P_home")
    logger2.log(ts.replace(hour=12), "reading", 0.65, [0, 0, 80, 80])

    csv_path = tmp_path / "2026-05-17.csv"
    header, data = _read_rows(str(csv_path))

    # Header appears exactly once
    assert _count_header_lines(str(csv_path)) == 1, "Header written more than once"
    assert header == HEADER

    # 4 data rows total; earlier rows intact
    assert len(data) == 4, f"Expected 4 data rows, got {len(data)}"
    assert data[0][1] == "walking"
    assert data[3][1] == "reading"


@pytest.mark.unit
def test_two_different_dates_two_files(tmp_path):
    """Events on two different dates produce two separate CSV files."""
    logger = ActivityLogger(log_dir=str(tmp_path), subject_id="P_home")

    logger.log(datetime(2026, 5, 17, 8, 0, 0), "walking", 0.88, [0, 0, 40, 40])
    logger.log(datetime(2026, 5, 18, 9, 0, 0), "sitting", 0.72, [0, 0, 50, 50])

    file_17 = tmp_path / "2026-05-17.csv"
    file_18 = tmp_path / "2026-05-18.csv"

    assert file_17.exists(), "2026-05-17.csv missing"
    assert file_18.exists(), "2026-05-18.csv missing"

    _, data_17 = _read_rows(str(file_17))
    _, data_18 = _read_rows(str(file_18))

    assert len(data_17) == 1 and data_17[0][1] == "walking"
    assert len(data_18) == 1 and data_18[0][1] == "sitting"
