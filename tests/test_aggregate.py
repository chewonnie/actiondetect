"""Unit tests for pipeline/aggregate.py — synthetic in-memory data only."""

import pandas as pd
import pytest

from pipeline.aggregate import summarize


def _make_df(events: list[tuple[str, str]]) -> pd.DataFrame:
    """Build a minimal log DataFrame from (timestamp_str, class) pairs."""
    rows = [
        {
            "timestamp": pd.Timestamp(ts),
            "class": cls,
            "confidence": 0.9,
            "bbox": "0,0,100,100",
            "subject_id": "P_home",
        }
        for ts, cls in events
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def one_day_df():
    """
    Synthetic 1-day log with KNOWN event counts:
        eating:  5 events  (all at 09:xx)
        walking: 12 events (8 at 14:xx, 4 at 15:xx)

    Expected 1D totals:
        eating  count=5,  cumulative_seconds=5,  peak_hour=9
        walking count=12, cumulative_seconds=12, peak_hour=14
    """
    events = []
    # 5 eating events in the 09:xx hour
    for m in range(5):
        events.append((f"2024-01-15 09:{m:02d}:00", "eating"))
    # 8 walking events in the 14:xx hour
    for m in range(8):
        events.append((f"2024-01-15 14:{m:02d}:00", "walking"))
    # 4 walking events in the 15:xx hour
    for m in range(4):
        events.append((f"2024-01-15 15:{m:02d}:00", "walking"))
    return _make_df(events)


# ---------------------------------------------------------------------------
# Tests: 1D rule
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_summarize_1D_counts(one_day_df):
    result = summarize(one_day_df, "1D")
    result = result.reset_index()

    eating_row = result[result["class"] == "eating"].iloc[0]
    walking_row = result[result["class"] == "walking"].iloc[0]

    assert eating_row["count"] == 5
    assert walking_row["count"] == 12


@pytest.mark.unit
def test_summarize_1D_cumulative_seconds(one_day_df):
    """cumulative_seconds == count (1 event = 1 second assumption)."""
    result = summarize(one_day_df, "1D").reset_index()

    eating_row = result[result["class"] == "eating"].iloc[0]
    walking_row = result[result["class"] == "walking"].iloc[0]

    assert eating_row["cumulative_seconds"] == 5
    assert walking_row["cumulative_seconds"] == 12


@pytest.mark.unit
def test_summarize_1D_peak_hour(one_day_df):
    """peak_hour = hour-of-day with most events."""
    result = summarize(one_day_df, "1D").reset_index()

    eating_row = result[result["class"] == "eating"].iloc[0]
    walking_row = result[result["class"] == "walking"].iloc[0]

    assert eating_row["peak_hour"] == 9    # all eating at 09:xx
    assert walking_row["peak_hour"] == 14  # 8 events at 14:xx vs 4 at 15:xx


# ---------------------------------------------------------------------------
# Tests: 1h rule
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_summarize_1h_counts(one_day_df):
    """
    1h bins:
        09:00 eating=5, walking=0
        14:00 eating=0, walking=8
        15:00 eating=0, walking=4
    """
    result = summarize(one_day_df, "1h").reset_index()

    # 09:00 hour — eating only
    eating_09 = result[
        (result["class"] == "eating") &
        (result["period_start"].dt.hour == 9)
    ]
    assert len(eating_09) == 1
    assert eating_09.iloc[0]["count"] == 5

    # 14:00 hour — walking
    walking_14 = result[
        (result["class"] == "walking") &
        (result["period_start"].dt.hour == 14)
    ]
    assert len(walking_14) == 1
    assert walking_14.iloc[0]["count"] == 8

    # 15:00 hour — walking
    walking_15 = result[
        (result["class"] == "walking") &
        (result["period_start"].dt.hour == 15)
    ]
    assert len(walking_15) == 1
    assert walking_15.iloc[0]["count"] == 4


@pytest.mark.unit
def test_summarize_1h_no_peak_hour(one_day_df):
    """peak_hour should be NaN for non-daily rules."""
    result = summarize(one_day_df, "1h").reset_index()
    assert result["peak_hour"].isna().all()


# ---------------------------------------------------------------------------
# Tests: 30min rule
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_summarize_30min_counts(one_day_df):
    """All 5 eating events fall within the 09:00–09:30 bin."""
    result = summarize(one_day_df, "30min").reset_index()

    eating_0900 = result[
        (result["class"] == "eating") &
        (result["period_start"].dt.hour == 9) &
        (result["period_start"].dt.minute == 0)
    ]
    assert len(eating_0900) == 1
    assert eating_0900.iloc[0]["count"] == 5


# ---------------------------------------------------------------------------
# Tests: empty input
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_summarize_empty():
    empty = pd.DataFrame(columns=["timestamp", "class", "confidence", "bbox", "subject_id"])
    result = summarize(empty, "1D")
    assert result.empty
