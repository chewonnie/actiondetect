"""Unit tests for pipeline/alerts.py — synthetic in-memory data only."""

import datetime

import pandas as pd
import pytest

from pipeline.alerts import compute_alerts, daily_class_seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily(rows: list[tuple]) -> pd.DataFrame:
    """Build a daily DataFrame from (date_str, class, seconds) tuples."""
    return pd.DataFrame(
        [
            {"date": datetime.date.fromisoformat(d), "class": cls, "seconds": secs}
            for d, cls, secs in rows
        ]
    )


# ---------------------------------------------------------------------------
# Fixture: 8-day series
#
# Days 1..7: stable baseline of 100 seconds/day for class "eating".
# Day 8: drops to 60 seconds (60% of 100 = 40% drop > 30% threshold).
#
# We need TWO consecutive "down" days to trigger an alert.
# So we'll use:
#   Days 1..7  = 100 s  (baseline building period)
#   Day 8      = 60 s   (first "down" day — no alert yet)
#   Day 9      = 60 s   (second consecutive "down" — alert here)
# That's 9 days total; days 1..7 have < 7 prior days -> insufficient_baseline.
# Day 8 has exactly 7 prior days -> baseline computed, first down.
# Day 9 has 8 prior days -> alert fires.
# ---------------------------------------------------------------------------

BASE_DATE = datetime.date(2024, 1, 1)


def _date(n: int) -> str:
    """Return ISO date string for day n (1-based)."""
    return (BASE_DATE + datetime.timedelta(days=n - 1)).isoformat()


@pytest.fixture
def two_day_drop_daily():
    """8-day stable + 2-day drop = alert on day 9 (the 2nd dropped day)."""
    rows = []
    for i in range(1, 8):          # days 1..7: stable at 100
        rows.append((_date(i), "eating", 100))
    rows.append((_date(8), "eating", 60))   # day 8: first down (40% drop)
    rows.append((_date(9), "eating", 60))   # day 9: second down -> alert
    return _make_daily(rows)


@pytest.fixture
def single_day_drop_daily():
    """7 stable days + 1 dropped day + 1 recovery -> no alert."""
    rows = []
    for i in range(1, 8):
        rows.append((_date(i), "eating", 100))
    rows.append((_date(8), "eating", 60))   # one dip
    rows.append((_date(9), "eating", 100))  # recovery
    return _make_daily(rows)


# ---------------------------------------------------------------------------
# Tests: two consecutive drops -> exactly one alert
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_alert_fires_on_second_consecutive_drop(two_day_drop_daily):
    """Alert emitted on day 9 (streak completes); only one alert total."""
    results = compute_alerts(
        two_day_drop_daily,
        drop_pct=0.30,
        baseline_days=7,
        consecutive_days=2,
    )
    alerts = [r for r in results if r["status"] == "alert"]
    assert len(alerts) == 1, f"Expected 1 alert, got {len(alerts)}: {alerts}"


@pytest.mark.unit
def test_alert_is_on_correct_day(two_day_drop_daily):
    """Alert date is day 9 (the day the streak completes)."""
    results = compute_alerts(
        two_day_drop_daily,
        drop_pct=0.30,
        baseline_days=7,
        consecutive_days=2,
    )
    alerts = [r for r in results if r["status"] == "alert"]
    assert alerts[0]["date"] == datetime.date.fromisoformat(_date(9))


@pytest.mark.unit
def test_alert_fields_present(two_day_drop_daily):
    """Alert dict contains baseline, actual, pct_drop."""
    results = compute_alerts(
        two_day_drop_daily,
        drop_pct=0.30,
        baseline_days=7,
        consecutive_days=2,
    )
    alert = next(r for r in results if r["status"] == "alert")
    assert "baseline" in alert
    assert "actual" in alert
    assert "pct_drop" in alert
    assert alert["actual"] == 60
    assert abs(alert["baseline"] - 100.0) < 1e-6
    assert abs(alert["pct_drop"] - 0.40) < 1e-6


# ---------------------------------------------------------------------------
# Tests: single-day dip -> NO alert
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_no_alert_for_single_day_dip(single_day_drop_daily):
    """One-day dip followed by recovery does not fire an alert."""
    results = compute_alerts(
        single_day_drop_daily,
        drop_pct=0.30,
        baseline_days=7,
        consecutive_days=2,
    )
    alerts = [r for r in results if r["status"] == "alert"]
    assert len(alerts) == 0, f"Expected 0 alerts, got {len(alerts)}: {alerts}"


# ---------------------------------------------------------------------------
# Tests: cold start (days 1..7 have < 7 prior days -> insufficient_baseline)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_warmup_days_insufficient_baseline(two_day_drop_daily):
    """Days 1..7 each have < 7 prior days -> all marked insufficient_baseline."""
    results = compute_alerts(
        two_day_drop_daily,
        drop_pct=0.30,
        baseline_days=7,
        consecutive_days=2,
    )
    insuf = [r for r in results if r["status"] == "insufficient_baseline"]
    # Days 1..7 = 7 days of warmup for the single class "eating"
    assert len(insuf) == 7, f"Expected 7 insufficient_baseline entries, got {len(insuf)}"


@pytest.mark.unit
def test_warmup_produces_no_crash(two_day_drop_daily):
    """compute_alerts must not raise during cold-start days."""
    # If this call raises, the test fails.
    compute_alerts(
        two_day_drop_daily,
        drop_pct=0.30,
        baseline_days=7,
        consecutive_days=2,
    )


@pytest.mark.unit
def test_warmup_produces_no_alerts(two_day_drop_daily):
    """No alert fires during the first baseline_days days."""
    results = compute_alerts(
        two_day_drop_daily,
        drop_pct=0.30,
        baseline_days=7,
        consecutive_days=2,
    )
    # Warmup days are days 1..7 (indices 0..6 in all_dates for "eating")
    warmup_dates = {datetime.date.fromisoformat(_date(i)) for i in range(1, 8)}
    warmup_alerts = [
        r for r in results
        if r["status"] == "alert" and r["date"] in warmup_dates
    ]
    assert len(warmup_alerts) == 0


# ---------------------------------------------------------------------------
# Tests: daily_class_seconds integration
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_daily_class_seconds_empty():
    """Empty input returns empty DataFrame without crashing."""
    empty = pd.DataFrame(columns=["timestamp", "class", "confidence", "bbox", "subject_id"])
    result = daily_class_seconds(empty)
    assert result.empty
