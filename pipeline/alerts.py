"""
Activity drop alerts based on a rolling baseline.

Default thresholds match pipeline/config.yaml:
    drop_pct:        0.30  (flag a class when seconds <= baseline * 0.70)
    baseline_days:   7     (days of history required before alerting)
    consecutive_days: 2    (how many consecutive "down" days trigger an alert)
"""

from __future__ import annotations

import pandas as pd

from pipeline.aggregate import summarize


def daily_class_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-day, per-class total seconds from a raw log DataFrame.

    Args:
        df: DataFrame from aggregate.load_logs (timestamp, class, ...).

    Returns:
        DataFrame with columns [date (date), class, seconds].
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "class", "seconds"])

    summary = summarize(df, "1D")
    summary = summary.reset_index()
    summary = summary.rename(
        columns={"period_start": "date", "cumulative_seconds": "seconds"}
    )
    summary["date"] = summary["date"].dt.date
    return summary[["date", "class", "seconds"]]


def compute_alerts(
    daily: pd.DataFrame,
    drop_pct: float = 0.30,
    baseline_days: int = 7,
    consecutive_days: int = 2,
) -> list[dict]:
    """Detect sustained activity drops vs a rolling baseline.

    For each (class, day) combination, a trailing baseline is computed from
    the `baseline_days` days immediately before that day (exclusive).  A class
    is "down" on a day when its seconds <= baseline * (1 - drop_pct).  An
    alert is emitted only when the class is down for `consecutive_days`
    consecutive days; it is reported on the day the streak completes.

    Days with fewer than `baseline_days` days of prior history produce a
    status-only entry (status="insufficient_baseline") — no alert, no crash.

    Args:
        daily:            DataFrame with columns [date, class, seconds].
                          date must be a comparable type (datetime.date or str).
        drop_pct:         Fraction drop that counts as "down" (default 0.30).
        baseline_days:    How many prior days are required before alerting.
        consecutive_days: How many consecutive "down" days trigger an alert.

    Returns:
        List of dicts, one per noteworthy (class, day).  Each dict has:
            date            - the day being evaluated
            class           - class name
            status          - "alert" | "insufficient_baseline" | "ok" | "down"
        Alert dicts additionally have:
            baseline        - mean seconds over the trailing window
            actual          - actual seconds on that day
            pct_drop        - fraction drop (positive = drop)
    """
    if daily.empty:
        return []

    results: list[dict] = []

    # Normalise date column to comparable type (keep as-is; sort works on date objects).
    all_classes = sorted(daily["class"].unique())
    all_dates = sorted(daily["date"].unique())

    # For each class, track consecutive "down" day count.
    for cls in all_classes:
        cls_data = daily[daily["class"] == cls].set_index("date")["seconds"]

        streak = 0  # consecutive "down" days so far

        for i, day in enumerate(all_dates):
            prior_dates = all_dates[:i]  # dates strictly before today

            if len(prior_dates) < baseline_days:
                # Cold start: not enough history.
                results.append({"date": day, "class": cls, "status": "insufficient_baseline"})
                streak = 0  # reset streak during warmup
                continue

            # Baseline = the normal level BEFORE the current down-streak.
            # Excluding the ongoing decline keeps the anomaly from dragging
            # its own reference down (a trailing mean that includes the drop
            # days would "chase" the decline and under-report it).
            pre_streak = prior_dates[: len(prior_dates) - streak]
            window = (pre_streak or prior_dates)[-baseline_days:]
            baseline_values = [cls_data.get(d, 0) for d in window]
            baseline = sum(baseline_values) / len(baseline_values)

            actual = cls_data.get(day, 0)
            threshold = baseline * (1.0 - drop_pct)

            if actual <= threshold:
                streak += 1
            else:
                streak = 0

            if streak >= consecutive_days and streak > 0:
                pct_drop = (baseline - actual) / baseline if baseline > 0 else 0.0
                if streak == consecutive_days:
                    # Emit alert only on the day the streak first completes.
                    results.append({
                        "date": day,
                        "class": cls,
                        "status": "alert",
                        "baseline": baseline,
                        "actual": actual,
                        "pct_drop": pct_drop,
                    })
                else:
                    # Streak continuing beyond threshold — still "down", no new alert.
                    results.append({"date": day, "class": cls, "status": "down"})
            elif actual <= threshold:
                # First day of a potential streak, not yet consecutive_days.
                results.append({"date": day, "class": cls, "status": "down"})
            else:
                results.append({"date": day, "class": cls, "status": "ok"})

    return results
