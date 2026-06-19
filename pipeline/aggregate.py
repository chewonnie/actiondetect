"""
Aggregate activity log CSV files into per-class summaries.

Log schema (produced by pipeline/activity_logger.py):
    timestamp,class,confidence,bbox,subject_id
    - timestamp: ISO 8601 string (e.g. "2024-01-15T14:30:00")
    - class: core class name string
    - confidence, bbox, subject_id: other fields (ignored here)

Assumption: each row represents exactly one 1-fps detection event = 1 second of
activity.  cumulative_seconds is therefore equal to count (number of events).
This is documented explicitly because it is an approximation: gaps in logging
(camera off, no detection) are not counted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

ACTIVITY_LOG_COLUMNS = ["timestamp", "class", "confidence", "bbox", "subject_id"]


def load_logs(log_dir: str, dates: Optional[list[str]] = None) -> pd.DataFrame:
    """Load one or more daily log CSVs from log_dir.

    Args:
        log_dir: Directory containing logs/YYYY-MM-DD.csv files.
        dates:   Optional list of date strings ("YYYY-MM-DD") to load.
                 If None, all CSV files in log_dir are loaded.

    Returns:
        DataFrame with columns: timestamp (datetime64), class, confidence,
        bbox, subject_id.  Empty DataFrame if no files found.
    """
    log_path = Path(log_dir)
    empty = pd.DataFrame(columns=ACTIVITY_LOG_COLUMNS)

    if dates is not None:
        files = [log_path / f"{d}.csv" for d in dates]
        files = [f for f in files if f.exists()]
    else:
        files = sorted(log_path.glob("*.csv"))

    if not files:
        return empty

    frames = []
    for f in files:
        df = pd.read_csv(f)
        if not set(ACTIVITY_LOG_COLUMNS).issubset(df.columns):
            continue
        df = df[ACTIVITY_LOG_COLUMNS]
        frames.append(df)

    if not frames:
        return empty

    result = pd.concat(frames, ignore_index=True)
    # ActivityLogger writes datetime.now().isoformat() — rows may or may not
    # carry microseconds. format="ISO8601" accepts both variants.
    result["timestamp"] = pd.to_datetime(result["timestamp"], format="ISO8601")
    return result


def summarize(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample log events and produce per-class activity statistics.

    Assumption: each row = 1 second of activity (1-fps logging, 1 event = 1s).
    So cumulative_seconds == count for each bin.

    Args:
        df:   DataFrame from load_logs (must have "timestamp" and "class" columns).
        rule: Pandas offset alias: "30min", "1h", or "1D".

    Returns:
        DataFrame indexed by (period_start, class) with columns:
            count              - number of detection events in the period
            cumulative_seconds - approximate seconds of activity (= count, see module docstring)
            peak_hour          - hour-of-day (0..23) with most events; only
                                 computed when rule == "1D", else NaN.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["period_start", "class", "count", "cumulative_seconds", "peak_hour"]
        ).set_index(["period_start", "class"])
    missing = {"timestamp", "class"} - set(df.columns)
    if missing:
        raise ValueError(f"activity log is missing required columns: {sorted(missing)}")

    df = df.copy()
    df = df.set_index("timestamp").sort_index()

    # Count events per (time bin, class).
    grouped = (
        df.groupby([pd.Grouper(freq=rule), "class"])
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"timestamp": "period_start"})
    )
    grouped["cumulative_seconds"] = grouped["count"]  # 1 event = 1 second (1-fps)

    # peak_hour: only meaningful for daily aggregation.
    if rule == "1D":
        # For each class, find the hour-of-day with the most events.
        df_h = df.copy()
        df_h["hour"] = df_h.index.hour
        # Day bucket (floor to day) so we can join back.
        df_h["day"] = df_h.index.floor("D")
        hour_counts = (
            df_h.groupby(["day", "class", "hour"])
            .size()
            .rename("n")
            .reset_index()
        )
        # For each (day, class) pick the hour with max events; ties -> lowest hour.
        peak = (
            hour_counts
            .sort_values("hour")
            .groupby(["day", "class"], sort=False)
            .apply(lambda g: g.loc[g["n"].idxmax(), "hour"])
            .rename("peak_hour")
            .reset_index()
            .rename(columns={"day": "period_start"})
        )
        grouped = grouped.merge(peak, on=["period_start", "class"], how="left")
    else:
        grouped["peak_hour"] = float("nan")

    return grouped.set_index(["period_start", "class"])
