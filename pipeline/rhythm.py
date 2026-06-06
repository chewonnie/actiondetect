"""pipeline/rhythm.py — 활동 로그에서 생활리듬 추정 (순수 함수).

입력: load_logs() 가 만든 DataFrame [timestamp(datetime), class, ...].
출력: 기상/취침 시각, 식사 시각(클러스터), 야간활동 횟수.

전부 로그 기반 휴리스틱 (추가 모델 없음). 시각은 그 날 첫/마지막 이벤트
및 행동 클래스 군집으로 추정 — GT 없으므로 정확도 미산정(근사치).
"""
from __future__ import annotations

import pandas as pd

MEAL_CLASSES = ("eating", "cooking_kitchen")          # 식사 관련
NIGHT_START, NIGHT_END = 0, 5                         # 야간활동 구간 [00,05)


def _cluster_times(ts: list, gap_min: int = 30) -> list:
    """정렬된 timestamp 리스트 → gap_min 이상 떨어지면 새 군집, 군집 시작시각."""
    if not ts:
        return []
    ts = sorted(ts)
    reps, prev = [ts[0]], ts[0]          # 군집 대표 = 각 군집의 첫 시각
    for t in ts[1:]:
        if (t - prev).total_seconds() > gap_min * 60:
            reps.append(t)
        prev = t
    return reps


def daily_rhythm(df: pd.DataFrame) -> dict:
    """하루치 로그 → 생활리듬 dict.

    Returns keys:
      wake_time, sleep_time          : "HH:MM" | None
      last_lying_time                : 마지막 posture_transition "HH:MM" | None
      meal_times                     : ["HH:MM", ...]  (식사 군집 시작시각)
      meal_count                     : int
      night_activity_count           : 00:00–05:00 이벤트 수 (불면/배회 단서)
      first_event, last_event        : ISO | None
    """
    empty = {"wake_time": None, "sleep_time": None, "last_lying_time": None,
             "meal_times": [], "meal_count": 0, "night_activity_count": 0,
             "first_event": None, "last_event": None}
    if df is None or df.empty:
        return empty

    d = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if d.empty:
        return empty
    t0, t1 = d["timestamp"].iloc[0], d["timestamp"].iloc[-1]

    lying = d[d["class"] == "posture_transition"]["timestamp"]
    last_lying = lying.iloc[-1] if len(lying) else None

    meals = _cluster_times(
        list(d[d["class"].isin(MEAL_CLASSES)]["timestamp"]), gap_min=30
    )
    hours = d["timestamp"].dt.hour
    night = int(((hours >= NIGHT_START) & (hours < NIGHT_END)).sum())

    def hm(t):
        return None if t is None else pd.Timestamp(t).strftime("%H:%M")

    return {
        "wake_time": hm(t0),                  # 그 날 첫 활동 = 기상 근사
        "sleep_time": hm(t1),                 # 마지막 활동 = 취침 근사
        "last_lying_time": hm(last_lying),
        "meal_times": [hm(t) for t in meals],
        "meal_count": len(meals),
        "night_activity_count": night,
        "first_event": pd.Timestamp(t0).isoformat(),
        "last_event": pd.Timestamp(t1).isoformat(),
    }
