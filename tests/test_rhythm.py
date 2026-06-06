"""pipeline.rhythm 단위테스트 (합성 로그, 데이터/모델 불필요)."""
import pandas as pd
import pytest

from pipeline.rhythm import daily_rhythm

pytestmark = pytest.mark.unit


def _df(rows):
    return pd.DataFrame(
        [{"timestamp": pd.Timestamp(t), "class": c} for t, c in rows]
    )


def test_empty():
    r = daily_rhythm(pd.DataFrame())
    assert r["wake_time"] is None and r["meal_count"] == 0


def test_wake_sleep_meals_night():
    rows = [
        ("2026-05-18 02:10", "mobility"),          # 야간활동 1
        ("2026-05-18 03:40", "phone"),             # 야간활동 2
        ("2026-05-18 07:05", "eating"),            # 아침식사 군집 A
        ("2026-05-18 07:15", "cooking_kitchen"),   # 같은 군집(30분내)
        ("2026-05-18 12:30", "eating"),            # 점심 군집 B
        ("2026-05-18 19:00", "eating"),            # 저녁 군집 C
        ("2026-05-18 22:50", "posture_transition"),# 마지막 눕기
    ]
    r = daily_rhythm(_df(rows))
    assert r["wake_time"] == "02:10"               # 첫 이벤트
    assert r["sleep_time"] == "22:50"              # 마지막 이벤트
    assert r["last_lying_time"] == "22:50"
    assert r["meal_count"] == 3                    # 07시 군집 1 + 12 + 19
    assert r["meal_times"][0] == "07:05"
    assert r["night_activity_count"] == 2          # 02,03시


def test_meal_clustering_merges_close_events():
    rows = [("2026-05-18 08:00", "eating"),
            ("2026-05-18 08:05", "eating"),
            ("2026-05-18 08:20", "cooking_kitchen")]
    r = daily_rhythm(_df(rows))
    assert r["meal_count"] == 1                    # 모두 30분 내 → 1군집
