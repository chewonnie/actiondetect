"""Unit tests for pipeline.urfd_fall_model.UrfdFallRecognizer.

We bypass ActionModel construction (torch + checkpoint dependency) and
exercise only the cooldown/threshold logic — UrfdFallRecognizer's contract
covers the same cooldown/threshold contract used by the dashboard and the
ActionModel wrapper itself has its own coverage in pipeline/action_model.py.
"""

from __future__ import annotations

import pytest

from pipeline.urfd_fall_model import UrfdFallRecognizer


def _fake_recognizer(prob_thr: float = 0.7, cooldown_s: float = 2.0):
    """Construct without invoking ActionModel (no torch/ckpt needed)."""
    rec = object.__new__(UrfdFallRecognizer)
    rec._m = None  # ActionModel stub — not touched by update()
    rec._prob_thr = float(prob_thr)
    rec._cooldown = float(cooldown_s)
    rec._last_event_ts = None
    rec.p_fall = 0.0
    rec.last_probs = None
    return rec


@pytest.mark.unit
def test_below_threshold_never_fires():
    rec = _fake_recognizer(prob_thr=0.7)
    rec.p_fall = 0.69                  # one tick below threshold
    assert rec.update(0.0) is False
    assert rec.update(10.0) is False   # time alone doesn't matter


@pytest.mark.unit
def test_at_or_above_threshold_fires_first_time():
    rec = _fake_recognizer(prob_thr=0.7)
    rec.p_fall = 0.70                  # exactly at threshold
    assert rec.update(0.0) is True
    # Subsequent ticks with the same elevated probability stay suppressed by
    # cooldown.
    assert rec.update(0.1) is False


@pytest.mark.unit
def test_cooldown_suppresses_then_releases():
    rec = _fake_recognizer(prob_thr=0.5, cooldown_s=2.0)
    rec.p_fall = 0.9
    assert rec.update(0.0) is True
    assert rec.update(1.9) is False    # within cooldown
    assert rec.update(2.0) is True     # exactly at cooldown boundary fires
    assert rec.update(3.5) is False    # within next cooldown
    assert rec.update(4.1) is True     # after next cooldown elapses


@pytest.mark.unit
def test_drop_below_threshold_resets_eligibility_not_cooldown():
    """Dropping p_fall below thr doesn't reset cooldown; coming back fires
    only after cooldown_s has actually elapsed."""
    rec = _fake_recognizer(prob_thr=0.6, cooldown_s=1.0)
    rec.p_fall = 0.9
    assert rec.update(0.0) is True
    rec.p_fall = 0.1
    assert rec.update(0.5) is False    # below threshold
    rec.p_fall = 0.9
    assert rec.update(0.8) is False    # back above thr but still in cooldown
    assert rec.update(1.05) is True    # cooldown expired


@pytest.mark.unit
def test_p_fall_attribute_used_for_threshold():
    """update() reads rec.p_fall — confirming the binding the wrapper depends on."""
    rec = _fake_recognizer(prob_thr=0.5)
    rec.p_fall = 0.49
    assert rec.update(0.0) is False
    rec.p_fall = 0.51
    assert rec.update(0.1) is True
