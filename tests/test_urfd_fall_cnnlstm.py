"""Unit tests for pipeline.urfd_fall_cnnlstm.UrfdFallCnnLstmRecognizer.

Same shape as tests/test_urfd_fall.py — we bypass MobileNetV3/CNNLSTM
construction (heavy torchvision dependency + ckpt) and exercise only the
cooldown/threshold logic. The update() contract is identical to the R3D-18
sibling, so the behavioural matrix is duplicated to lock both wrappers in.
"""

from __future__ import annotations

import pytest

from pipeline.urfd_fall_cnnlstm import UrfdFallCnnLstmRecognizer


def _fake_recognizer(prob_thr: float = 0.7, cooldown_s: float = 2.0):
    """Construct without loading torchvision/ckpt — exercises update() only."""
    rec = object.__new__(UrfdFallCnnLstmRecognizer)
    rec._cnn = None
    rec._model = None
    rec._buf = None
    rec._prob_thr = float(prob_thr)
    rec._cooldown = float(cooldown_s)
    rec._last_event_ts = None
    rec.p_fall = 0.0
    rec.last_probs = None
    rec.num_classes = 2
    return rec


@pytest.mark.unit
def test_below_threshold_never_fires():
    rec = _fake_recognizer(prob_thr=0.7)
    rec.p_fall = 0.69
    assert rec.update(0.0) is False
    assert rec.update(10.0) is False


@pytest.mark.unit
def test_at_or_above_threshold_fires_first_time():
    rec = _fake_recognizer(prob_thr=0.7)
    rec.p_fall = 0.70
    assert rec.update(0.0) is True
    assert rec.update(0.1) is False    # cooldown suppresses immediate refire


@pytest.mark.unit
def test_cooldown_suppresses_then_releases():
    rec = _fake_recognizer(prob_thr=0.5, cooldown_s=2.0)
    rec.p_fall = 0.9
    assert rec.update(0.0) is True
    assert rec.update(1.9) is False
    assert rec.update(2.0) is True     # exactly at cooldown boundary fires
    assert rec.update(3.5) is False
    assert rec.update(4.1) is True


@pytest.mark.unit
def test_drop_below_threshold_resets_eligibility_not_cooldown():
    rec = _fake_recognizer(prob_thr=0.6, cooldown_s=1.0)
    rec.p_fall = 0.9
    assert rec.update(0.0) is True
    rec.p_fall = 0.1
    assert rec.update(0.5) is False
    rec.p_fall = 0.9
    assert rec.update(0.8) is False
    assert rec.update(1.05) is True


@pytest.mark.unit
def test_p_fall_attribute_used_for_threshold():
    rec = _fake_recognizer(prob_thr=0.5)
    rec.p_fall = 0.49
    assert rec.update(0.0) is False
    rec.p_fall = 0.51
    assert rec.update(0.1) is True


@pytest.mark.unit
def test_fall_class_idx_matches_training_label_convention():
    """train_urfd_cnnlstm.py URFD_LABELS = {'fall': 1, 'adl': 0} — class 1 must be FALL."""
    assert UrfdFallCnnLstmRecognizer.FALL_CLASS_IDX == 1
