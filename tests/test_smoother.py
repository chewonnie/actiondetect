"""Unit tests for pipeline.smoother.MajorityVoteSmoother."""

import pytest
from pipeline.smoother import MajorityVoteSmoother


@pytest.mark.unit
def test_single_outlier_does_not_flip_output():
    """Feeding [3,3,7,3,3,3] with window=5 should keep output at 3.

    The lone 7 enters the buffer but is always outvoted by 3s.
    We check that the output never becomes 7.
    """
    smoother = MajorityVoteSmoother(window=5)
    labels = [3, 3, 7, 3, 3, 3]
    outputs = [smoother.update(lbl) for lbl in labels]

    # Output should never be 7.
    assert 7 not in outputs, f"Unexpected 7 in outputs: {outputs}"

    # Once we have at least one label, output should always be 3.
    assert all(o == 3 for o in outputs), f"Expected all 3s, got: {outputs}"


@pytest.mark.unit
def test_pre_full_buffer_returns_mode_of_seen_so_far():
    """Before the buffer fills, mode is computed over what's been seen."""
    smoother = MajorityVoteSmoother(window=5)

    # After one label, the only option is that label.
    assert smoother.update(4) == 4

    # After [4, 4], mode is clearly 4.
    assert smoother.update(4) == 4

    # After [4, 4, 9], mode is 4 (count 2 vs 1).
    assert smoother.update(9) == 4


@pytest.mark.unit
def test_none_input_returns_none():
    """None input should pass through as None without touching the buffer."""
    smoother = MajorityVoteSmoother(window=3)
    smoother.update(1)
    result = smoother.update(None)
    assert result is None


@pytest.mark.unit
def test_sliding_window_evicts_old_values():
    """Old values fall off the deque as the window slides."""
    smoother = MajorityVoteSmoother(window=3)
    # Fill with 1s.
    smoother.update(1)
    smoother.update(1)
    smoother.update(1)
    # Now push three 2s — window becomes [2, 2, 2].
    smoother.update(2)
    smoother.update(2)
    result = smoother.update(2)
    assert result == 2


@pytest.mark.unit
def test_tie_broken_by_recency():
    """With window=2, a tie [1, 2] resolves to the most recent label (2)."""
    smoother = MajorityVoteSmoother(window=2)
    smoother.update(1)
    result = smoother.update(2)
    # Buffer is [1, 2], each count=1; recency picks 2.
    assert result == 2


@pytest.mark.unit
def test_window_one_always_returns_latest():
    """Window=1 is a passthrough: always returns the most recent label."""
    smoother = MajorityVoteSmoother(window=1)
    for lbl in [5, 3, 7, 2]:
        assert smoother.update(lbl) == lbl


@pytest.mark.unit
def test_invalid_window_raises():
    with pytest.raises(ValueError):
        MajorityVoteSmoother(window=0)
