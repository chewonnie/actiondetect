"""Unit tests for pipeline/class_map.py."""

import csv
import os
import sys

import pytest

# Ensure the repo root is on sys.path so pipeline.class_map is importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline.class_map import remap  # noqa: E402


@pytest.mark.unit
def test_all_55_mapped():
    """Every action_idx 0..54 maps to an int in 0..11."""
    for i in range(55):
        result = remap(i)
        assert isinstance(result, int), f"remap({i}) returned {result!r}, expected int"
        assert 0 <= result <= 11, f"remap({i}) = {result} is out of core_idx range"


@pytest.mark.unit
def test_dense_codomain():
    """All 12 core classes appear in the mapping (no gaps)."""
    assert set(remap(i) for i in range(55)) == set(range(12))


@pytest.mark.unit
def test_out_of_range_none():
    """Out-of-range inputs return None."""
    assert remap(-1) is None
    assert remap(55) is None
    assert remap(100) is None


@pytest.mark.unit
def test_csv_consistency():
    """pipeline/etri_actions.csv has exactly 55 data rows with action_idx 0..54."""
    csv_path = os.path.join(_REPO_ROOT, "pipeline", "etri_actions.csv")
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(ln for ln in f if not ln.lstrip().startswith("#"))
        indices = [int(row["action_idx"]) for row in reader]
    assert len(indices) == 55, f"Expected 55 rows, got {len(indices)}"
    assert sorted(indices) == list(range(55)), "action_idx values are not contiguous 0..54"
