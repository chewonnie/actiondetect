"""US-005 — proof that the single dataset.py edit is a flag-gated no-op.

PLAN.md §3.2 / §8.2: with the class map INACTIVE (env var ETRI_CLASS_MAP
unset) `scan_etri_root` must reproduce the original 55-class baseline
bit-for-bit; with it ACTIVE every label must fall in the dense 0..11 core
space and no clip is lost (the confirmed map drops nothing).

Smoke test: needs ./etri extracted (P01 is enough). Uses P01 only for speed.
"""

import importlib
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
_CSV = os.path.join(_REPO, "pipeline", "etri_actions.csv")
_ETRI = os.path.join(_REPO, "etri")

if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

pytestmark = pytest.mark.smoke


def _fresh_dataset():
    """Reimport dataset so the module-level class-map cache is clean."""
    sys.modules.pop("dataset", None)
    return importlib.import_module("dataset")


@pytest.mark.skipif(not os.path.isdir(_ETRI), reason="./etri not extracted")
def test_remap_inactive_is_bitfor_bit_baseline(monkeypatch):
    monkeypatch.delenv("ETRI_CLASS_MAP", raising=False)
    ds = _fresh_dataset()
    samples = ds.scan_etri_root(_ETRI, participants=["P01"])
    assert samples, "no clips found under ./etri/RGB/P01"
    # No-op proof: every label equals the raw filename parse (no remap applied).
    for s in samples:
        assert s.action_idx == ds.parse_action_index(os.path.basename(s.rgb_path))
    # 55-class space is preserved (labels exceed the 12-core range).
    assert max(s.action_idx for s in samples) > 11


@pytest.mark.skipif(not os.path.isdir(_ETRI), reason="./etri not extracted")
def test_remap_active_maps_into_dense_core_space(monkeypatch):
    monkeypatch.delenv("ETRI_CLASS_MAP", raising=False)
    ds = _fresh_dataset()
    base = ds.scan_etri_root(_ETRI, participants=["P01"])

    monkeypatch.setenv("ETRI_CLASS_MAP", _CSV)
    ds = _fresh_dataset()
    mapped = ds.scan_etri_root(_ETRI, participants=["P01"])

    assert {s.action_idx for s in mapped} <= set(range(12))
    # Confirmed map drops nothing: same clip count as the 55-class scan.
    assert len(mapped) == len(base)
