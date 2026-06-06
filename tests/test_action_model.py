"""tests/test_action_model.py — Tests for pipeline/action_model.py."""

import os
import numpy as np
import pytest


# ── Unit tests (no checkpoint needed) ────────────────────────────────────────

@pytest.mark.unit
def test_buffer_span_default():
    """buffer_span == clip_length * sampling_rate for default parameters."""
    # Import only the class constants; do NOT instantiate (would need a ckpt).
    from pipeline.action_model import ActionModel
    assert ActionModel._DEFAULT_CLIP_LENGTH == 16
    assert ActionModel._DEFAULT_SAMPLING_RATE == 2
    expected = ActionModel._DEFAULT_CLIP_LENGTH * ActionModel._DEFAULT_SAMPLING_RATE
    assert expected == 32, f"Default buffer_span should be 32, got {expected}"


@pytest.mark.smoke
def test_buffer_span_wired_to_instance_params():
    """The temporal contract is a real instance property + deque sizing,
    not just a formula: instantiate with custom clip_length/sampling_rate
    and assert buffer_span and the rolling buffer maxlen both reflect them.
    Needs the trained ckpt (ActionModel.__init__ loads weights)."""
    from pipeline.action_model import ActionModel
    ckpt = "runs/baseline12/best.pt"
    if not os.path.exists(ckpt):
        pytest.skip("needs trained R3D-18 ckpt from P1")
    m = ActionModel(ckpt, clip_length=8, sampling_rate=4, device="cpu")
    assert m.buffer_span == 32          # 8*4, non-default path
    assert m._buffer.maxlen == 32       # buffer actually sized to the contract
    m2 = ActionModel(ckpt, clip_length=4, sampling_rate=2, device="cpu")
    assert m2.buffer_span == 8 and m2._buffer.maxlen == 8


# ── Smoke test (needs trained R3D-18 checkpoint from P1) ─────────────────────

@pytest.mark.smoke
def test_action_model_inference_with_ckpt():
    """Push buffer_span frames and verify infer() returns (int, float) with hold-last."""
    import yaml

    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "pipeline", "config.yaml"
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ckpt_path = cfg["paths"]["r3d18_ckpt"]
    # Resolve relative path from repo root.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(repo_root, ckpt_path)

    if not os.path.exists(ckpt_path):
        pytest.skip("needs trained R3D-18 ckpt from P1")

    from pipeline.action_model import ActionModel

    model = ActionModel(ckpt_path, num_classes=12)

    # Verify temporal contract.
    assert model.buffer_span == 32

    # Before buffer is full, infer() must return None.
    dummy_frame = np.zeros((112, 112, 3), dtype=np.uint8)
    for _ in range(model.buffer_span - 1):
        model.push(dummy_frame)
    assert model.infer() is None
    assert model.last_label is None

    # Push one more to fill the buffer.
    model.push(dummy_frame)
    result = model.infer()
    assert result is not None
    class_idx, prob = result
    assert isinstance(class_idx, int)
    assert 0 <= class_idx < 12
    assert 0.0 <= prob <= 1.0

    # hold-last: last_label is now set.
    assert model.last_label == result
