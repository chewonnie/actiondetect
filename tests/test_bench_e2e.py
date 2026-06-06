"""tests/test_bench_e2e.py — Smoke test for the end-to-end fps benchmark."""

import pytest


@pytest.mark.smoke
def test_run_benchmark_synthetic():
    """Benchmark with synthetic frames; no video file or R3D-18 ckpt required."""
    from pipeline.bench_e2e import run_benchmark

    result = run_benchmark(video_path=None, n_frames=24)

    assert isinstance(result, dict)
    required_keys = {"frames", "seconds", "fps", "yolo_only_fps", "action_model_loaded", "device", "note"}
    assert required_keys <= result.keys(), f"Missing keys: {required_keys - result.keys()}"
    assert result["fps"] > 0, "fps must be positive"
    assert result["yolo_only_fps"] > 0, "yolo_only_fps must be positive"
    assert result["frames"] == 24
