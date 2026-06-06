"""tests/test_frames.py — Smoke tests for pipeline/frames.py."""

import numpy as np
import cv2
import pytest

from pipeline.frames import FrameSource


@pytest.mark.smoke
def test_from_video_yields_correct_frames(tmp_path):
    """Synthesise a 2s @30fps video and check FrameSource.from_video at 1 fps."""
    video_path = str(tmp_path / "test.mp4")
    fps = 30
    duration_s = 2
    total_frames = fps * duration_s  # 60 frames
    size = 640

    # Write a synthetic video: each frame is a solid colour that varies by frame.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (size, size))
    for i in range(total_frames):
        colour = int(i * 255 / total_frames)
        frame = np.full((size, size, 3), colour, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    frames = list(FrameSource.from_video(video_path, src_fps=fps, target_fps=1, size=size))

    # At 30fps -> 1fps, step = 30, so we expect floor(60/30) = 2 frames.
    assert len(frames) == 2, f"Expected ~2 frames, got {len(frames)}"

    for frame in frames:
        assert frame.shape == (size, size, 3), f"Wrong shape: {frame.shape}"
        assert frame.dtype == np.uint8, f"Wrong dtype: {frame.dtype}"
