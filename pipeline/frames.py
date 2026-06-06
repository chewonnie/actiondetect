"""pipeline/frames.py — Frame source: video file or single webrtc frame.

NOTE ON DESIGN (dual-rate architecture):
  This 1-fps stream is for YOLO overlay and activity logging ONLY.
  The R3D-18 action model maintains its own separate native-fps rolling
  buffer (see pipeline/action_model.py). The two streams are independent.
"""

from __future__ import annotations

from typing import Generator

import cv2
import numpy as np


class FrameSource:
    """Yields frames downsampled from src_fps to target_fps."""

    @staticmethod
    def from_video(
        path: str,
        src_fps: float | None = None,
        target_fps: float = 1.0,
        size: int = 640,
    ) -> Generator[np.ndarray, None, None]:
        """Yield every (src_fps/target_fps)-th frame from a video file.

        Args:
            path: Path to the video file.
            src_fps: Source frame rate. If None, read from the video.
            target_fps: Desired output frame rate (default 1 fps).
            size: Resize each frame to size×size pixels.

        Yields:
            np.ndarray of shape (size, size, 3), dtype uint8, RGB.
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

        if src_fps is None:
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Take every step-th frame to achieve target_fps.
        step = max(1, round(src_fps / target_fps))

        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if frame_idx % step == 0:
                    frame = cv2.resize(frame, (size, size))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    yield frame
                frame_idx += 1
        finally:
            cap.release()

    @staticmethod
    def from_webrtc_frame(
        frame: np.ndarray,
        size: int = 640,
    ) -> np.ndarray:
        """Resize and convert a single incoming webrtc frame to RGB uint8.

        The streamlit-webrtc callback loop lives in app/dashboard.py.
        This helper only does the frame transform so action_model.py and
        detector.py receive a consistent format.

        Args:
            frame: HxWx3 uint8 array. May be BGR (from av/webrtc) or RGB.
                   Caller must pass the correct format; no conversion here
                   because webrtc delivers RGB via VideoFrame.to_ndarray(format="rgb24").
            size: Resize target (square).

        Returns:
            np.ndarray of shape (size, size, 3), dtype uint8, RGB.
        """
        if frame.shape[0] != size or frame.shape[1] != size:
            frame = cv2.resize(frame, (size, size))
        return frame.astype(np.uint8)
