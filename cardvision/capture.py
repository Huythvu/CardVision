"""Frame sources: grab BGR frames from the screen or load from an image file.

Returns OpenCV-style BGR ``numpy`` arrays so the rest of the pipeline is
source-agnostic.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config


def load_image(path: str) -> np.ndarray:
    """Load an image file as a BGR frame."""
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return frame


def grab_screen(region: dict | None = None) -> np.ndarray:
    """Grab a single frame from the screen region as BGR.

    ``region`` is an mss-style dict {left, top, width, height}; defaults to
    ``config.SCREEN_REGION``.
    """
    import mss  # imported lazily so non-capture use works without a display

    region = region or config.SCREEN_REGION
    with mss.mss() as sct:
        shot = sct.grab(region)
        # mss returns BGRA; drop alpha to get BGR.
        frame = np.asarray(shot)[:, :, :3]
    return np.ascontiguousarray(frame)


def crop(frame: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """Crop ``rect`` = (x, y, w, h) from a frame, clamped to bounds."""
    x, y, w, h = rect
    H, W = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    return frame[y0:y1, x0:x1]
