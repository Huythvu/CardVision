"""Automatic card detection: find card-shaped quadrilaterals in a frame and
perspective-warp each to a flat, canonical card image.

This replaces the fixed ``CARD_REGIONS`` crops: cards can sit anywhere. The
warp normalizes scale and orientation, so downstream rank/suit reading is far
more stable than reading a raw screen crop.

Assumes upright cards (the long axis vertical). Heavily rotated or upside-down
cards are out of scope for this temporary detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config


@dataclass
class DetectedCard:
    quad: np.ndarray  # 4x2 float32, ordered tl, tr, br, bl
    warped: np.ndarray = field(repr=False)  # canonical BGR card image


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl has smallest x+y
    rect[2] = pts[np.argmax(s)]  # br has largest x+y
    d = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(d)]  # tr has smallest y-x
    rect[3] = pts[np.argmax(d)]  # bl has largest y-x
    return rect


def warp_card(frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-warp a quad to the canonical card size, kept portrait."""
    w, h = config.CARD_WARP_SIZE
    rect = order_points(quad)

    # If the quad is wider than it is tall, the card is sideways — rotate the
    # destination so the result stays portrait.
    (tl, tr, br, bl) = rect
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if width > height:
        dst = np.array([[w, 0], [w, h], [0, h], [0, 0]], dtype="float32")
    else:
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(frame, M, (w, h))


def find_cards(frame: np.ndarray) -> list[DetectedCard]:
    """Detect card-shaped quads in a frame, left-to-right.

    Pipeline: grayscale -> blur -> Canny edges -> dilate to close the outline
    -> external contours -> keep large 4-corner convex quads.
    """
    H, W = frame.shape[:2]
    frame_area = float(H * W)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, *config.DETECT_CANNY)
    edges = cv2.dilate(edges, None, iterations=config.DETECT_DILATE_ITERS)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards: list[DetectedCard] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < config.DETECT_MIN_AREA_FRAC * frame_area:
            continue
        if area > config.DETECT_MAX_AREA_FRAC * frame_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, config.DETECT_APPROX_EPS * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = approx.reshape(4, 2).astype("float32")
        cards.append(DetectedCard(order_points(quad), warp_card(frame, quad)))

    # Stable left-to-right ordering by the top-left corner's x.
    cards.sort(key=lambda d: d.quad[0][0])
    return cards
