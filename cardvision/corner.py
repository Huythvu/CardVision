"""Extract the rank and suit glyphs from a card's top-left corner.

Pipeline per card crop:
  1. Take the top-left corner strip (rank stacked above suit).
  2. Threshold to isolate ink.
  3. Split the strip vertically into the rank glyph (top) and suit glyph
     (bottom) using the row-wise ink profile, then tight-crop each.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config


def extract_corner(card: np.ndarray) -> np.ndarray:
    """Return the resized top-left corner strip of a card crop."""
    h, w = card.shape[:2]
    cw = max(1, int(w * config.CORNER_WIDTH_FRAC))
    ch = max(1, int(h * config.CORNER_HEIGHT_FRAC))
    corner = card[0:ch, 0:cw]
    return cv2.resize(corner, config.CORNER_SIZE, interpolation=cv2.INTER_CUBIC)


def binarize(img: np.ndarray) -> np.ndarray:
    """Grayscale -> binary with ink as white (255) on black background."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if config.BINARY_THRESHOLD <= 0:
        _, binimg = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        _, binimg = cv2.threshold(
            gray, config.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV
        )
    return binimg


def _tight_crop(binimg: np.ndarray) -> np.ndarray | None:
    """Crop to the bounding box of all ink, or None if the strip is blank."""
    ys, xs = np.where(binimg > 0)
    if len(xs) == 0:
        return None
    return binimg[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def split_rank_suit(corner: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Split a corner strip into (rank_glyph, suit_glyph), each normalized.

    Uses the gap in the vertical ink profile between the rank (top) and the
    suit pip (bottom) as the split point.
    """
    binimg = binarize(corner)

    # Row-wise ink counts; find the largest empty band in the middle third.
    row_ink = binimg.sum(axis=1)
    h = len(row_ink)
    search_lo, search_hi = int(h * 0.30), int(h * 0.80)
    band = row_ink[search_lo:search_hi]

    if band.size and band.min() == 0:
        # Pick the empty row nearest the vertical center of the search band.
        empty = np.where(band == 0)[0]
        center = band.size / 2
        split = search_lo + int(empty[np.argmin(np.abs(empty - center))])
    else:
        split = h // 2  # fallback: halve the strip

    rank = _tight_crop(binimg[:split, :])
    suit = _tight_crop(binimg[split:, :])

    rank = _normalize(rank)
    suit = _normalize(suit)
    return rank, suit


def _normalize(glyph: np.ndarray | None) -> np.ndarray | None:
    """Resize an isolated glyph to the fixed matching size."""
    if glyph is None or glyph.size == 0:
        return None
    return cv2.resize(glyph, config.GLYPH_SIZE, interpolation=cv2.INTER_CUBIC)
