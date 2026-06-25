"""Generate a starter set of rank/suit templates synthetically.

This gives CardVision card knowledge with zero calibration so it works out of
the box. Glyphs are rendered with OpenCV only (no font files, no extra deps),
then pushed through the same binarize -> tight-crop -> normalize pipeline as
real captured glyphs, so they live in the same representation as calibrated
templates.

Accuracy note: synthetic glyphs are a baseline. For best results on a specific
source, overwrite individual glyphs with `calibrate` / `calibrate-all`.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config
from .corner import binarize

# Render on a generous canvas, then the pipeline tight-crops + normalizes.
_CANVAS = (220, 300)  # (w, h)
_INK = (0, 0, 0)


def _finalize(canvas: np.ndarray) -> np.ndarray | None:
    """Binarize a black-on-white canvas and normalize to GLYPH_SIZE."""
    binimg = binarize(canvas)
    ys, xs = np.where(binimg > 0)
    if len(xs) == 0:
        return None
    crop = binimg[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    return cv2.resize(crop, config.GLYPH_SIZE, interpolation=cv2.INTER_CUBIC)


def render_rank(rank: str) -> np.ndarray | None:
    """Render a rank string (A,2..10,J,Q,K) as a normalized binary glyph."""
    w, h = _CANVAS
    canvas = np.full((h, w, 3), 255, np.uint8)
    scale = 5.0 if len(rank) == 1 else 3.5  # shrink the two-char "10" to fit
    thick = 8 if len(rank) == 1 else 6
    (tw, th), base = cv2.getTextSize(rank, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    org = ((w - tw) // 2, (h + th) // 2)
    cv2.putText(canvas, rank, org, cv2.FONT_HERSHEY_DUPLEX, scale, _INK, thick, cv2.LINE_AA)
    return _finalize(canvas)


def _suit_canvas() -> np.ndarray:
    w, h = _CANVAS
    return np.full((h, w, 3), 255, np.uint8)


def render_suit(suit: str) -> np.ndarray | None:
    """Render a suit symbol (S,H,D,C) as a normalized binary glyph."""
    w, h = _CANVAS
    cx, cy = w // 2, h // 2
    canvas = _suit_canvas()
    s = 70  # nominal half-size

    if suit == "D":  # diamond
        pts = np.array([[cx, cy - s], [cx + s * 0.7, cy], [cx, cy + s], [cx - s * 0.7, cy]])
        cv2.fillConvexPoly(canvas, pts.astype(np.int32), _INK)

    elif suit == "H":  # heart: two lobes + point
        r = int(s * 0.5)
        cv2.circle(canvas, (cx - r // 1, cy - r // 2), r, _INK, -1)
        cv2.circle(canvas, (cx + r // 1, cy - r // 2), r, _INK, -1)
        pts = np.array([[cx - s, cy - r // 2], [cx + s, cy - r // 2], [cx, cy + s]])
        cv2.fillConvexPoly(canvas, pts.astype(np.int32), _INK)

    elif suit == "S":  # spade: inverted heart + stem
        r = int(s * 0.5)
        cv2.circle(canvas, (cx - r, cy + r // 2), r, _INK, -1)
        cv2.circle(canvas, (cx + r, cy + r // 2), r, _INK, -1)
        pts = np.array([[cx - s, cy + r // 2], [cx + s, cy + r // 2], [cx, cy - s]])
        cv2.fillConvexPoly(canvas, pts.astype(np.int32), _INK)
        stem = np.array([[cx - s // 3, cy + s], [cx + s // 3, cy + s],
                         [cx + s // 6, cy + r // 2], [cx - s // 6, cy + r // 2]])
        cv2.fillConvexPoly(canvas, stem.astype(np.int32), _INK)

    elif suit == "C":  # club: three lobes + stem
        r = int(s * 0.45)
        cv2.circle(canvas, (cx, cy - r), r, _INK, -1)
        cv2.circle(canvas, (cx - r, cy + r // 2), r, _INK, -1)
        cv2.circle(canvas, (cx + r, cy + r // 2), r, _INK, -1)
        stem = np.array([[cx - s // 3, cy + s], [cx + s // 3, cy + s],
                         [cx + s // 6, cy], [cx - s // 6, cy]])
        cv2.fillConvexPoly(canvas, stem.astype(np.int32), _INK)
    else:
        return None

    return _finalize(canvas)


def generate(force: bool = False) -> tuple[list[str], list[str]]:
    """Generate and save the full starter set.

    Returns (saved_labels, skipped_labels). Existing templates are kept unless
    ``force`` is True, so calibrated glyphs are not clobbered.
    """
    import os

    from . import templates

    saved, skipped = [], []

    def maybe_save(kind, label, glyph):
        path = os.path.join(
            templates.RANK_DIR if kind == "rank" else templates.SUIT_DIR,
            f"{label}.png",
        )
        if glyph is None:
            skipped.append(f"{label} (render failed)")
            return
        if os.path.exists(path) and not force:
            skipped.append(f"{label} (exists)")
            return
        (templates.save_rank if kind == "rank" else templates.save_suit)(label, glyph)
        saved.append(label)

    for rank in config.RANKS:
        maybe_save("rank", rank, render_rank(rank))
    for suit in config.SUITS:
        maybe_save("suit", suit, render_suit(suit))
    return saved, skipped
