"""Visual overlays: highlight the captured card regions and show, for each
slot, the detected label, confidence, and the glyphs the matcher actually saw.

Note: OpenCV's built-in (Hershey) font cannot render unicode suit symbols, so
on-image text always uses ASCII suits (S/H/D/C). The terminal output still
uses ♠♥♦♣.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config, formatter

GREEN = (0, 200, 0)
AMBER = (0, 165, 255)
GREY = (160, 160, 160)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate_frame(frame: np.ndarray, results: dict | None = None) -> np.ndarray:
    """Draw a box + label around each configured card region.

    ``results`` maps slot -> CardResult; if None, just outlines the regions
    (useful for aligning crops before any templates exist).
    """
    out = frame.copy()
    for slot, (x, y, w, h) in config.CARD_REGIONS.items():
        color = GREY
        label = slot
        if results and slot in results:
            r = results[slot]
            color = GREEN if r.confident else AMBER
            text = formatter.format_card(r, ascii_only=True)
            label = f"{slot}: {text}  r{r.rank_score:.2f} s{r.suit_score:.2f}"
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        _text_with_bg(out, label, (x, max(14, y - 8)), color)
    return out


def annotate_detections(frame: np.ndarray, cards: list, results: list | None = None) -> np.ndarray:
    """Outline each auto-detected card quad and label it.

    ``cards`` is a list of detect.DetectedCard; ``results`` (optional) is a
    parallel list of CardResult. With no results, just outlines detections
    (no templates needed) — handy for testing placement.
    """
    out = frame.copy()
    for i, card in enumerate(cards):
        r = results[i] if results and i < len(results) else None
        color = GREY
        label = f"card {i + 1}"
        if r is not None:
            color = GREEN if r.confident else AMBER
            text = formatter.format_card(r, ascii_only=True)
            label = f"{text}  r{r.rank_score:.2f} s{r.suit_score:.2f}"
        pts = card.quad.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, color, 2)
        tl = card.quad[0].astype(int)
        _text_with_bg(out, label, (int(tl[0]), max(14, int(tl[1]) - 8)), color)
    return out


def _text_with_bg(img, text, org, color, scale=0.5, thick=1):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = org
    cv2.rectangle(img, (x, y - th - base), (x + tw, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _to_bgr(glyph: np.ndarray | None, size) -> np.ndarray:
    """Render a (possibly missing) binary glyph as a fixed-size BGR tile."""
    w, h = size
    tile = np.zeros((h, w, 3), np.uint8)
    if glyph is None or glyph.size == 0:
        cv2.putText(tile, "none", (4, h // 2), FONT, 0.4, GREY, 1, cv2.LINE_AA)
        return tile
    g = cv2.resize(glyph, size)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def debug_panel(results: dict) -> np.ndarray:
    """Compose a panel showing each slot's rank/suit glyph + scores stacked."""
    tile = config.GLYPH_SIZE  # (w, h)
    rows = []
    pad = 8
    label_h = 22
    for slot, r in results.items():
        rank_tile = _to_bgr(r.rank_glyph, tile)
        suit_tile = _to_bgr(r.suit_glyph, tile)
        gap = np.zeros((tile[1], pad, 3), np.uint8)
        glyphs = np.hstack([rank_tile, gap, suit_tile])

        text = formatter.format_card(r, ascii_only=True)
        color = GREEN if r.confident else AMBER
        header = np.zeros((label_h, glyphs.shape[1], 3), np.uint8)
        cv2.putText(header, f"{slot}: {text}", (2, 16), FONT, 0.5, color, 1, cv2.LINE_AA)
        rows.append(np.vstack([header, glyphs]))
        rows.append(np.zeros((pad, glyphs.shape[1], 3), np.uint8))

    if not rows:
        return np.zeros((40, 160, 3), np.uint8)
    return np.vstack(rows)
