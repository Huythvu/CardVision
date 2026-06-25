"""Classify a card crop into a rank and suit via template matching.

Matching score is the normalized cross-correlation peak between a normalized
glyph and each reference template; the best label wins. Suit color (red vs
black) is used as a prior to disambiguate similarly shaped pips.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config, corner, templates


@dataclass
class CardResult:
    rank: str | None
    suit: str | None
    rank_score: float
    suit_score: float
    is_red: bool

    @property
    def confident(self) -> bool:
        return (
            self.rank is not None
            and self.suit is not None
            and self.rank_score >= config.MIN_MATCH_CONFIDENCE
            and self.suit_score >= config.MIN_MATCH_CONFIDENCE
        )


def _best_match(glyph: np.ndarray, refs: dict[str, np.ndarray]) -> tuple[str | None, float]:
    """Return (label, score) of the best-matching template, or (None, 0)."""
    if glyph is None or not refs:
        return None, 0.0
    best_label, best_score = None, -1.0
    for label, ref in refs.items():
        res = cv2.matchTemplate(glyph, ref, cv2.TM_CCOEFF_NORMED)
        score = float(res.max())
        if score > best_score:
            best_label, best_score = label, score
    return best_label, max(0.0, best_score)


def _is_red_corner(card: np.ndarray) -> bool:
    """Detect whether the corner ink is red (hearts/diamonds)."""
    c = corner.extract_corner(card)
    binimg = corner.binarize(c)
    mask = binimg > 0
    if not mask.any():
        return False
    bgr = c.reshape(-1, 3)[mask.reshape(-1)]
    b, g, r = bgr[:, 0].mean(), bgr[:, 1].mean(), bgr[:, 2].mean()
    return r > g + 25 and r > b + 25


def classify_card(
    card: np.ndarray,
    rank_refs: dict[str, np.ndarray],
    suit_refs: dict[str, np.ndarray],
) -> CardResult:
    """Classify a single card crop."""
    c = corner.extract_corner(card)
    rank_glyph, suit_glyph = corner.split_rank_suit(c)

    is_red = _is_red_corner(card)

    rank, rank_score = _best_match(rank_glyph, rank_refs)

    # Restrict suit candidates by detected color, then match shape.
    if is_red:
        candidates = {k: v for k, v in suit_refs.items() if k in config.RED_SUITS}
    else:
        candidates = {k: v for k, v in suit_refs.items() if k not in config.RED_SUITS}
    candidates = candidates or suit_refs  # fall back if color filtering empties it
    suit, suit_score = _best_match(suit_glyph, candidates)

    return CardResult(rank, suit, rank_score, suit_score, is_red)
