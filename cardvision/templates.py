"""Load and save reference rank/suit templates used for matching.

Templates are normalized binary glyphs (white ink on black) stored as PNGs:
  templates/ranks/A.png, templates/ranks/10.png, ...
  templates/suits/S.png, templates/suits/H.png, ...

Build them once with the ``calibrate`` command (see main.py), then reuse.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from . import config

RANK_DIR = os.path.join(config.TEMPLATE_DIR, "ranks")
SUIT_DIR = os.path.join(config.TEMPLATE_DIR, "suits")


def _ensure_dirs() -> None:
    os.makedirs(RANK_DIR, exist_ok=True)
    os.makedirs(SUIT_DIR, exist_ok=True)


def save_rank(label: str, glyph: np.ndarray) -> str:
    _ensure_dirs()
    path = os.path.join(RANK_DIR, f"{label}.png")
    cv2.imwrite(path, glyph)
    return path


def save_suit(label: str, glyph: np.ndarray) -> str:
    _ensure_dirs()
    path = os.path.join(SUIT_DIR, f"{label}.png")
    cv2.imwrite(path, glyph)
    return path


def _load_dir(dir_path: str, labels: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for label in labels:
        path = os.path.join(dir_path, f"{label}.png")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                out[label] = cv2.resize(img, config.GLYPH_SIZE)
    return out


def load_rank_templates() -> dict[str, np.ndarray]:
    return _load_dir(RANK_DIR, config.RANKS)


def load_suit_templates() -> dict[str, np.ndarray]:
    return _load_dir(SUIT_DIR, config.SUITS)


def templates_available() -> bool:
    return bool(load_rank_templates()) and bool(load_suit_templates())
