"""Local ML card detection via a YOLO model (ultralytics).

Runs fully on your machine — no cloud, no API, no tokens. Robust to overlap,
perspective and lighting, which is what a live-table feed needs.

Get a model one of two ways:
  - drop pretrained weights at ``config.ML_MODEL_PATH`` (default models/cards.pt)
  - train one on a public 52-class playing-card dataset (see train_yolo.py)

YOLO class names are expected to be card codes like ``AS``, ``KH``, ``10D``.
``parse_card_code`` also understands word forms ("ace of spades") and a few
common variants; anything unrecognized is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config

# Word/synonym maps for non-coded class names.
_RANK_WORDS = {
    "ace": "A", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "jack": "J", "queen": "Q", "king": "K",
}
_SUIT_WORDS = {"spade": "S", "heart": "H", "diamond": "D", "club": "C"}


@dataclass
class MLDetection:
    rank: str
    suit: str
    conf: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2

    @property
    def quad(self) -> np.ndarray:
        """Axis-aligned box as a 4-point quad (so viz overlays can reuse it)."""
        x1, y1, x2, y2 = self.bbox
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")


def parse_card_code(name: str) -> tuple[str, str] | None:
    """Map a YOLO class name to (rank, suit), or None if not a card.

    Handles 'AS', '10d', 'ace of spades', 'ace_spades', etc.
    """
    if not name:
        return None
    raw = name.strip()

    # Explicit override map from config wins.
    if raw in config.ML_CLASS_MAP:
        return config.ML_CLASS_MAP[raw]

    s = raw.upper()
    # Compact code form: rank chars + trailing suit letter (AS, 10D, KH).
    if len(s) >= 2 and s[-1] in ("S", "H", "D", "C"):
        rank, suit = s[:-1], s[-1]
        if rank in config.RANKS:
            return rank, suit

    # Word form: "ace of spades", "ace_of_spades", "tenhearts", ...
    low = raw.lower()
    rank = next((v for w, v in _RANK_WORDS.items() if w in low), None)
    suit = next((v for w, v in _SUIT_WORDS.items() if w in low), None)
    if rank and suit:
        return rank, suit
    return None


class MLCardDetector:
    """Lazy wrapper around an ultralytics YOLO model."""

    def __init__(self, model_path: str | None = None, conf: float | None = None):
        self.model_path = model_path or config.ML_MODEL_PATH
        self.conf = conf if conf is not None else config.ML_CONF_THRESHOLD
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            import os

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"No model at '{self.model_path}'. Place pretrained weights "
                    f"there or train one (see train_yolo.py)."
                )
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, frame: np.ndarray) -> list[MLDetection]:
        """Run detection on a BGR frame; return cards left-to-right."""
        model = self._ensure_model()
        # ultralytics accepts BGR numpy arrays directly.
        results = model.predict(frame, conf=self.conf, verbose=False)
        names = model.names  # class index -> name

        dets: list[MLDetection] = []
        for res in results:
            if res.boxes is None:
                continue
            for box in res.boxes:
                cls = int(box.cls[0])
                parsed = parse_card_code(names.get(cls, str(cls)))
                if parsed is None:
                    continue
                rank, suit = parsed
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                dets.append(MLDetection(rank, suit, float(box.conf[0]), (x1, y1, x2, y2)))

        dets.sort(key=lambda d: d.bbox[0])
        return dets
