"""Configuration: card labels, fixed crop regions, and tunable constants.

Crop regions are intentionally fixed (manual) for v1 per the project plan.
Automatic card detection comes later. Adjust ``CARD_REGIONS`` and
``SCREEN_REGION`` to match your source, then calibrate templates.
"""

from __future__ import annotations

import os

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "templates")

# --- Card vocabulary -----------------------------------------------------
# 13 ranks. "10" is the only two-character rank; treat it specially anywhere
# single-glyph assumptions are made.
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["S", "H", "D", "C"]  # spades, hearts, diamonds, clubs

# Pretty unicode suit symbols for clean output (A♠, 10♥, K♦, ...).
SUIT_SYMBOLS = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
RED_SUITS = {"H", "D"}

# --- Screen capture region (mss) -----------------------------------------
# Pixels of the monitor to grab. Set to the area where cards appear.
# left/top is the top-left corner; width/height the size.
SCREEN_REGION = {"left": 0, "top": 0, "width": 1280, "height": 720}

# --- Fixed card crop regions ---------------------------------------------
# Rectangles (x, y, w, h) WITHIN the captured frame, one per card slot.
# These are placeholders — set them to where each card sits in your source.
CARD_REGIONS = {
    "card_1": (40, 40, 140, 200),
    "card_2": (200, 40, 140, 200),
}

# --- Corner geometry -----------------------------------------------------
# The rank+suit live in the top-left corner. These fractions define how much
# of the card's width/height the corner strip occupies.
CORNER_WIDTH_FRAC = 0.22
CORNER_HEIGHT_FRAC = 0.30

# Working size each corner crop is resized to before contour splitting.
# Keeps template matching scale-stable.
CORNER_SIZE = (200, 280)  # (width, height)

# Size each isolated rank / suit glyph is normalized to for matching.
GLYPH_SIZE = (70, 100)  # (width, height)

# --- Matching thresholds -------------------------------------------------
# Binary threshold for separating ink from background (0 = use Otsu).
BINARY_THRESHOLD = 0

# Minimum match confidence (0..1) below which a result is reported as unknown.
MIN_MATCH_CONFIDENCE = 0.55

# --- Automatic card detection --------------------------------------------
# Canonical flat-card size each detected card is perspective-warped to.
# Detection normalizes scale/orientation, so downstream corner reading is
# stable regardless of where the card sat in the frame. (width, height)
CARD_WARP_SIZE = (200, 300)

# A detected quadrilateral counts as a card when its area is within this
# fraction of the whole frame (filters out tiny noise and full-frame blobs).
DETECT_MIN_AREA_FRAC = 0.01
DETECT_MAX_AREA_FRAC = 0.90

# Canny edge thresholds and the dilation iterations used to close gaps in the
# card outline before contour finding.
DETECT_CANNY = (50, 150)
DETECT_DILATE_ITERS = 2

# approxPolyDP epsilon as a fraction of contour perimeter (corner snapping).
DETECT_APPROX_EPS = 0.02
