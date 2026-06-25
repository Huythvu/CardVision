# CardVision

Local computer-vision tool that identifies standard playing cards from an
image or live screen capture and outputs clean labels like `A♠`, `10♥`, `K♦`.

Pure, local, free: Python + OpenCV only — no cloud or paid vision APIs.

## Scope (v1)

- General card identification only (no blackjack / poker / strategy).
- Two ways to locate cards:
  - **Fixed/manual crop positions** (`CARD_REGIONS`) — deterministic.
  - **Automatic detection** (`--auto`) — finds card-shaped quads and
    perspective-warps each to a canonical upright image, so cards can sit
    anywhere. Great for testing; assumes roughly upright cards.
- Cards are identified by reading the **top-left corner**: rank + suit.
  (Auto-detection warps to a canonical size first, so the corner is always
  in the same place — far more stable than reading a raw screen crop.)
- Classification is **template matching** (not OCR) for robustness on
  stylized single glyphs. Suit color (red vs black) is used as a prior.

## How it works

```
frame (screen grab or image)
  → crop fixed card slots            (config.CARD_REGIONS)
    → extract top-left corner        (corner.extract_corner)
      → split into rank + suit glyph (corner.split_rank_suit)
        → match each vs templates    (classify.classify_card)
          → format                   (formatter → "A♠")
```

Because exact-render templates match best, you **calibrate** once against
your own source, then run against it.

## Install

```bash
pip install -r requirements.txt
```

## Desktop UI (Phase 1)

A PySide6 window with live preview, Start/Stop, drag-to-select capture region,
and a backend switch (detect / template / ml). Isolated from the core — the
CLI keeps working unchanged.

```bash
pip install -r requirements-ui.txt
python main.py ui
```

- **Select Region**: drag a rectangle over the screen area to capture.
- **Start/Stop**: toggle the live capture+detection loop.
- **Backend**: `detect` (outline cards, no templates needed), `template`
  (corner reading), or `ml` (local YOLO).
- **Interval**: capture cadence.

Planned: Phase 2 settings/profiles, Phase 3 visual card-learning trainer,
Phase 4 output panel + packaging to .exe.

## Usage (CLI)

### Quick start with automatic detection (recommended for testing)

No fixed coordinates needed — just get cards into the capture region.

```bash
python main.py preview --auto              # see what gets detected (no templates)
python main.py gen-templates               # instant starter knowledge (synthetic)
python main.py run --auto --loop --show    # identify everything, live
python main.py run --auto --loop --show --debug   # + glyph diagnostic panel
```

### Card knowledge (templates)

CardVision matches cards against reference glyphs (13 ranks + 4 suits = 17).
Three ways to populate them, lowest-effort first:

```bash
# A. Synthetic starter set — works instantly, but it's only a baseline; a font
#    that differs from your cards will score low and report '??' (never a wrong
#    guess). Re-run with --force to overwrite.
python main.py gen-templates

# B. Guided walkthrough — capture all 17 glyphs from YOUR source in one pass.
#    Highest accuracy. SPACE captures, N skips, B goes back, Q quits.
python main.py calibrate-all

# C. One glyph at a time (fixed slot or --auto):
python main.py calibrate --auto --rank A --suit S
```

You can mix these: generate a starter set, then overwrite individual glyphs
with `calibrate` where the synthetic one matches poorly.

Detection tuning lives in `config.py` under "Automatic card detection"
(`DETECT_*`, `CARD_WARP_SIZE`).

### Live / messy feeds: local ML detector (`--ml`)

Classic CV (corner reading) is great for clean digital cards but struggles
with **live-table feeds**: overlapping fanned hands, perspective, busy felt,
small glyphs. For those, use a locally-run YOLO detector — **100% local and
free** (no cloud, no API, no tokens), just heavier deps.

```bash
pip install -r requirements-ml.txt          # ultralytics (pulls PyTorch)
# get weights: drop a trained model at models/cards.pt, OR train one:
python train_yolo.py --data path/to/data.yaml --epochs 80 --export models/cards.pt

python main.py preview --ml --image samples/blackjack.png   # see detections
python main.py run --ml --image samples/blackjack.png        # print labels
python main.py run --ml --loop --show                        # live
```

- Get a **52-class playing-card dataset** (YOLO format) with class names as
  card codes (`AS`, `KH`, `10D`). `mldetect.parse_card_code` also understands
  word forms ("ace of spades"); add odd names to `config.ML_CLASS_MAP`.
- Tune `ML_MODEL_PATH` / `ML_CONF_THRESHOLD` in `config.py`.

### Fixed-region mode

1. **Aim the capture.** Edit `cardvision/config.py`:
   - `SCREEN_REGION` — the monitor area to grab.
   - `CARD_REGIONS` — one `(x, y, w, h)` rectangle per card slot within the
     captured frame.

   Tune without a display by dumping crops to disk:
   ```bash
   python main.py show-crops                 # from screen
   python main.py show-crops --image hand.png
   ```
   Inspect `crops/frame.png`, `crops/card_1.png`, `crops/card_1_corner.png`.

2. **Calibrate templates** (once per glyph). Point a slot at a known card:
   ```bash
   python main.py calibrate --slot card_1 --rank A --suit S
   python main.py calibrate --slot card_1 --rank K --suit H
   # ...repeat to cover the 13 ranks and 4 suits you need
   ```
   Add `--image hand.png` to calibrate from a file instead of the screen.

3. **Run** the identifier:
   ```bash
   python main.py run                 # one screen grab
   python main.py run --loop          # continuous
   python main.py run --image hand.png
   python main.py run --ascii         # S/H/D/C instead of ♠♥♦♣
   ```
   Output: `card_1=A♠  card_2=K♥`. Low-confidence results show as `??`.

## Configuration knobs (`cardvision/config.py`)

- `CORNER_WIDTH_FRAC` / `CORNER_HEIGHT_FRAC` — size of the corner strip.
- `GLYPH_SIZE` — normalized glyph size used for matching.
- `MIN_MATCH_CONFIDENCE` — below this, a card is reported as `??`.
- `BINARY_THRESHOLD` — `0` uses Otsu auto-threshold.

## Roadmap

- [x] Automatic card detection (contour finding + perspective warp).
- [x] Confidence / overlay debug view.
- [x] Local ML detector (YOLO) for live/messy feeds.
- [x] Desktop UI Phase 1: live preview, Start/Stop, drag-to-select region.
- [ ] UI Phase 2: live settings + saved config profiles.
- [ ] UI Phase 3: visual card-learning trainer.
- [ ] UI Phase 4: output panel, status, packaging to .exe.
- [ ] Whole-card template matching as an alternative to corner reading.
- [ ] Multi-template averaging per glyph for robustness across decks.
- [ ] Camera-feed source.

## Project layout

```
main.py                 CLI: calibrate / run / show-crops
cardvision/
  config.py             labels, crop regions, detection + match thresholds
  capture.py            screen grab (mss) + image loading + cropping
  detect.py             auto card detection: contours → 4-corner warp
  mldetect.py           local YOLO detector + card-code parsing
  corner.py             corner extraction, binarize, rank/suit split
  synth.py              synthetic starter-template generator
  templates.py          load/save reference glyphs
  classify.py           template matching → CardResult
  viz.py                region/detection/ML overlays + glyph debug panel
  gui.py                PySide6 desktop UI (Phase 1)
  formatter.py          CardResult → "A♠"
train_yolo.py           train/fine-tune a local YOLO card model
templates/              calibrated reference glyphs (generated)
```
