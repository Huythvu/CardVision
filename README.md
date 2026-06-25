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

## Usage

### Quick start with automatic detection (recommended for testing)

No fixed coordinates needed — just get cards into the capture region.

```bash
python main.py preview --auto              # see what gets detected (no templates)
python main.py calibrate --auto --rank A --suit S   # show ONE card, label it
# ...repeat calibrate for the ranks/suits you need...
python main.py run --auto --loop --show    # identify everything, live
python main.py run --auto --loop --show --debug   # + glyph diagnostic panel
```

Detection tuning lives in `config.py` under "Automatic card detection"
(`DETECT_*`, `CARD_WARP_SIZE`).

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
- [ ] Whole-card template matching as an alternative to corner reading.
- [ ] Multi-template averaging per glyph for robustness across decks.
- [ ] Camera-feed source.
- [ ] Dedicated app shell (GUI / packaged executable).

## Project layout

```
main.py                 CLI: calibrate / run / show-crops
cardvision/
  config.py             labels, crop regions, detection + match thresholds
  capture.py            screen grab (mss) + image loading + cropping
  detect.py             auto card detection: contours → 4-corner warp
  corner.py             corner extraction, binarize, rank/suit split
  templates.py          load/save reference glyphs
  classify.py           template matching → CardResult
  viz.py                region/detection overlays + glyph debug panel
  formatter.py          CardResult → "A♠"
templates/              calibrated reference glyphs (generated)
```
