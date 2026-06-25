# CardVision

Local computer-vision tool that identifies standard playing cards from an
image or live screen capture and outputs clean labels like `A♠`, `10♥`, `K♦`.

Pure, local, free: Python + OpenCV only — no cloud or paid vision APIs.

## Scope (v1)

- General card identification only (no blackjack / poker / strategy).
- **Fixed/manual crop positions** for each card slot (automatic card
  detection comes later).
- Cards are identified by reading the **top-left corner**: rank + suit.
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

- [ ] Automatic card detection (contour finding + perspective warp).
- [ ] Multi-template averaging per glyph for robustness across decks.
- [ ] Camera-feed source.
- [ ] Confidence/overlay debug view.

## Project layout

```
main.py                 CLI: calibrate / run / show-crops
cardvision/
  config.py             labels, crop regions, thresholds
  capture.py            screen grab (mss) + image loading + cropping
  corner.py             corner extraction, binarize, rank/suit split
  templates.py          load/save reference glyphs
  classify.py           template matching → CardResult
  formatter.py          CardResult → "A♠"
templates/              calibrated reference glyphs (generated)
```
