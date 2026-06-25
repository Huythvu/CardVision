#!/usr/bin/env python3
"""CardVision CLI: identify standard playing cards from screen captures / images.

Commands
--------
  calibrate   Capture rank & suit glyphs from a known card and save them as
              reference templates. Run once per glyph you want to recognize.
  run         Capture/load a frame, classify the configured card slots, and
              print clean labels (A♠, 10♥, K♦, ...).
  show-crops  Save the configured card + corner crops to disk so you can tune
              CARD_REGIONS / SCREEN_REGION without a display.

Examples
--------
  # Save the rank+suit of the ace of spades sitting in slot card_1 on screen:
  python main.py calibrate --slot card_1 --rank A --suit S

  # Same from an image file:
  python main.py calibrate --image hand.png --slot card_1 --rank K --suit H

  # Classify everything once:
  python main.py run

  # Continuously classify the screen:
  python main.py run --loop
"""

from __future__ import annotations

import argparse
import time

from cardvision import capture, config, formatter, templates
from cardvision.classify import classify_card
from cardvision.corner import extract_corner, split_rank_suit


def _get_frame(image_path: str | None):
    if image_path:
        return capture.load_image(image_path)
    return capture.grab_screen()


def _largest_card(cards):
    """Pick the detected card with the largest quad (the one held up front)."""
    import cv2

    if not cards:
        return None
    return max(cards, key=lambda d: cv2.contourArea(d.quad.astype("float32")))


def cmd_gen_templates(args: argparse.Namespace) -> int:
    """Generate a synthetic starter set so the program works with no calibration."""
    from cardvision import synth

    saved, skipped = synth.generate(force=args.force)
    print(f"Generated {len(saved)} template(s): {', '.join(saved) or '(none)'}")
    if skipped:
        print(f"Skipped {len(skipped)}: {', '.join(skipped)}")
        if any("exists" in s for s in skipped):
            print("  (use --force to overwrite existing/calibrated templates)")
    return 0


def cmd_calibrate_all(args: argparse.Namespace) -> int:
    """Guided walkthrough: capture every rank then every suit from your source.

    Shows the live detected card; press SPACE to capture the highlighted glyph,
    N to skip, B to go back, Q/Esc to quit.
    """
    import cv2
    import numpy as np

    from cardvision import detect

    targets = [("rank", r) for r in config.RANKS] + [("suit", s) for s in config.SUITS]
    win = "cardvision-calibrate-all"
    i = 0
    while 0 <= i < len(targets):
        kind, label = targets[i]
        frame = _get_frame(args.image)
        card = _largest_card(detect.find_cards(frame))

        glyph = None
        warped = None
        if card is not None:
            warped = card.warped
            rank_g, suit_g = split_rank_suit(extract_corner(warped))
            glyph = rank_g if kind == "rank" else suit_g

        disp = _calib_display(warped, glyph, i, len(targets), kind, label)
        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("n"),):
            i += 1
        elif key in (ord("b"),):
            i = max(0, i - 1)
        elif key in (32, 13):  # space / enter -> capture
            if glyph is None:
                print(f"[{label}] no glyph to capture (no card / blank corner).")
                continue
            if kind == "rank":
                templates.save_rank(label, glyph)
            else:
                templates.save_suit(label, glyph)
            print(f"[{label}] captured.")
            i += 1
    cv2.destroyAllWindows()
    print("Calibration walkthrough ended.")
    return 0


def _calib_display(warped, glyph, idx, total, kind, label):
    """Compose the guided-calibration window: card | candidate glyph | prompt."""
    import cv2
    import numpy as np

    from cardvision import viz

    panel_w = 520
    disp = np.zeros((360, panel_w, 3), np.uint8)
    # Left: the detected/warped card (or a placeholder).
    if warped is not None:
        card_view = cv2.resize(warped, (180, 270))
    else:
        card_view = np.zeros((270, 180, 3), np.uint8)
        cv2.putText(card_view, "no card", (40, 135), viz.FONT, 0.6, viz.GREY, 1, cv2.LINE_AA)
    disp[70:340, 20:200] = card_view
    # Right: the candidate glyph that SPACE would save.
    gtile = viz._to_bgr(glyph, (120, 170))
    disp[70:240, 240:360] = gtile

    color = viz.GREEN if glyph is not None else viz.AMBER
    cv2.putText(disp, f"[{idx + 1}/{total}] Show a card, capture {kind.upper()} = {label}",
                (20, 30), viz.FONT, 0.6, color, 1, cv2.LINE_AA)
    cv2.putText(disp, "SPACE capture   N skip   B back   Q quit",
                (20, 55), viz.FONT, 0.5, viz.GREY, 1, cv2.LINE_AA)
    cv2.putText(disp, "candidate ->", (240, 60), viz.FONT, 0.5, viz.GREY, 1, cv2.LINE_AA)
    return disp


def cmd_calibrate(args: argparse.Namespace) -> int:
    frame = _get_frame(args.image)
    if args.auto:
        from cardvision import detect

        cards = detect.find_cards(frame)
        if not cards:
            print("No card detected in frame.")
            return 1
        if len(cards) > 1:
            print(f"! {len(cards)} cards detected; using the left-most. "
                  "Show one card at a time when calibrating with --auto.")
        card = cards[0].warped
    else:
        if not args.slot:
            print("Pass --slot <name> (fixed mode) or --auto (detect the card).")
            return 1
        if args.slot not in config.CARD_REGIONS:
            print(f"Unknown slot '{args.slot}'. Known: {list(config.CARD_REGIONS)}")
            return 1
        card = capture.crop(frame, config.CARD_REGIONS[args.slot])
    corner = extract_corner(card)
    rank_glyph, suit_glyph = split_rank_suit(corner)

    saved = []
    if args.rank:
        if rank_glyph is None:
            print("! Could not isolate a rank glyph (corner blank?).")
        else:
            saved.append(templates.save_rank(args.rank, rank_glyph))
    if args.suit:
        if suit_glyph is None:
            print("! Could not isolate a suit glyph (corner blank?).")
        else:
            saved.append(templates.save_suit(args.suit, suit_glyph))

    if saved:
        print("Saved templates:")
        for p in saved:
            print(f"  {p}")
    else:
        print("Nothing saved. Pass --rank and/or --suit.")
    return 0


def cmd_show_crops(args: argparse.Namespace) -> int:
    import os

    import cv2

    frame = _get_frame(args.image)
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "frame.png"), frame)
    for slot, rect in config.CARD_REGIONS.items():
        card = capture.crop(frame, rect)
        cv2.imwrite(os.path.join(out_dir, f"{slot}.png"), card)
        cv2.imwrite(os.path.join(out_dir, f"{slot}_corner.png"), extract_corner(card))
    print(f"Wrote frame + {len(config.CARD_REGIONS)} card/corner crops to {out_dir}/")
    return 0


def _classify_frame(frame, rank_refs, suit_refs):
    results = {}
    for slot, rect in config.CARD_REGIONS.items():
        card = capture.crop(frame, rect)
        results[slot] = classify_card(card, rank_refs, suit_refs)
    return results


def cmd_run(args: argparse.Namespace) -> int:
    if not templates.templates_available():
        print("No templates found. Run 'calibrate' first to build them.")
        return 1
    rank_refs = templates.load_rank_templates()
    suit_refs = templates.load_suit_templates()
    show = args.show or args.debug

    if show:
        import cv2

        from cardvision import viz

    if args.auto:
        from cardvision import detect

    def step():
        frame = _get_frame(args.image)
        if args.auto:
            cards = detect.find_cards(frame)
            results = [classify_card(c.warped, rank_refs, suit_refs) for c in cards]
            labels = [formatter.format_card(r, ascii_only=args.ascii) for r in results]
            line = "  ".join(f"card_{i+1}={lbl}" for i, lbl in enumerate(labels)) or "(no cards)"
            print(line, flush=True)
            if show:
                cv2.imshow("cardvision", viz.annotate_detections(frame, cards, results))
                if args.debug:
                    cv2.imshow("cardvision-debug",
                               viz.debug_panel({f"card_{i+1}": r for i, r in enumerate(results)}))
            return results
        results = _classify_frame(frame, rank_refs, suit_refs)
        labels = formatter.format_hand(results, ascii_only=args.ascii)
        line = "  ".join(f"{slot}={label}" for slot, label in labels.items())
        print(line, flush=True)
        if show:
            cv2.imshow("cardvision", viz.annotate_frame(frame, results))
            if args.debug:
                cv2.imshow("cardvision-debug", viz.debug_panel(results))
        return results

    if show:
        # Window-driven loop; press q or Esc to quit.
        single = bool(args.image) and not args.loop
        while True:
            step()
            key = cv2.waitKey(0 if single else max(1, int(args.interval * 1000)))
            if single or key in (ord("q"), 27):
                break
        cv2.destroyAllWindows()
    elif args.loop and not args.image:
        try:
            while True:
                step()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nstopped.")
    else:
        step()
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    """Show the captured frame with card outlines — no templates needed.

    --auto draws auto-detected card quads (test placement freely); otherwise
    outlines the fixed CARD_REGIONS for aligning SCREEN_REGION / CARD_REGIONS.
    """
    import cv2

    from cardvision import viz

    if args.auto:
        from cardvision import detect

    single = bool(args.image)
    while True:
        frame = _get_frame(args.image)
        if args.auto:
            cards = detect.find_cards(frame)
            view = viz.annotate_detections(frame, cards, None)
        else:
            view = viz.annotate_frame(frame, None)
        cv2.imshow("cardvision-preview", view)
        key = cv2.waitKey(0 if single else max(1, int(args.interval * 1000)))
        if single or key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cardvision", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("calibrate", help="save rank/suit reference templates")
    cal.add_argument("--slot", help="card slot from config.CARD_REGIONS (fixed mode)")
    cal.add_argument("--auto", action="store_true",
                     help="auto-detect the card instead of using a fixed slot")
    cal.add_argument("--rank", help="rank label to save (A,2..10,J,Q,K)")
    cal.add_argument("--suit", help="suit label to save (S,H,D,C)")
    cal.add_argument("--image", help="use an image file instead of screen capture")
    cal.set_defaults(func=cmd_calibrate)

    gen = sub.add_parser("gen-templates",
                         help="generate a synthetic starter template set (no calibration)")
    gen.add_argument("--force", action="store_true",
                     help="overwrite existing/calibrated templates")
    gen.set_defaults(func=cmd_gen_templates)

    cal_all = sub.add_parser("calibrate-all",
                             help="guided walkthrough to capture all 17 glyphs from your source")
    cal_all.add_argument("--image", help="use an image file instead of screen capture")
    cal_all.set_defaults(func=cmd_calibrate_all)

    run = sub.add_parser("run", help="classify cards (fixed slots or --auto)")
    run.add_argument("--image", help="use an image file instead of screen capture")
    run.add_argument("--auto", action="store_true",
                     help="auto-detect cards instead of using fixed CARD_REGIONS")
    run.add_argument("--loop", action="store_true", help="classify continuously")
    run.add_argument("--interval", type=float, default=0.5, help="loop delay (s)")
    run.add_argument("--ascii", action="store_true", help="use S/H/D/C, not symbols")
    run.add_argument("--show", action="store_true",
                     help="open a window highlighting cards + labels")
    run.add_argument("--debug", action="store_true",
                     help="also show a panel of the rank/suit glyphs being matched")
    run.set_defaults(func=cmd_run)

    prev = sub.add_parser("preview", help="show card outlines (no templates needed)")
    prev.add_argument("--image", help="use an image file instead of screen capture")
    prev.add_argument("--auto", action="store_true",
                      help="draw auto-detected cards instead of fixed regions")
    prev.add_argument("--interval", type=float, default=0.1, help="refresh delay (s)")
    prev.set_defaults(func=cmd_preview)

    sc = sub.add_parser("show-crops", help="dump frame/card/corner crops to disk")
    sc.add_argument("--image", help="use an image file instead of screen capture")
    sc.add_argument("--out", default="crops", help="output directory")
    sc.set_defaults(func=cmd_show_crops)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
