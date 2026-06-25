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


def cmd_calibrate(args: argparse.Namespace) -> int:
    frame = _get_frame(args.image)
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

    def step():
        frame = _get_frame(args.image)
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
    """Show the captured frame with card regions outlined — no templates needed.

    Useful for aligning SCREEN_REGION / CARD_REGIONS before calibrating.
    """
    import cv2

    from cardvision import viz

    single = bool(args.image)
    while True:
        frame = _get_frame(args.image)
        cv2.imshow("cardvision-preview", viz.annotate_frame(frame, None))
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
    cal.add_argument("--slot", required=True, help="card slot from config.CARD_REGIONS")
    cal.add_argument("--rank", help="rank label to save (A,2..10,J,Q,K)")
    cal.add_argument("--suit", help="suit label to save (S,H,D,C)")
    cal.add_argument("--image", help="use an image file instead of screen capture")
    cal.set_defaults(func=cmd_calibrate)

    run = sub.add_parser("run", help="classify configured card slots")
    run.add_argument("--image", help="use an image file instead of screen capture")
    run.add_argument("--loop", action="store_true", help="classify continuously")
    run.add_argument("--interval", type=float, default=0.5, help="loop delay (s)")
    run.add_argument("--ascii", action="store_true", help="use S/H/D/C, not symbols")
    run.add_argument("--show", action="store_true",
                     help="open a window highlighting card regions + labels")
    run.add_argument("--debug", action="store_true",
                     help="also show a panel of the rank/suit glyphs being matched")
    run.set_defaults(func=cmd_run)

    prev = sub.add_parser("preview", help="show capture regions (no templates needed)")
    prev.add_argument("--image", help="use an image file instead of screen capture")
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
