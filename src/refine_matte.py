#!/usr/bin/env python3
"""Turn SAM 2's binary mask into a soft alpha with MatAnyone.

Thin CLI over cleanplate.refine - the app calls the same function.

    python src/refine_matte.py --shot walk --masks outputs/walk_v2/masks \
        --out-name walk_v2

MatAnyone is licensed S-Lab 1.0 (non-commercial). It is an optional stage:
CleanPlate without it still produces the binary SAM 2 matte. See THIRD_PARTY.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from cleanplate import refine, runs
from cleanplate.paths import MATANYONE_CKPT, out_dir, rel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--masks", type=Path, default=None,
                    help="folder of binary masks (default outputs/<out-name>/masks)")
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--anchor-frame", type=int, default=0)
    ap.add_argument("--reanchor", type=int, action="append", default=[], metavar="FRAME")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--dilate", type=int, default=10)
    ap.add_argument("--erode", type=int, default=10)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    ap.add_argument("--checkpoint", type=Path, default=MATANYONE_CKPT)
    args = ap.parse_args()

    name = args.out_name or args.shot
    masks_dir = args.masks.resolve() if args.masks else out_dir(name) / "masks"
    ok, why = refine.available()
    if not ok:
        sys.exit(why)
    try:
        masks = runs.load_seq(masks_dir) > 127
    except FileNotFoundError as e:
        sys.exit(f"{e}\nRun src/track_matte.py --shot {args.shot} first.")

    print(f"[refine] shot={args.shot}  frames={len(masks)}")
    print(f"[refine] masks: {rel(masks_dir)}  anchor frame {args.anchor_frame}")
    print(f"[refine] warmup={args.warmup} dilate={args.dilate} erode={args.erode}"
          + (f"  reanchor at {args.reanchor}" if args.reanchor else ""))
    print(f"[refine] outputs -> {rel(out_dir(name) / 'alpha')}")

    last = [""]

    def show(frac: float, msg: str) -> None:
        if msg != last[0]:
            print(f"\r[refine] {msg}".ljust(70), end="", flush=True)
            last[0] = msg

    try:
        alphas, stats = refine.refine(args.shot, masks, anchor_frame=args.anchor_frame,
                                      device=args.device, warmup=args.warmup,
                                      dilate=args.dilate, erode=args.erode,
                                      reanchor=args.reanchor, progress=show)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print()
        sys.exit(str(e))
    print()

    runs.save_alpha(name, alphas, stats)
    print(f"[refine] {len(alphas)} soft alphas -> {rel(out_dir(name) / 'alpha')}")
    print(f"[refine] {stats['refine_s']}s = {stats['seconds_per_frame']:.3f} s/frame "
          f"({stats['fps']} fps)")
    print(f"[refine] soft (0<a<255) pixels: mean "
          f"{100 * stats['soft_pixel_fraction']['mean']:.3f}% of frame  "
          "[exactly 0% with the binary matte]")
    print(f"[refine] MPS fallback ops: {stats['mps_fallback_ops'] or 'none'}")
    print(f"[refine] stats -> {rel(out_dir(name) / 'refine.json')}")


if __name__ == "__main__":
    main()
