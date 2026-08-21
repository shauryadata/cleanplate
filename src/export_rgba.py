#!/usr/bin/env python3
"""Merge source frames and tracked masks into an RGBA PNG sequence.

    python src/export_rgba.py --shot walk

Reads   shots/<shot>/frames/NNNNN.jpg  +  an alpha sequence, which is either
        outputs/<shot>/masks/ (binary, from SAM 2) or, with --alpha, a soft alpha
        from outputs/<shot>/alpha/ (MatAnyone).
Writes  outputs/<shot>/rgba/NNNNN.png   (8-bit RGBA, straight/unpremultiplied)

Frame numbering is carried straight through from the source frames, so
outputs/<shot>/rgba/00042.png is the matte for shots/<shot>/frames/00042.jpg.

RGB is kept intact everywhere, including under alpha=0. Straight (unpremultiplied)
alpha is what the PNG spec defines, and keeping the original colour under the matte
means a later feather or edge-blur picks up real pixels instead of black.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--out-name", default=None,
                    help="read/write under outputs/<out-name>/ (default: the shot name)")
    ap.add_argument("--alpha", type=Path, default=None,
                    help="folder of alpha PNGs to use instead of outputs/<name>/masks; "
                         "point this at outputs/<name>/alpha for the soft matte")
    ap.add_argument("--despill", choices=["none", "green"], default="none",
                    help="suppress background colour bleed in the soft edge band")
    ap.add_argument("--despill-strength", type=float, default=1.0,
                    help="0 = off, 1 = full suppression (default 1.0)")
    ap.add_argument("--despill-band", type=int, default=2,
                    help="grow the edge band by N px inward/outward before despilling "
                         "(default 2)")
    ap.add_argument("--feather", type=float, default=0.0,
                    help="gaussian-blur the alpha by this many pixels (default 0 = "
                         "raw binary matte, flaws and all)")
    ap.add_argument("--erode", type=int, default=0,
                    help="pull the alpha in by N pixels before feathering, to bite off "
                         "a halo of background (default 0)")
    args = ap.parse_args()

    out_name = args.out_name or args.shot
    frames_dir = ROOT / "shots" / args.shot / "frames"
    masks_dir = (args.alpha.resolve() if args.alpha
                 else ROOT / "outputs" / out_name / "masks")
    if not frames_dir.is_dir():
        sys.exit(f"no frames: {frames_dir}")
    if not masks_dir.is_dir():
        sys.exit(f"no alpha: {masks_dir}\nRun src/track_matte.py --shot {args.shot} first.")

    frames = {int(p.stem): p for p in frames_dir.glob("*.jpg")}
    masks = {int(p.stem): p for p in masks_dir.glob("*.png")}

    missing = sorted(set(frames) - set(masks))
    extra = sorted(set(masks) - set(frames))
    if missing:
        sys.exit(f"{len(missing)} frames have no mask (first: {missing[:8]}). "
                 "Re-run track_matte.py - a partial matte would export silently wrong.")
    if extra:
        print(f"[rgba] note: {len(extra)} masks with no source frame, ignored: {extra[:8]}")

    out_dir = ROOT / "outputs" / out_name / "rgba"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    alpha_means: list[float] = []
    despilled_px = 0
    size = None
    for idx in sorted(frames):
        rgb = Image.open(frames[idx]).convert("RGB")
        alpha = Image.open(masks[idx]).convert("L")
        if alpha.size != rgb.size:
            sys.exit(f"frame {idx}: mask {alpha.size} != frame {rgb.size}")
        if size is None:
            size = rgb.size

        if args.erode > 0:
            # MinFilter over a (2n+1) window is a binary erosion by n pixels.
            alpha = alpha.filter(ImageFilter.MinFilter(2 * args.erode + 1))
        if args.feather > 0:
            alpha = alpha.filter(ImageFilter.GaussianBlur(args.feather))

        rgb_arr = np.asarray(rgb, dtype=np.float32)
        a_arr = np.asarray(alpha, dtype=np.float32)

        if args.despill == "green" and args.despill_strength > 0:
            # Edge band = anywhere the matte is fractional, grown a little so pixels that
            # quantised to 0 or 255 but still carry bled colour are included.
            band = (a_arr > 0) & (a_arr < 255)
            if args.despill_band > 0:
                k = np.ones((2 * args.despill_band + 1,) * 2, np.uint8)
                band = cv2.dilate(band.astype(np.uint8), k, iterations=1).astype(bool)
            band &= a_arr > 0                      # never touch fully transparent pixels
            r, g, b = rgb_arr[..., 0], rgb_arr[..., 1], rgb_arr[..., 2]
            # Classic average despill: green is not allowed to exceed the mean of the
            # other two channels. Only the excess is removed, so genuinely green
            # subject pixels keep their colour.
            limit = 0.5 * r + 0.5 * b
            excess = np.maximum(0.0, g - limit)
            rgb_arr[..., 1] = np.where(band, g - args.despill_strength * excess, g)
            despilled_px += int((band & (excess > 1)).sum())

        out = Image.fromarray(np.clip(rgb_arr, 0, 255).astype(np.uint8), "RGB")
        out.putalpha(Image.fromarray(a_arr.astype(np.uint8), "L"))
        out.save(out_dir / f"{idx:05d}.png")
        alpha_means.append(float(np.asarray(alpha, dtype=np.float32).mean() / 255.0))

    # Sanity: an RGBA that is fully opaque or fully clear means the matte is broken.
    stats = {
        "shot": args.shot,
        "out_name": out_name,
        "alpha_source": str(masks_dir.relative_to(ROOT)) if masks_dir.is_relative_to(ROOT) else str(masks_dir),
        "despill": args.despill,
        "despill_strength": args.despill_strength,
        "despill_band_px": args.despill_band,
        "despilled_pixels_total": despilled_px,
        "frames": len(frames),
        "resolution": list(size),
        "feather_px": args.feather,
        "erode_px": args.erode,
        "premultiplied": False,
        "alpha_coverage": {
            "min": round(min(alpha_means), 5),
            "max": round(max(alpha_means), 5),
            "mean": round(float(np.mean(alpha_means)), 5),
        },
    }
    (ROOT / "outputs" / out_name / "rgba.json").write_text(json.dumps(stats, indent=2) + "\n")

    probe = Image.open(out_dir / f"{min(frames):05d}.png")
    a = np.asarray(probe)[..., 3]
    print(f"[rgba] {len(frames)} RGBA PNGs at {size[0]}x{size[1]} -> "
          f"{out_dir.relative_to(ROOT)}")
    print(f"[rgba] mode={probe.mode}  alpha range on first frame: {a.min()}..{a.max()}  "
          f"({(a > 127).mean():.1%} opaque)")
    print(f"[rgba] alpha coverage across shot: min {stats['alpha_coverage']['min']:.4f} "
          f"max {stats['alpha_coverage']['max']:.4f}")
    if args.despill != "none":
        print(f"[rgba] despill={args.despill} strength={args.despill_strength} "
              f"band={args.despill_band}px -> {despilled_px} pixels altered")
    if a.min() == a.max():
        print("[rgba] WARNING: first frame alpha is uniform - the matte is not doing "
              "anything", file=sys.stderr)


if __name__ == "__main__":
    main()
