#!/usr/bin/env python3
"""Merge source frames and a matte into an RGBA PNG sequence.

Thin CLI over cleanplate.compose - the app calls the same functions.

    python src/export_rgba.py --shot walk --out-name walk_v2 \
        --alpha outputs/walk_v2/alpha --despill green

Straight (unpremultiplied) alpha, which is what the PNG spec defines. RGB is kept
intact under alpha=0 so a later feather picks up real pixels rather than black.
Frame numbering carries straight through from the source frames.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from cleanplate import compose, runs
from cleanplate.ingest import frame_paths, load_frame
from cleanplate.paths import out_dir, rel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--alpha", type=Path, default=None,
                    help="folder of alpha PNGs; point at outputs/<name>/alpha for soft")
    ap.add_argument("--despill", choices=["none", "green"], default="none")
    ap.add_argument("--despill-strength", type=float, default=1.0)
    ap.add_argument("--despill-band", type=int, default=2)
    args = ap.parse_args()

    name = args.out_name or args.shot
    src = args.alpha.resolve() if args.alpha else out_dir(name) / "masks"
    frames = frame_paths(args.shot)
    if not frames:
        sys.exit(f"no frames for shot {args.shot!r}")
    try:
        alphas = runs.load_seq(src)
    except FileNotFoundError as e:
        sys.exit(f"{e}\nRun src/track_matte.py --shot {args.shot} first.")
    if len(alphas) != len(frames):
        sys.exit(f"{len(frames)} frames but {len(alphas)} alphas in {rel(src)}. "
                 "Re-run the track - a partial matte would export silently wrong.")

    out = []
    despilled = 0
    for i in range(len(frames)):
        rgb = load_frame(args.shot, i)
        a = alphas[i]
        if args.despill == "green":
            rgb, n = compose.despill_green(rgb, a, args.despill_strength,
                                           args.despill_band)
            despilled += n
        out.append(compose.to_rgba(rgb, a))

    compose.write_sequence(out, out_dir(name) / "rgba", mode="RGBA")
    cov = [float(a.mean() / 255.0) for a in alphas]
    stats = {"shot": args.shot, "out_name": name, "alpha_source": rel(src),
             "despill": args.despill, "despill_strength": args.despill_strength,
             "despill_band_px": args.despill_band, "despilled_pixels_total": despilled,
             "frames": len(frames), "premultiplied": False,
             "resolution": list(alphas.shape[1:][::-1]),
             "alpha_coverage": {"min": round(min(cov), 5), "max": round(max(cov), 5),
                                "mean": round(float(np.mean(cov)), 5)}}
    runs.save_json(stats, out_dir(name) / "rgba.json")

    a0 = out[0][..., 3]
    print(f"[rgba] {len(out)} RGBA PNGs at {a0.shape[1]}x{a0.shape[0]} -> "
          f"{rel(out_dir(name) / 'rgba')}")
    print(f"[rgba] alpha range on first frame: {a0.min()}..{a0.max()}  "
          f"({(a0 > 127).mean():.1%} opaque)")
    print(f"[rgba] alpha coverage across shot: min {stats['alpha_coverage']['min']:.4f} "
          f"max {stats['alpha_coverage']['max']:.4f}")
    if args.despill != "none":
        print(f"[rgba] despill={args.despill} strength={args.despill_strength} "
              f"band={args.despill_band}px -> {despilled} pixels altered")
    if a0.min() == a0.max():
        print("[rgba] WARNING: first frame alpha is uniform - the matte does nothing",
              file=sys.stderr)


if __name__ == "__main__":
    main()
