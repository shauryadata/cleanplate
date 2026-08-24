#!/usr/bin/env python3
"""Make the README's removal GIF from a finished demo in outputs/_demos.

    python scripts/removal_gif.py --demo C_dialogue_one_actor

Reads the before/after arrays the demo parked on disk, so it costs no inpainting.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate.paths import ROOT, rel                                 # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="C_dialogue_one_actor")
    ap.add_argument("--width", type=int, default=760, help="total width of the pair")
    ap.add_argument("--step", type=int, default=3, help="frame stride")
    ap.add_argument("--colors", type=int, default=96)
    ap.add_argument("--no-dither", dest="dither", action="store_const",
                    const=Image.NONE, default=Image.FLOYDSTEINBERG,
                    help="photographic GIFs compress far better undithered")
    ap.add_argument("--out", default="docs/img/removal.gif")
    args = ap.parse_args()

    d = ROOT / "outputs" / "_demos" / args.demo
    before, after = np.load(d / "before.npy"), np.load(d / "cleaned.npy")
    idx = range(0, len(before), args.step)
    w = args.width // 2
    frames = []
    for i in idx:
        pair = []
        for arr, label in ((before[i], "before"), (after[i], "after")):
            im = Image.fromarray(arr)
            im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
            g = ImageDraw.Draw(im)
            g.rectangle([0, im.height - 18, 54, im.height], fill=(0, 0, 0))
            g.text((6, im.height - 15), label, fill=(255, 255, 255))
            pair.append(np.asarray(im))
        frames.append(Image.fromarray(np.hstack(pair)))

    # One palette for the whole clip, not one per frame. With per-frame palettes the
    # encoder cannot diff consecutive frames and the file roughly triples.
    base = frames[0].quantize(colors=args.colors, method=Image.MEDIANCUT)
    frames = [f.quantize(palette=base, dither=args.dither) for f in frames]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    # Play at the speed the shot was filmed at: a stride of N frames at 24 fps is
    # N/24 of a second per GIF frame, otherwise the subsampling reads as fast motion.
    ms = round(1000 * args.step / 24)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=ms, loop=0,
                   optimize=True)
    print(f"[gif] {rel(out)}  {len(frames)} frames at {ms} ms, "
          f"{frames[0].size[0]}x{frames[0].size[1]}, {out.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
