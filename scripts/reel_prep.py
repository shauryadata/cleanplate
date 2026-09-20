#!/usr/bin/env python3
"""Build the two reel segments that need composing first. Everything from repo data.

    python scripts/reel_prep.py

  hair_ab       P01 (the professional-truth green-screen shot), head crop, the default
                matte against the high-quality one, both over a neutral card. Replaces
                an earlier version of this segment that was comped over a personal photo
                with no clear licence - fine on a laptop, not in a public reel.
  markers_zoom  the green-screen marker removal, zoomed to where the markers are, so
                the point is legible at reel scale instead of being three pixels wide.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import compose                                        # noqa: E402
from cleanplate.paths import ROOT, rel                                # noqa: E402

SRC = ROOT / "outputs" / "_reel_src"
CLIP = "P01_08_3a"
HEAD = (300, 40, 1180, 600)          # the old man's head and shoulders, 1920x1012 space
BACKDROP = (18, 21, 28)


def seq(d: Path) -> list[Path]:
    return sorted(d.glob("*.png"), key=lambda p: int(p.stem))


def hair_ab() -> None:
    frames = seq(ROOT / "truth" / CLIP / "frames")
    if not frames:
        frames = sorted((ROOT / "truth" / CLIP / "frames").glob("*.jpg"),
                        key=lambda p: int(p.stem))
    a_dir = ROOT / "outputs" / "_bench_p" / "baseline_960" / CLIP
    b_dir = ROOT / "outputs" / "_bench_p" / "hairzoom2_960" / CLIP
    for name, ad in (("a", a_dir), ("b", b_dir)):
        out = SRC / "hair_ab" / name
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        alphas = seq(ad)
        l, t, r, b = HEAD
        for i, (fp, ap) in enumerate(zip(frames, alphas)):
            rgb = np.asarray(Image.open(fp).convert("RGB"))[t:b, l:r]
            al = np.asarray(Image.open(ap).convert("L"))[t:b, l:r]
            # despill, as the pipeline does before any comp: without it both sides carry
            # a green rim from the screen and the comparison shows spill, not matte quality
            rgb, _ = compose.despill_green(rgb, al)
            bg = np.zeros_like(rgb); bg[:] = BACKDROP
            Image.fromarray(compose.over(rgb, al, bg)).save(out / f"{i:05d}.jpg", quality=95)
    print(f"[prep] {rel(SRC / 'hair_ab')}  default vs high-quality, {len(frames)} frames")


def markers_zoom() -> None:
    d = ROOT / "outputs" / "_demos" / "B_greenscreen_markers"
    before, after = np.load(d / "before.npy"), np.load(d / "cleaned.npy")
    # where the removal actually changed pixels, across the whole clip
    diff = np.abs(before.astype(np.int16) - after.astype(np.int16)).max(3).max(0)
    ys, xs = np.nonzero(diff > 12)
    cy, cx = int(ys.mean()), int(xs.mean())
    h, w = before.shape[1:3]
    ch, cw = 300, 480
    y0 = max(0, min(h - ch, cy - ch // 2)); x0 = max(0, min(w - cw, cx - cw // 2))
    out = SRC / "markers_zoom"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for i, (b_, a_) in enumerate(zip(before, after)):
        pair = np.hstack([b_[y0:y0 + ch, x0:x0 + cw], a_[y0:y0 + ch, x0:x0 + cw]])
        Image.fromarray(pair).save(out / f"{i:05d}.jpg", quality=95)
    print(f"[prep] {rel(out)}  zoomed to the markers at x{x0}-{x0 + cw}, y{y0}-{y0 + ch} "
          f"({len(before)} frames)")


if __name__ == "__main__":
    SRC.mkdir(parents=True, exist_ok=True)
    hair_ab()
    markers_zoom()
