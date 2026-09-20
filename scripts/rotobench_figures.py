#!/usr/bin/env python3
"""Every figure in docs/ROTOBENCH.md, regenerated from Tier P. Nothing hand-made.

    python scripts/rotobench_figures.py              # all figures
    python scripts/rotobench_figures.py alignment    # one of them

  alignment   why plate N-1, not N: the residual at the same frame number lights up
              with motion edges; at N-1 it is sensor-noise dark
  stills      professional key against our mattes, hair crops, at the frame where the
              methods disagree most
  selfcheck   self-consistency against accuracy, per method, plus a frozen control
  dropout     the coverage repair, before and after, where the base matte dropped out
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import bench                                         # noqa: E402
from cleanplate.paths import ROOT, rel                               # noqa: E402

IMG = ROOT / "docs" / "img"
OUT = ROOT / "outputs" / "_bench_p"


def label(im: np.ndarray, text: str, h: int = 22) -> np.ndarray:
    pil = Image.fromarray(im); g = ImageDraw.Draw(pil)
    g.rectangle([0, 0, pil.width, h], fill=(0, 0, 0))
    g.text((8, 5), text, fill=(255, 255, 255))
    return np.asarray(pil)


def alignment() -> None:
    import cv2
    from cleanplate import protruth as P
    rec = json.loads((ROOT / "truth" / "P01_08_3a" / "recipe.json").read_text())
    shot, (s, e) = rec["shot"], rec["window"]
    raw = ROOT / "datasets" / "tos_pro" / shot
    c = rec["alignment"]["evidence"][1]["cleaned_frame"]
    crgb, ca = P.read_exr(raw / "cleaned" / "linear_hd" / f"{c:04d}.exr")
    same, _ = P.read_exr(raw / "plate" / f"{c:05d}.exr")
    prev, _ = P.read_exr(raw / "plate" / f"{c - 1:05d}.exr")
    ys, xs = np.nonzero(ca > 0.02)
    y0, x0 = max(0, ys.min() - 20), max(0, xs.min() + 60)
    box = (slice(y0, y0 + 460), slice(x0, x0 + 700))
    m = ca > 0.99

    def diff(p):
        d = np.abs(p - crgb).mean(2); d[~m] = 0
        return cv2.cvtColor(np.clip(d * 40 * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    pl = P.to_display(prev).copy()
    cnt, _ = cv2.findContours((ca > 0.5).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(pl, cnt, -1, (255, 0, 160), 2)
    ev = rec["alignment"]["evidence"][1]
    tiles = [label(pl[box], f"plate {c - 1} + professional key contour"),
             label(diff(same)[box], f"|plate {c} - key {c}| x40  same frame number: motion edges"),
             label(diff(prev)[box], f"|plate {c - 1} - key {c}| x40  N-1: noise floor "
                                    f"(residual {ev['residual_best']:.5f} vs "
                                    f"{ev['residual_runner_up']:.5f})")]
    out = IMG / "rotobench_alignment.jpg"
    Image.fromarray(np.hstack(tiles)).save(out, quality=88)
    print(f"[fig] {rel(out)}")


def _load(d: Path) -> np.ndarray:
    return np.stack([np.asarray(Image.open(q)) for q in
                     sorted(d.glob("*.png"), key=lambda q: int(q.stem))])


def _err(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Signed error: red where the matte has too little alpha, cyan where too much."""
    d = pred.astype(np.float32) - gt.astype(np.float32)
    out = np.zeros((*d.shape, 3), np.uint8)
    out[..., 0] = np.clip(-d * 2, 0, 255)
    out[..., 1] = np.clip(d * 2, 0, 255)
    out[..., 2] = np.clip(d * 2, 0, 255)
    return out


def stills(clips=("P01_08_3a", "P05_09_1a", "P06_08_4a"),
           methods=("baseline_960", "hairzoom2_960"), frame: str = "mid") -> None:
    """Professional key against our mattes. The MIDDLE frame of each clip, by rule -
    not the frame where a method looks best."""
    import cv2
    rows = []
    for c in clips:
        gt = bench.load_alpha(ROOT / "truth" / c / "alpha")
        i = len(gt) // 2
        l, t, r, b = bench.hair_box(gt)
        f = np.asarray(Image.open(ROOT / "truth" / c / "frames" / f"{i:05d}.jpg").convert("RGB"))
        crop = lambda x: x[t:b, l:r]
        tiles = [label(crop(f), f"{c}  plate, frame {i}"),
                 label(np.dstack([crop(gt[i])] * 3), "professional key")]
        for m in methods:
            pa = np.asarray(Image.open(OUT / m / c / f"{i:05d}.png"))
            tiles.append(label(np.dstack([crop(pa)] * 3), m))
        tiles.append(label(crop(_err(pa, gt[i])), f"{methods[-1]} error: red = missing, cyan = extra"))
        h = 300
        tiles = [cv2.resize(x, (round(x.shape[1] * h / x.shape[0]), h)) for x in tiles]
        rows.append(np.hstack(tiles))
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0))) for r in rows]
    out = IMG / "rotobench_stills.jpg"
    Image.fromarray(np.vstack(rows)).save(out, quality=88)
    print(f"[fig] {rel(out)}")


def dropout(base: str = "hairzoom2_960", repair: str = "cover_960", n: int = 2) -> None:
    """The frames where the base matte lost coverage, before and after the repair.

    The frames are chosen by the worst base dropout, not by where the repair looks
    good."""
    import cv2
    from cleanplate import bench as B
    cands = []
    for c in B.clips("P"):
        if not (OUT / repair / c.name).is_dir():
            continue
        gt = B.load_alpha(ROOT / "truth" / c.name / "alpha")
        bp = _load(OUT / base / c.name)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        for i, (g, b) in enumerate(zip(gt, bp)):
            core = cv2.erode((g >= 250).astype(np.uint8), k) > 0
            cands.append((int((core & (b < 128)).sum()), c.name, i))
    if not cands:
        print("[fig] no repair output yet"); return
    best, rows = sorted(cands, reverse=True), []
    picked, seen = [], set()
    for px, cn, i in best:                       # one frame per clip, worst first
        if cn in seen:
            continue
        seen.add(cn); picked.append((px, cn, i))
        if len(picked) >= n:
            break
    for px, cn, i in picked:
        gt = np.asarray(Image.open(ROOT / "truth" / cn / "alpha" / f"{i:05d}.png"))
        f = np.asarray(Image.open(ROOT / "truth" / cn / "frames" / f"{i:05d}.jpg").convert("RGB"))
        bp = np.asarray(Image.open(OUT / base / cn / f"{i:05d}.png"))
        rp = np.asarray(Image.open(OUT / repair / cn / f"{i:05d}.png"))
        ys, xs = np.nonzero((gt >= 250) & (bp < 128))
        cy, cx = int(ys.mean()), int(xs.mean())
        h = 260
        sl = (slice(max(0, cy - h), max(0, cy - h) + 2 * h),
              slice(max(0, cx - h), max(0, cx - h) + 2 * h))
        tiles = [label(f[sl], f"{cn} frame {i}: plate"),
                 label(np.dstack([gt[sl]] * 3), "professional key"),
                 label(np.dstack([bp[sl]] * 3), f"{base} ({px:,} px dropped)"),
                 label(np.dstack([rp[sl]] * 3), repair),
                 label(_err(bp, gt)[sl], "before: red = missing"),
                 label(_err(rp, gt)[sl], "after")]
        rows.append(np.hstack(tiles))
    out = IMG / "rotobench_dropout.jpg"
    Image.fromarray(np.vstack(rows)).save(out, quality=88)
    print(f"[fig] {rel(out)}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["alignment"]
    for w in which:
        globals()[w]()
