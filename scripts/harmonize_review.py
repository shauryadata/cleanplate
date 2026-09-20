#!/usr/bin/env python3
"""Before/after renders for the harmonize v0 review stop. Nothing ships until it helps.

    python scripts/harmonize_review.py

Builds, on the walk shot over the Mars panorama:
  A  colour/exposure + grain match off vs on (stills and a clip)
  B  static background vs background panned by the plate's own measured track

Writes outputs/_harmonize/ and prints the numbers behind each. Whatever the review
rejects ships disabled, with the result recorded in docs/DECISIONS.md.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import compose, harmonize                              # noqa: E402
from cleanplate.ingest import frame_paths                              # noqa: E402
from cleanplate.paths import ROOT, rel                                 # noqa: E402

OUT = ROOT / "outputs" / "_harmonize"
SHOT = "walk"
BG = ROOT / "datasets" / "backgrounds" / "mars_curiosity_360_pano.jpg"
# Task 1's comp used --bg-crop 380,125,1680,600 on this 2960x765 panorama. For the
# tracked version the background has to pan ~1170 px, so the source is the full-width
# band at the same height; both the static and tracked comps are windows on ONE canvas,
# so the A/B cannot differ in scale. (A first pass used a crop beyond the image bounds,
# which put black across half the plate and, worse, made the colour match "learn" that
# the background was nearly black.)
# The largest black-free band of the panorama that needs the least upscaling to cover
# the frame plus the measured travel (found by search, not by eye): 1357x307 at 1.60x.
CROP = (545, 117, 1902, 424)
STRENGTH = 0.5


def load(shot: str):
    ps = frame_paths(shot)
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in ps]
    ad = ROOT / "outputs" / f"{shot}_v2" / "alpha"
    if not ad.is_dir():
        sys.exit(f"no matte at {rel(ad)} - run the pipeline on '{shot}' first")
    alphas = np.stack([np.asarray(Image.open(p).convert("L")) for p in
                       sorted(ad.glob("*.png"), key=lambda q: int(q.stem))])
    return frames, alphas


def strip(rows: list[np.ndarray], path: Path, labels: list[str]) -> None:
    from PIL import ImageDraw
    out = []
    for im, lab in zip(rows, labels):
        p = Image.fromarray(im); d = ImageDraw.Draw(p)
        d.rectangle([0, 0, p.width, 22], fill=(0, 0, 0))
        d.text((8, 5), lab, fill=(255, 255, 255))
        out.append(np.asarray(p))
    Image.fromarray(np.vstack(out)).save(path, quality=92)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    frames, alphas = load(SHOT)
    h, w = frames[0].shape[:2]
    size = (w, h)
    bg_img = Image.open(BG).convert("RGB")
    track = harmonize.track_translation(frames, alphas)
    pad = harmonize.pan_pad(track)
    plate_static = harmonize.panned_background(bg_img, size, np.zeros_like(track),
                                               CROP, pad=pad)[0]

    # ---- A: colour + grain -------------------------------------------------
    variants = {"cast": harmonize.estimate(frames, alphas, [plate_static] * len(frames),
                                           strength=0.4, mode="cast"),
                "full": harmonize.estimate(frames, alphas, [plate_static] * len(frames),
                                           strength=STRENGTH, mode="full")}
    hm = variants["cast"]
    for name, v in variants.items():
        print(f"[harmonize/{name}] gain {[round(g, 3) for g in v.gain]}  "
              f"offset {[round(o, 1) for o in v.offset]}  grain +{v.grain_sigma:.2f}")
    print(f"            fg mean {hm.stats.get('fg_mean')} -> bg mean {hm.stats.get('bg_mean')}; "
          f"noise fg {hm.stats.get('fg_noise_sigma')} vs bg {hm.stats.get('bg_noise_sigma')}")

    off_dir = OUT / "_off"; off_dir.mkdir()
    dirs = {k: OUT / f"_{k}" for k in variants}
    for d in dirs.values():
        d.mkdir()
    for i, (f, a) in enumerate(zip(frames, alphas)):
        Image.fromarray(compose.over(f, a, plate_static)).save(off_dir / f"{i:05d}.png")
        for k, v in variants.items():
            Image.fromarray(compose.over(harmonize.apply(f, v, a, seed=i), a,
                                         plate_static)).save(dirs[k] / f"{i:05d}.png")
    for i in (12, 48, 84):
        strip([np.asarray(Image.open(off_dir / f"{i:05d}.png"))]
              + [np.asarray(Image.open(dirs[k] / f"{i:05d}.png")) for k in variants],
              OUT / f"colour_f{i:05d}.jpg",
              [f"A1. harmonize OFF  (frame {i})",
               f"A2. colour cast only, strength 0.4  (+ grain "
               f"{variants['cast'].grain_sigma:.2f})",
               f"A3. full match (cast + level), strength {STRENGTH}  (+ grain "
               f"{variants['full'].grain_sigma:.2f})"])
    compose.encode(off_dir, OUT / "colour_off.mp4", 24.0)
    compose.encode(dirs["cast"], OUT / "colour_cast.mp4", 24.0)
    compose.encode(dirs["full"], OUT / "colour_full.mp4", 24.0)

    # ---- B: background track ----------------------------------------------
    print(f"[track] plate travels {track[:, 0].min():.0f} to {track[:, 0].max():.0f} px "
          f"in x, {track[:, 1].min():.0f} to {track[:, 1].max():.0f} px in y over "
          f"{len(track)} frames")
    panned = harmonize.panned_background(bg_img, size, track, CROP, pad=pad)
    tr_dir = OUT / "_tracked"; tr_dir.mkdir()
    for i, (f, a) in enumerate(zip(frames, alphas)):
        Image.fromarray(compose.over(f, a, panned[i])).save(tr_dir / f"{i:05d}.png")
    compose.encode(tr_dir, OUT / "track_on.mp4", 24.0)
    for i in (0, 48, 95):
        strip([np.asarray(Image.open(off_dir / f"{i:05d}.png")),
               np.asarray(Image.open(tr_dir / f"{i:05d}.png"))],
              OUT / f"track_f{i:05d}.jpg",
              [f"B. background STATIC  (frame {i})",
               f"B. background TRACKED  (plate moved {track[i, 0]:+.0f}, {track[i, 1]:+.0f} px)"])

    # side by side clip, static left / tracked right: the whole point is the motion
    sbs = OUT / "_sbs"; sbs.mkdir()
    for i in range(len(frames)):
        a_ = np.asarray(Image.open(off_dir / f"{i:05d}.png"))
        b_ = np.asarray(Image.open(tr_dir / f"{i:05d}.png"))
        Image.fromarray(np.hstack([a_, b_])).save(sbs / f"{i:05d}.png")
    compose.encode(sbs, OUT / "track_static_vs_tracked.mp4", 24.0)

    (OUT / "harmonize.json").write_text(json.dumps(
        {"shot": SHOT, "background": str(rel(BG)), "crop": list(CROP),
         "colour": {k: v.to_dict() for k, v in variants.items()},
         "track": {"cumulative_px": [[round(float(x), 2), round(float(y), 2)]
                                     for x, y in track],
                   "x_range": [float(track[:, 0].min()), float(track[:, 0].max())],
                   "y_range": [float(track[:, 1].min()), float(track[:, 1].max())]}},
        indent=2) + "\n")
    shutil.rmtree(sbs)          # the side-by-side frames; the mp4 is the artefact
    print(f"[harmonize] -> {rel(OUT)}")


if __name__ == "__main__":
    main()
