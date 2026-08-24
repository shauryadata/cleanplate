#!/usr/bin/env python3
"""Tier R — removal ground truth. Deterministic from the recipes below.

The RotoBench move for removal: take a clip of clean background, composite a tracked
RGBA foreground INTO it, and keep the clean background. Removing the foreground should
give the background back, so the background *is* the answer — exactly and by
construction, the same trick that makes Tier A's alpha exact.

    python scripts/build_removal_truth.py

Per clip, under truth_removal/<name>/:
    frames/NNNNN.jpg   the dirty plate (background + intruder) — what removal sees
    clean/NNNNN.jpg    the background alone — the answer
    alpha/NNNNN.png    the intruder's alpha, so the hole is reproducible
    recipe.json
"""
from __future__ import annotations

import json, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import compose, truth                                  # noqa: E402
from cleanplate.paths import ROOT, rel                                 # noqa: E402

OUT = ROOT / "truth_removal"
BG = "datasets/backgrounds"
FG_RGBA = "outputs/walk_v2/rgba"          # the tracked walk actor, 96 frames RGBA

# Positions were chosen once so the intruder sits fully inside frame, then frozen.
RECIPES = [
    {"name": "R1_actor_over_mars", "bg": f"{BG}/mars_clean.jpg", "bg_kind": "image",
     "scale": 0.85, "dx": -0.18, "dy": 0.06, "start": 0,
     "note": "static background, rocky texture. Easiest case: nothing to track."},
    {"name": "R2_actor_over_city", "bg": f"{BG}/tos_city_96s.jpg", "bg_kind": "image",
     "scale": 0.80, "dx": 0.16, "dy": 0.08, "start": 0,
     "note": "static background with hard architectural lines the fill must continue."},
    {"name": "R3_actor_over_canal", "bg": f"{BG}/tos_canal_move.mp4", "bg_kind": "video",
     "scale": 0.85, "dx": 0.0, "dy": 0.05, "start": 0,
     "note": "MOVING background. The fill has to be consistent with camera motion."},
    {"name": "R4_big_over_harbour", "bg": f"{BG}/tos_harbour_move.mp4", "bg_kind": "video",
     "scale": 1.35, "dx": 0.05, "dy": 0.0, "start": 0,
     "note": "moving background AND a much larger intruder: the big-occlusion case."},
]
WIDTH, HEIGHT, FRAMES = 960, 400, 96


def main() -> None:
    import cv2
    rgba_dir = ROOT / FG_RGBA
    ps = sorted(rgba_dir.glob("*.png"), key=lambda p: int(p.stem))[:FRAMES]
    if not ps:
        sys.exit(f"no RGBA at {rel(rgba_dir)} — run the walk pipeline first")
    fg = np.stack([np.asarray(Image.open(p).convert("RGBA")) for p in ps])
    OUT.mkdir(exist_ok=True)

    for r in RECIPES:
        # background
        if r["bg_kind"] == "video":
            bg = truth.read_frames(ROOT / r["bg"], r["start"], FRAMES, width=WIDTH)
            bg = np.stack([cv2.resize(b, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                           for b in bg])
        else:
            one = compose.cover_fit(Image.open(ROOT / r["bg"]).convert("RGB"),
                                    (WIDTH, HEIGHT))
            bg = np.repeat(one[None], FRAMES, axis=0)

        # place the intruder
        th = max(2, int(round(HEIGHT * r["scale"]))); th -= th % 2
        tw = max(2, int(round(fg.shape[2] * th / fg.shape[1]))); tw -= tw % 2
        ox = int(round((WIDTH - tw) / 2 + r["dx"] * WIDTH))
        oy = int(round((HEIGHT - th) / 2 + r["dy"] * HEIGHT))
        sx0, sy0 = max(0, -ox), max(0, -oy)
        dx0, dy0 = max(0, ox), max(0, oy)
        cw = min(tw - sx0, WIDTH - dx0); ch = min(th - sy0, HEIGHT - dy0)

        dirty = bg.copy()
        alpha = np.zeros((FRAMES, HEIGHT, WIDTH), np.uint8)
        for i in range(FRAMES):
            f = cv2.resize(fg[i], (tw, th), interpolation=cv2.INTER_AREA)
            sub = f[sy0:sy0 + ch, sx0:sx0 + cw]
            a = sub[..., 3:4].astype(np.float32) / 255.0
            region = dirty[i, dy0:dy0 + ch, dx0:dx0 + cw].astype(np.float32)
            dirty[i, dy0:dy0 + ch, dx0:dx0 + cw] = np.clip(
                sub[..., :3].astype(np.float32) * a + region * (1 - a), 0, 255)
            alpha[i, dy0:dy0 + ch, dx0:dx0 + cw] = sub[..., 3]

        d = OUT / r["name"]
        if d.exists():
            shutil.rmtree(d)
        for sub_name in ("frames", "clean", "alpha"):
            (d / sub_name).mkdir(parents=True)
        for i in range(FRAMES):
            Image.fromarray(dirty[i]).save(d / "frames" / f"{i:05d}.jpg", quality=95)
            Image.fromarray(bg[i]).save(d / "clean" / f"{i:05d}.jpg", quality=95)
            Image.fromarray(alpha[i], mode="L").save(d / "alpha" / f"{i:05d}.png")

        rec = dict(r, tier="R",
                   kind="synthetic removal: the clean background IS the answer, by construction",
                   width=WIDTH, height=HEIGHT, frames=FRAMES,
                   foreground=FG_RGBA,
                   placement={"target_h": th, "target_w": tw, "ox": ox, "oy": oy},
                   alpha_area_fraction=round(float((alpha > 127).mean()), 5),
                   credits="backgrounds: datasets/backgrounds/SOURCES.txt; "
                           "foreground: Tears of Steel, (CC) Blender Foundation")
        (d / "recipe.json").write_text(json.dumps(rec, indent=2) + "\n")
        print(f"  {r['name']:26s} intruder covers "
              f"{100 * rec['alpha_area_fraction']:5.2f}% of frame  -> {rel(d)}")


if __name__ == "__main__":
    main()
