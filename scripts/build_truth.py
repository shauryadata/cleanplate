#!/usr/bin/env python3
"""Regenerate every truth clip from a committed recipe. Deterministic.

    python scripts/build_truth.py --tier a      # synthetic composites, exact alpha
    python scripts/build_truth.py --tier b      # chroma key on the ToS green plate
    python scripts/build_truth.py --tier all

Nothing here is random at run time: the recipe is written out in full, so the same
recipe always produces the same pixels. The seed only ever chose the layout, once, and
the chosen values are frozen in RECIPES below.

Output, per clip, under truth/<name>/:
    frames/NNNNN.jpg   the composite - this is what the pipeline sees
    alpha/NNNNN.png    the reference alpha
    recipe.json        every parameter needed to rebuild it
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import truth                                          # noqa: E402
from cleanplate.paths import ROOT, rel                                # noqa: E402

TRUTH = ROOT / "truth"
VM = "datasets/vm240k/VideoMatte240K/test"
BG = "datasets/backgrounds"

# --- Tier A: five VideoMatte240K test clips, each over a different background.
# Layout numbers were picked once with seed 20260823 so each subject sits fully in
# frame, then frozen here. Backgrounds alternate static / moving on purpose: a static
# plate isolates matte flicker, a moving one is the realistic case.
RECIPES = [
    truth.ClipRecipe(
        name="A1_man_wave_mars", fgr=f"{VM}/fgr/0000.mp4", pha=f"{VM}/pha/0000.mp4",
        bg=f"{BG}/mars_clean.jpg", bg_kind="image",
        start=40, frames=96, width=1920, fg_scale=0.92, fg_dx=-0.06, fg_dy=0.04,
        note="older man, short hair and beard, waving. Static PD background."),
    truth.ClipRecipe(
        name="A2_pair_city", fgr=f"{VM}/fgr/0001.mp4", pha=f"{VM}/pha/0001.mp4",
        bg=f"{BG}/tos_city_96s.jpg", bg_kind="image",
        start=60, frames=96, width=1920, fg_scale=0.88, fg_dx=0.0, fg_dy=0.05,
        note="woman and child, loose hair on the child, two subjects. Static bg."),
    truth.ClipRecipe(
        name="A3_longhair_canal", fgr=f"{VM}/fgr/0002.mp4", pha=f"{VM}/pha/0002.mp4",
        bg=f"{BG}/tos_canal_move.mp4", bg_kind="video",
        start=30, bg_start=0, frames=96, width=1920, fg_scale=0.95, fg_dx=0.0, fg_dy=0.02,
        note="THE HAIR CASE: long loose dark hair, front on. Moving background."),
    truth.ClipRecipe(
        name="A4_longhair_ruins", fgr=f"{VM}/fgr/0003.mp4", pha=f"{VM}/pha/0003.mp4",
        bg=f"{BG}/tos_ruins_580s.jpg", bg_kind="image",
        start=120, frames=96, width=1920, fg_scale=0.90, fg_dx=0.05, fg_dy=0.05,
        note="long dark hair over dark clothing, low contrast edge. Static bg."),
    truth.ClipRecipe(
        name="A5_coat_harbour", fgr=f"{VM}/fgr/0004.mp4", pha=f"{VM}/pha/0004.mp4",
        bg=f"{BG}/tos_harbour_move.mp4", bg_kind="video",
        start=20, bg_start=0, frames=96, width=1920, fg_scale=0.93, fg_dx=-0.02, fg_dy=0.03,
        note="white coat, arms raised, hair tied back. Moving background, bright fg."),
]

# --- Tier B: chroma key on the ToS green plate. The garbage box excludes the grey
# flag standing at frame right; without it the flag keys in as foreground.
TIER_B = {
    "name": "B1_tos_greenscreen_hair",
    "plates": "datasets/tos_plates/08_3a",
    "frames": 96,
    "width": 1920,
    "gamma": 2.2,
    "bg": f"{BG}/tos_ruins_580s.jpg",
    "key": truth.KeyParams(
        key_hue=120.0, hue_tol=44.0, sat_min=0.16, val_min=0.05,
        alpha_lo=0.08, alpha_hi=0.38, garbage=(0, 0, 1240, 1012),
        despill=True, median=3,
        note=("Thresholds are set from the measured greenness distribution on frame 48, "
              "not guessed: subject p99 = 0.051, backing p1 = 0.417, so alpha_lo sits "
              "just above the subject and alpha_hi just below the backing. The garbage "
              "box drops the grey flag at frame right, which is not green and would "
              "otherwise key in as foreground. KEYED REFERENCE, NOT GOSPEL.")),
    "note": ("Tears of Steel plate 08_3a, (CC) Blender Foundation | mango.blender.org. "
             "Same actor as our `hair` shot, long backlit grey hair, real green screen. "
             "The reference alpha is produced by the simple keyer in cleanplate.truth, "
             "so it carries that keyer's errors and must not be treated as exact."),
}


def write_clip(name: str, comp: np.ndarray, alpha: np.ndarray, recipe: dict) -> Path:
    d = TRUTH / name
    if d.exists():
        shutil.rmtree(d)
    (d / "frames").mkdir(parents=True)
    (d / "alpha").mkdir(parents=True)
    for i, (c, a) in enumerate(zip(comp, alpha)):
        Image.fromarray(c).save(d / "frames" / f"{i:05d}.jpg", quality=95)
        Image.fromarray(a, mode="L").save(d / "alpha" / f"{i:05d}.png")
    soft = ((alpha > 0) & (alpha < 255))
    recipe["built"] = {
        "frames": int(len(comp)),
        "resolution": [int(comp.shape[2]), int(comp.shape[1])],
        "alpha_soft_pixel_fraction": round(float(soft.mean()), 6),
        "alpha_distinct_values": int(len(np.unique(alpha))),
        "alpha_coverage_mean": round(float(alpha.mean() / 255.0), 5),
    }
    (d / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    return d


def build_tier_a() -> None:
    for r in RECIPES:
        t0 = time.perf_counter()
        print(f"[truth A] {r.name}: {r.note}")
        out = truth.build_clip(r, ROOT)
        rec = {"tier": "A", "kind": "synthetic composite, alpha exact by construction",
               "recipe": r.to_dict(),
               "source_dataset": "VideoMatte240K (BackgroundMattingV2, UW GRAIL) - "
                                 "licensed for commercial and non-commercial use",
               "background_credits": "see datasets/backgrounds/SOURCES.txt"}
        d = write_clip(r.name, out["comp"], out["alpha"], rec)
        b = json.loads((d / "recipe.json").read_text())["built"]
        print(f"[truth A]   -> {rel(d)}  {b['frames']}f {b['resolution']}  "
              f"soft {100 * b['alpha_soft_pixel_fraction']:.3f}%  "
              f"{b['alpha_distinct_values']} alpha values  "
              f"({time.perf_counter() - t0:.1f}s)")


def build_tier_b() -> None:
    import cv2
    spec = TIER_B
    src = ROOT / spec["plates"]
    exrs = sorted(src.glob("*.exr"))[: spec["frames"]]
    if not exrs:
        sys.exit(f"no plates in {rel(src)} - run ./scripts/download.sh plates")
    print(f"[truth B] {spec['name']}: {len(exrs)} plates from {rel(src)}")

    import os
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    comps, alphas = [], []
    t0 = time.perf_counter()
    for p in exrs:
        lin = cv2.imread(str(p), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH
                         | cv2.IMREAD_ANYCOLOR)
        if lin is None:
            sys.exit(f"could not read {p} - OpenEXR support missing in this OpenCV build")
        lin = cv2.cvtColor(lin[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        # linear scene EXR -> display: clamp then gamma, so the keyer works in a
        # perceptual space rather than on linear light.
        disp = np.clip(lin, 0, 1) ** (1.0 / spec["gamma"])
        rgb = (disp * 255).astype(np.uint8)
        if rgb.shape[1] != spec["width"]:
            h = int(round(rgb.shape[0] * spec["width"] / rgb.shape[1])); h -= h % 2
            rgb = cv2.resize(rgb, (spec["width"], h), interpolation=cv2.INTER_AREA)
        a, despilled = truth.chroma_key(rgb, spec["key"])
        comps.append(despilled)
        alphas.append(a)
    fg = np.stack(comps); alpha = np.stack(alphas)

    # Composite the despilled foreground over a new background using the keyed alpha,
    # exactly as Tier A does. The pipeline then sees an ordinary-looking shot rather
    # than a green screen, so the two tiers are directly comparable.
    from PIL import Image as PILImage
    H, W = fg.shape[1], fg.shape[2]
    bim = PILImage.open(ROOT / spec["bg"]).convert("RGB")
    sc = max(W / bim.width, H / bim.height)
    bim = bim.resize((max(W, int(bim.width * sc)), max(H, int(bim.height * sc))),
                     PILImage.LANCZOS)
    l, t = (bim.width - W) // 2, (bim.height - H) // 2
    bg1 = np.asarray(bim.crop((l, t, l + W, t + H)))
    af = alpha.astype(np.float32)[..., None] / 255.0
    comp = np.clip(fg.astype(np.float32) * af + bg1.astype(np.float32) * (1 - af),
                   0, 255).astype(np.uint8)
    rec = {"tier": "B", "kind": "keyed reference, NOT gospel",
           "note": spec["note"], "plates": spec["plates"], "background": spec["bg"],
           "gamma": spec["gamma"], "width": spec["width"],
           "key_params": spec["key"].to_dict(),
           "footage_credit": "(CC) Blender Foundation | mango.blender.org, CC BY 3.0"}
    d = write_clip(spec["name"], comp, alpha, rec)
    b = json.loads((d / "recipe.json").read_text())["built"]
    print(f"[truth B]   -> {rel(d)}  {b['frames']}f {b['resolution']}  "
          f"soft {100 * b['alpha_soft_pixel_fraction']:.3f}%  "
          f"{b['alpha_distinct_values']} alpha values  "
          f"({time.perf_counter() - t0:.1f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=["a", "b", "all"], default="all")
    args = ap.parse_args()
    TRUTH.mkdir(exist_ok=True)
    if args.tier in ("a", "all"):
        build_tier_a()
    if args.tier in ("b", "all"):
        build_tier_b()


if __name__ == "__main__":
    main()
