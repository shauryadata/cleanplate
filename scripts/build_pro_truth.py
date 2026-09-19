#!/usr/bin/env python3
"""Build Tier P - professional reference truth - from the Tears of Steel archive.

    python scripts/build_pro_truth.py --init       # recipes from the window selection
    python scripts/build_pro_truth.py              # fetch, prove alignment, build all
    python scripts/build_pro_truth.py --clip P01_08_3a

Input is the raw plate (tearsofsteel-footage-exr), truth is the compositor's key
(tearsofsteel-cleaned-exr, channel 4). No compositing: the frames a method sees are the
real footage the key was pulled from.

Nothing is scored until the plate and the key are proven to be the same frame. The
two archives do not share numbering - on 08_3a the key for frame N was pulled from
plate N-1 - so every clip measures its own offset from pixels and records the evidence
in its recipe. A clip whose offset is ambiguous, inconsistent across the window, or
spatially shifted is refused, not built.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import protruth as P                                   # noqa: E402
from cleanplate import truth                                           # noqa: E402
from cleanplate.paths import ROOT, rel                                 # noqa: E402

TRUTH = ROOT / "truth"
DATA = ROOT / "datasets" / "tos_pro"
SURVEY = DATA / "_survey"
HD = (1920, 1012)
SIG = 0.005                   # significant component: >= 0.5% of the frame
OFFSETS = range(-3, 4)        # plate = cleaned + k, searched per clip
MAX_RATIO = 0.6               # best residual must beat the runner-up by this factor
CREDIT = "(CC) Blender Foundation | mango.blender.org, CC BY 3.0"

# The in-house keyer from Task 4, with exactly the Tier B settings, run on the same
# frames. Same predictions scored against both keys is the cleanest test of whether
# our earlier conclusions were artefacts of our own reference.
INHOUSE = {"08_3a": truth.KeyParams(
    key_hue=120.0, hue_tol=44.0, sat_min=0.16, val_min=0.05, alpha_lo=0.08,
    alpha_hi=0.38, garbage=(0, 0, 1240, 1012), despill=True, median=3,
    note="Tier B settings from Task 4, unchanged. KEYED REFERENCE, NOT GOSPEL.")}


def init() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from select_pro import CANDIDATES
    win = json.loads((SURVEY / "windows.json").read_text())
    for cand in CANDIDATES:
        pid, shot, axes = cand[:3]
        w = win.get(shot)
        if not w or "window" not in w:
            print(f"[init] {shot}: not selected ({(w or {}).get('rejected', 'no window')})")
            continue
        d = TRUTH / f"{pid}_{shot}"
        rp = d / "recipe.json"
        if rp.exists():
            print(f"[init] {d.name}: recipe exists, left alone")
            continue
        d.mkdir(parents=True, exist_ok=True)
        rec = {
            "tier": "P",
            "kind": "professional reference: the compositor's own key. A professional "
                    "key, still a key - not a hand-painted or rendered ground truth.",
            "group": ("short" if "short" in axes else
                      "stress" if any(a.startswith("stress") for a in axes) else "core"),
            "shot": shot, "axes": axes,
            "source": {"cleaned": f"tearsofsteel-cleaned-exr/{shot}/{w['source']}",
                       "res": w["source"], "digits": w["digits"],
                       "plate": f"tearsofsteel-footage-exr/{shot}/linear_hd"},
            "window": w["window"],
            "window_rule": "contiguous 96 frames nearest the shot centre in which every "
                           "significant component descends from one on the first frame "
                           "(scripts/select_pro.py)",
            "gamma": P.GAMMA,
            "footage_credit": CREDIT,
            "oracle_objects": None,
        }
        if shot in INHOUSE:
            rec["inhouse_key"] = INHOUSE[shot].to_dict()
        rp.write_text(json.dumps(rec, indent=2) + "\n")
        print(f"[init] {d.name}: window {w['window']}, {rec['group']}")


# ------------------------------------------------------------------ helpers
def to_hd(x: np.ndarray) -> np.ndarray:
    import cv2
    if x.shape[1] == HD[0]:
        return x
    return cv2.resize(x, HD, interpolation=cv2.INTER_AREA)


def fetch_all(jobs: list[tuple[str, Path]], label: str) -> None:
    todo = [(u, p) for u, p in jobs if not (p.exists() and p.stat().st_size > 0)]
    if not todo:
        return
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: P.fetch(*j), todo))
    mb = sum(p.stat().st_size for _, p in todo) / 1e6
    print(f"   fetched {len(todo)} {label} frames, {mb:.0f} MB in "
          f"{time.perf_counter() - t0:.0f}s", flush=True)


def components(a: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    from scipy import ndimage
    m = a > 0.5
    lab, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, n + 1)) if n else np.array([])
    return lab, n, np.asarray(sizes)


# ------------------------------------------------------------------ build
def build(d: Path) -> dict:
    rec = json.loads((d / "recipe.json").read_text())
    shot, (s, e) = rec["shot"], rec["window"]
    res, digits = rec["source"]["res"], rec["source"]["digits"]
    raw = DATA / shot
    print(f"[build] {d.name}: {shot} {res} window {s}-{e}", flush=True)

    cjobs = [(P.cleaned_url(shot, n, digits, res), raw / "cleaned" / res / f"{n:0{digits}d}.exr")
             for n in range(s, e + 1)]
    pjobs = [(P.footage_url(shot, n), raw / "plate" / f"{n:05d}.exr")
             for n in range(max(0, s + min(OFFSETS) - 1), e + max(OFFSETS) + 2)]
    fetch_all(cjobs, "key"); fetch_all(pjobs, "plate")

    # --- 1. alignment, proven before anything is written ----------------------
    def plate(n):
        rgb, _ = P.read_exr(raw / "plate" / f"{n:05d}.exr")
        return rgb
    tests = [s + 8, s + (e - s) // 2, e - 8]
    ev, offs = [], []
    for t in tests:
        crgb, ca = P.read_exr(raw / "cleaned" / res / f"{t:0{digits}d}.exr")
        crgb, ca = to_hd(crgb), to_hd(ca)
        r = P.offset_residuals(crgb, ca, {t + k: plate(t + k) for k in OFFSETS})
        order = sorted(r, key=r.get)
        best, second = order[0], order[1]
        k = best - t
        sh = P.shift_residuals(crgb, ca, plate(best))
        ev.append({"cleaned_frame": t, "best_plate_frame": best, "offset": k,
                   "residual_best": round(r[best], 6), "residual_runner_up": round(r[second], 6),
                   "ratio": round(r[best] / r[second], 3),
                   "residual_by_offset": {str(f - t): round(v, 6) for f, v in sorted(r.items())},
                   "spatial_best_shift": list(min(sh, key=sh.get)),
                   "spatial_ratio": round(sh[(0, 0)] / sorted(sh.values())[1], 3)
                                    if min(sh, key=sh.get) == (0, 0) else None})
        offs.append(k)
    problems = []
    if len(set(offs)) != 1:
        problems.append(f"offset differs across the window: {offs}")
    for x in ev:
        if x["ratio"] > MAX_RATIO:
            problems.append(f"frame {x['cleaned_frame']}: no clear minimum "
                            f"(best/runner-up {x['ratio']})")
        if x["spatial_best_shift"] != [0, 0]:
            problems.append(f"frame {x['cleaned_frame']}: best spatial shift "
                            f"{x['spatial_best_shift']}, not (0, 0)")
    rec["alignment"] = {"method": "median |plate - key| over opaque key pixels, red "
                                  "channel, linear light; offsets " + str(list(OFFSETS)),
                        "offset": offs[0] if len(set(offs)) == 1 else None,
                        "verdict": "PROVEN" if not problems else "REFUSED",
                        "problems": problems, "evidence": ev}
    (d / "recipe.json").write_text(json.dumps(rec, indent=2) + "\n")
    for x in ev:
        print(f"   align: key {x['cleaned_frame']} <-> plate {x['best_plate_frame']} "
              f"(offset {x['offset']:+d}), residual {x['residual_best']:.5f} vs runner-up "
              f"{x['residual_runner_up']:.5f} (ratio {x['ratio']}), spatial best "
              f"{tuple(x['spatial_best_shift'])}", flush=True)
    if problems:
        print(f"   REFUSED: {'; '.join(problems)}")
        return rec
    k = offs[0]

    # --- 2. frames and truth ---------------------------------------------------
    for sub in ("frames", "alpha", "ignore", "alpha_inhouse"):
        if (d / sub).exists():
            shutil.rmtree(d / sub)
    (d / "frames").mkdir(); (d / "alpha").mkdir()
    alphas = []
    for i, n in enumerate(range(s, e + 1)):
        _, a = P.read_exr(raw / "cleaned" / res / f"{n:0{digits}d}.exr")
        a = np.clip(to_hd(a), 0, 1)
        alphas.append(a)
        Image.fromarray(P.to_display(plate(n + k))).save(d / "frames" / f"{i:05d}.jpg",
                                                         quality=95)
        Image.fromarray((a * 255 + 0.5).astype(np.uint8)).save(d / "alpha" / f"{i:05d}.png")

    # --- 3. full-rate lineage: who is the subject, what is ignored -------------
    # Only discrete non-subject OBJECTS are ignored: a component whose opaque core is
    # not descended from the subject (a flag edge, a floor marker), together with its
    # own soft edge. Soft pixels connected to the subject - hair wisps, the halo - are
    # never ignored; they are the point of the exercise. (A first version ignored every
    # soft pixel more than 5 px from the opaque core, which silently removed the outer
    # hair from scoring. Caught on P01 before anything was scored.)
    import cv2
    from scipy import ndimage
    reach_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    pad_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    lab, n, sizes = components(alphas[0])
    subject = np.isin(lab, [j + 1 for j, z in enumerate(sizes) if z >= SIG * lab.size])
    ignore_px, entrants = [], []
    for i, a in enumerate(alphas):
        lab, n, sizes = components(a)
        low, _ = ndimage.label(a > 0.05)            # soft extent of every object
        reach = cv2.dilate(subject.astype(np.uint8), reach_k) > 0
        now = np.zeros_like(subject); specks = []
        for j, z in enumerate(sizes, start=1):
            c = lab == j
            if (c & reach).any():
                now |= c
            elif z >= SIG * c.size:
                entrants.append(i); now |= c        # recorded; clip is refused below
            else:
                specks.append(c)
        subject = now
        subj_low = np.isin(low, np.unique(low[subject & (low > 0)]))
        ign = np.zeros_like(subject)
        for c in specks:
            ids = np.unique(low[c & (low > 0)])
            ign |= np.isin(low, ids) & ~subj_low
        ign = (cv2.dilate(ign.astype(np.uint8), pad_k) > 0) & ~subj_low
        ignore_px.append(int(ign.sum()))
        if ign.any():
            (d / "ignore").mkdir(exist_ok=True)
            Image.fromarray(ign.astype(np.uint8) * 255).save(d / "ignore" / f"{i:05d}.png")
    if entrants:
        rec["built"] = {"refused": f"significant late entrant at frame(s) "
                                   f"{sorted(set(entrants))[:8]} at full frame rate"}
        (d / "recipe.json").write_text(json.dumps(rec, indent=2) + "\n")
        print(f"   REFUSED: {rec['built']['refused']}")
        return rec

    # --- 4. the in-house key on the same frames, where one is defined ----------
    if rec.get("inhouse_key"):
        kp = truth.KeyParams(**{kk: (tuple(v) if kk == "garbage" and v else v)
                                for kk, v in rec["inhouse_key"].items()})
        (d / "alpha_inhouse").mkdir()
        for i in range(len(alphas)):
            rgb = np.asarray(Image.open(d / "frames" / f"{i:05d}.jpg").convert("RGB"))
            a, _ = truth.chroma_key(rgb, kp)
            Image.fromarray(a).save(d / "alpha_inhouse" / f"{i:05d}.png")

    A = np.stack(alphas)
    rec["built"] = {
        "frames": len(alphas), "resolution": list(HD),
        "plate_frames": [s + k, e + k], "key_frames": [s, e],
        "alpha_coverage_mean": round(float((A > 0.5).mean()), 5),
        "alpha_soft_pixel_fraction": round(float(((A > 0.02) & (A < 0.98)).mean()), 6),
        "alpha_distinct_values": int(len(np.unique((A * 255 + 0.5).astype(np.uint8)))),
        "ignored_pixels_mean": round(float(np.mean(ignore_px)), 1),
        "ignored_frames": int(sum(1 for x in ignore_px if x)),
        "downscaled_from_4k": res == "linear",
    }
    (d / "recipe.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(f"   built: coverage {100 * rec['built']['alpha_coverage_mean']:.1f}%, soft px "
          f"{100 * rec['built']['alpha_soft_pixel_fraction']:.2f}%, ignored "
          f"{rec['built']['ignored_pixels_mean']:.0f} px/frame on "
          f"{rec['built']['ignored_frames']} frames", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--clip", action="append")
    args = ap.parse_args()
    if args.init:
        init(); return
    clips = sorted(p for p in TRUTH.glob("P*") if (p / "recipe.json").exists()
                   and "excluded" not in json.loads((p / "recipe.json").read_text()))
    if args.clip:
        clips = [c for c in clips if c.name in set(args.clip)]
    for d in clips:
        build(d)


if __name__ == "__main__":
    main()
