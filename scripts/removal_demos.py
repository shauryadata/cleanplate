#!/usr/bin/env python3
"""Three real removals, not synthetic ones. Before/after side-by-sides.

  A  walk       an element of opportunity - the cast-iron lamppost at frame left
  B  08_3a      the tracking markers on the green screen (genuine VFX prep)
  C  dialogue   one of two actors, leaving the other (selective removal)

    python scripts/removal_demos.py --demo A --demo B --demo C
"""
from __future__ import annotations

import os
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import compose, refine, remove, track                  # noqa: E402
from cleanplate.ingest import frame_paths, load_frame, resolve_frames_dir  # noqa: E402
from cleanplate.memguard import MemoryAbort                            # noqa: E402
from cleanplate.paths import ROOT, rel                                 # noqa: E402
from cleanplate.session import Prompt                                  # noqa: E402

OUT = ROOT / "outputs" / "_demos"

# Demo B's budget, set by measurement rather than taste — see the note in main().
B_FRAMES = 48
B_CHUNK = 6


def marker_masks(plate_dir: Path, alpha_dir: Path, n: int) -> tuple[np.ndarray, float]:
    """Tracking markers on a green screen, found rather than clicked.

    The markers are the small non-green objects stuck to the backing: a white paper
    square on the backing itself, and the paper and tape tabs on the flag at frame
    right. The Tier B keyer already had to identify and discard them; here the same
    signal is the target instead of the nuisance, which shows that a removal mask does
    not have to come from a click.

    The subject is excluded using the Tier B keyed alpha rather than a fixed box,
    because the actor moves across the plate over the 96 frames.
    """
    import cv2
    from scipy import ndimage
    ps = sorted(plate_dir.glob("*.exr"), key=lambda p: int(p.stem))[:n]
    aps = sorted(alpha_dir.glob("*.png"), key=lambda p: int(p.stem))[:n]
    if len(ps) != len(aps):
        raise SystemExit(f"{len(ps)} plates but {len(aps)} alphas")
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    out, kept = [], 0
    k_sub = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    for p, ap in zip(ps, aps):
        lin = cv2.imread(str(p), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH
                         | cv2.IMREAD_ANYCOLOR)
        if lin is None:
            raise SystemExit(f"could not read {p} - OpenEXR support missing")
        rgb = np.clip(cv2.cvtColor(lin[..., :3], cv2.COLOR_BGR2RGB), 0, 1) ** (1 / 2.2)
        hsv = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        # Same greenness test the Tier B keyer uses, same thresholds.
        green = (np.abs(((h - 120.0 + 180) % 360) - 180) < 44) & (s > 0.16) & (v > 0.05)
        subj = cv2.dilate((np.asarray(Image.open(ap)) > 8).astype(np.uint8), k_sub, 1) > 0
        # The largest marker on the backing is badly out of focus, so the green bleeds
        # through it and it still passes the greenness test. What it cannot fake is
        # saturation: it sits far below the backing around it. A global threshold will
        # not do, because the backing's own saturation falls off with the lighting, so
        # compare each pixel to a local mean of the backing instead. Measured on frame
        # 0: the marker's drop is 0.287 median, clean backing is 0.182 at p99.9, so
        # 0.20 separates them.
        num = cv2.blur(np.where(subj, 0, s).astype(np.float32), (151, 151))
        den = cv2.blur((~subj).astype(np.float32), (151, 151))
        drop = num / np.maximum(den, 1e-3) - s
        cand = ((~green) | (drop > 0.20)) & (~subj)
        cand[:6] = cand[-6:] = False           # frame edges are not markers
        cand[:, :6] = cand[:, -6:] = False
        lab, nk = ndimage.label(cand)
        keep = np.zeros_like(cand)
        if nk:
            sizes = ndimage.sum(cand, lab, range(1, nk + 1))
            for j, sz in enumerate(sizes, start=1):
                if 40 <= sz <= 12000:          # marker-sized: not the flag, not noise
                    keep |= (lab == j); kept += 1
        out.append(cv2.dilate(keep.astype(np.uint8) * 255, np.ones((7, 7), np.uint8), 1))
    return np.stack(out), kept / max(1, len(ps))


def matte_for(shot: str, clicks: list[tuple[int, int]], model="matanyone2") -> np.ndarray:
    p = Prompt()
    for xy in clicks:
        p.add(0, *xy, True)
    masks, _ = track.track(shot, p, progress=None)
    alpha, _ = refine.refine(shot, masks, model=model, progress=None)
    return alpha


def stage(images: np.ndarray, d: Path, ext: str = ".jpg") -> Path:
    """Write frames where another tool can read them, numbered from zero.

    ProPainter is happy with JPEG; `compose.encode` builds its ffmpeg input pattern as
    `%05d.png` and is not, so the side-by-side has to be staged as PNG.
    """
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for i, im in enumerate(images):
        Image.fromarray(im).save(d / f"{i:05d}{ext}", quality=97)
    return d


def in_shot_span(alpha: np.ndarray, floor: float = 0.15) -> int:
    """How many frames the object is actually in shot for.

    The walk shot is a tracking shot: the camera pans right the whole way, so a piece
    of street furniture at frame left is gone within a second. Running removal over
    frames where the mask is empty would pad the demo with 70 frames of nothing
    happening, so the demo covers the frames where the object exists and says so.
    """
    area = (alpha > 12).reshape(len(alpha), -1).sum(1).astype(float)
    if area[0] <= 0:
        raise SystemExit("empty mask on frame 0 — check the clicks")
    live = area >= floor * area[0]
    n = int(np.argmin(live)) if not live.all() else len(alpha)
    return max(n, 8)


def render(name: str, before: np.ndarray, cleaned: np.ndarray, holes: np.ndarray,
           stats: dict, caption: str) -> None:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    sbs = stage(np.stack([np.hstack([b, c]) for b, c in zip(before, cleaned)]),
                d / "_sbs", ext=".png")
    compose.encode(sbs, d / "before_after.mp4", 24.0)
    shutil.rmtree(sbs)
    for i in (0, len(cleaned) // 2, len(cleaned) - 1):
        Image.fromarray(np.vstack([before[i], cleaned[i]])).save(d / f"still_{i:05d}.png")
    stats = dict(stats, demo=caption)
    (d / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"   -> {rel(d)}  before_after.mp4 + 3 stills")


def load_plate(n: int, width: int = 960) -> np.ndarray:
    """08_3a is linear half-float OpenEXR; convert to display-referred 8-bit."""
    import cv2
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    ps = sorted((ROOT / "datasets" / "tos_plates" / "08_3a").glob("*.exr"),
                key=lambda q: int(q.stem))[:n]
    out = []
    for q in ps:
        lin = cv2.imread(str(q), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH
                         | cv2.IMREAD_ANYCOLOR)
        rgb = np.clip(cv2.cvtColor(lin[..., :3], cv2.COLOR_BGR2RGB), 0, 1) ** (1 / 2.2)
        rgb = (rgb * 255).astype(np.uint8)
        if width and rgb.shape[1] != width:
            h = int(round(rgb.shape[0] * width / rgb.shape[1])); h -= h % 2
            rgb = cv2.resize(rgb, (width, h), interpolation=cv2.INTER_AREA)
        out.append(rgb)
    return np.stack(out)


def frames_of(shot: str, n: int | None = None) -> np.ndarray:
    ps = sorted(resolve_frames_dir(shot).glob("*.jpg"), key=lambda q: int(q.stem))[:n]
    return np.stack([np.asarray(Image.open(q).convert("RGB")) for q in ps])


DEMOS = {
    "A": ("A_walk_lamppost", "walk shot: the cast-iron lamppost at frame left"),
    "B": ("B_greenscreen_markers", "08_3a plate: tracking markers on the green backing"),
    "C": ("C_dialogue_one_actor", "dialogue: one of the two actors, leaving the other"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="append", default=None)
    ap.add_argument("--preflight", action="store_true",
                    help="dump the masks and stop, before spending inpainting time")
    args = ap.parse_args()
    want = [d for d in ["A", "B", "C"] if d in set(args.demo or ["A", "B", "C"])]
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "_tmp"

    for which in want:
        name, caption = DEMOS[which]
        print(f"[demo {which}] {caption}")
        t0 = time.time()

        if which == "A":
            chunk = 8
            alpha = matte_for("walk", [(88, 60), (85, 190), (82, 330)])
            n = in_shot_span(alpha)
            print(f"   lamppost is in shot for {n} of {len(alpha)} frames "
                  f"(the camera pans off it); removing over those {n}")
            alpha, before = alpha[:n], frames_of("walk", n)
            src, dil = stage(before, tmp), 14
        elif which == "B":
            # Half length, full resolution. The full 96 frames at 960x506 is 1.27x the
            # pixel budget of the Tier R clips and it tripped the memory guard at
            # 465 MB reclaimable — see docs/REMOVAL_BENCH.md. Of the two ways to make
            # it fit, dropping resolution would blunt the thing the demo is about
            # (the markers are small), while dropping duration costs nothing: this is
            # a locked-off green screen and two seconds shows the removal as well as
            # four. So the frames go, not the pixels.
            before = load_plate(B_FRAMES, width=960)
            holes, per = marker_masks(ROOT / "datasets" / "tos_plates" / "08_3a",
                                      ROOT / "truth" / "B1_tos_greenscreen_hair" / "alpha",
                                      len(before))
            import cv2
            alpha = np.stack([cv2.resize(h, before.shape[1:3][::-1],
                                         interpolation=cv2.INTER_NEAREST)
                              for h in holes])
            print(f"   found {per:.0f} marker blobs per frame covering "
                  f"{100 * (alpha > 127).mean():.2f}% of frame — detected, not clicked")
            src, dil, chunk = stage(before, tmp), 6, B_CHUNK
        else:
            chunk = 8
            alpha = matte_for("dialogue", [(320, 60), (320, 165), (330, 315)])
            before = frames_of("dialogue")
            src, dil = resolve_frames_dir("dialogue"), 14

        if args.preflight:
            holes = remove.make_hole(alpha, dilate=dil)
            sheet = []
            for i in (0, len(alpha) // 2, len(alpha) - 1):
                o = before[i].copy(); m = holes[i] > 127
                o[m] = (0.3 * o[m] + 0.7 * np.array([255, 0, 90])).astype(np.uint8)
                sheet.append(o)
            Image.fromarray(np.vstack(sheet)).save(OUT / f"preflight_{which}.png")
            print(f"   preflight -> {rel(OUT / f'preflight_{which}.png')}  "
                  f"hole {100 * (holes > 127).mean():.2f}% of frame")
            continue

        try:
            r = remove.remove(src, alpha, dilate=dil, subvideo_length=chunk,
                              guard_label=f"demo {which}", progress=None)
        except MemoryAbort as e:
            print(f"   ABORTED: {e}")
            continue
        print(f"   {r.stats['seconds_per_frame']:.2f} s/f, hole "
              f"{100 * r.stats['hole_area_fraction_mean']:.2f}% of frame, "
              f"{time.time() - t0:.0f}s total", flush=True)
        # Park the arrays before rendering. Inpainting is minutes; a video-encode bug
        # should not cost that twice.
        (OUT / name).mkdir(parents=True, exist_ok=True)
        np.save(OUT / name / "cleaned.npy", r.frames)
        np.save(OUT / name / "before.npy", before)
        render(name, before, r.frames, r.holes, r.stats, caption)

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
