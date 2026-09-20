#!/usr/bin/env python3
"""One command: clicks in, matte and comp out. The engine behind the Colab notebook.

    python scripts/colab_demo.py --shot walk --click 470,150 --click 455,300
    python scripts/colab_demo.py --video my.mp4 --start 3 --seconds 4 --click 600,400

The notebook is a thin wrapper around this so the thing Colab runs is the thing that
can be tested on a laptop. Everything it does is the repo's own pipeline: SAM 2 for the
mask, MatAnyone 2 for the soft alpha, despill and comp from cleanplate.compose.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import compose, ingest, refine, track                  # noqa: E402
from cleanplate.paths import ROOT, out_dir, rel                        # noqa: E402
from cleanplate.session import Prompt                                  # noqa: E402


def parse_click(s: str) -> tuple[int, int]:
    x, y = s.replace(" ", "").split(",")
    return int(x), int(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", help="an existing shot name under shots/")
    ap.add_argument("--video", type=Path, help="or a video file to cut a shot from")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--click", action="append", required=True,
                    metavar="X,Y", help="repeatable; coordinates in the extracted frame")
    ap.add_argument("--negative", action="append", default=[], metavar="X,Y")
    ap.add_argument("--bg", type=Path, help="background image for the comp")
    ap.add_argument("--bg-colour", default="#D4571E")
    ap.add_argument("--no-harmonize", action="store_true")
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    if not args.shot and not args.video:
        sys.exit("pass --shot or --video")
    shot = args.shot
    if args.video:
        shot = args.video.stem.lower().replace(" ", "_")[:24] or "clip"
        print(f"[1/5] cutting {args.seconds:.0f}s from {args.video.name} -> shot '{shot}'")
        ingest.extract_shot(shot, start=args.start, duration=args.seconds,
                            source=args.video, width=args.width, force=True)
    n = len(ingest.frame_paths(shot))
    w, h = ingest.frame_size(shot)
    print(f"      {n} frames at {w}x{h}")

    prompt = Prompt()
    for c in args.click:
        prompt.add(0, *parse_click(c), positive=True)
    for c in args.negative:
        prompt.add(0, *parse_click(c), positive=False)
    print(f"[2/5] tracking from {prompt.total_clicks} click(s) on frame 0")
    t0 = time.perf_counter()
    masks, tstats = track.track(shot, prompt, progress=None)
    print(f"      SAM 2: {tstats['seconds_per_frame']:.2f} s/frame on {tstats['device']}")

    print("[3/5] refining to a soft alpha (MatAnyone 2)")
    alphas, rstats = refine.refine(shot, masks, model="matanyone2", progress=None)
    print(f"      {rstats['seconds_per_frame']:.2f} s/frame on {rstats['device']}")

    out = out_dir(args.out_name or f"{shot}_colab")
    for sub in ("matte", "comp"):
        d = out / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    print("[4/5] despill, RGBA, comp")
    plate = None
    if args.bg and args.bg.exists():
        plate = compose.cover_fit(Image.open(args.bg).convert("RGB"), (w, h))
    else:
        plate = compose.solid((w, h), compose.hex_rgb(args.bg_colour))
    hm = None
    if not args.no_harmonize and args.bg:
        from cleanplate import harmonize
        frames = [ingest.load_frame(shot, i) for i in range(0, n, 8)]
        hm = harmonize.estimate(frames, alphas[::8], [plate] * len(frames),
                                strength=0.5, mode="full", sample=1)
    for i in range(n):
        rgb = ingest.load_frame(shot, i)
        rgb, _ = compose.despill_green(rgb, alphas[i])
        fg = harmonize.apply(rgb, hm, alphas[i], seed=i) if hm is not None else rgb
        Image.fromarray(alphas[i]).save(out / "matte" / f"{i:05d}.png")
        Image.fromarray(compose.over(fg, alphas[i], plate)).save(
            out / "comp" / f"{i:05d}.png")

    print("[5/5] encoding")
    compose.encode(out / "matte", out / "matte.mp4", 24.0)
    compose.encode(out / "comp", out / "comp.mp4", 24.0)
    sheet = np.hstack([ingest.load_frame(shot, n // 2),
                       np.dstack([alphas[n // 2]] * 3),
                       np.asarray(Image.open(out / "comp" / f"{n // 2:05d}.png"))])
    Image.fromarray(sheet).save(out / "contact.jpg", quality=92)
    stats = {"shot": shot, "frames": n, "resolution": [w, h],
             "clicks": [parse_click(c) for c in args.click],
             "track_s_per_frame": tstats["seconds_per_frame"],
             "refine_s_per_frame": rstats["seconds_per_frame"],
             "device": tstats["device"],
             "harmonize": hm.to_dict() if hm is not None else None,
             "total_s": round(time.perf_counter() - t0, 1)}
    (out / "colab_demo.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"\ndone in {stats['total_s']:.0f}s -> {rel(out)}")
    print(f"  {rel(out / 'matte.mp4')}\n  {rel(out / 'comp.mp4')}\n"
          f"  {rel(out / 'contact.jpg')}")


if __name__ == "__main__":
    main()
