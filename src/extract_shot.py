#!/usr/bin/env python3
"""Cut one shot out of a source movie into a folder of JPEG frames.

Thin CLI over cleanplate.ingest.extract_shot - the app calls the same function.

    python src/extract_shot.py --name dialogue --start 00:00:30 --duration 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from cleanplate.ingest import extract_shot
from cleanplate.paths import SOURCE_MOVIE, rel, shot_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=SOURCE_MOVIE)
    ap.add_argument("--start", required=True, help="timecode, e.g. 00:01:23.5 or 83.5")
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--name", required=True, help="shot name -> shots/<name>/frames/")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--quality", type=int, default=2, help="ffmpeg -q:v (2 = near-lossless)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print(f"[extract] {args.name}: {args.start} +{args.duration}s "
          f"-> {args.width}px @ {args.fps}fps")
    try:
        meta = extract_shot(args.name, args.start, args.duration, args.source,
                            args.width, args.fps, args.quality, args.force)
    except (FileNotFoundError, FileExistsError, RuntimeError) as e:
        sys.exit(str(e))
    w, h = meta["resolution"]
    print(f"[extract] {meta['frame_count']} frames at {w}x{h} -> "
          f"{rel(shot_dir(args.name) / 'frames')}")
    print(f"[extract] metadata -> {rel(shot_dir(args.name) / 'shot.json')}")


if __name__ == "__main__":
    main()
