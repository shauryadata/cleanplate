#!/usr/bin/env python3
"""Cut one shot out of a source movie into a folder of JPEG frames.

SAM 2's video predictor wants a directory of JPEGs whose stems are integers, so
frames are written as ``00000.jpg``, ``00001.jpg``, ... A ``shot.json`` sidecar
records exactly where the frames came from, so a shot is reproducible from the
source movie alone.

    python src/extract_shot.py --start 00:01:23.5 --name dialogue
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "shots" / "source" / "tears_of_steel_1080p.webm"
FOOTAGE_CREDIT = "(CC) Blender Foundation | mango.blender.org"


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{proc.stderr[-2000:]}")
    return proc.stdout


def probe(source: Path) -> dict:
    out = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-show_entries", "format=duration",
        "-of", "json", str(source),
    ])
    return json.loads(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--start", required=True,
                    help="start timecode, e.g. 00:01:23.5 or 83.5")
    ap.add_argument("--duration", type=float, default=4.0, help="seconds (default 4)")
    ap.add_argument("--name", required=True, help="shot name -> shots/<name>/frames/")
    ap.add_argument("--width", type=int, default=960, help="output width (default 960)")
    ap.add_argument("--fps", type=float, default=24.0, help="output fps (default 24)")
    ap.add_argument("--quality", type=int, default=2,
                    help="ffmpeg -q:v, 2 = near-lossless JPEG (default 2)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing shot")
    args = ap.parse_args()

    if not args.source.exists():
        sys.exit(f"source movie not found: {args.source}\nRun ./scripts/download.sh footage")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH (brew install ffmpeg)")

    shot_dir = ROOT / "shots" / args.name
    frames_dir = shot_dir / "frames"
    if frames_dir.exists() and any(frames_dir.iterdir()):
        if not args.force:
            sys.exit(f"{frames_dir} already has frames; pass --force to overwrite")
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    info = probe(args.source)
    stream = info["streams"][0]

    # -ss before -i seeks fast; -copyts is deliberately NOT used so the output
    # timeline restarts at 0 and frame 00000 is the first frame of the shot.
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(args.start),
        "-i", str(args.source),
        "-t", str(args.duration),
        "-vf", f"fps={args.fps},scale={args.width}:-2:flags=lanczos",
        "-q:v", str(args.quality),
        "-start_number", "0",
        str(frames_dir / "%05d.jpg"),
    ]
    print(f"[extract] {args.name}: {args.start} +{args.duration}s "
          f"-> {args.width}px @ {args.fps}fps")
    run(cmd)

    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        sys.exit("ffmpeg produced no frames - is --start past the end of the movie?")

    from PIL import Image
    with Image.open(frames[0]) as im:
        out_w, out_h = im.size

    meta = {
        "name": args.name,
        "source_movie": str(args.source.relative_to(ROOT)),
        "source_url": "https://media.xiph.org/tearsofsteel/tears_of_steel_1080p.webm",
        "credit": FOOTAGE_CREDIT,
        "start": args.start,
        "duration_s": args.duration,
        "fps": args.fps,
        "frame_count": len(frames),
        "resolution": [out_w, out_h],
        "source_resolution": [stream.get("width"), stream.get("height")],
        "source_fps": stream.get("r_frame_rate"),
        "ffmpeg_cmd": " ".join(cmd),
    }
    (shot_dir / "shot.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[extract] {len(frames)} frames at {out_w}x{out_h} -> "
          f"{frames_dir.relative_to(ROOT)}")
    print(f"[extract] metadata -> {(shot_dir / 'shot.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
