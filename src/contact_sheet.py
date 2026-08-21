#!/usr/bin/env python3
"""Build contact sheets, either to scout a movie for shots or to review one shot.

    # scout: one thumbnail every 5s across the whole film, timecode-stamped
    python src/contact_sheet.py --movie shots/source/tears_of_steel_1080p.webm \
        --every 5 --out outputs/scout.jpg

    # review: every Nth frame of an extracted shot, frame-numbered
    python src/contact_sheet.py --shot dialogue --out outputs/dialogue_sheet.jpg
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent

# DejaVuSans ships with matplotlib, so it is always present in this venv.
def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        import matplotlib
        p = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        return ImageFont.truetype(str(p), size)
    except Exception:
        return ImageFont.load_default()


def tile(images: list[tuple[Image.Image, str]], cols: int, thumb_w: int,
         pad: int = 6, bg=(18, 18, 20), fg=(255, 255, 255)) -> Image.Image:
    if not images:
        sys.exit("nothing to tile")
    ar = images[0][0].height / images[0][0].width
    thumb_h = int(round(thumb_w * ar))
    label_h = max(16, thumb_w // 11)
    rows = (len(images) + cols - 1) // cols
    cell_w, cell_h = thumb_w + pad, thumb_h + label_h + pad
    sheet = Image.new("RGB", (cols * cell_w + pad, rows * cell_h + pad), bg)
    draw = ImageDraw.Draw(sheet)
    font = _font(int(label_h * 0.72))

    for i, (im, label) in enumerate(images):
        r, c = divmod(i, cols)
        x, y = pad + c * cell_w, pad + r * cell_h
        sheet.paste(im.resize((thumb_w, thumb_h), Image.LANCZOS), (x, y))
        draw.text((x + 3, y + thumb_h + 1), label, font=font, fill=fg)
    return sheet


def from_movie(movie: Path, every: float, thumb_w: int, cols: int,
               start: float, end: float | None) -> list[tuple[Image.Image, str]]:
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH")
    out: list[tuple[Image.Image, str]] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        dur = (end - start) if end else None
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-ss", str(start), "-i", str(movie)]
        if dur:
            cmd += ["-t", str(dur)]
        cmd += ["-vf", f"fps=1/{every},scale={thumb_w}:-2", "-q:v", "4",
                "-start_number", "0", str(tmp / "%05d.jpg")]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(proc.stderr[-2000:])
        for p in sorted(tmp.glob("*.jpg"), key=lambda q: int(q.stem)):
            # ffmpeg's fps filter emits its first sample at t=0 of the trimmed
            # stream, so thumbnail n sits at start + n*every seconds.
            t = start + int(p.stem) * every
            out.append((Image.open(p).copy(), f"{int(t)//60:d}:{int(t)%60:02d} ({t:.0f}s)"))
    return out


def from_shot(shot: str, step: int, thumb_w: int) -> list[tuple[Image.Image, str]]:
    frames_dir = ROOT / "shots" / shot / "frames"
    if not frames_dir.is_dir():
        sys.exit(f"no such shot folder: {frames_dir}")
    frames = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if not frames:
        sys.exit(f"no frames in {frames_dir}")
    picked = frames[::step]
    return [(Image.open(p).copy(), f"f{int(p.stem):04d}") for p in picked]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--movie", type=Path, help="scout mode: sample a whole movie")
    g.add_argument("--shot", help="review mode: sample shots/<name>/frames")
    ap.add_argument("--every", type=float, default=5.0, help="scout: seconds between thumbs")
    ap.add_argument("--start", type=float, default=0.0, help="scout: start second")
    ap.add_argument("--end", type=float, default=None, help="scout: end second")
    ap.add_argument("--step", type=int, default=6, help="review: take every Nth frame")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--thumb-width", type=int, default=240)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.movie:
        imgs = from_movie(args.movie, args.every, args.thumb_width, args.cols,
                          args.start, args.end)
    else:
        imgs = from_shot(args.shot, args.step, args.thumb_width)

    sheet = tile(imgs, args.cols, args.thumb_width)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=92)
    print(f"[sheet] {len(imgs)} tiles, {sheet.width}x{sheet.height} -> {args.out}")


if __name__ == "__main__":
    main()
