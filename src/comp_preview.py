#!/usr/bin/env python3
"""Composite the tracked actor over new backgrounds and render preview videos.

Thin CLI over cleanplate.compose - the app calls the same functions.

    python src/comp_preview.py --shot walk --out-name walk_v2 \
        --bg-crop 380,125,1680,600 --compare-with walk

Writes, under outputs/<out-name>/:
    comp_solid/, comp_image/   comp frames
    comp.mp4, comp_solid.mp4   the reveal, and the flat-colour edge stress test
    side_by_side.mp4           original | matte | comp
    <old>_vs_<new>.mp4         2x2, mattes above and comps below (with --compare-with)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import _bootstrap  # noqa: F401
from cleanplate import compose
from cleanplate.ingest import load_frame
from cleanplate.paths import ROOT, out_dir, rel


def _font(size: int):
    try:
        import matplotlib
        p = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        return ImageFont.truetype(str(p), size)
    except Exception:
        return ImageFont.load_default()


def label(im: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(im)
    f = _font(max(14, im.width // 42))
    pad = max(6, im.width // 90)
    box = d.textbbox((0, 0), text, font=f)
    d.rectangle([pad - 4, pad - 4, pad + box[2] - box[0] + 10, pad + box[3] - box[1] + 12],
                fill=(0, 0, 0))
    d.text((pad + 2, pad), text, font=f, fill=(255, 255, 255))
    return im


def parse_crop(s: str) -> tuple[int, int, int, int]:
    try:
        l, t, r, b = (int(v) for v in s.replace(" ", "").split(","))
        return l, t, r, b
    except Exception:
        raise argparse.ArgumentTypeError(f"expected LEFT,TOP,RIGHT,BOTTOM - got {s!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--bg-image", type=Path)
    ap.add_argument("--bg-crop", type=parse_crop, metavar="L,T,R,B")
    ap.add_argument("--color", default="#D4571E", metavar="#RRGGBB",
                    help="flat plate; a warm hue makes cool edge halos obvious")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--compare-with", metavar="RUN")
    ap.add_argument("--compare-labels", nargs=2, default=["v1", "v2"])
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args()

    name = args.out_name or args.shot
    root = out_dir(name)
    rgba_dir = root / "rgba"
    if not rgba_dir.is_dir():
        sys.exit(f"no RGBA sequence: {rel(rgba_dir)}\n"
                 f"Run src/export_rgba.py --shot {args.shot} --out-name {name}")
    paths = sorted(rgba_dir.glob("*.png"), key=lambda p: int(p.stem))
    size = Image.open(paths[0]).size

    bg_path = args.bg_image
    if bg_path is None:
        cands = [p for p in (root / "backgrounds").glob("*")
                 if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        if len(cands) != 1:
            sys.exit(f"pass --bg-image (found {len(cands)} in {rel(root / 'backgrounds')})")
        bg_path = cands[0]

    print(f"[comp] shot={args.shot}  {len(paths)} frames at {size[0]}x{size[1]}")
    print(f"[comp] plate: {rel(bg_path)}")
    print(f"[comp] flat colour: {args.color}")
    plate = compose.cover_fit(Image.open(bg_path).convert("RGB"), size, args.bg_crop)
    if args.bg_crop:
        print(f"[comp] plate cropped from {args.bg_crop}")
    flat = compose.solid(size, compose.hex_rgb(args.color))

    dirs = {k: root / k for k in ("comp_solid", "comp_image", "_sbs")}
    for d in dirs.values():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for p in paths:
        idx = int(p.stem)
        arr = np.asarray(Image.open(p).convert("RGBA"))
        fg, a = arr[..., :3], arr[..., 3]
        Image.fromarray(compose.over(fg, a, flat)).save(dirs["comp_solid"] / f"{idx:05d}.png")
        comp = compose.over(fg, a, plate)
        Image.fromarray(comp).save(dirs["comp_image"] / f"{idx:05d}.png")
        strip = Image.new("RGB", (size[0] * 3, size[1]))
        strip.paste(label(Image.fromarray(load_frame(args.shot, idx)), "ORIGINAL"), (0, 0))
        strip.paste(label(Image.fromarray(a).convert("RGB"), "MATTE"), (size[0], 0))
        strip.paste(label(Image.fromarray(comp), "COMP"), (size[0] * 2, 0))
        strip.save(dirs["_sbs"] / f"{idx:05d}.png")

    if args.compare_with:
        other = out_dir(args.compare_with)
        o_alpha = other / "alpha" if (other / "alpha").is_dir() else other / "masks"
        o_comp = other / "comp_image"
        missing = [rel(d) for d in (o_alpha, o_comp) if not d.is_dir()]
        if missing:
            sys.exit(f"--compare-with {args.compare_with}: missing {missing}")
        old_lab, new_lab = args.compare_labels
        cmp_dir = root / "_cmp"
        if cmp_dir.exists():
            shutil.rmtree(cmp_dir)
        cmp_dir.mkdir(parents=True)
        for p in paths:
            idx = int(p.stem)
            grid = Image.new("RGB", (size[0] * 2, size[1] * 2))
            grid.paste(label(Image.open(o_alpha / f"{idx:05d}.png").convert("RGB"),
                             f"{old_lab}  MATTE"), (0, 0))
            grid.paste(label(Image.open(p).convert("RGBA").getchannel("A").convert("RGB"),
                             f"{new_lab}  MATTE"), (size[0], 0))
            grid.paste(label(Image.open(o_comp / f"{idx:05d}.png").convert("RGB"),
                             f"{old_lab}  COMP"), (0, size[1]))
            grid.paste(label(Image.open(dirs["comp_image"] / f"{idx:05d}.png").convert("RGB"),
                             f"{new_lab}  COMP"), (size[0], size[1]))
            grid.save(cmp_dir / f"{idx:05d}.png")
        out = compose.encode(cmp_dir, root / f"{old_lab}_vs_{new_lab}.mp4", args.fps)
        print(f"[comp] {rel(out)}  ({out.stat().st_size / 1e6:.1f} MB)")
        if not args.keep_frames:
            shutil.rmtree(cmp_dir)

    for src, dst in [("comp_image", "comp.mp4"), ("comp_solid", "comp_solid.mp4"),
                     ("_sbs", "side_by_side.mp4")]:
        out = compose.encode(dirs[src], root / dst, args.fps)
        print(f"[comp] {rel(out)}  ({out.stat().st_size / 1e6:.1f} MB)")
    if not args.keep_frames:
        shutil.rmtree(dirs["_sbs"])
    print(f"[comp] comp frames kept in {rel(dirs['comp_image'])} and "
          f"{rel(dirs['comp_solid'])}")


if __name__ == "__main__":
    main()
