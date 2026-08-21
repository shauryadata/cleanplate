#!/usr/bin/env python3
"""Composite the tracked actor over new backgrounds and render preview videos.

    python src/comp_preview.py --shot walk \
        --bg-image outputs/walk/backgrounds/mars_curiosity_afternoon.png

Writes, under outputs/<shot>/:
    comp_solid/NNNNN.png     actor over a flat colour  (edge/halo check)
    comp_image/NNNNN.png     actor over the free plate (the reveal)
    comp_solid.mp4
    comp.mp4                 the image comp
    side_by_side.mp4         original | matte | comp, labelled

Compositing is straight-alpha over: out = fg*a + bg*(1-a), done in float32 so the
edge pixels are not quantised twice. The background plate is cover-fitted (scaled
to fill, centre-cropped) rather than squashed.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def font(size: int) -> ImageFont.FreeTypeFont:
    try:
        import matplotlib
        p = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        return ImageFont.truetype(str(p), size)
    except Exception:
        return ImageFont.load_default()


def hex_rgb(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    if len(s) != 6:
        raise argparse.ArgumentTypeError(f"expected #RRGGBB, got {s!r}")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def parse_crop(s: str) -> tuple[int, int, int, int]:
    try:
        l, t, r, b = (int(v) for v in s.replace(" ", "").split(","))
        return l, t, r, b
    except Exception:
        raise argparse.ArgumentTypeError(f"expected LEFT,TOP,RIGHT,BOTTOM - got {s!r}")


def cover_fit(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale to fill the target and centre-crop, preserving aspect ratio."""
    tw, th = size
    scale = max(tw / im.width, th / im.height)
    nw, nh = max(tw, int(round(im.width * scale))), max(th, int(round(im.height * scale)))
    im = im.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - tw) // 2, (nh - th) // 2
    return im.crop((left, top, left + tw, top + th))


def over(fg_rgb: np.ndarray, alpha: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return np.clip(fg_rgb * a + bg_rgb * (1.0 - a), 0, 255).astype(np.uint8)


def label(im: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(im)
    f = font(max(14, im.width // 42))
    pad = max(6, im.width // 90)
    box = d.textbbox((0, 0), text, font=f)
    w, h = box[2] - box[0], box[3] - box[1]
    d.rectangle([pad - 4, pad - 4, pad + w + 10, pad + h + 12], fill=(0, 0, 0))
    d.text((pad + 2, pad), text, font=f, fill=(255, 255, 255))
    return im


def encode(frames_dir: Path, out: Path, fps: float) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-framerate", str(fps), "-start_number", "0",
           "-i", str(frames_dir / "%05d.png"),
           "-c:v", "libx264", "-preset", "slow", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"ffmpeg failed:\n{p.stderr[-3000:]}")
    print(f"[comp] {rel(out)}  ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--out-name", default=None,
                    help="read/write under outputs/<out-name>/ (default: the shot name)")
    ap.add_argument("--bg-image", type=Path, help="background plate (default: the only "
                                                  "image in outputs/<shot>/backgrounds/)")
    ap.add_argument("--color", type=hex_rgb, default="#D4571E", metavar="#RRGGBB",
                    help="flat colour plate; a warm hue makes cool-toned edge halos "
                         "obvious (default #D4571E)")
    ap.add_argument("--bg-crop", type=parse_crop, metavar="L,T,R,B",
                    help="crop the plate before cover-fitting, e.g. to cut the black "
                         "stitching borders off a stitched panorama")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--compare-with", metavar="RUN",
                    help="also render a 2x2 comparison against another run's matte and "
                         "comp, e.g. --compare-with walk. Top row mattes, bottom row "
                         "comps, left = that run, right = this one.")
    ap.add_argument("--compare-labels", nargs=2, default=["v1", "v2"],
                    metavar=("OLD", "NEW"), help="labels for the 2x2 comparison")
    ap.add_argument("--keep-frames", action="store_true",
                    help="keep the intermediate side-by-side PNGs")
    args = ap.parse_args()

    out_root = ROOT / "outputs" / (args.out_name or args.shot)
    rgba_dir = out_root / "rgba"
    frames_dir = ROOT / "shots" / args.shot / "frames"
    if not rgba_dir.is_dir():
        sys.exit(f"no RGBA sequence: {rgba_dir}\nRun src/export_rgba.py --shot {args.shot} "
                 f"--out-name {args.out_name or args.shot}")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH")

    rgbas = sorted(rgba_dir.glob("*.png"), key=lambda p: int(p.stem))
    size = Image.open(rgbas[0]).size

    bg_path = args.bg_image
    if bg_path is None:
        cands = [p for p in (out_root / "backgrounds").glob("*")
                 if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        if len(cands) != 1:
            sys.exit("pass --bg-image (found "
                     f"{len(cands)} candidates in {rel(out_root / 'backgrounds')})")
        bg_path = cands[0]
    print(f"[comp] shot={args.shot}  {len(rgbas)} frames at {size[0]}x{size[1]}")
    print(f"[comp] plate: {rel(bg_path)}")
    print(f"[comp] flat colour: #{''.join(f'{c:02X}' for c in args.color)}")

    plate_src = Image.open(bg_path).convert("RGB")
    if args.bg_crop:
        l, t, r, b = args.bg_crop
        if not (0 <= l < r <= plate_src.width and 0 <= t < b <= plate_src.height):
            sys.exit(f"--bg-crop {args.bg_crop} is outside the plate {plate_src.size}")
        plate_src = plate_src.crop((l, t, r, b))
        print(f"[comp] plate cropped to {plate_src.size} from {args.bg_crop}")
    plate = cover_fit(plate_src, size)
    plate_a = np.asarray(plate, dtype=np.float32)
    solid_a = np.zeros((size[1], size[0], 3), dtype=np.float32) + np.array(
        args.color, dtype=np.float32)

    dirs = {k: out_root / k for k in ("comp_solid", "comp_image", "_sbs")}
    for d in dirs.values():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for p in rgbas:
        idx = int(p.stem)
        rgba = np.asarray(Image.open(p).convert("RGBA"), dtype=np.float32)
        fg, a = rgba[..., :3], rgba[..., 3] / 255.0

        Image.fromarray(over(fg, a, solid_a)).save(dirs["comp_solid"] / f"{idx:05d}.png")
        comp = over(fg, a, plate_a)
        Image.fromarray(comp).save(dirs["comp_image"] / f"{idx:05d}.png")

        orig = Image.open(frames_dir / f"{idx:05d}.jpg").convert("RGB")
        matte = Image.fromarray((a * 255).astype(np.uint8), mode="L").convert("RGB")
        strip = Image.new("RGB", (size[0] * 3, size[1]))
        strip.paste(label(orig, "ORIGINAL"), (0, 0))
        strip.paste(label(matte, "MATTE"), (size[0], 0))
        strip.paste(label(Image.fromarray(comp), "COMP"), (size[0] * 2, 0))
        strip.save(dirs["_sbs"] / f"{idx:05d}.png")

    if args.compare_with:
        other = ROOT / "outputs" / args.compare_with
        o_alpha = other / "alpha" if (other / "alpha").is_dir() else other / "masks"
        o_comp = other / "comp_image"
        missing = [str(d) for d in (o_alpha, o_comp) if not d.is_dir()]
        if missing:
            sys.exit(f"--compare-with {args.compare_with}: missing {missing}. "
                     "Run export_rgba.py and comp_preview.py for that run first.")
        cmp_dir = out_root / "_cmp"
        if cmp_dir.exists():
            shutil.rmtree(cmp_dir)
        cmp_dir.mkdir(parents=True)
        old_lab, new_lab = args.compare_labels
        for p2 in rgbas:
            idx = int(p2.stem)
            oa = Image.open(o_alpha / f"{idx:05d}.png").convert("L").convert("RGB")
            na = Image.open(rgba_dir / f"{idx:05d}.png").convert("RGBA").getchannel("A")
            na = na.convert("RGB")
            oc = Image.open(o_comp / f"{idx:05d}.png").convert("RGB")
            nc = Image.open(dirs["comp_image"] / f"{idx:05d}.png").convert("RGB")
            grid = Image.new("RGB", (size[0] * 2, size[1] * 2))
            grid.paste(label(oa, f"{old_lab}  MATTE"), (0, 0))
            grid.paste(label(na, f"{new_lab}  MATTE"), (size[0], 0))
            grid.paste(label(oc, f"{old_lab}  COMP"), (0, size[1]))
            grid.paste(label(nc, f"{new_lab}  COMP"), (size[0], size[1]))
            grid.save(cmp_dir / f"{idx:05d}.png")
        encode(cmp_dir, out_root / f"{old_lab}_vs_{new_lab}.mp4", args.fps)
        if not args.keep_frames:
            shutil.rmtree(cmp_dir)

    encode(dirs["comp_image"], out_root / "comp.mp4", args.fps)
    encode(dirs["comp_solid"], out_root / "comp_solid.mp4", args.fps)
    encode(dirs["_sbs"], out_root / "side_by_side.mp4", args.fps)

    if not args.keep_frames:
        shutil.rmtree(dirs["_sbs"])
    print(f"[comp] comp frames kept in {rel(dirs['comp_image'])} and "
          f"{rel(dirs['comp_solid'])}")


if __name__ == "__main__":
    main()
