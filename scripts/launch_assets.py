#!/usr/bin/env python3
"""The README's hero GIF and the repo's social preview card, both from the reel.

    python scripts/launch_assets.py

  docs/img/reel.gif           the click-to-matte stretch, for the top of the README
  docs/img/social_preview.png 1280x640, the image GitHub shows when the repo is shared
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate.paths import ROOT, rel                                 # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from build_reel import font, hexrgb                                    # noqa: E402

REEL = ROOT / "outputs" / "reel.mp4"
IMG = ROOT / "docs" / "img"


def reel_gif(start: float = 9.0, seconds: float = 5.5, width: int = 520,
             fps: int = 8, colors: int = 64) -> None:
    """The click-to-matte stretch: plate, the click, the matte, the comp.

    Kept under 2 MB: a README hero that takes a second to load is worse than a smaller
    one. 720p/12fps/128 colours was 6.7 MB.
    """
    tmp = ROOT / "outputs" / "_gif"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(seconds),
                    "-i", str(REEL), "-vf", f"fps={fps},scale={width}:-2",
                    str(tmp / "%04d.png")], check=True)
    frames = [Image.open(p).convert("RGB") for p in sorted(tmp.glob("*.png"))]
    base = frames[0].quantize(colors=colors, method=Image.MEDIANCUT)
    frames = [f.quantize(palette=base, dither=Image.FLOYDSTEINBERG) for f in frames]
    out = IMG / "reel.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=round(1000 / fps), loop=0, optimize=True)
    shutil.rmtree(tmp)
    print(f"[assets] {rel(out)}  {len(frames)} frames, {out.stat().st_size / 1e6:.2f} MB")


def social_preview() -> None:
    """1280x640. GitHub crops the edges on some surfaces, so nothing vital goes near them."""
    spec = json.loads((ROOT / "docs" / "reel.json").read_text())
    W, H = 1280, 640
    bg, fg, dim, accent = (hexrgb(spec[k]) for k in ("bg", "fg", "dim", "accent"))
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)

    # a real comp along the bottom, faded into the card: the Mars shot this repo makes
    shot = Image.open(ROOT / "outputs" / "walk_harmonized" / "comp_image" / "00048.png")
    band = shot.convert("RGB").resize((W, int(shot.height * W / shot.width)), Image.LANCZOS)
    bh = band.height
    im.paste(band, (0, H - bh))
    grad = Image.new("L", (1, bh))
    for y in range(bh):
        grad.putpixel((0, y), int(255 * max(0.0, 1 - y / (bh * 0.75))))
    im.paste(Image.new("RGB", (W, bh), bg), (0, H - bh), grad.resize((W, bh)))

    d.text((64, 92), "CleanPlate", font=font(96, True), fill=fg)
    d.text((70, 214), "Rotoscoping, matting and object removal — entirely local.",
           font=font(32), fill=dim)
    for i, line in enumerate([
            "One click to a production matte",
            "Remove a thing and fill what was behind it",
            "RotoBench: scored against the compositors' own keys"]):
        y = 286 + i * 46
        d.ellipse([70, y + 12, 82, y + 24], fill=accent)
        d.text((100, y), line, font=font(30), fill=fg)
    url = "github.com/shauryadata/cleanplate"
    uf = font(26, True)
    tw = d.textlength(url, font=uf)
    d.rectangle([56, 436, 56 + tw + 28, 436 + 46], fill=bg)
    d.text((70, 446), url, font=uf, fill=accent)
    out = IMG / "social_preview.png"
    im.save(out)
    print(f"[assets] {rel(out)}  {W}x{H}, {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    reel_gif()
    social_preview()
