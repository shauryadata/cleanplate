#!/usr/bin/env python3
"""Hair close-ups: plate, reference alpha, and one or more methods, side by side.

    python scripts/hair_stills.py --clip B1_tos_greenscreen_hair \
        --method baseline_960 --method fullres_1920 --frame 48
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import bench                                          # noqa: E402
from cleanplate.paths import ROOT, rel                                # noqa: E402

BENCH = ROOT / "outputs" / "_bench"


def font(sz: int):
    import matplotlib
    return ImageFont.truetype(
        str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"), sz)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--method", action="append", required=True)
    ap.add_argument("--frame", type=int, default=48)
    ap.add_argument("--zoom", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    clip = next((c for c in bench.clips() if c.name == args.clip), None)
    if clip is None:
        sys.exit(f"no clip {args.clip!r}")
    gt = bench.load_alpha(clip.alpha_dir)
    l, t, r, b = bench.hair_box(gt)
    i = args.frame
    Z = args.zoom

    def z(im: Image.Image) -> Image.Image:
        c = im.crop((l, t, r, b))
        return c.resize((int(c.width * Z), int(c.height * Z)), Image.NEAREST)

    panels = [("plate", z(Image.open(clip.frames_dir / f"{i:05d}.jpg").convert("RGB"))),
              ("TRUTH alpha", z(Image.fromarray(gt[i]).convert("RGB")))]
    for m in args.method:
        p = BENCH / m / clip.name / f"{i:05d}.png"
        if not p.exists():
            print(f"  missing {rel(p)}, skipping", file=sys.stderr)
            continue
        a = np.asarray(Image.open(p).convert("L"))
        panels.append((m, z(Image.fromarray(a).convert("RGB"))))
        # error map: |pred - truth|, so the difference is visible not inferred
        err = np.abs(a.astype(np.int16) - gt[i].astype(np.int16)).astype(np.uint8)
        panels.append((f"{m}  |error|", z(Image.fromarray(err).convert("RGB"))))

    W, H = panels[0][1].size
    cols = 2
    rows = (len(panels) + cols - 1) // cols
    f = font(max(15, int(W / 26)))
    sheet = Image.new("RGB", (cols * (W + 6) + 6, rows * (H + 6) + 34), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    d.text((6, 7), f"{clip.name}  frame {i}  hair box {(l, t, r, b)}  {Z}x",
           font=f, fill=(255, 235, 0))
    for k, (name, im) in enumerate(panels):
        rr, cc = divmod(k, cols)
        x, y = 6 + cc * (W + 6), 30 + rr * (H + 6)
        sheet.paste(im, (x, y))
        d.rectangle([x, y, x + W, y + 26], fill=(0, 0, 0))
        d.text((x + 5, y + 3), name, font=f, fill=(255, 235, 0))
    out = args.out or (ROOT / "outputs" / "_scout" /
                       f"hair_{clip.name}_f{i:03d}.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)
    print(f"-> {rel(out)}  {sheet.size}")


if __name__ == "__main__":
    main()
