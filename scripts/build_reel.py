#!/usr/bin/env python3
"""Build the CleanPlate reel from the committed shot list. No timeline, no NLE.

    python scripts/build_reel.py                 # 1080p horizontal + 1080x1920 vertical
    python scripts/build_reel.py --only rotobench   # one segment, for iterating

Everything the reel shows is produced by this repository. The shot list is
docs/reel.json; every segment renders to a normalised intermediate and the whole thing
concatenates, so the reel re-renders from the repo rather than existing only as an
export someone made once.

Segment types:
  clip    an existing mp4 (the removal before/afters, the double-role shot)
  reveal  computed here: plate -> the click -> the matte -> the comp, on one shot
  wipe    two image sequences under a moving divider (hair, default vs HQ)
  still   a screenshot with a slow push in
  card    title/body text
  table   the RotoBench self-consistency table, including the frozen-control row
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate.paths import ROOT, rel                                 # noqa: E402

SPEC = ROOT / "docs" / "reel.json"
WORK = ROOT / "outputs" / "_reel"
FONT_DIR = Path(__import__("matplotlib").get_data_path()) / "fonts" / "ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """DejaVu Sans: Bitstream Vera licence, free to use and redistribute."""
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def hexrgb(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


class Reel:
    def __init__(self, spec: dict, vertical: bool = False):
        self.s = spec
        self.vertical = vertical
        self.w, self.h = spec["vertical_size" if vertical else "size"]
        self.fps = spec["fps"]
        self.bg = hexrgb(spec["bg"]); self.fg = hexrgb(spec["fg"])
        self.dim = hexrgb(spec["dim"]); self.accent = hexrgb(spec["accent"])
        self.scale = self.w / 1920.0

    # ---------------------------------------------------------------- helpers
    def canvas(self) -> Image.Image:
        return Image.new("RGB", (self.w, self.h), self.bg)

    def px(self, v: float) -> int:
        return max(1, int(round(v * (self.w / 1080.0 if self.vertical else self.scale))))

    def wrap(self, d: ImageDraw.ImageDraw, text: str, f, maxw: int) -> list[str]:
        out = []
        for para in text.split("\n"):
            line = ""
            for word in para.split():
                t = (line + " " + word).strip()
                if d.textlength(t, font=f) <= maxw or not line:
                    line = t
                else:
                    out.append(line); line = word
            out.append(line)
        return out

    def caption(self, im: Image.Image, text: str, y: int) -> None:
        d = ImageDraw.Draw(im)
        f = font(self.px(30))
        for i, line in enumerate(self.wrap(d, text, f, int(self.w * 0.86))):
            tw = d.textlength(line, font=f)
            d.text(((self.w - tw) / 2, y + i * self.px(40)), line, font=f, fill=self.fg)

    def block(self, im: Image.Image, frame: np.ndarray, caption: str,
              max_h_frac: float = 0.62, extra: int = 0, heading: str = "") -> int:
        """Centre an image and its caption as one block. Vertical needs this badly:
        laying out from a fixed top left the bottom half of a 1080x1920 frame empty."""
        d = ImageDraw.Draw(im)
        src_w, src_h = frame.shape[1], frame.shape[0]
        tw = int(self.w * (0.98 if self.vertical else 1.0))
        th = int(round(src_h * tw / src_w))
        if th > int(self.h * max_h_frac):
            th = int(self.h * max_h_frac); tw = int(round(src_w * th / src_h))
        lines = self.wrap(d, caption, font(self.px(30)), int(self.w * 0.86))
        cap_h = len(lines) * self.px(40)
        hf = font(self.px(58), True)
        head_h = (self.px(96) if (heading and self.vertical) else 0)
        top = max(self.px(50),
                  (self.h - (th + self.px(50) + cap_h + extra + head_h)) // 2) + head_h
        if head_h:
            d.text(((self.w - d.textlength(heading, font=hf)) / 2,
                    top - self.px(86)), heading, font=hf, fill=self.fg)
        im.paste(Image.fromarray(frame).resize((tw, th), Image.LANCZOS),
                 ((self.w - tw) // 2, top))
        self.caption(im, caption, top + th + self.px(50) + extra)
        return top + th

    def place(self, im: Image.Image, frame: np.ndarray, top: int, max_h: int | None = None):
        """Fit a frame to the canvas width (or a height cap) and paste it at `top`."""
        src = Image.fromarray(frame)
        tw = int(self.w * (0.98 if self.vertical else 1.0))
        th = int(round(src.height * tw / src.width))
        if max_h and th > max_h:
            th = max_h; tw = int(round(src.width * th / src.height))
        src = src.resize((tw, th), Image.LANCZOS)
        im.paste(src, ((self.w - tw) // 2, top))
        return top + th

    # ---------------------------------------------------------------- segments
    def seg_card(self, sp: dict) -> list[Image.Image]:
        im = self.canvas(); d = ImageDraw.Draw(im)
        tf, bf = font(self.px(76), True), font(self.px(34))
        title = self.wrap(d, sp["title"], tf, int(self.w * 0.86))
        body = self.wrap(d, sp.get("body", ""), bf, int(self.w * 0.8)) if sp.get("body") else []
        block = len(title) * self.px(92) + (self.px(30) + len(body) * self.px(48) if body else 0)
        y = (self.h - block) // 2
        for line in title:
            d.text(((self.w - d.textlength(line, font=tf)) / 2, y), line, font=tf,
                   fill=self.accent if sp.get("credits") else self.fg)
            y += self.px(92)
        y += self.px(30)
        for line in body:
            d.text(((self.w - d.textlength(line, font=bf)) / 2, y), line, font=bf, fill=self.dim)
            y += self.px(48)
        if sp.get("credits"):
            cf = font(self.px(24))
            for i, line in enumerate([self.s["credit"],
                                      "Music and fonts: see THIRD_PARTY.md"]):
                d.text(((self.w - d.textlength(line, font=cf)) / 2,
                        self.h - self.px(120) + i * self.px(34)), line, font=cf, fill=self.dim)
        return [im]

    def seg_table(self, sp: dict) -> list[Image.Image]:
        res = json.loads((ROOT / "outputs" / "_bench_p" / "_selfconsistency.json").read_text())
        rows = [("MatAnyone 2", res["matanyone2_960"]["iou_consecutive"], 156.3),
                ("hairzoom2 (best matte)", res["hairzoom2_960"]["iou_consecutive"], 151.4),
                ("binary (SAM 2 only)", res["binary_960"]["iou_consecutive"], 202.0),
                ("frozen matte (control)", res["frozen"]["iou_consecutive"],
                 res["frozen"].get("band_MAD", float("nan")))]
        im = self.canvas(); d = ImageDraw.Draw(im)
        tf, hf, rf = font(self.px(64), True), font(self.px(26), True), font(self.px(30))
        y = int(self.h * (0.20 if not self.vertical else 0.24))
        c0 = int(self.w * 0.06)
        d.text((c0, y), sp["title"], font=tf, fill=self.fg); y += self.px(86)
        for line in self.wrap(d, sp["body"], font(self.px(30)), int(self.w * 0.86)):
            d.text((c0, y), line, font=font(self.px(30)), fill=self.dim)
            y += self.px(44)
        y += self.px(40)
        c1, c2, c3 = int(self.w * 0.06), int(self.w * 0.60), int(self.w * 0.84)
        d.text((c1, y), "METHOD", font=hf, fill=self.dim)
        d.text((c2, y), "STABLE?", font=hf, fill=self.dim)
        d.text((c3, y), "RIGHT?", font=hf, fill=self.dim)
        y += self.px(52)
        d.line([(c1, y), (self.w - self.px(110), y)], fill=(40, 46, 58), width=2)
        y += self.px(24)
        for name, iou, bmad in rows:
            frozen = name.startswith("frozen")
            col = self.accent if frozen else self.fg
            d.text((c1, y), name, font=rf, fill=col)
            d.text((c2, y), f"{iou:.4f}", font=rf, fill=col)
            d.text((c3, y), f"{bmad:.0f}", font=rf, fill=col)
            y += self.px(58)
        y += self.px(24)
        ff = font(self.px(28))
        for line in self.wrap(d, sp["footer"], ff, int(self.w * 0.82)):
            d.text((c1, y), line, font=ff, fill=self.accent); y += self.px(40)
        sf = font(self.px(24))
        sub = ("STABLE? = consecutive-frame IoU, higher looks calmer. "
               "RIGHT? = error against the key, lower is better.")
        for line in self.wrap(d, sub, sf, int(self.w * 0.88)):
            d.text((c1, y + self.px(10)), line, font=sf, fill=self.dim); y += self.px(32)
        return [im]

    def seg_still(self, sp: dict, n: int) -> list[Image.Image]:
        src = Image.open(ROOT / sp["src"]).convert("RGB")
        out = []
        for i in range(n):
            z = 1.0 + (sp.get("zoom", 1.06) - 1.0) * (i / max(1, n - 1))
            w, h = int(src.width / z), int(src.height / z)
            box = ((src.width - w) // 2, (src.height - h) // 2)
            crop = src.crop((box[0], box[1], box[0] + w, box[1] + h))
            im = self.canvas()
            self.block(im, np.asarray(crop), sp["caption"], max_h_frac=0.74)
            out.append(im)
        return out

    def _seq(self, d: Path) -> list[Path]:
        return sorted(d.glob("*.png"), key=lambda p: int(p.stem)) or \
               sorted(d.glob("*.jpg"), key=lambda p: int(p.stem))

    def seg_reveal(self, sp: dict, n: int) -> list[Image.Image]:
        from cleanplate.ingest import frame_paths
        from cleanplate.session import Prompt
        frames = [np.asarray(Image.open(p).convert("RGB")) for p in frame_paths(sp["shot"])]
        alph = [np.asarray(Image.open(p).convert("L")) for p in self._seq(ROOT / sp["alpha"])]
        comps = [np.asarray(Image.open(p).convert("RGB")) for p in self._seq(ROOT / sp["comp"])]
        pr = Prompt.load(sp["shot"])
        pt = next(iter(pr.frames.values())).positive[0]
        out = []
        for i in range(n):
            t = i / (n - 1)
            k = min(len(frames) - 1, int(t * (len(frames) - 1)))
            base = frames[k].astype(np.float32)
            if t < 0.18:                                    # plate, then the click lands
                img = base
            elif t < 0.45:                                  # matte wipes across
                u = (t - 0.18) / 0.27
                a = alph[k].astype(np.float32)[..., None] / 255.0
                lit = base * (0.25 + 0.75 * a) + np.array([61, 220, 132]) * 0.22 * a
                x = int(u * base.shape[1])
                img = base.copy(); img[:, :x] = lit[:, :x]
            else:                                           # comp wipes in
                u = min(1.0, (t - 0.45) / 0.3)
                a = alph[k].astype(np.float32)[..., None] / 255.0
                lit = base * (0.25 + 0.75 * a) + np.array([61, 220, 132]) * 0.22 * a
                x = int(u * base.shape[1])
                img = lit.copy(); img[:, :x] = comps[k][:, :x]
            frame = np.clip(img, 0, 255).astype(np.uint8)
            im = self.canvas()
            top = self.block(im, frame, sp["caption"],
                             heading=sp.get("heading", "")) - int(round(
                frame.shape[0] * (self.w * (0.98 if self.vertical else 1.0)) / frame.shape[1]))
            if t < 0.30:                                    # the click marker
                d = ImageDraw.Draw(im)
                sx = self.w / frames[0].shape[1] * (0.98 if self.vertical else 1.0)
                cx = (self.w - frames[0].shape[1] * sx) / 2 + pt[0] * sx
                cy = top + pt[1] * sx
                r = self.px(16) + self.px(26) * (1 - min(1.0, t / 0.18))
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=self.accent,
                          width=self.px(5))
            out.append(im)
        return out

    def seg_wipe(self, sp: dict, n: int) -> list[Image.Image]:
        A = [np.asarray(Image.open(p).convert("RGB")) for p in self._seq(ROOT / sp["a"])]
        B = [np.asarray(Image.open(p).convert("RGB")) for p in self._seq(ROOT / sp["b"])]
        l, t_, r, b = sp["crop"]
        out = []
        for i in range(n):
            u = i / (n - 1)
            k = min(len(A) - 1, int(u * (len(A) - 1)))
            a, bb = A[k][t_:b, l:r], B[k][t_:b, l:r]
            x = int((0.15 + 0.7 * (0.5 - 0.5 * np.cos(u * 2 * np.pi))) * a.shape[1])
            mix = a.copy(); mix[:, x:] = bb[:, x:]
            mix[:, max(0, x - 2):x + 2] = np.array(self.accent, np.uint8)
            im = self.canvas()
            bot = self.block(im, mix, sp["caption"], max_h_frac=0.52, extra=self.px(46),
                             heading=sp.get("heading", ""))
            d = ImageDraw.Draw(im); lf = font(self.px(28), True)
            d.text((self.px(90), bot + self.px(18)), sp["a_label"], font=lf, fill=self.dim)
            tw = d.textlength(sp["b_label"], font=lf)
            d.text((self.w - self.px(90) - tw, bot + self.px(18)), sp["b_label"],
                   font=lf, fill=self.accent)
            out.append(im)
        return out

    def seg_clip(self, sp: dict, n: int) -> list[Image.Image]:
        src = ROOT / sp["src"]
        if src.is_dir():
            fs = self._seq(src)
            out = []
            for i in range(n):
                k = min(len(fs) - 1, int(i / n * len(fs)))
                im = self.canvas()
                self.block(im, np.asarray(Image.open(fs[k]).convert("RGB")), sp["caption"],
                           heading=sp.get("heading", ""))
                out.append(im)
            return out
        if not src.exists():
            im = self.canvas(); d = ImageDraw.Draw(im)
            f = font(self.px(44), True)
            lines = sp.get("placeholder", "MISSING").split("\n")
            y = (self.h - len(lines) * self.px(64)) // 2
            for line in lines:
                d.text(((self.w - d.textlength(line, font=f)) / 2, y), line, font=f,
                       fill=self.dim)
                y += self.px(64)
            d.rectangle([self.px(60), self.px(60), self.w - self.px(60), self.h - self.px(60)],
                        outline=(60, 68, 84), width=self.px(3))
            self.caption(im, sp["caption"], self.h - self.px(170))
            return [im] * n
        tmp = WORK / f"_frames_{sp['id']}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                        "-vf", f"fps={self.fps}", str(tmp / "%05d.jpg")], check=True)
        fs = sorted(tmp.glob("*.jpg"))
        out = []
        for i in range(n):
            k = min(len(fs) - 1, int(i / n * len(fs)))
            im = self.canvas()
            self.block(im, np.asarray(Image.open(fs[k]).convert("RGB")), sp["caption"],
                       heading=sp.get("heading", ""))
            out.append(im)
        shutil.rmtree(tmp)
        return out

    # ---------------------------------------------------------------- build
    def render(self, only: str | None = None) -> list[Path]:
        WORK.mkdir(parents=True, exist_ok=True)
        tag = "v" if self.vertical else "h"
        parts = []
        for sp in self.s["segments"]:
            if only and sp["id"] != only:
                continue
            if self.vertical and not sp.get("vertical", True):
                continue
            n = max(1, int(round(sp["seconds"] * self.fps)))
            kind = sp["type"]
            imgs = (self.seg_card(sp) if kind == "card" else
                    self.seg_table(sp) if kind == "table" else
                    self.seg_still(sp, n) if kind == "still" else
                    self.seg_reveal(sp, n) if kind == "reveal" else
                    self.seg_wipe(sp, n) if kind == "wipe" else
                    self.seg_clip(sp, n))
            d = WORK / f"{tag}_{sp['id']}"
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
            if len(imgs) == 1:
                imgs[0].save(d / "00000.jpg", quality=95)
                mp4 = WORK / f"{tag}_{sp['id']}.mp4"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate",
                                str(self.fps), "-i", str(d / "00000.jpg"), "-t",
                                str(sp["seconds"]), "-c:v", "libx264", "-preset", "medium",
                                "-crf", "18", "-pix_fmt", "yuv420p", str(mp4)], check=True)
            else:
                for i, im in enumerate(imgs):
                    im.save(d / f"{i:05d}.jpg", quality=95)
                mp4 = WORK / f"{tag}_{sp['id']}.mp4"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(self.fps),
                                "-i", str(d / "%05d.jpg"), "-c:v", "libx264", "-preset",
                                "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(mp4)],
                               check=True)
            shutil.rmtree(d)
            parts.append(mp4)
            print(f"   [{tag}] {sp['id']:18s} {sp['seconds']:4.1f}s  {rel(mp4)}", flush=True)
        return parts

    def concat(self, parts: list[Path], out: Path) -> Path:
        lst = WORK / f"{out.stem}.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in parts))
        total = sum(sp["seconds"] for sp in self.s["segments"]
                    if not (self.vertical and not sp.get("vertical", True)))
        vf = f"fade=t=in:st=0:d=0.6,fade=t=out:st={total - 0.8:.2f}:d=0.8"
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst)]
        music = self.s.get("music")
        if music and (ROOT / music["file"]).exists():
            cmd += ["-i", str(ROOT / music["file"]),
                    "-filter_complex",
                    f"[0:v]{vf}[v];[1:a]afade=t=in:st=0:d=1.5,"
                    f"afade=t=out:st={total - 2.5:.2f}:d=2.5,volume=0.5[a]",
                    "-map", "[v]", "-map", "[a]", "-shortest", "-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-vf", vf]
        cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(out)]
        subprocess.run(cmd, check=True)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--horizontal-only", action="store_true")
    args = ap.parse_args()
    spec = json.loads(SPEC.read_text())
    outs = []
    for vertical in ([False] if args.horizontal_only else [False, True]):
        r = Reel(spec, vertical=vertical)
        print(f"[reel] {'vertical' if vertical else 'horizontal'} "
              f"{r.w}x{r.h} @{r.fps}", flush=True)
        parts = r.render(args.only)
        if args.only:
            continue
        out = ROOT / "outputs" / ("reel_vertical.mp4" if vertical else "reel.mp4")
        r.concat(parts, out)
        secs = sum(sp["seconds"] for sp in spec["segments"]
                   if not (vertical and not sp.get("vertical", True)))
        print(f"[reel] {rel(out)}  {secs:.0f}s  "
              f"{out.stat().st_size / 1e6:.1f} MB", flush=True)
        outs.append(out)


if __name__ == "__main__":
    main()
