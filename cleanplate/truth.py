"""Building reference alpha to score against.

Two tiers, kept apart on purpose:

Tier A - synthetic, exact.
    A foreground clip with a known alpha is composited over a background:
        comp = fgr * a + bg * (1 - a)
    `a` is then ground truth *by construction*, because it is literally the number used
    to make the pixel. This holds even though VideoMatte240K ships its alpha as an
    HEVC-compressed mp4: whatever the codec did, the decoded alpha is what we
    composite with, so it is what the correct answer is.

Tier B - keyed reference, NOT gospel.
    A chroma key on a real green-screen plate. Real hair, real optics, but the
    reference carries the keyer's own errors. Anything scored against it is labelled.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ video I/O
def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(path)],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-800:])
    s = json.loads(out.stdout)["streams"][0]
    return {"width": int(s["width"]), "height": int(s["height"]),
            "frames": int(s.get("nb_frames") or 0), "fps": s.get("r_frame_rate")}


def read_frames(path: Path, start: int, count: int, width: int | None = None,
                gray: bool = False) -> np.ndarray:
    """Decode `count` frames starting at `start`, as uint8 (N, H, W[, 3]).

    Decoded straight from ffmpeg's rawvideo so nothing is re-encoded on the way in.
    """
    info = probe(path)
    w = width or info["width"]
    h = int(round(info["height"] * w / info["width"]))
    h -= h % 2
    pix = "gray" if gray else "rgb24"
    bpp = 1 if gray else 3
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-vf", f"select='gte(n\\,{start})',scale={w}:{h}:flags=lanczos",
           "-vsync", "0", "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", pix, "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[-800:])
    n = len(p.stdout) // (w * h * bpp)
    if n < count:
        raise RuntimeError(f"{path.name}: asked for {count} frames from {start}, got {n}")
    arr = np.frombuffer(p.stdout[:n * w * h * bpp], np.uint8)
    return arr.reshape((n, h, w) if gray else (n, h, w, 3))


# ------------------------------------------------------------------ tier A
@dataclass
class ClipRecipe:
    """Everything needed to regenerate one synthetic truth clip, bit for bit."""
    name: str
    fgr: str
    pha: str
    bg: str                       # path to a background image or video
    bg_kind: str                  # "image" | "video"
    start: int = 0                # first frame of the foreground clip
    bg_start: int = 0
    frames: int = 96
    width: int = 1920
    fg_scale: float = 1.0         # relative to the composite height
    fg_dx: float = 0.0            # fraction of width, from centre
    fg_dy: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _place(fgr: np.ndarray, pha: np.ndarray, out_hw: tuple[int, int],
           scale: float, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale the foreground to `scale` x output height and paste it at an offset.

    Uses cv2 so the alpha and the colour are resampled identically.
    """
    import cv2
    H, W = out_hw
    n = len(fgr)
    th = max(2, int(round(H * scale)))
    tw = max(2, int(round(fgr.shape[2] * th / fgr.shape[1])))
    ox = int(round((W - tw) / 2 + dx * W))
    oy = int(round((H - th) / 2 + dy * H))

    out_f = np.zeros((n, H, W, 3), np.uint8)
    out_a = np.zeros((n, H, W), np.uint8)
    # source/destination windows, clipped to the canvas
    sx0, sy0 = max(0, -ox), max(0, -oy)
    dx0, dy0 = max(0, ox), max(0, oy)
    cw = min(tw - sx0, W - dx0)
    ch = min(th - sy0, H - dy0)
    if cw <= 0 or ch <= 0:
        raise ValueError("foreground placed entirely off canvas")
    for i in range(n):
        f = cv2.resize(fgr[i], (tw, th), interpolation=cv2.INTER_AREA)
        a = cv2.resize(pha[i], (tw, th), interpolation=cv2.INTER_AREA)
        out_f[i, dy0:dy0 + ch, dx0:dx0 + cw] = f[sy0:sy0 + ch, sx0:sx0 + cw]
        out_a[i, dy0:dy0 + ch, dx0:dx0 + cw] = a[sy0:sy0 + ch, sx0:sx0 + cw]
    return out_f, out_a


def build_clip(r: ClipRecipe, root: Path) -> dict:
    """Make one synthetic truth clip. Returns (and does not write) the arrays."""
    import cv2
    fgr = read_frames(root / r.fgr, r.start, r.frames)
    pha = read_frames(root / r.pha, r.start, r.frames, gray=True)
    if fgr.shape[:3] != pha.shape[:3]:
        raise ValueError(f"{r.name}: fgr {fgr.shape} vs pha {pha.shape}")

    H = int(round(r.width * fgr.shape[1] / fgr.shape[2]))
    H -= H % 2
    out_hw = (H, r.width)

    if r.bg_kind == "video":
        bg = read_frames(root / r.bg, r.bg_start, r.frames, width=r.width)
        if bg.shape[1] != H:
            bg = np.stack([cv2.resize(b, (r.width, H), interpolation=cv2.INTER_AREA)
                           for b in bg])
    else:
        from PIL import Image
        im = Image.open(root / r.bg).convert("RGB")
        s = max(r.width / im.width, H / im.height)
        im = im.resize((max(r.width, int(im.width * s)), max(H, int(im.height * s))),
                       Image.LANCZOS)
        l, t = (im.width - r.width) // 2, (im.height - H) // 2
        one = np.asarray(im.crop((l, t, l + r.width, t + H)))
        bg = np.repeat(one[None], r.frames, axis=0)

    f, a = _place(fgr, pha, out_hw, r.fg_scale, r.fg_dx, r.fg_dy)
    af = a.astype(np.float32)[..., None] / 255.0
    comp = np.clip(f.astype(np.float32) * af + bg.astype(np.float32) * (1 - af),
                   0, 255).astype(np.uint8)
    return {"comp": comp, "alpha": a, "fgr": f, "bg": bg}


# ------------------------------------------------------------------ tier B
@dataclass
class KeyParams:
    """Chroma key settings. Every one of these is reported, because the reference
    alpha is only as good as they are."""
    key_hue: float = 120.0        # degrees; pure green
    hue_tol: float = 42.0         # +/- degrees counted as backing
    sat_min: float = 0.15         # below this the pixel is too grey to be backing
    val_min: float = 0.05
    alpha_lo: float = 0.20        # greenness at/below which alpha = 1 (full subject)
    alpha_hi: float = 0.72        # greenness at/above which alpha = 0 (full backing)
    garbage: tuple[int, int, int, int] | None = None   # keep only inside this box
    despill: bool = True
    median: int = 3               # px, removes single-pixel key noise
    keep_largest: bool = True     # drop everything not attached to the subject
    keep_dilate: int = 9          # px; bridges hair strands to the head before labelling
    note: str = ""

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["garbage"] = list(self.garbage) if self.garbage else None
        return d


def chroma_key(rgb: np.ndarray, p: KeyParams) -> tuple[np.ndarray, np.ndarray]:
    """Key a green screen. Returns (alpha uint8, despilled rgb uint8).

    A deliberately simple, inspectable keyer: measure how much a pixel looks like the
    backing hue, then ramp alpha between two thresholds. Not Keylight. Good enough to
    be a reference, and simple enough that its failures are legible.
    """
    import cv2
    f = rgb.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)          # H in degrees, S/V in 0..1
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    dh = np.abs(((h - p.key_hue + 180.0) % 360.0) - 180.0)
    hue_score = np.clip(1.0 - dh / max(p.hue_tol, 1e-6), 0.0, 1.0)
    greenness = hue_score * np.clip((s - p.sat_min) / max(1 - p.sat_min, 1e-6), 0, 1)
    greenness = np.where(v < p.val_min, 0.0, greenness)

    alpha = 1.0 - np.clip((greenness - p.alpha_lo) / max(p.alpha_hi - p.alpha_lo, 1e-6),
                          0.0, 1.0)

    if p.garbage:
        l, t, r_, b = p.garbage
        m = np.zeros_like(alpha, bool)
        m[t:b, l:r_] = True
        alpha = np.where(m, alpha, 0.0)

    a8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    if p.median and p.median >= 3 and p.median % 2 == 1:
        a8 = cv2.medianBlur(a8, p.median)

    if p.keep_largest:
        # A green screen is covered in tracking markers, and markers are not green, so
        # they survive the key as little foreground blobs. Keep only what is attached
        # to the subject. Dilating before labelling lets wispy hair stay connected to
        # the head instead of being discarded as its own component.
        from scipy import ndimage
        seed = a8 > 12
        if p.keep_dilate > 0:
            k = np.ones((p.keep_dilate,) * 2, np.uint8)
            grown = cv2.dilate(seed.astype(np.uint8), k, iterations=1).astype(bool)
        else:
            grown = seed
        lab, n = ndimage.label(grown)
        if n > 1:
            sizes = ndimage.sum(grown, lab, range(1, n + 1))
            keep = lab == (1 + int(np.argmax(sizes)))
            a8 = np.where(keep, a8, 0).astype(np.uint8)

    out = rgb.copy()
    if p.despill:
        r_, g_, b_ = [out[..., i].astype(np.float32) for i in range(3)]
        limit = np.maximum(r_, b_)
        out[..., 1] = np.minimum(g_, limit).astype(np.uint8)
    return a8, out
