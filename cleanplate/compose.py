"""Everything downstream of the matte: despill, RGBA, comps, video encoding."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

CHECKER_LIGHT = (210, 210, 215)
CHECKER_DARK = (120, 120, 128)


def as_alpha8(alpha: np.ndarray) -> np.ndarray:
    """Accept a bool mask or a uint8 alpha, always return uint8 0..255."""
    if alpha.dtype == bool:
        return (alpha.astype(np.uint8) * 255)
    if alpha.dtype != np.uint8:
        return np.clip(alpha, 0, 255).astype(np.uint8)
    return alpha


def despill_green(rgb: np.ndarray, alpha: np.ndarray, strength: float = 1.0,
                  band: int = 2) -> tuple[np.ndarray, int]:
    """Suppress green background bleed inside the fractional-alpha band.

    Green is capped at the mean of red and blue, and only the excess is removed, so
    genuinely green subject pixels keep their colour. Restricted to the soft edge
    band (grown by `band` px) because that is where background actually bleeds in;
    applying it to the whole subject would shift real colours.

    Returns (rgb, n_pixels_changed).
    """
    import cv2
    if strength <= 0:
        return rgb, 0
    a = as_alpha8(alpha)
    out = rgb.astype(np.float32).copy()

    edge = (a > 0) & (a < 255)
    if not edge.any():
        # A hard binary matte has no fractional band at all; fall back to a ring
        # just inside the silhouette so despill still has somewhere to act.
        solid = (a > 127).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(band, 1) + 1,) * 2)
        edge = (solid - cv2.erode(solid, k, iterations=1)).astype(bool)
    elif band > 0:
        k = np.ones((2 * band + 1,) * 2, np.uint8)
        edge = cv2.dilate(edge.astype(np.uint8), k, iterations=1).astype(bool)
    edge &= a > 0                                   # never touch fully clear pixels

    r, g, b = out[..., 0], out[..., 1], out[..., 2]
    limit = 0.5 * r + 0.5 * b
    excess = np.maximum(0.0, g - limit)
    out[..., 1] = np.where(edge, g - strength * excess, g)
    return np.clip(out, 0, 255).astype(np.uint8), int((edge & (excess > 1)).sum())


def to_rgba(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Straight (unpremultiplied) RGBA.

    RGB is kept intact under alpha=0 so a later feather picks up real pixels
    rather than black.
    """
    return np.dstack([rgb, as_alpha8(alpha)])


def over(rgb: np.ndarray, alpha: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Straight-alpha composite, in float so edge pixels are not quantised twice."""
    a = as_alpha8(alpha).astype(np.float32)[..., None] / 255.0
    return np.clip(rgb.astype(np.float32) * a + bg.astype(np.float32) * (1 - a),
                   0, 255).astype(np.uint8)


def checkerboard(size: tuple[int, int], cell: int = 20) -> np.ndarray:
    """Transparency backdrop, so alpha is unmistakable."""
    w, h = size
    board = (np.indices((h, w)).sum(axis=0) // cell) % 2
    return np.where(board[..., None], np.array(CHECKER_LIGHT), np.array(CHECKER_DARK)
                    ).astype(np.uint8)


def solid(size: tuple[int, int], colour: tuple[int, int, int]) -> np.ndarray:
    w, h = size
    return np.tile(np.array(colour, dtype=np.uint8), (h, w, 1))


def hex_rgb(s: str) -> tuple[int, int, int]:
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"expected #RRGGBB, got {s!r}")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def cover_fit(im: Image.Image, size: tuple[int, int],
              crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Scale to fill the target and centre-crop, preserving aspect ratio."""
    if crop:
        im = im.crop(crop)
    tw, th = size
    scale = max(tw / im.width, th / im.height)
    nw = max(tw, int(round(im.width * scale)))
    nh = max(th, int(round(im.height * scale)))
    im = im.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - tw) // 2, (nh - th) // 2
    return np.asarray(im.crop((left, top, left + tw, top + th)).convert("RGB"))


def write_sequence(arrs, out_dir: Path, mode: str = "RGBA", start: int = 0) -> int:
    """Write a numbered PNG sequence, replacing anything already there."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    n = 0
    for i, a in enumerate(arrs):
        Image.fromarray(a, mode=mode).save(out_dir / f"{start + i:05d}.png")
        n += 1
    return n


def encode(frames_dir: Path, out: Path, fps: float = 24.0) -> Path:
    """PNG sequence -> H.264 mp4. Raises with ffmpeg's own message on failure."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-framerate", str(fps), "-start_number", "0",
           "-i", str(Path(frames_dir) / "%05d.png"),
           "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{p.stderr[-3000:]}")
    return out
