"""Professional reference truth from the Tears of Steel VFX archive.

The Mango team's compositing output is public on media.xiph.org under CC BY 3.0:

    tearsofsteel-footage-exr/<shot>/linear_hd/<shot>_NNNNN.exr   raw plates, 1920x1012
    tearsofsteel-cleaned-exr/<shot>/linear_hd/<shot>_NNNN.exr    "cleaned" plates

"Cleaned" is not one thing. For some shots it is a rig-removed full plate (RGB, no
alpha). For others it is the compositor's KEY: the despilled foreground premultiplied
over black, with the matte in a real fourth channel. Only the second kind is truth.
Every shot is classified from its EXR header before it is trusted, never from the
directory name.

Provenance, stated plainly: this is a professional key, which is still a key. It is
far better than our in-house chroma keyer and independent of our pipeline, but it is
not a hand-painted or rendered ground truth. Tier label: "professional reference".

Frame numbering between the two archives is NOT a same-number match. On 08_3a the
cleaned frame N is footage frame N-1 (proven by a residual sweep, see
`offset_residuals`). Every shot's offset is measured, not assumed.
"""
from __future__ import annotations

import os
import re
import struct
import subprocess
import urllib.request
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

BASE = "https://media.xiph.org/tearsofsteel"
FOOTAGE = f"{BASE}/tearsofsteel-footage-exr"
CLEANED = f"{BASE}/tearsofsteel-cleaned-exr"
GAMMA = 2.2          # same display transform as Tier B, so the tiers are comparable


# ------------------------------------------------------------------ archive layout
def listing(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", "ignore")


def frame_numbers(url: str) -> tuple[list[int], int | None]:
    """Frame numbers in a directory listing, and their zero-padding width."""
    names = re.findall(r'href="[^"]*?_(\d+)\.exr"', listing(url))
    nums = sorted(int(n) for n in names)
    return nums, (len(names[0]) if names else None)


def cleaned_url(shot: str, n: int, digits: int = 4, res: str = "linear_hd") -> str:
    return f"{CLEANED}/{shot}/{res}/{shot}_{n:0{digits}d}.exr"


def footage_url(shot: str, n: int, res: str = "linear_hd") -> str:
    return f"{FOOTAGE}/{shot}/{res}/{shot}_{n:05d}.exr"


# ------------------------------------------------------------------ EXR header
def exr_header(url: str, nbytes: int = 8192) -> dict:
    """Channel names and resolution from the first few KB of a remote EXR.

    OpenEXR puts every attribute before the pixel data, so a ranged GET of the head of
    the file answers "does this frame carry an alpha channel?" without downloading
    megabytes of pixels. Parses the `channels` (chlist) and `dataWindow` (box2i)
    attributes only.
    """
    raw = subprocess.run(["curl", "-sfL", "--max-time", "60", "-r", f"0-{nbytes - 1}",
                          url], capture_output=True).stdout
    if raw[:4] != b"\x76\x2f\x31\x01":
        raise ValueError(f"not an OpenEXR file: {url}")
    i, out = 8, {}
    while i < len(raw) and raw[i] != 0:
        j = raw.index(b"\0", i); name = raw[i:j].decode(); i = j + 1
        j = raw.index(b"\0", i); typ = raw[i:j].decode(); i = j + 1
        size = struct.unpack("<i", raw[i:i + 4])[0]; i += 4
        val = raw[i:i + size]; i += size
        if typ == "chlist":
            chans, k = [], 0
            while k < len(val) and val[k] != 0:
                e = val.index(b"\0", k); chans.append(val[k:e].decode()); k = e + 1 + 16
            out["channels"] = chans
        elif name == "dataWindow" and typ == "box2i":
            x0, y0, x1, y1 = struct.unpack("<4i", val)
            out["resolution"] = [x1 - x0 + 1, y1 - y0 + 1]
        elif name == "compression":
            out["compression"] = val[0]
    return out


# ------------------------------------------------------------------ pixels
def fetch(url: str, dest: Path) -> Path:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    r = subprocess.run(["curl", "-sfL", "--retry", "2", "--max-time", "300",
                        "-o", str(tmp), url])
    if r.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"download failed ({r.returncode}): {url}")
    tmp.rename(dest)
    return dest


def read_exr(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Linear float32 RGB (H, W, 3) and alpha (H, W) or None."""
    import cv2
    x = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH
                   | cv2.IMREAD_ANYCOLOR)
    if x is None:
        raise RuntimeError(f"could not read {path} - OpenEXR support missing?")
    if x.ndim == 2:
        x = x[..., None]
    rgb = cv2.cvtColor(x[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)
    a = x[..., 3].astype(np.float32) if x.shape[2] >= 4 else None
    return rgb, a


def to_display(lin: np.ndarray) -> np.ndarray:
    """Linear scene light -> 8-bit display: clamp, then gamma. Same as Tier B."""
    return (np.clip(lin, 0, 1) ** (1.0 / GAMMA) * 255.0 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------ alignment
def offset_residuals(clean_rgb: np.ndarray, clean_a: np.ndarray,
                     plates: dict[int, np.ndarray]) -> dict[int, float]:
    """Median |plate - cleaned| (red channel, linear) over the opaque foreground.

    Where the key is fully opaque, the premultiplied foreground IS the plate pixel,
    apart from despill and half-float rounding. Sensor noise is independent frame to
    frame, so only the one plate frame the key was pulled from matches closely; a
    neighbour differs by at least the noise even when nothing moves. That makes the
    minimum sharp, and the ratio to the runner-up a measure of confidence.
    Red, because despill modifies green and red is the cleanest of the other two.
    """
    m = clean_a > 0.99
    m[:8] = m[-8:] = False
    m[:, :8] = m[:, -8:] = False
    if m.sum() < 1000:
        raise ValueError("too few opaque key pixels to measure alignment")
    return {f: float(np.median(np.abs(p[..., 0][m] - clean_rgb[..., 0][m])))
            for f, p in plates.items()}


def shift_residuals(clean_rgb: np.ndarray, clean_a: np.ndarray, plate: np.ndarray,
                    radius: int = 2) -> dict[tuple[int, int], float]:
    """Same residual under small spatial shifts: proves (0, 0) registration."""
    m = clean_a > 0.99
    m[:8] = m[-8:] = False
    m[:, :8] = m[:, -8:] = False
    out = {}
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sh = np.roll(np.roll(plate[..., 0], dy, 0), dx, 1)
            out[(dx, dy)] = float(np.median(np.abs(sh[m] - clean_rgb[..., 0][m])))
    return out
