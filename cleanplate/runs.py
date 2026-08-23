"""Persisting a run to outputs/<name>/ and reading it back.

Layout, all optional except masks:
    masks/    binary mattes  (8-bit L, 0/255)
    alpha/    soft mattes    (8-bit L, 0..255)
    rgba/     RGBA sequence  (straight alpha)
    *.json    per-stage stats
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .paths import out_dir


def save_seq(arr: np.ndarray, d: Path, mode: str = "L") -> int:
    from .compose import write_sequence
    if mode == "L" and arr.dtype == bool:
        arr = (arr.astype(np.uint8) * 255)
    return write_sequence(list(arr), Path(d), mode=mode)


def load_seq(d: Path) -> np.ndarray:
    d = Path(d)
    ps = sorted(d.glob("*.png"), key=lambda p: int(p.stem))
    if not ps:
        raise FileNotFoundError(f"no PNGs in {d}")
    return np.stack([np.asarray(Image.open(p).convert("L")) for p in ps])


def save_json(obj: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
    return path


def save_masks(name: str, masks: np.ndarray, stats: dict | None = None) -> Path:
    d = out_dir(name)
    save_seq(masks, d / "masks")
    if stats:
        save_json(stats, d / "track.json")
    return d / "masks"


def save_alpha(name: str, alphas: np.ndarray, stats: dict | None = None) -> Path:
    d = out_dir(name)
    save_seq(alphas, d / "alpha")
    if stats:
        save_json(stats, d / "refine.json")
    return d / "alpha"


def zip_dir(src: Path, out: Path, arcprefix: str = "") -> Path:
    """Zip a directory of files. Used by the app's export panel."""
    src, out = Path(src), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.iterdir() if p.is_file())
    if not files:
        raise FileNotFoundError(f"nothing to zip in {src}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, Path(arcprefix or src.name) / p.name)
    return out
