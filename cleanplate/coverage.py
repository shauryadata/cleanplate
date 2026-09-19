"""A matting stage that can ADD coverage: the Task 5 face dropout, attacked with truth.

The dropout is a segmentation failure. SAM 2's mask loses part of the subject for a
stretch of frames (the user's shadowed jaw), and MatAnyone, which refines what it is
given, cannot bring back a region that is not in its input. So the repair has to be
allowed to say "this is foreground" where the mask said nothing.

How it finds where to look, with no truth at inference time:

  suspect(t) = [ pixels that were subject in any frame t-k..t+k ]
               U [ holes enclosed by this frame's mask ]
               minus this frame's mask

A jaw that was in the mask three frames ago and is gone now is exactly that.

Two variants, both reported, both kept in the table whatever happens:

  cover_960   the brief's design: a trimap whose unknown region is the suspect area
              (plus a narrow edge band), solved by ViTMatte, merged with max() INSIDE
              the suspect area only. It can add coverage; it can never remove any.
  cover2_960  the blunt alternative: a wide unknown band around the whole mask,
              ViTMatte's answer used across that band as-is. It can add and remove.

Base is hairzoom2 - the app's default output since Task 5 - so the repair is judged
on what a user actually gets. ViTMatte is Apache-2.0; nothing here is non-commercial
beyond the base it composes with.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import methods as M
from .ingest import frame_paths
from .paths import peak_rss_mb


def suspect_regions(masks: np.ndarray, k: int = 6) -> np.ndarray:
    """(N, H, W) bool: where coverage may have been dropped. See the module doc."""
    from scipy import ndimage
    n = len(masks)
    out = np.zeros_like(masks, dtype=bool)
    for t in range(n):
        lo, hi = max(0, t - k), min(n, t + k + 1)
        support = masks[lo:hi].any(axis=0)
        holes = ndimage.binary_fill_holes(masks[t]) & ~masks[t]
        out[t] = (support | holes) & ~masks[t]
    return out


def _trimap(mask: np.ndarray, unknown_extra: np.ndarray, core_erode: int,
            edge_band: int, bg_margin: int) -> np.ndarray:
    import cv2
    ell = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    m = mask.astype(np.uint8)
    fg = cv2.erode(m, ell(core_erode)) > 0
    near = cv2.dilate(m, ell(edge_band)) > 0
    reach = cv2.dilate((mask | unknown_extra).astype(np.uint8), ell(bg_margin)) > 0
    tri = np.zeros(mask.shape, np.uint8)            # background
    tri[near | reach] = 128                          # unknown: edge band + suspect area
    tri[fg] = 255                                    # confident core
    return tri


def _vitmatte_seq(frames960: Path, trimaps: np.ndarray, device: str) -> np.ndarray:
    import torch
    proc, net = M._vitmatte(device)
    ps = sorted(Path(frames960).glob("*.jpg"), key=lambda p: int(p.stem))
    out = []
    with torch.inference_mode():
        for p, tri in zip(ps, trimaps):
            if not (tri == 128).any():
                out.append((tri == 255).astype(np.float32)); continue
            img = np.asarray(Image.open(p).convert("RGB"))
            inp = proc(images=img, trimaps=tri, return_tensors="pt").to(device)
            a = net(**inp).alphas[0, 0].float().cpu().numpy()[: tri.shape[0], : tri.shape[1]]
            a[tri == 255] = 1.0; a[tri == 0] = 0.0
            out.append(a)
    return np.stack(out)


def _smooth(a: np.ndarray, w: float = 0.35) -> np.ndarray:
    """One-pole filter forward then backward: no phase lag. As in trimap_vitmatte."""
    s = a.copy()
    for i in range(1, len(s)):
        s[i] = w * s[i - 1] + (1 - w) * s[i]
    for i in range(len(s) - 2, -1, -1):
        s[i] = w * s[i + 1] + (1 - w) * s[i]
    return s


def cover(clip, prompt, variant: str = "suspect", k: int = 6, width: int = 960,
          device: str = "auto") -> tuple[np.ndarray, dict]:
    import cv2
    t_all = time.perf_counter()
    base, stats = M.hair_zoom(clip, prompt, base_width=width, zoom=2.0,
                              device=device, model="matanyone2")
    fdir, s = M._scaled_frames(Path(clip.frames_dir), width)
    masks, _ = M._cached_track(fdir, M._scale_prompt(prompt, s), device)   # cache hit
    dev = M._track.pick_device(device)
    t0 = time.perf_counter()
    if variant == "suspect":
        sus = suspect_regions(masks, k=k)
        tris = np.stack([_trimap(m, u, core_erode=6, edge_band=4, bg_margin=8)
                         for m, u in zip(masks, sus)])
    else:                                            # "wideband"
        sus = None
        tris = np.stack([_trimap(m, np.zeros_like(m), core_erode=12, edge_band=24,
                                 bg_margin=0) for m in masks])
    a960 = _smooth(_vitmatte_seq(fdir, tris, dev))
    H, W = base.shape[1:]
    up = np.stack([cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR) for a in a960])
    b = base.astype(np.float32) / 255.0
    if variant == "suspect":
        region = np.stack([cv2.resize(u.astype(np.uint8), (W, H),
                                      interpolation=cv2.INTER_NEAREST) > 0 for u in sus])
        out = np.where(region, np.maximum(b, up), b)            # add, never remove
        added = float((region & (up > b + 0.5)).sum()) / len(out)
    else:
        band = np.stack([cv2.resize((t == 128).astype(np.uint8), (W, H),
                                    interpolation=cv2.INTER_NEAREST) > 0 for t in tris])
        out = np.where(band, up, b)
        added = float((band & (up > b + 0.5)).sum()) / len(out)
    stats.update({"cover_variant": variant, "cover_k": k,
                  "cover_s_per_frame": round((time.perf_counter() - t0) / len(out), 4),
                  "cover_px_added_per_frame": round(added, 1),
                  "cover_total_s": round(time.perf_counter() - t_all, 2),
                  "peak_rss_after_cover_mb": round(peak_rss_mb(), 1),
                  "vitmatte_model": "hustvl/vitmatte-small-composition-1k (Apache-2.0)"})
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8), stats
