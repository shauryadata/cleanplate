"""Harmonize v0: tie a cut-out into the plate it is being dropped onto.

Two separate, optional things, both deliberately crude and both judged by eye before
anything is switched on by default:

**Colour and exposure match.** One gain and offset per channel for the WHOLE shot,
estimated from a sample of frames, blended toward identity by `strength`. Global and
static on purpose: a per-frame match is a flicker generator, and this stage runs on
mattes whose edges already move.

**Grain match.** The plate has sensor noise; a cut-out that came from another camera
usually has less, and the difference reads as "pasted on". v0 measures the
high-frequency noise of both and adds the shortfall to the foreground.

What v0 does NOT do, and what will still look wrong:
  * nothing directional - a subject lit from the left comped into a plate lit from the
    right stays wrong, and no global grade fixes that
  * no contact shadow, no light wrap, no interactive bounce
  * no per-region control: a face and a dark jacket get the same treatment, so a strong
    match can push skin somewhere unpleasant. Hence `strength`, and hence the default
    being whatever the review decided rather than 1.0

The background tracker is here too, for the opposite problem: when the camera moves and
the new background does not, the comp reads as a sticker. It measures the plate's own
2D translation from background pixels and pans the new background by the same amount.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


# --------------------------------------------------------------- colour + grain
@dataclass
class Harmony:
    gain: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    offset: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    grain_sigma: float = 0.0
    strength: float = 0.5
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"gain": [round(g, 4) for g in self.gain],
                "offset": [round(o, 2) for o in self.offset],
                "grain_sigma": round(self.grain_sigma, 3),
                "strength": self.strength, "stats": self.stats}


def _noise_sigma(rgb: np.ndarray) -> float:
    """Immerkaer's noise estimate: mean |I * L| over a Laplacian-like kernel, scaled.

    A first version took the MAD of image-minus-median-blur over the whole frame and
    reported 0.00 for every image, because most of a frame is flat and a median filter
    changes nothing there - so the median of the residual was zero and the grain match
    silently did nothing. This estimator is the standard one and does not have that
    failure mode.
    """
    import cv2
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    lap = cv2.filter2D(g, -1, k)
    h, w = g.shape
    return float(np.sqrt(np.pi / 2) * np.abs(lap[1:-1, 1:-1]).mean() / 6.0)


def estimate(frames: list[np.ndarray], alphas: np.ndarray, plates: list[np.ndarray],
             strength: float = 0.5, clamp: tuple[float, float] = (0.75, 1.35),
             sample: int = 8, mode: str = "cast") -> Harmony:
    """One transform for the shot, from every `sample`-th frame.

    The foreground statistics come from pixels the matte calls solid (alpha > 0.9), so
    edge pixels - which are part background already - cannot drag the estimate.

    mode="full"  match the foreground's per-channel mean and spread to the plate's.
                 Textbook colour transfer, and on a lit person over a dark landscape it
                 is far too strong: it matches a face to the average of a desert.
    mode="cast"  transfer only the plate's colour CAST, preserving the foreground's own
                 luminance. Much closer to what a compositor actually does first, and
                 the reason there is a choice here at all.
    """
    fg_px, bg_px, fg_n, bg_n = [], [], [], []
    for i in range(0, len(frames), max(1, sample)):
        a = alphas[i]
        m = (a > 229) if a.dtype == np.uint8 else (a > 0.9)
        if m.sum() < 500:
            continue
        fg_px.append(frames[i][m].reshape(-1, 3).astype(np.float32))
        p = plates[i] if isinstance(plates, list) else plates
        bg_px.append(p.reshape(-1, 3).astype(np.float32))
        fg_n.append(_noise_sigma(frames[i])); bg_n.append(_noise_sigma(p))
    if not fg_px:
        return Harmony(stats={"note": "matte never solid enough to estimate from"})
    f, b = np.concatenate(fg_px), np.concatenate(bg_px)
    fm, fs = f.mean(0), f.std(0) + 1e-6
    bm, bs = b.mean(0), b.std(0) + 1e-6
    if mode == "cast":
        # per-channel ratio relative to each image's own luminance: the cast, not the level
        gain = np.clip((bm / bm.mean()) / (fm / fm.mean()), *clamp)
        offset = np.zeros(3, np.float32)
    else:
        gain = np.clip(bs / fs, *clamp)
        offset = bm - gain * fm
    # Blend toward identity. At strength 0 this is a no-op by construction.
    s = float(strength)
    gain = 1.0 + s * (gain - 1.0)
    offset = s * offset
    grain = max(0.0, float(np.median(bg_n) - np.median(fg_n)))
    return Harmony(gain=gain.tolist(), offset=offset.tolist(), grain_sigma=grain,
                   strength=s,
                   stats={"mode": mode, "fg_mean": [round(x, 1) for x in fm.tolist()],
                          "bg_mean": [round(x, 1) for x in bm.tolist()],
                          "fg_std": [round(x, 1) for x in fs.tolist()],
                          "bg_std": [round(x, 1) for x in bs.tolist()],
                          "fg_noise_sigma": round(float(np.median(fg_n)), 3),
                          "bg_noise_sigma": round(float(np.median(bg_n)), 3),
                          "frames_sampled": len(fg_px)})


def apply(rgb: np.ndarray, h: Harmony, alpha: np.ndarray | None = None,
          grain: bool = True, seed: int = 0) -> np.ndarray:
    """Apply the shot's transform to one frame. Grain only where the matte is non-zero."""
    out = rgb.astype(np.float32) * np.asarray(h.gain) + np.asarray(h.offset)
    if grain and h.grain_sigma > 0.05:
        rng = np.random.default_rng(seed)
        n = rng.normal(0.0, h.grain_sigma, rgb.shape[:2]).astype(np.float32)[..., None]
        if alpha is not None:
            a = (alpha.astype(np.float32) / 255.0) if alpha.dtype == np.uint8 else alpha
            n = n * a[..., None]
        out = out + n
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------- background track
def track_translation(frames: list[np.ndarray], alphas: np.ndarray,
                      dilate: int = 32) -> np.ndarray:
    """Per-frame cumulative 2D translation of the plate, in pixels, from BACKGROUND only.

    Lucas-Kanade on features found outside the (dilated) matte, median of the per-feature
    displacement. Median rather than a fit: a handful of features stuck to the subject's
    edge would drag a mean, and a 2D translation is all this is claiming to measure.

    Returns (N, 2) cumulative [dx, dy]; frame 0 is [0, 0]. A pan right gives negative dx,
    because the content moves left.
    """
    import cv2
    n = len(frames)
    cum = np.zeros((n, 2), np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1,) * 2)
    prev_g = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for i in range(1, n):
        a = alphas[i - 1]
        m = ((a > 12) if a.dtype == np.uint8 else (a > 0.05)).astype(np.uint8)
        bg_mask = (1 - (cv2.dilate(m, k) > 0).astype(np.uint8)) * 255
        g = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        p0 = cv2.goodFeaturesToTrack(prev_g, maxCorners=400, qualityLevel=0.01,
                                     minDistance=12, mask=bg_mask)
        d = np.array([0.0, 0.0], np.float32)
        if p0 is not None and len(p0) >= 8:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, g, p0, None,
                                                 winSize=(21, 21), maxLevel=3)
            ok = (st.reshape(-1) == 1)
            if ok.sum() >= 8:
                d = np.median((p1 - p0).reshape(-1, 2)[ok], axis=0)
        cum[i] = cum[i - 1] + d
        prev_g = g
    return cum


def pan_pad(offsets: np.ndarray, margin: int = 2) -> tuple[int, int, int, int]:
    """Canvas padding (left, top, right, bottom) for a track, in the direction it goes.

    Padding symmetrically by max|offset| doubles the canvas and forces the source to be
    upscaled twice as much as it needs to be. A pan that only ever goes left needs room
    on the right only.
    """
    dx, dy = offsets[:, 0], offsets[:, 1]
    return (int(np.ceil(max(0.0, float(dx.max())))) + margin,
            int(np.ceil(max(0.0, float(dy.max())))) + margin,
            int(np.ceil(max(0.0, float(-dx.min())))) + margin,
            int(np.ceil(max(0.0, float(-dy.min())))) + margin)


def panned_background(bg: Image.Image, size: tuple[int, int], offsets: np.ndarray,
                      crop: tuple[int, int, int, int] | None = None,
                      pad: tuple[int, int, int, int] | None = None) -> list[np.ndarray]:
    """One background frame per offset, panned so it sticks to the world.

    The source is scaled to cover the frame plus the travel of the track, so the pan
    reveals real background instead of running off the edge. Pass `pad` to build a
    static and a tracked version on the same canvas, so an A/B cannot differ in scale.
    """
    from . import compose
    if crop:
        bg = bg.crop(crop)
    tw, th = size
    left, top, right, bottom = pad if pad is not None else pan_pad(offsets)
    big = compose.cover_fit(bg, (tw + left + right, th + top + bottom))
    out = []
    for dx, dy in offsets:
        x0 = int(round(left - dx)); y0 = int(round(top - dy))
        x0 = max(0, min(left + right, x0)); y0 = max(0, min(top + bottom, y0))
        out.append(big[y0:y0 + th, x0:x0 + tw])
    return out
