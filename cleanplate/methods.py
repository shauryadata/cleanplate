"""The matting methods RotoBench compares.

Each is a callable (clip, prompt) -> (alpha uint8 (N,H,W) at the clip's resolution,
stats dict). They share everything else: the same oracle prompt, the same metrics.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import refine as _refine
from . import track as _track
from .ingest import frame_paths
from .paths import ROOT, peak_rss_mb
from .session import Prompt

WORK = ROOT / "outputs" / "_bench_work"


def _cached_track(frames_dir: Path, prompt: Prompt, device: str
                  ) -> tuple[np.ndarray, dict]:
    """SAM 2 masks, cached on disk.

    Every experiment starts from the same SAM 2 pass over the same frames with the
    same oracle prompt, so recomputing it once per method wastes about eighty seconds
    a clip. The key covers everything that changes the result.
    """
    key = json.dumps({"dir": str(frames_dir),
                      "prompt": [prompt.frames[f].to_dict()
                                 for f in prompt.prompt_frames]}, sort_keys=True)
    tag = hashlib.sha1(key.encode()).hexdigest()[:12]
    cdir = WORK / "trackcache" / tag
    npz, meta = cdir / "masks.npz", cdir / "stats.json"
    if npz.exists() and meta.exists():
        st = json.loads(meta.read_text())
        st["cached"] = True
        return np.load(npz)["m"], st
    masks, st = _track.track(frames_dir, prompt, device=device, progress=None)
    cdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, m=masks)
    meta.write_text(json.dumps(st, indent=2))
    st["cached"] = False
    return masks, st


def _scaled_frames(frames_dir: Path, width: int) -> tuple[Path, float]:
    """Write a downscaled copy of a frame folder, cached. Returns (dir, scale)."""
    ps = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    with Image.open(ps[0]) as im:
        w0, h0 = im.size
    if width >= w0:
        return frames_dir, 1.0
    scale = width / w0
    h = int(round(h0 * scale)); h -= h % 2
    tag = hashlib.sha1(f"{frames_dir}|{width}".encode()).hexdigest()[:10]
    out = WORK / f"scaled_{width}_{tag}"
    if out.is_dir() and len(list(out.glob("*.jpg"))) == len(ps):
        return out, scale
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for p in ps:
        Image.open(p).convert("RGB").resize((width, h), Image.LANCZOS).save(
            out / p.name, quality=95)
    return out, scale


def _scale_prompt(prompt: Prompt, s: float) -> Prompt:
    if s == 1.0:
        return prompt
    q = Prompt()
    for f, e in prompt.frames.items():
        for x, y in e.positive:
            q.add(f, int(round(x * s)), int(round(y * s)), True)
        for x, y in e.negative:
            q.add(f, int(round(x * s)), int(round(y * s)), False)
    return q


def _upscale(alpha: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bring an alpha sequence back to the clip's resolution. (w, h)."""
    import cv2
    w, h = size
    if alpha.shape[1:] == (h, w):
        return alpha
    return np.stack([cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
                     for a in alpha])


def sam2_matanyone(clip, prompt: Prompt, width: int = 960, device: str = "auto",
                   refine_on: bool = True, model: str = "matanyone"
                   ) -> tuple[np.ndarray, dict]:
    """The pipeline as it stands: SAM 2 binary mask, then MatAnyone soft alpha."""
    ps = frame_paths(clip.frames_dir)
    with Image.open(ps[0]) as im:
        full = im.size                                   # (w, h)

    fdir, s = _scaled_frames(Path(clip.frames_dir), width)
    p = _scale_prompt(prompt, s)

    t0 = time.perf_counter()
    masks, tstats = _cached_track(fdir, p, device)
    t_track = time.perf_counter() - t0
    rss_track = peak_rss_mb()

    stats = {"width": width, "scale": round(s, 4),
             "track_cached": bool(tstats.get("cached")),
             "track_s_per_frame": tstats["seconds_per_frame"],
             "track_device": tstats["device"],
             "track_fallback_ops": tstats["mps_fallback_ops"],
             "track_total_s": round(t_track, 2),
             "peak_rss_after_track_mb": round(rss_track, 1)}

    if not refine_on:
        return _upscale((masks.astype(np.uint8) * 255), full), stats

    t1 = time.perf_counter()
    alpha, rstats = _refine.refine(fdir, masks, device=device, model=model,
                                   progress=None)
    stats.update({"refine_model": rstats["model"],
                  "refine_s_per_frame": rstats["seconds_per_frame"],
                  "refine_device": rstats["device"],
                  "refine_fallback_ops": rstats["mps_fallback_ops"],
                  "refine_total_s": round(time.perf_counter() - t1, 2),
                  "peak_rss_after_refine_mb": round(peak_rss_mb(), 1)})
    return _upscale(alpha, full), stats


# --------------------------------------------------------------- 5a: hair zoom
def _hair_box_from_alpha(alpha: np.ndarray, top_fraction: float = 0.30,
                         pad: int = 24) -> tuple[int, int, int, int]:
    """Head box from a predicted alpha (no truth involved - this runs at inference)."""
    m = alpha.max(axis=0) > 127
    ys, xs = np.where(m)
    if len(ys) == 0:
        raise ValueError("empty alpha")
    y0, y1 = int(ys.min()), int(ys.max())
    cut = int(y0 + (y1 - y0) * top_fraction)
    sel = m[y0:cut]
    xs2 = np.where(sel.any(axis=0))[0]
    H, W = m.shape
    return (max(0, int(xs2.min()) - pad), max(0, y0 - pad),
            min(W, int(xs2.max()) + pad), min(H, cut + pad))


def hair_zoom(clip, prompt: Prompt, base_width: int = 960, zoom: float = 2.0,
              device: str = "auto", feather: int = 12,
              model: str = "matanyone") -> tuple[np.ndarray, dict]:
    """Baseline everywhere, then re-run the matting stage zoomed into the head.

    The hypothesis behind 5a: MatAnyone downsamples internally, so hair detail is lost
    to resolution rather than to the model. Running it again on a magnified crop of
    just the head gives the same model far more pixels per strand. The crop result is
    feathered back in so the seam does not show.
    """
    import cv2
    alpha, stats = sam2_matanyone(clip, prompt, width=base_width, device=device,
                                  model=model)
    H, W = alpha.shape[1:]
    box = _hair_box_from_alpha(alpha)
    l, t, r, b = box
    cw, ch = r - l, b - t

    # crop the plate around the head, magnified
    ps = frame_paths(clip.frames_dir)
    tw = int(round(cw * zoom)); tw -= tw % 2
    th = int(round(ch * zoom)); th -= th % 2
    tag = hashlib.sha1(f"{clip.frames_dir}|zoom|{box}|{zoom}".encode()).hexdigest()[:10]
    cdir = WORK / f"hairzoom_{tag}"
    if cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir(parents=True)
    for i, p in enumerate(ps):
        im = np.asarray(Image.open(p).convert("RGB"))[t:b, l:r]
        Image.fromarray(cv2.resize(im, (tw, th), interpolation=cv2.INTER_CUBIC)).save(
            cdir / f"{i:05d}.jpg", quality=96)

    # seed the crop with the baseline alpha, so no new click is needed
    seed = cv2.resize(alpha[0][t:b, l:r], (tw, th), interpolation=cv2.INTER_LINEAR)
    seed_mask = np.stack([(seed > 127)] + [(seed > 127)] * (len(ps) - 1))
    t0 = time.perf_counter()
    zoomed, rstats = _refine.refine(cdir, seed_mask, device=device, model=model,
                                    progress=None)
    stats.update({"hair_box": list(box), "zoom": zoom,
                  "hairzoom_s_per_frame": rstats["seconds_per_frame"],
                  "hairzoom_total_s": round(time.perf_counter() - t0, 2),
                  "peak_rss_after_hairzoom_mb": round(peak_rss_mb(), 1)})

    # feather the crop back in
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    f = max(1, feather)
    w = np.minimum.reduce([xx, cw - 1 - xx, yy, ch - 1 - yy]) / f
    w = np.clip(w, 0, 1)[None]
    out = alpha.copy()
    for i in range(len(out)):
        back = cv2.resize(zoomed[i], (cw, ch), interpolation=cv2.INTER_AREA)
        cur = out[i, t:b, l:r].astype(np.float32)
        out[i, t:b, l:r] = np.clip(cur * (1 - w[0]) + back.astype(np.float32) * w[0],
                                   0, 255).astype(np.uint8)
    return out, stats


def fullres_matte(clip, prompt: Prompt, track_width: int = 960, device: str = "auto",
                  model: str = "matanyone") -> tuple[np.ndarray, dict]:
    """Track at 960, then run the MATTING stage at full resolution.

    `fullres_1920` - running both stages at 1920 - turned out to answer the wrong
    question. SAM 2's multi-mask head chooses a different granularity at a different
    input resolution: on A3 the same relative click selected the woman's SKIN (face,
    neck, chest) at 1920 where it selected the whole person at 960. The matting stage
    was then handed a mask of the wrong object, and the clip scored MAD 149.

    So the resolution hypothesis has to be tested with the segmentation held constant:
    same 960 SAM 2 mask, upscaled, matted at full res. This isolates "does the matting
    stage benefit from more pixels" from "does SAM 2 behave differently at 1920".
    """
    import cv2
    ps = frame_paths(clip.frames_dir)
    with Image.open(ps[0]) as im:
        W, H = im.size

    fdir, sc = _scaled_frames(Path(clip.frames_dir), track_width)
    p = _scale_prompt(prompt, sc)
    t0 = time.perf_counter()
    masks, tstats = _cached_track(fdir, p, device)
    t_track = time.perf_counter() - t0

    big = np.stack([cv2.resize(m.astype(np.uint8) * 255, (W, H),
                               interpolation=cv2.INTER_NEAREST) > 127 for m in masks])
    t1 = time.perf_counter()
    alpha, rstats = _refine.refine(Path(clip.frames_dir), big, device=device,
                                   model=model, progress=None)
    return alpha, {"track_width": track_width, "matte_width": W,
                   "track_cached": bool(tstats.get("cached")),
                   "track_s_per_frame": tstats["seconds_per_frame"],
                   "track_total_s": round(t_track, 2),
                   "refine_model": rstats["model"],
                   "refine_s_per_frame": rstats["seconds_per_frame"],
                   "refine_total_s": round(time.perf_counter() - t1, 2),
                   "refine_fallback_ops": rstats["mps_fallback_ops"],
                   "peak_rss_mb": round(peak_rss_mb(), 1)}


# --------------------------------------------------------------- 5c: guided filter
def guided_filter(guide_gray: np.ndarray, src: np.ndarray, radius: int = 8,
                  eps: float = 1e-4) -> np.ndarray:
    """He et al. guided filter, single channel. Both inputs float32 in [0, 1].

    Implemented here rather than pulled from opencv-contrib: it is a dozen box filters
    and adding a whole extra OpenCV build for it is not worth it.
    """
    import cv2
    k = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.blur(guide_gray, k)
    mean_p = cv2.blur(src, k)
    corr_i = cv2.blur(guide_gray * guide_gray, k)
    corr_ip = cv2.blur(guide_gray * src, k)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return cv2.blur(a, k) * guide_gray + cv2.blur(b, k)


def guided(clip, prompt: Prompt, base_width: int = 960, radius: int = 8,
           eps: float = 1e-4, device: str = "auto",
           model: str = "matanyone") -> tuple[np.ndarray, dict]:
    """Baseline, then pull the alpha towards the plate's own edges with a guided filter.

    The cheap hypothesis: the plate already knows where the hair is; a guided filter
    transfers that structure onto the alpha for almost no compute.
    """
    import cv2
    alpha, stats = sam2_matanyone(clip, prompt, width=base_width, device=device,
                                  model=model)
    ps = frame_paths(clip.frames_dir)
    t0 = time.perf_counter()
    out = np.empty_like(alpha)
    for i, p in enumerate(ps):
        g = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        a = alpha[i].astype(np.float32) / 255.0
        out[i] = np.clip(guided_filter(g, a, radius, eps) * 255.0, 0, 255).astype(np.uint8)
    stats.update({"guided_radius": radius, "guided_eps": eps,
                  "guided_total_s": round(time.perf_counter() - t0, 2),
                  "guided_s_per_frame": round((time.perf_counter() - t0) / len(ps), 4)})
    return out, stats


# --------------------------------------------------------------- 5b: trimap+ViTMatte
_VITMATTE = {}


def _vitmatte(device: str):
    """Load ViTMatte once. Apache-2.0, unlike the MatAnyone family."""
    if "m" not in _VITMATTE:
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor
        mid = "hustvl/vitmatte-small-composition-1k"
        _VITMATTE["p"] = VitMatteImageProcessor.from_pretrained(mid)
        _VITMATTE["m"] = VitMatteForImageMatting.from_pretrained(mid).to(device).eval()
    return _VITMATTE["p"], _VITMATTE["m"]


def make_trimap(alpha: np.ndarray, band: int = 18) -> np.ndarray:
    """0 = background, 128 = unknown, 255 = foreground.

    The unknown band is what ViTMatte actually solves in, so it must be wide enough to
    contain the hair. Too narrow and the model is not allowed to find strands; too wide
    and it has to re-solve the whole subject.
    """
    import cv2
    a = alpha
    fg = (a >= 250).astype(np.uint8)
    bg = (a <= 5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1,) * 2)
    fg_sure = cv2.erode(fg, k, iterations=1)
    bg_sure = cv2.erode(bg, k, iterations=1)
    tri = np.full(a.shape, 128, np.uint8)
    tri[fg_sure > 0] = 255
    tri[bg_sure > 0] = 0
    return tri


def trimap_vitmatte(clip, prompt: Prompt, base_width: int = 960, band: int = 18,
                    temporal: float = 0.35, device: str = "auto",
                    model: str = "matanyone") -> tuple[np.ndarray, dict]:
    """Baseline, then re-solve the head region with ViTMatte on a trimap.

    ViTMatte is a per-frame image matter, so it has no temporal memory at all. Its
    output is smoothed along time with a one-pole filter before being blended back,
    otherwise it reintroduces exactly the boil that MatAnyone removed.
    """
    import cv2
    import torch
    alpha, stats = sam2_matanyone(clip, prompt, width=base_width, device=device,
                                  model=model)
    dev = _track.pick_device(device)
    proc, net = _vitmatte(dev)

    box = _hair_box_from_alpha(alpha)
    l, t, r, b = box
    ps = frame_paths(clip.frames_dir)
    t0 = time.perf_counter()

    patch_alpha = []
    with torch.inference_mode():
        for i, p in enumerate(ps):
            img = np.asarray(Image.open(p).convert("RGB"))[t:b, l:r]
            tri = make_trimap(alpha[i][t:b, l:r], band)
            if not (tri == 128).any():          # nothing uncertain: keep the baseline
                patch_alpha.append(alpha[i][t:b, l:r].astype(np.float32) / 255.0)
                continue
            inp = proc(images=img, trimaps=tri, return_tensors="pt").to(dev)
            out = net(**inp).alphas[0, 0].float().cpu().numpy()
            patch_alpha.append(out[: b - t, : r - l])
    pa = np.stack(patch_alpha)

    # one-pole temporal smoothing, forward then backward so there is no phase lag
    if temporal > 0:
        sm = pa.copy()
        for i in range(1, len(sm)):
            sm[i] = temporal * sm[i - 1] + (1 - temporal) * sm[i]
        for i in range(len(sm) - 2, -1, -1):
            sm[i] = temporal * sm[i + 1] + (1 - temporal) * sm[i]
        pa = sm

    # blend the patch back only where the trimap was uncertain, feathered at the box
    ch, cw = b - t, r - l
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    edge = np.clip(np.minimum.reduce([xx, cw - 1 - xx, yy, ch - 1 - yy]) / 12.0, 0, 1)
    out = alpha.copy()
    for i in range(len(out)):
        tri = make_trimap(alpha[i][t:b, l:r], band)
        w = edge * (tri == 128)
        cur = out[i, t:b, l:r].astype(np.float32) / 255.0
        blended = cur * (1 - w) + np.clip(pa[i], 0, 1) * w
        out[i, t:b, l:r] = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    stats.update({"vitmatte_model": "hustvl/vitmatte-small-composition-1k (Apache-2.0)",
                  "trimap_band_px": band, "temporal_alpha": temporal,
                  "hair_box": list(box),
                  "vitmatte_total_s": round(time.perf_counter() - t0, 2),
                  "vitmatte_s_per_frame": round((time.perf_counter() - t0) / len(ps), 4),
                  "peak_rss_after_vitmatte_mb": round(peak_rss_mb(), 1)})
    return out, stats


# --------------------------------------------------------------- registry
def make(name: str):
    """Look up a method by name."""
    table = {
        # baseline: the pipeline as Task 3 shipped it
        "binary_960":    lambda c, p: sam2_matanyone(c, p, width=960, refine_on=False),
        "baseline_960":  lambda c, p: sam2_matanyone(c, p, width=960),
        # 5a resolution
        "fullres_1920":  lambda c, p: sam2_matanyone(c, p, width=1920),
        "fullmatte_1920": lambda c, p: fullres_matte(c, p, track_width=960),
        "fullmatte2_1920": lambda c, p: fullres_matte(c, p, track_width=960,
                                                      model="matanyone2"),
        "hairzoom_960":  lambda c, p: hair_zoom(c, p, base_width=960, zoom=2.0),
        # 5b trimap -> per-frame ViTMatte on the head, temporally smoothed
        "vitmatte_960":  lambda c, p: trimap_vitmatte(c, p, base_width=960),
        # 5c cheap refinement
        "guided_960":    lambda c, p: guided(c, p, base_width=960),
        # 5d newer model
        "matanyone2_960":  lambda c, p: sam2_matanyone(c, p, width=960,
                                                       model="matanyone2"),
        "matanyone2_1920": lambda c, p: sam2_matanyone(c, p, width=1920,
                                                       model="matanyone2"),
    }
    if name not in table:
        raise KeyError(f"unknown method {name!r}; have {sorted(table)}")
    return table[name]
