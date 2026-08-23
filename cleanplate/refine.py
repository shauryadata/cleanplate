"""MatAnyone: binary mask in, soft alpha out.

MatAnyone is licensed S-Lab 1.0, NON-COMMERCIAL ONLY, so it is never bundled -
scripts/download.sh clones it into gitignored vendor/. This module is a thin
adapter and CleanPlate runs fine without it, returning the binary matte instead.
See THIRD_PARTY.md and docs/DECISIONS.md.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import contextlib
import re
import sys
import time
import warnings
from typing import Callable

import numpy as np

from .ingest import frame_paths
from .paths import MATANYONE_CKPT
from .track import MPS_FALLBACK_RE, pick_device

ProgressFn = Callable[[float, str], None] | None

LICENCE_NOTE = ("MatAnyone - S-Lab License 1.0, NON-COMMERCIAL USE ONLY. "
                "Not bundled; see THIRD_PARTY.md.")


def available() -> tuple[bool, str]:
    """Can we refine? Returns (ok, human-readable reason if not)."""
    if not MATANYONE_CKPT.exists():
        return False, (f"checkpoint missing: {MATANYONE_CKPT.name}. "
                       "Run ./scripts/download.sh matanyone")
    try:
        import matanyone  # noqa: F401
    except Exception as e:
        return False, (f"matanyone not importable ({type(e).__name__}: {e}). "
                       "Run ./scripts/download.sh matanyone")
    return True, ""


@contextlib.contextmanager
def _hydra_for_matanyone():
    """Let MatAnyone initialise Hydra, then hand it back to SAM 2.

    sam2/__init__.py registers its config module at import time; MatAnyone's
    get_matanyone_model() calls hydra.initialize() unconditionally. Both in one
    process raises "GlobalHydra is already initialized". This never surfaced while
    the two stages ran as separate CLI processes, but the app runs them together.
    """
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    try:
        yield
    finally:
        GlobalHydra.instance().clear()
        if "sam2" in sys.modules:                    # restore SAM 2's config module
            from hydra import initialize_config_module
            initialize_config_module("sam2", version_base="1.2")


def _condition_mask(mask: np.ndarray, dilate: int, erode: int) -> np.ndarray:
    """Close small holes, then pull the edge in, as upstream does.

    The matting model wants an unambiguous interior, not a noisy boundary.
    """
    import cv2
    m = (mask.astype(np.float32) * 255.0) if mask.dtype == bool else mask.astype(np.float32)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        m = cv2.dilate((m != 0).astype(np.float32), k, iterations=1) * 255
    if erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode))
        m = cv2.erode((m == 255).astype(np.float32), k, iterations=1) * 255
    return m


def refine(shot: str, masks: np.ndarray, anchor_frame: int = 0,
           device: str = "auto", warmup: int = 10, dilate: int = 10, erode: int = 10,
           reanchor: list[int] | None = None,
           progress: ProgressFn = None) -> tuple[np.ndarray, dict]:
    """Refine binary masks into a soft alpha.

    Returns (alphas, stats) where alphas is uint8 (N, H, W) in 0..255.
    """
    ok, why = available()
    if not ok:
        raise FileNotFoundError(why)

    frames = frame_paths(shot)
    if len(frames) != len(masks):
        raise ValueError(f"{len(frames)} frames but {len(masks)} masks")

    import torch
    from PIL import Image

    device = pick_device(device)

    # matanyone/model/matanyone.py binds `device = get_default_device()` at import
    # time, and encode_image() pushes pixel_mean/pixel_std to THAT global rather than
    # to the model's device. Without this rebind, --device cpu on a Mac with MPS
    # available mixes devices and crashes.
    import matanyone.model.matanyone as _ma_model
    if str(_ma_model.device) != str(torch.device(device)):
        _ma_model.device = torch.device(device)
    from matanyone.inference.inference_core import InferenceCore
    from matanyone.utils.get_default_model import get_matanyone_model

    reanchor_set = set(reanchor or [])
    fallback_ops: set[str] = set()
    if progress:
        progress(0.0, f"loading MatAnyone on {device}")

    def frame_tensor(i: int) -> "torch.Tensor":
        arr = np.asarray(Image.open(frames[i]).convert("RGB"), dtype=np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1).to(device) / 255.0

    def mask_tensor(i: int) -> "torch.Tensor":
        return torch.from_numpy(_condition_mask(masks[i], dilate, erode)).float().to(device)

    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        with _hydra_for_matanyone():
            model = get_matanyone_model(str(MATANYONE_CKPT), device)
        processor = InferenceCore(model, cfg=model.cfg, device=device)
        t_build = time.perf_counter() - t0

        out = np.empty((len(frames),) + masks.shape[1:], dtype=np.uint8)
        t1 = time.perf_counter()
        per_frame: list[float] = []

        with torch.inference_mode():
            # Settle the first-frame alpha by showing the anchor repeatedly. These
            # passes are not recorded.
            if progress:
                progress(0.05, f"warmup x{warmup}")
            anchor_img = frame_tensor(anchor_frame)
            processor.step(anchor_img, mask_tensor(anchor_frame), objects=[1])
            prob = processor.step(anchor_img, first_frame_pred=True)
            for _ in range(warmup):
                prob = processor.step(anchor_img, first_frame_pred=True)

            t_prev = time.perf_counter()
            for i in range(len(frames)):
                img = frame_tensor(i)
                if i == anchor_frame:
                    prob = processor.step(img, first_frame_pred=True)
                elif i in reanchor_set:
                    prob = processor.step(img, mask_tensor(i), objects=[1])
                else:
                    prob = processor.step(img)
                alpha = processor.output_prob_to_mask(prob)          # soft, 0..1
                out[i] = np.clip(alpha.float().cpu().numpy() * 255.0, 0, 255).round()
                now = time.perf_counter()
                per_frame.append(now - t_prev)
                t_prev = now
                if progress:
                    progress(0.1 + 0.9 * (i + 1) / len(frames),
                             f"refining {i + 1}/{len(frames)} "
                             f"({np.mean(per_frame):.2f} s/frame)")
        t_run = time.perf_counter() - t1

    for w in caught:
        m = MPS_FALLBACK_RE.search(str(w.message))
        if m:
            fallback_ops.add(m.group(1))

    soft = ((out > 0) & (out < 255)).reshape(len(out), -1).mean(1)
    stats = {
        "stage": "refine",
        "shot": shot,
        "model": "MatAnyone v1.0.0",
        "licence": LICENCE_NOTE,
        "device": device,
        "checkpoint": MATANYONE_CKPT.name,
        "anchor_frame": anchor_frame,
        "reanchor_frames": sorted(reanchor_set),
        "warmup": warmup, "dilate": dilate, "erode": erode,
        "frames": len(frames),
        "model_build_s": round(t_build, 2),
        "refine_s": round(t_run, 2),
        "seconds_per_frame": round(t_run / len(frames), 4),
        "fps": round(len(frames) / t_run, 2),
        "mps_fallback_ops": sorted(fallback_ops),
        "soft_pixel_fraction": {"min": round(float(soft.min()), 6),
                                "max": round(float(soft.max()), 6),
                                "mean": round(float(soft.mean()), 6)},
    }
    if progress:
        progress(1.0, f"refined {len(frames)} frames in {t_run:.1f}s")
    return out, stats
