"""SAM 2 video segmentation: clicks in, a binary mask per frame out.

Two SAM 2 behaviours are load-bearing here and are enforced/documented rather than
left as traps (both cost a wasted run to discover in Task 2):

1. A conditioning frame carrying only negative points is read as "the object is
   absent in this frame" and yields an empty mask. `validate_prompt` refuses it.
2. Conditioning frames are GLOBAL, not causal. `select_closest_cond_frames` puts
   every conditioning frame into the memory bank at every timestep, so a corrective
   click at t=67 changes the mask at t=24. A corrective frame must therefore
   describe the whole subject, not just the part being fixed. Callers should diff
   against the previous run across the entire shot - see `metrics.per_frame_iou`.
"""
from __future__ import annotations

import os

# Must be set before torch initialises its MPS backend.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import re
import time
import warnings
from typing import Callable

import numpy as np

from .ingest import frame_paths, resolve_frames_dir
from .paths import SAM2_CFG, SAM2_CKPT
from .session import Prompt

MPS_FALLBACK_RE = re.compile(
    r"operator '([^']+)' is not currently supported on the MPS backend", re.I)

ProgressFn = Callable[[float, str], None] | None


def pick_device(requested: str = "auto") -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class PromptError(ValueError):
    """The prompt cannot be run as given, and running it would mislead."""


def validate_prompt(prompt: Prompt, n_frames: int) -> None:
    """Raise if SAM 2 would silently do something surprising."""
    if prompt.total_clicks == 0:
        raise PromptError("No points yet. Click the subject on a frame first.")
    bad = [f for f in prompt.prompt_frames if not (0 <= f < n_frames)]
    if bad:
        raise PromptError(f"Prompt frame(s) {bad} are outside the shot (0-{n_frames - 1}).")
    lonely = prompt.negative_only_frames()
    if lonely:
        raise PromptError(
            f"Frame(s) {lonely} have only negative points. SAM 2 reads a conditioning "
            "frame with no positive point as 'object absent' and will blank that frame. "
            "Add a positive click on the subject in the same frame.")


def track(shot: "str | object", prompt: Prompt, device: str = "auto",
          progress: ProgressFn = None,
          offload_video_to_cpu: bool = False,
          offload_state_to_cpu: bool = False) -> tuple[np.ndarray, dict]:
    """Propagate the prompt through every frame.

    Returns (masks, stats) where masks is a bool array of shape (N, H, W).
    """
    frames = frame_paths(shot)
    if not frames:
        raise FileNotFoundError(f"no frames for shot {shot!r}")
    validate_prompt(prompt, len(frames))
    if not SAM2_CKPT.exists():
        raise FileNotFoundError(
            f"checkpoint missing: {SAM2_CKPT}\nRun ./scripts/download.sh checkpoints")

    import torch
    from sam2.build_sam import build_sam2_video_predictor

    device = pick_device(device)
    fallback_ops: set[str] = set()
    if progress:
        progress(0.0, f"loading SAM 2 on {device}")

    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        predictor = build_sam2_video_predictor(SAM2_CFG, str(SAM2_CKPT), device=device)
        t_build = time.perf_counter() - t0

        if progress:
            progress(0.05, f"decoding {len(frames)} frames")
        t1 = time.perf_counter()
        state = predictor.init_state(
            video_path=str(resolve_frames_dir(shot)),
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu)
        t_init = time.perf_counter() - t1

        by_frame = prompt.for_sam2()
        with torch.inference_mode():
            for f, e in by_frame.items():
                pts = [*e["positive"], *e["negative"]]
                labs = [1] * len(e["positive"]) + [0] * len(e["negative"])
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=f, obj_id=1,
                    points=np.array(pts, dtype=np.float32),
                    labels=np.array(labs, dtype=np.int32))

            t2 = time.perf_counter()
            masks: dict[int, np.ndarray] = {}
            per_frame: list[float] = []
            t_prev = t2
            for frame_idx, _obj_ids, video_res_masks in predictor.propagate_in_video(state):
                masks[frame_idx] = (video_res_masks[0, 0] > 0.0).cpu().numpy()
                now = time.perf_counter()
                per_frame.append(now - t_prev)
                t_prev = now
                if progress:
                    done = len(masks)
                    progress(0.1 + 0.9 * done / len(frames),
                             f"tracking {done}/{len(frames)} "
                             f"({np.mean(per_frame):.2f} s/frame)")
            t_prop = time.perf_counter() - t2

    for w in caught:
        m = MPS_FALLBACK_RE.search(str(w.message))
        if m:
            fallback_ops.add(m.group(1))

    if len(masks) != len(frames):
        missing = sorted(set(range(len(frames))) - set(masks))
        raise RuntimeError(
            f"propagation covered {len(masks)}/{len(frames)} frames; missing {missing[:8]}")

    arr = np.stack([masks[i] for i in range(len(frames))])
    areas = arr.reshape(len(arr), -1).mean(1)

    stats = {
        "stage": "track",
        "shot": str(shot),
        "device": device,
        "torch": torch.__version__,
        "model_cfg": SAM2_CFG,
        "checkpoint": SAM2_CKPT.name,
        "frames": len(frames),
        "prompts": [prompt.frames[f].to_dict() for f in prompt.prompt_frames],
        "total_clicks": prompt.total_clicks,
        "model_build_s": round(t_build, 2),
        "frame_load_s": round(t_init, 2),
        "propagate_s": round(t_prop, 2),
        "seconds_per_frame": round(t_prop / len(frames), 4),
        "fps": round(len(frames) / t_prop, 2),
        "mps_fallback_ops": sorted(fallback_ops),
        "mask_area_fraction": {"min": round(float(areas.min()), 5),
                               "max": round(float(areas.max()), 5),
                               "mean": round(float(areas.mean()), 5)},
        "empty_mask_frames": [int(i) for i in np.where(areas == 0)[0]],
    }
    if progress:
        progress(1.0, f"tracked {len(frames)} frames in {t_prop:.1f}s")
    return arr, stats
