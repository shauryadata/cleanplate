#!/usr/bin/env python3
"""Turn SAM 2's binary mask into a soft alpha with MatAnyone.

SAM 2 answers "which pixels are the subject"; it answers in whole pixels. MatAnyone
takes that answer as a first-frame anchor and re-solves the boundary as a matting
problem, so hair, motion blur and edge pixels come back as real fractional coverage,
propagated with its own memory for temporal stability.

    python src/refine_matte.py --shot walk --masks outputs/walk_v2/masks \
        --out-name walk_v2

In    shots/<shot>/frames/NNNNN.jpg  +  a binary mask for the anchor frame
Out   outputs/<out-name>/alpha/NNNNN.png    8-bit soft alpha, 0-255
      outputs/<out-name>/refine.json        device, timings, settings

MatAnyone is licensed S-Lab 1.0 (non-commercial). It is an optional stage: CleanPlate
without it still produces the binary SAM 2 matte. See docs/DECISIONS.md and
THIRD_PARTY.md.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import re
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "checkpoints" / "matanyone.pth"
MPS_FALLBACK_RE = re.compile(
    r"operator '([^']+)' is not currently supported on the MPS backend", re.I)


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def pick_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--masks", type=Path, default=None,
                    help="folder of binary masks (default outputs/<out-name>/masks)")
    ap.add_argument("--out-name", default=None,
                    help="write to outputs/<out-name>/ (default: the shot name)")
    ap.add_argument("--anchor-frame", type=int, default=0,
                    help="frame whose binary mask seeds MatAnyone (default 0)")
    ap.add_argument("--reanchor", type=int, action="append", default=[],
                    metavar="FRAME",
                    help="also re-inject the binary mask at this frame (repeatable). "
                         "MatAnyone normally runs off the first frame alone; use this "
                         "only if measurement shows the matte drifting.")
    ap.add_argument("--warmup", type=int, default=10,
                    help="repeat the anchor frame N times before recording, to settle "
                         "the first-frame alpha (default 10, as upstream)")
    ap.add_argument("--dilate", type=int, default=10, help="dilate the input mask (px)")
    ap.add_argument("--erode", type=int, default=10, help="erode the input mask (px)")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    args = ap.parse_args()

    out_name = args.out_name or args.shot
    out_root = ROOT / "outputs" / out_name
    masks_dir = (args.masks.resolve() if args.masks else out_root / "masks")
    frames_dir = ROOT / "shots" / args.shot / "frames"

    if not frames_dir.is_dir():
        sys.exit(f"no frames: {frames_dir}")
    if not masks_dir.is_dir():
        sys.exit(f"no masks: {masks_dir}\nRun src/track_matte.py first.")
    if not args.checkpoint.exists():
        sys.exit(f"checkpoint missing: {args.checkpoint}\n"
                 "Run ./scripts/download.sh matanyone")

    frames = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if not frames:
        sys.exit(f"no frames in {frames_dir}")

    import torch
    import cv2
    from PIL import Image

    device = pick_device(args.device)

    # matanyone/model/matanyone.py binds `device = get_default_device()` at import time
    # and encode_image() pushes pixel_mean/pixel_std to THAT global rather than to the
    # model's device. Rebind it before anything runs, or --device cpu mixes devices.
    import matanyone.model.matanyone as _ma_model
    if str(_ma_model.device) != str(torch.device(device)):
        print(f"[refine] rebinding matanyone module device "
              f"{_ma_model.device} -> {device}")
        _ma_model.device = torch.device(device)
    from matanyone.inference.inference_core import InferenceCore
    from matanyone.utils.get_default_model import get_matanyone_model

    anchor = args.anchor_frame
    anchor_mask_path = masks_dir / f"{anchor:05d}.png"
    if not anchor_mask_path.exists():
        sys.exit(f"no mask for anchor frame {anchor}: {anchor_mask_path}")

    alpha_dir = out_root / "alpha"
    if alpha_dir.exists():
        shutil.rmtree(alpha_dir)
    alpha_dir.mkdir(parents=True)

    print(f"[refine] shot={args.shot}  frames={len(frames)}  device={device}")
    print(f"[refine] anchor mask: {rel(anchor_mask_path)} (frame {anchor})")
    print(f"[refine] warmup={args.warmup}  dilate={args.dilate}  erode={args.erode}"
          + (f"  reanchor at {args.reanchor}" if args.reanchor else ""))
    print(f"[refine] outputs -> {rel(alpha_dir)}")

    def load_mask(idx: int) -> torch.Tensor:
        m = np.array(Image.open(masks_dir / f"{idx:05d}.png").convert("L")).astype(np.float32)
        # Same conditioning upstream uses: close small holes, then pull the edge in so the
        # matting model is given an unambiguous interior rather than a noisy boundary.
        if args.dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate, args.dilate))
            m = cv2.dilate((m != 0).astype(np.float32), k, iterations=1) * 255
        if args.erode > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.erode, args.erode))
            m = cv2.erode((m == 255).astype(np.float32), k, iterations=1) * 255
        return torch.from_numpy(m).float().to(device)

    def load_frame(p: Path) -> torch.Tensor:
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1).to(device) / 255.0

    fallback_ops: set[str] = set()
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        model = get_matanyone_model(str(args.checkpoint), device)
        processor = InferenceCore(model, cfg=model.cfg, device=device)
        t_build = time.perf_counter() - t0

        reanchor = set(args.reanchor)
        per_frame: list[float] = []
        t_run0 = time.perf_counter()

        with torch.inference_mode():
            # Settle the first-frame alpha by showing the anchor frame repeatedly. The
            # warmup passes are not written out.
            anchor_img = load_frame(frames[anchor])
            prob = processor.step(anchor_img, load_mask(anchor), objects=[1])
            prob = processor.step(anchor_img, first_frame_pred=True)
            for _ in range(args.warmup):
                prob = processor.step(anchor_img, first_frame_pred=True)

            t_prev = time.perf_counter()
            for i, p in enumerate(frames):
                img = load_frame(p)
                if i == anchor:
                    prob = processor.step(img, first_frame_pred=True)
                elif i in reanchor:
                    prob = processor.step(img, load_mask(i), objects=[1])
                else:
                    prob = processor.step(img)

                alpha = processor.output_prob_to_mask(prob)          # soft, 0..1
                a8 = np.clip(alpha.float().cpu().numpy() * 255.0, 0, 255).round().astype(np.uint8)
                Image.fromarray(a8, mode="L").save(alpha_dir / f"{i:05d}.png")

                now = time.perf_counter()
                per_frame.append(now - t_prev)
                t_prev = now
                if (i + 1) % 24 == 0 or i == len(frames) - 1:
                    print(f"[refine]   {i + 1}/{len(frames)} frames "
                          f"({np.mean(per_frame):.2f} s/frame)", flush=True)
        t_run = time.perf_counter() - t_run0

    for w in caught:
        m = MPS_FALLBACK_RE.search(str(w.message))
        if m:
            fallback_ops.add(m.group(1))

    # Softness is the whole point of this stage: count pixels that are neither 0 nor 255.
    soft_frac, cov = [], []
    for p in sorted(alpha_dir.glob("*.png"), key=lambda q: int(q.stem)):
        a = np.asarray(Image.open(p))
        soft_frac.append(float(((a > 0) & (a < 255)).mean()))
        cov.append(float(a.mean() / 255.0))

    stats = {
        "shot": args.shot,
        "out_name": out_name,
        "model": "MatAnyone v1.0.0 (S-Lab License 1.0, non-commercial)",
        "checkpoint": args.checkpoint.name,
        "device": device,
        "torch": torch.__version__,
        "masks_in": rel(masks_dir),
        "anchor_frame": anchor,
        "reanchor_frames": sorted(reanchor),
        "warmup": args.warmup,
        "dilate": args.dilate,
        "erode": args.erode,
        "frames": len(frames),
        "model_build_s": round(t_build, 2),
        "refine_s": round(t_run, 2),
        "seconds_per_frame": round(t_run / len(frames), 4),
        "fps": round(len(frames) / t_run, 2),
        "mps_fallback_ops": sorted(fallback_ops),
        "soft_pixel_fraction": {
            "min": round(min(soft_frac), 6),
            "max": round(max(soft_frac), 6),
            "mean": round(float(np.mean(soft_frac)), 6),
        },
        "alpha_coverage_mean": round(float(np.mean(cov)), 5),
    }
    (out_root / "refine.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(f"\n[refine] {len(frames)} soft alphas -> {rel(alpha_dir)}")
    print(f"[refine] {t_run:.1f}s = {stats['seconds_per_frame']:.3f} s/frame "
          f"({stats['fps']} fps)")
    print(f"[refine] soft (0<a<255) pixels: mean {100 * stats['soft_pixel_fraction']['mean']:.3f}% "
          f"of frame  [was exactly 0% with the binary matte]")
    print(f"[refine] MPS fallback ops: {sorted(fallback_ops) or 'none'}")
    print(f"[refine] stats -> {rel(out_root / "refine.json")}")


if __name__ == "__main__":
    main()
