#!/usr/bin/env python3
"""Propagate a matte through a shot from one click, using the SAM 2 video predictor.

Takes the point you clicked on the prompt frame, runs SAM 2's video predictor over
the whole frame folder, and writes one binary mask PNG per frame.

    python src/track_matte.py --shot walk --point 360,200

Corrective clicks can be added on any later frame, which is how you kill a patch of
background that only latches on part-way through a shot:

    python src/track_matte.py --shot walk --point 360,200 \
        --at 67:321,379:-  --out-name walk_v2

With no --point/--neg/--at, reads shots/<shot>/point.json written by src/pick_point.py.

Outputs (under outputs/<out-name>, defaulting to the shot name)
    masks/00000.png ...   8-bit binary masks, source resolution
    track.json            device, timings, MPS fallback ops, every prompt used
"""
from __future__ import annotations

# Must be set before torch loads its MPS backend: lets unimplemented MPS ops run
# on the CPU instead of hard-failing.
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import contextlib
import json
import re
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
DEFAULT_CKPT = ROOT / "checkpoints" / "sam2.1_hiera_small.pt"

MPS_FALLBACK_RE = re.compile(
    r"operator '([^']+)' is not currently supported on the MPS backend", re.I)


def parse_xy(s: str) -> tuple[int, int]:
    try:
        x, y = s.replace(" ", "").split(",")
        return int(x), int(y)
    except Exception:
        raise argparse.ArgumentTypeError(f"expected X,Y - got {s!r}")


def parse_at(s: str) -> tuple[int, int, int, int]:
    """FRAME:X,Y:SIGN  ->  (frame, x, y, label).  SIGN is + (keep) or - (exclude)."""
    try:
        frame, xy, sign = s.replace(" ", "").split(":")
        x, y = xy.split(",")
        if sign not in ("+", "-"):
            raise ValueError(sign)
        return int(frame), int(x), int(y), 1 if sign == "+" else 0
    except Exception:
        raise argparse.ArgumentTypeError(
            f"expected FRAME:X,Y:+ or FRAME:X,Y:- - got {s!r}")


def pick_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_prompt(args, shot_dir: Path) -> dict[int, dict[str, list]]:
    """-> {frame: {"positive": [(x,y), ...], "negative": [...]}}

    CLI wins if anything was passed; otherwise fall back to point.json.
    """
    by_frame: dict[int, dict[str, list]] = {}

    def add(frame: int, x: int, y: int, label: int) -> None:
        e = by_frame.setdefault(frame, {"positive": [], "negative": []})
        e["positive" if label == 1 else "negative"].append((x, y))

    for x, y in args.point:
        add(args.prompt_frame, x, y, 1)
    for x, y in args.neg:
        add(args.prompt_frame, x, y, 0)
    for frame, x, y, label in args.at:
        add(frame, x, y, label)
    if by_frame:
        return by_frame

    pj = shot_dir / "point.json"
    if pj.exists():
        d = json.loads(pj.read_text())
        if "prompts" in d:                      # current format
            for e in d["prompts"]:
                for x, y in e.get("positive", []):
                    add(int(e["frame"]), int(x), int(y), 1)
                for x, y in e.get("negative", []):
                    add(int(e["frame"]), int(x), int(y), 0)
        else:                                    # Task 1 flat format
            f = int(d.get("prompt_frame", 0))
            for x, y in d.get("positive", []):
                add(f, int(x), int(y), 1)
            for x, y in d.get("negative", []):
                add(f, int(x), int(y), 0)
        if by_frame:
            print(f"[track] prompt from {pj.relative_to(ROOT)}")
            return by_frame

    sys.exit("no prompt point. Pass --point X,Y (and optionally --at FRAME:X,Y:-) "
             f"or run src/pick_point.py --shot {args.shot} first.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--point", type=parse_xy, action="append", default=[],
                    metavar="X,Y", help="positive point (repeatable)")
    ap.add_argument("--neg", type=parse_xy, action="append", default=[],
                    metavar="X,Y", help="negative point (repeatable)")
    ap.add_argument("--at", type=parse_at, action="append", default=[],
                    metavar="FRAME:X,Y:+|-",
                    help="corrective click on a specific frame; + keeps, - excludes "
                         "(repeatable). Use this to remove background that only latches "
                         "on part-way through a shot.")
    ap.add_argument("--prompt-frame", type=int, default=0,
                    help="frame that bare --point/--neg refer to (default 0)")
    ap.add_argument("--out-name", default=None,
                    help="write to outputs/<out-name>/ instead of outputs/<shot>/, so a "
                         "new run does not clobber an old one")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    ap.add_argument("--model-cfg", default=DEFAULT_CFG)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--autocast", action="store_true",
                    help="run under torch.autocast(bfloat16); off by default because "
                         "bf16 autocast is not dependable on MPS")
    ap.add_argument("--offload-video-to-cpu", action="store_true",
                    help="keep decoded frames in system RAM (slower, much less VRAM)")
    ap.add_argument("--offload-state-to-cpu", action="store_true",
                    help="keep the memory bank in system RAM")
    ap.add_argument("--overlay", action="store_true",
                    help="also write a magenta matte overlay per frame for QC")
    args = ap.parse_args()

    shot_dir = ROOT / "shots" / args.shot
    frames_dir = shot_dir / "frames"
    if not frames_dir.is_dir():
        sys.exit(f"no such shot: {frames_dir}\nRun src/extract_shot.py first.")
    if not args.checkpoint.exists():
        sys.exit(f"checkpoint missing: {args.checkpoint}\n"
                 "Run ./scripts/download.sh checkpoints")

    frames = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if not frames:
        sys.exit(f"no frames in {frames_dir}")

    prompts = load_prompt(args, shot_dir)
    n_frames = len(frames)
    for f in prompts:
        if not (0 <= f < n_frames):
            sys.exit(f"prompt frame {f} is outside the shot (0..{n_frames - 1})")

    # A conditioning frame carrying only negative points reads as "the object is not
    # in this frame", and SAM 2 will emit an empty mask there. A corrective click must
    # be paired with a positive point that re-anchors the subject on that same frame.
    lonely = [f for f, e in prompts.items() if e["negative"] and not e["positive"]]
    if lonely:
        sys.exit(
            f"frame(s) {lonely} have only negative points. SAM 2 treats a conditioning "
            "frame with no positive point as 'object absent' and will blank that frame.\n"
            "Add a positive click on the subject in the same frame, e.g.\n"
            f"    --at {lonely[0]}:<X>,<Y>:+  --at {lonely[0]}:<X>,<Y>:-")

    out_dir = ROOT / "outputs" / (args.out_name or args.shot)
    masks_dir = out_dir / "masks"
    if masks_dir.exists():
        shutil.rmtree(masks_dir)
    masks_dir.mkdir(parents=True)
    overlay_dir = out_dir / "overlay"
    if args.overlay:
        if overlay_dir.exists():
            shutil.rmtree(overlay_dir)
        overlay_dir.mkdir(parents=True)

    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor

    device = pick_device(args.device)
    print(f"[track] shot={args.shot}  frames={len(frames)}  device={device}")
    print(f"[track] outputs -> {out_dir.relative_to(ROOT)}")
    for f in sorted(prompts):
        e = prompts[f]
        print(f"[track] frame {f}: keep {e['positive'] or '-'}  exclude {e['negative'] or '-'}")
    print(f"[track] model {args.model_cfg}  ckpt {args.checkpoint.name}")

    fallback_ops: set[str] = set()
    t_build0 = time.perf_counter()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        predictor = build_sam2_video_predictor(
            args.model_cfg, str(args.checkpoint), device=device)
        t_build = time.perf_counter() - t_build0

        t_init0 = time.perf_counter()
        state = predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=args.offload_video_to_cpu,
            offload_state_to_cpu=args.offload_state_to_cpu,
        )
        t_init = time.perf_counter() - t_init0
        print(f"[track] model built in {t_build:.1f}s, "
              f"{len(frames)} frames loaded in {t_init:.1f}s")

        amp = (torch.autocast(device_type=device, dtype=torch.bfloat16)
               if args.autocast else contextlib.nullcontext())

        with torch.inference_mode(), amp:
            # One call per conditioning frame. propagate_in_video then starts from the
            # earliest of them and runs forward over the whole shot.
            for f in sorted(prompts):
                e = prompts[f]
                pts = [*e["positive"], *e["negative"]]
                labs = [1] * len(e["positive"]) + [0] * len(e["negative"])
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=f,
                    obj_id=1,
                    points=np.array(pts, dtype=np.float32),
                    labels=np.array(labs, dtype=np.int32),
                )

            t_prop0 = time.perf_counter()
            written = 0
            per_frame: list[float] = []
            t_prev = t_prop0
            for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(state):
                mask = (video_res_masks[0, 0] > 0.0).cpu().numpy()
                Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
                    masks_dir / f"{frame_idx:05d}.png")
                if args.overlay:
                    base = np.array(Image.open(frames[frame_idx]).convert("RGB"),
                                    dtype=np.float32)
                    tint = np.array([255, 0, 200], dtype=np.float32)
                    m = mask[..., None].astype(np.float32) * 0.45
                    Image.fromarray((base * (1 - m) + tint * m).astype(np.uint8)).save(
                        overlay_dir / f"{frame_idx:05d}.jpg", quality=90)
                now = time.perf_counter()
                per_frame.append(now - t_prev)
                t_prev = now
                written += 1
            t_prop = time.perf_counter() - t_prop0

    for w in caught:
        m = MPS_FALLBACK_RE.search(str(w.message))
        if m:
            fallback_ops.add(m.group(1))

    if written != len(frames):
        print(f"[track] WARNING: {written} masks for {len(frames)} frames "
              "- propagation did not cover the whole shot", file=sys.stderr)

    covered = [int(p.stem) for p in sorted(masks_dir.glob("*.png"))]
    areas = []
    for p in sorted(masks_dir.glob("*.png"), key=lambda q: int(q.stem)):
        a = np.array(Image.open(p))
        areas.append(float((a > 127).mean()))

    stats = {
        "shot": args.shot,
        "device": device,
        "torch": torch.__version__,
        "model_cfg": args.model_cfg,
        "checkpoint": args.checkpoint.name,
        "autocast_bf16": args.autocast,
        "offload_video_to_cpu": args.offload_video_to_cpu,
        "offload_state_to_cpu": args.offload_state_to_cpu,
        "prompts": [
            {"frame": f,
             "positive": [list(p) for p in prompts[f]["positive"]],
             "negative": [list(p) for p in prompts[f]["negative"]]}
            for f in sorted(prompts)
        ],
        "total_clicks": sum(len(e["positive"]) + len(e["negative"])
                            for e in prompts.values()),
        "frame_count": len(frames),
        "masks_written": written,
        "frames_covered": [min(covered), max(covered)] if covered else [],
        "model_build_s": round(t_build, 2),
        "frame_load_s": round(t_init, 2),
        "propagate_s": round(t_prop, 2),
        "seconds_per_frame": round(t_prop / max(written, 1), 4),
        "fps": round(written / t_prop, 2) if t_prop else None,
        "slowest_frame_s": round(max(per_frame), 3) if per_frame else None,
        "mps_fallback_ops": sorted(fallback_ops),
        "mask_area_fraction": {
            "min": round(min(areas), 5) if areas else None,
            "max": round(max(areas), 5) if areas else None,
            "mean": round(float(np.mean(areas)), 5) if areas else None,
        },
        "empty_mask_frames": [i for i, a in zip(covered, areas) if a == 0.0],
    }
    (out_dir / "track.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(f"\n[track] {written} masks -> {masks_dir.relative_to(ROOT)}")
    if args.overlay:
        print(f"[track] overlays   -> {overlay_dir.relative_to(ROOT)}")
    print(f"[track] propagate {t_prop:.1f}s  =  {stats['seconds_per_frame']:.3f} s/frame  "
          f"({stats['fps']} fps)")
    print(f"[track] mask area  min {stats['mask_area_fraction']['min']:.4f}  "
          f"max {stats['mask_area_fraction']['max']:.4f}")
    if stats["empty_mask_frames"]:
        print(f"[track] WARNING: {len(stats['empty_mask_frames'])} EMPTY masks: "
              f"{stats['empty_mask_frames'][:12]}", file=sys.stderr)
    print(f"[track] MPS fallback ops: {sorted(fallback_ops) or 'none'}")
    print(f"[track] stats -> {(out_dir / 'track.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
