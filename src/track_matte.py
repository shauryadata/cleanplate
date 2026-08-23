#!/usr/bin/env python3
"""Propagate a matte through a shot from clicks, using the SAM 2 video predictor.

Thin CLI over cleanplate.track - the app calls the same function.

    python src/track_matte.py --shot walk --point 360,200

Corrective clicks land on any frame, which is how you kill background that only
latches on part-way through a shot:

    python src/track_matte.py --shot walk --point 360,200 \
        --at 67:352,250:+ --at 67:321,379:-  --out-name walk_v2

With no clicks on the command line, reads shots/<shot>/point.json.
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from cleanplate import runs, track
from cleanplate.ingest import n_frames
from cleanplate.paths import out_dir, rel
from cleanplate.session import Prompt


def parse_xy(s: str) -> tuple[int, int]:
    try:
        x, y = s.replace(" ", "").split(",")
        return int(x), int(y)
    except Exception:
        raise argparse.ArgumentTypeError(f"expected X,Y - got {s!r}")


def parse_at(s: str) -> tuple[int, int, int, int]:
    """FRAME:X,Y:SIGN  ->  (frame, x, y, label). SIGN is + (keep) or - (exclude)."""
    try:
        frame, xy, sign = s.replace(" ", "").split(":")
        x, y = xy.split(",")
        if sign not in ("+", "-"):
            raise ValueError(sign)
        return int(frame), int(x), int(y), 1 if sign == "+" else 0
    except Exception:
        raise argparse.ArgumentTypeError(f"expected FRAME:X,Y:+ or FRAME:X,Y:- - got {s!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--point", type=parse_xy, action="append", default=[], metavar="X,Y")
    ap.add_argument("--neg", type=parse_xy, action="append", default=[], metavar="X,Y")
    ap.add_argument("--at", type=parse_at, action="append", default=[],
                    metavar="FRAME:X,Y:+|-", help="corrective click on a specific frame")
    ap.add_argument("--prompt-frame", type=int, default=0,
                    help="frame that bare --point/--neg refer to (default 0)")
    ap.add_argument("--out-name", default=None,
                    help="write to outputs/<out-name>/ instead of outputs/<shot>/")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    ap.add_argument("--offload-video-to-cpu", action="store_true")
    ap.add_argument("--offload-state-to-cpu", action="store_true")
    ap.add_argument("--overlay", action="store_true",
                    help="also write a magenta matte overlay per frame for QC")
    args = ap.parse_args()

    prompt = Prompt()
    for x, y in args.point:
        prompt.add(args.prompt_frame, x, y, True)
    for x, y in args.neg:
        prompt.add(args.prompt_frame, x, y, False)
    for f, x, y, lab in args.at:
        prompt.add(f, x, y, lab == 1)
    if prompt.total_clicks == 0:
        prompt = Prompt.load(args.shot)
        if prompt.total_clicks:
            print(f"[track] prompt from shots/{args.shot}/point.json")

    name = args.out_name or args.shot
    nf = n_frames(args.shot)
    print(f"[track] shot={args.shot}  frames={nf}")
    print(f"[track] outputs -> {rel(out_dir(name))}")
    for f in prompt.prompt_frames:
        e = prompt.frames[f]
        print(f"[track] frame {f}: keep {e.positive or '-'}  exclude {e.negative or '-'}")

    last = [""]

    def show(frac: float, msg: str) -> None:
        if msg != last[0]:
            print(f"\r[track] {msg}".ljust(70), end="", flush=True)
            last[0] = msg

    try:
        masks, stats = track.track(args.shot, prompt, device=args.device, progress=show,
                                   offload_video_to_cpu=args.offload_video_to_cpu,
                                   offload_state_to_cpu=args.offload_state_to_cpu)
    except (track.PromptError, FileNotFoundError, RuntimeError) as e:
        print()
        sys.exit(str(e))
    print()

    runs.save_masks(name, masks, stats)
    if args.overlay:
        import numpy as np
        from cleanplate.compose import write_sequence
        from cleanplate.ingest import load_frame
        tint = np.array([255, 0, 200], dtype=np.float32)
        frames = (load_frame(args.shot, i).astype(np.float32) for i in range(len(masks)))
        blend = [(f * (1 - m[..., None] * 0.45) + tint * (m[..., None] * 0.45)
                  ).astype(np.uint8) for f, m in zip(frames, masks.astype(np.float32))]
        write_sequence(blend, out_dir(name) / "overlay", mode="RGB")
        print(f"[track] overlays   -> {rel(out_dir(name) / 'overlay')}")

    print(f"[track] {len(masks)} masks -> {rel(out_dir(name) / 'masks')}")
    print(f"[track] propagate {stats['propagate_s']}s = "
          f"{stats['seconds_per_frame']:.3f} s/frame ({stats['fps']} fps)")
    print(f"[track] mask area  min {stats['mask_area_fraction']['min']:.4f}  "
          f"max {stats['mask_area_fraction']['max']:.4f}")
    if stats["empty_mask_frames"]:
        print(f"[track] WARNING: empty masks on {stats['empty_mask_frames'][:12]}",
              file=sys.stderr)
    print(f"[track] MPS fallback ops: {stats['mps_fallback_ops'] or 'none'}")
    print(f"[track] stats -> {rel(out_dir(name) / 'track.json')}")


if __name__ == "__main__":
    main()
