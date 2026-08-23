#!/usr/bin/env python3
"""Re-run a shot through the cleanplate package and compare to a stored run.

The point is to prove the Task 3 refactor did not change the numbers. It rebuilds
the matte from the shot's committed point.json alone and diffs against the masks
and alpha already on disk.

    python scripts/regression_check.py --shot walk --against walk_v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import metrics, refine, track                      # noqa: E402
from cleanplate.paths import out_dir, rel                          # noqa: E402
from cleanplate.session import Prompt                              # noqa: E402


def load_seq(d: Path) -> np.ndarray:
    ps = sorted(d.glob("*.png"), key=lambda p: int(p.stem))
    return np.stack([np.asarray(Image.open(p).convert("L")) for p in ps])


def cmp_block(name: str, new: np.ndarray, old: np.ndarray) -> bool:
    if new.shape != old.shape:
        print(f"  {name}: SHAPE MISMATCH {new.shape} vs {old.shape}")
        return False
    identical = bool((new == old).all())
    agree = float((new == old).mean())
    iou = metrics.per_frame_iou(new, old)
    print(f"  {name}: {'IDENTICAL' if identical else 'differs'}  "
          f"pixel agreement {100 * agree:.4f}%  "
          f"per-frame IoU min {iou.min():.5f} mean {iou.mean():.5f}")
    return identical


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default="walk")
    ap.add_argument("--against", default="walk_v2", help="existing outputs/<name> to diff")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-refine", action="store_true")
    args = ap.parse_args()

    ref = out_dir(args.against)
    prompt = Prompt.load(args.shot)
    print(f"[regression] shot={args.shot}  reference={rel(ref)}")
    print(f"[regression] prompt from shots/{args.shot}/point.json: "
          f"{prompt.total_clicks} clicks on frames {prompt.prompt_frames}")

    ok = True
    masks, tstats = track.track(args.shot, prompt, device=args.device,
                                progress=lambda f, m: None)
    print(f"[regression] track: {tstats['seconds_per_frame']} s/frame on "
          f"{tstats['device']}, fallback ops {tstats['mps_fallback_ops'] or 'none'}")
    old_masks = load_seq(ref / "masks") > 127
    ok &= cmp_block("binary masks", masks, old_masks)

    alphas = None
    if not args.skip_refine and (ref / "alpha").is_dir():
        avail, why = refine.available()
        if not avail:
            print(f"  refine skipped: {why}")
        else:
            alphas, rstats = refine.refine(args.shot, masks, device=args.device,
                                           progress=lambda f, m: None)
            print(f"[regression] refine: {rstats['seconds_per_frame']} s/frame on "
                  f"{rstats['device']}, fallback ops {rstats['mps_fallback_ops'] or 'none'}")
            ok &= cmp_block("soft alpha  ", alphas, load_seq(ref / "alpha"))

    print("\n[regression] metrics, recomputed vs stored:")
    stored_path = ref / "metrics.json"
    stored = json.loads(stored_path.read_text()) if stored_path.exists() else None
    live = metrics.measure(alphas if alphas is not None else masks, "recomputed")

    keys = [("area_change_pct", "mean"), ("area_change_pct", "max"),
            ("iou_consecutive", "mean"), ("iou_consecutive", "min"),
            ("perimeter", "norm_cv_pct"),
            ("softness", "soft_pixel_fraction_mean"),
            ("components", "significant_mean")]
    if stored:
        print(f"  {'metric':<46} {'stored':>12} {'now':>12}   match")
        for a, b in keys:
            s, n = stored[a][b], live[a][b]
            same = abs(float(s) - float(n)) < 1e-9
            ok &= same
            print(f"  {a + '.' + b:<46} {s:>12} {n:>12}   {'yes' if same else 'NO'}")
    else:
        print(f"  no stored metrics.json at {rel(stored_path)}")
        for a, b in keys:
            print(f"  {a + '.' + b:<46} {live[a][b]:>12}")

    print("\n[regression] " + ("PASS - refactor is numerically identical"
                               if ok else "FAIL - see mismatches above"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
