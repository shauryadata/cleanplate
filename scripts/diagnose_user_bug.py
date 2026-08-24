#!/usr/bin/env python3
"""Checkpoint 0a: how much of the user's hair problem is recoverable, measured.

Scores three configurations on B1 — the ToS green-screen clip of the SAME ACTOR as the
`hair` shot, the one clip where we hold reference alpha for this hair.

  their_run          one click, MatAnyone v1        (what the sidecar says they ran)
  single_click_hq    one click, MatAnyone 2 + zoom  (toggle only, no better clicking)
  wellformed_hq      clicks on hair/face/torso, MatAnyone 2 + zoom

Separating those two says whether the user should change what they click, flip a
toggle, or both.
"""
from __future__ import annotations

import os
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json, sys, time
from pathlib import Path
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import accuracy, bench, methods, refine, track          # noqa: E402
from cleanplate.memguard import MemoryGuard                             # noqa: E402
from cleanplate.paths import ROOT, peak_rss_mb                          # noqa: E402
from cleanplate.session import Prompt                                   # noqa: E402


def wellformed_prompt(gt: np.ndarray) -> Prompt:
    """One click per anatomical region, derived from the reference so it is repeatable.

    The point of the exercise: the user's single click landed on the plaid shirt, which
    under-specifies a subject whose hair and shadowed beard are the hard parts. This
    puts a click in each.
    """
    m = gt[0] > 127
    ys, xs = np.where(m)
    y0, y1 = int(ys.min()), int(ys.max())
    p = Prompt()
    # three horizontal bands: hair, face, torso. Interior peak of each.
    for lo, hi in ((0.02, 0.16), (0.18, 0.38), (0.55, 0.85)):
        band = np.zeros_like(m)
        a, b = int(y0 + (y1 - y0) * lo), int(y0 + (y1 - y0) * hi)
        band[a:b] = m[a:b]
        if not band.any():
            continue
        d = ndimage.distance_transform_edt(band)
        yy, xx = np.unravel_index(int(np.argmax(d)), d.shape)
        p.add(0, int(xx), int(yy), True)
    return p


def main() -> None:
    clip = next(c for c in bench.clips() if c.name == "B1_tos_greenscreen_hair")
    gt = bench.load_alpha(clip.alpha_dir)
    hb = bench.hair_box(gt)
    single = bench.oracle_prompt(gt, override=clip.oracle_clicks)
    wf = wellformed_prompt(gt)
    print(f"single click : {single.frames[0].positive}")
    print(f"well-formed  : {wf.frames[0].positive}")

    configs = [
        ("their_run (1 click, MatAnyone v1)", single, "matanyone", False),
        ("1 click + HQ (MatAnyone 2 + zoom)", single, "matanyone2", True),
        ("well-formed clicks + HQ", wf, "matanyone2", True),
    ]
    scored, guards = [], {}
    for label, prompt, model, hq in configs:
        # Run at 960, which is what the app and the user's pipeline actually do, then
        # upscale for scoring. Running at B1's native 1920 is the Task 4 configuration
        # that thrashes, and it is not what the user ran.
        t0 = time.perf_counter()
        with MemoryGuard(label) as g:
            alpha, st = methods.sam2_matanyone(clip, prompt, width=960, model=model)
            g.check()
            if hq:
                fdir, _sc = methods._scaled_frames(Path(clip.frames_dir), 960)
                small = np.stack([__import__("cv2").resize(
                    a, (fdir_w := 960, int(round(a.shape[0] * 960 / a.shape[1]))))
                    for a in alpha])
                small, zst = methods.refine_hair_zoom(fdir, small, model=model)
                import cv2
                alpha = np.stack([cv2.resize(a, (alpha.shape[2], alpha.shape[1]),
                                             interpolation=cv2.INTER_LINEAR)
                                  for a in small])
                g.check()
        wall = time.perf_counter() - t0
        sc = accuracy.score(alpha, gt, label, hair_box=hb)
        sc["seconds_per_frame"] = round(wall / len(gt), 4)
        sc["peak_rss_mb"] = round(peak_rss_mb(), 1)
        scored.append(sc)
        guards[label] = g.stats()
        out = ROOT / "outputs" / "_userbug" / label.split(" ")[0]
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "alpha.npy", alpha)
        hr = sc["hair_region"]
        print(f"  {label:38s} hair MAD {hr['MAD']:6.2f}  Grad {hr['Grad']:.3f}  "
              f"BF {hr['BF']:.4f}  | {sc['seconds_per_frame']:.3f} s/f")
        print(f"      {g.report()}")

    md = ["# Checkpoint 0a — the hair symptom, measured on B1", "",
          "B1 is the Tears of Steel green-screen plate of the **same actor** as the",
          "`hair` demo shot, so it is the one clip where we hold reference alpha for",
          "this hair. Tier B: keyed reference, not gospel.", "",
          "## Hair region", "", accuracy.table(scored, region="hair_region"), "",
          "## Whole frame", "", accuracy.table(scored), "",
          "## Cost and memory", "",
          "| Config | s/frame | peak RSS | lowest available | swap |",
          "|---|---|---|---|---|"]
    for sc in scored:
        g = guards[sc["label"]]
        md.append(f"| {sc['label']} | {sc['seconds_per_frame']:.3f} | "
                  f"{sc['peak_rss_mb']:.0f} MB | {g['lowest_available_mb']:.0f} MB | "
                  f"{g['swap_start_mb']:.0f} → {g['swap_peak_mb']:.0f} MB |")
    md += ["", "Peak RSS understates pressure — a process being paged out reports a "
               "small resident size — so the guard's available-memory and swap columns "
               "are the ones to read.", ""]
    (ROOT / "docs" / "USERBUG_HAIR.md").write_text("\n".join(md))
    (ROOT / "outputs" / "_userbug" / "scores.json").write_text(
        json.dumps({"scored": scored, "guards": guards}, indent=2) + "\n")
    print("\n-> docs/USERBUG_HAIR.md")


if __name__ == "__main__":
    main()
