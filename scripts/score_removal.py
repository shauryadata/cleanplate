#!/usr/bin/env python3
"""Score removal on the Tier R synthetic clips, where the clean background is the answer."""
from __future__ import annotations

import os
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import accuracy, remove                                # noqa: E402
from cleanplate.paths import ROOT, rel                                 # noqa: E402

TR = ROOT / "truth_removal"


def load(d: Path, ext: str) -> np.ndarray:
    ps = sorted(d.glob(f"*.{ext}"), key=lambda p: int(p.stem))
    mode = "L" if ext == "png" else "RGB"
    return np.stack([np.asarray(Image.open(p).convert(mode)) for p in ps])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", default=None)
    ap.add_argument("--dilate", type=int, default=12)
    args = ap.parse_args()

    clips = sorted(d for d in TR.iterdir() if (d / "recipe.json").exists())
    if args.clip:
        clips = [c for c in clips if c.name in set(args.clip)]
    scored, out_json = [], {}
    for c in clips:
        rec = json.loads((c / "recipe.json").read_text())
        alpha = load(c / "alpha", "png")
        clean = load(c / "clean", "jpg")
        print(f"[remove] {c.name}: {len(alpha)} frames, intruder "
              f"{100*rec['alpha_area_fraction']:.1f}% of frame", flush=True)
        t0 = time.perf_counter()
        try:
            r = remove.remove(c / "frames", alpha, dilate=args.dilate, progress=None,
                              guard_label=f"remove {c.name}")
        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        sc = accuracy.score_removal(r.frames, clean, r.holes, c.name)
        sc["seconds_per_frame"] = r.stats["seconds_per_frame"]
        sc["memory"] = r.stats["memory"]
        sc["note"] = rec.get("note", "")
        scored.append(sc)
        out_json[c.name] = sc
        m = r.stats["memory"]
        print(f"   PSNR hole {sc['PSNR_hole']:.2f} dB  SSIM {sc['SSIM_hole']:.4f}  "
              f"warp {sc['warp_err_hole']:.3f} (truth floor {sc['warp_err_gt']:.3f})  "
              f"| {sc['seconds_per_frame']:.2f} s/f  "
              f"swap {m['swap_start_mb']:.0f}->{m['swap_peak_mb']:.0f} MB  "
              f"avail min {m['lowest_available_mb']:.0f} MB  "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
        d = ROOT / "outputs" / "_removal" / c.name
        d.mkdir(parents=True, exist_ok=True)
        for i in (0, len(r.frames)//2, len(r.frames)-1):
            Image.fromarray(r.frames[i]).save(d / f"{i:05d}.png")
        np.save(d / "cleaned.npy", r.frames)

    md = ["# Removal — scored on Tier R synthetic truth", "",
          "Each clip is a clean background with a tracked foreground composited in, so",
          "the background is the correct answer **by construction**. PSNR and SSIM are",
          "measured inside the removal hole only: the rest of the frame is untouched and",
          "would score infinity, which tells you nothing.", "",
          "`warp_err_gt` is the same temporal metric computed on the true background —",
          "the floor. A fill cannot be more temporally stable than the footage it",
          "replaces, so read the two together.", "",
          accuracy.removal_table(scored), "", "## Cost and memory", "",
          "| Clip | s/frame | swap during stage | lowest available |", "|---|---|---|---|"]
    for sc in scored:
        m = sc["memory"]
        md.append(f"| {sc['label']} | {sc['seconds_per_frame']:.2f} | "
                  f"{m['swap_start_mb']:.0f} → {m['swap_peak_mb']:.0f} MB | "
                  f"{m['lowest_available_mb']:.0f} MB |")
    md += ["", "Settings: ProPainter, fp16, chunks of 8 frames, hole dilated 12 px. "
               "Those defaults were measured, not guessed — ProPainter's own defaults "
               "(fp32, chunks of 80) grew swap by 6 GB on 24 frames and the guard "
               "killed the run.", ""]
    (ROOT / "docs" / "REMOVAL_BENCH.md").write_text("\n".join(md))
    (ROOT / "outputs" / "_removal" / "scores.json").write_text(
        json.dumps(out_json, indent=2) + "\n")
    print("\n-> docs/REMOVAL_BENCH.md")


if __name__ == "__main__":
    main()
