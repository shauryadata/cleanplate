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



def demo_section() -> list:
    """The three real removals, read back from what the demo run actually wrote."""
    dd = ROOT / "outputs" / "_demos"
    stats = []
    for j in sorted(dd.glob("*/stats.json")):
        st = json.loads(j.read_text())
        st["dir"] = j.parent.name
        stats.append(st)
    if not stats:
        return []
    md = ["## Three real removals", "",
          "The scored clips above are synthetic, because that is the only way to have a "
          "correct answer to compare against. These three are real shots with no ground "
          "truth, which is what the tool will actually meet.", "",
          "| Demo | Frames | Hole | s/frame | swap during stage | lowest available |",
          "|---|---|---|---|---|---|"]
    for st in stats:
        m = st["memory"]
        md.append("| " + st["dir"] + " — " + st.get("demo", "") + " | "
                  + str(st["frames"]) + " | "
                  + "%.2f%%" % (100 * st["hole_area_fraction_mean"]) + " | "
                  + "%.2f" % st["seconds_per_frame"] + " | "
                  + "%.0f → %.0f MB" % (m["swap_start_mb"], m["swap_peak_mb"]) + " | "
                  + "%.0f MB" % m["lowest_available_mb"] + " |")
    md += ["",
           "**A, the lamppost.** The walk shot is a tracking shot, so a piece of street "
           "furniture at frame left is gone within a second: the lamppost is in shot for "
           "24 of the 96 frames and the demo covers those. That camera move is the "
           "reason this is the best-looking of the three. Panning past an object is "
           "parallax, and parallax genuinely reveals what is behind it, so the fill has "
           "evidence instead of having to invent. The building, the doorway, the kerb "
           "and the railing all continue correctly; the railing is slightly soft where "
           "it crosses the hole.", "",
           "**B, the tracking markers.** The mask here comes from a detector, not a "
           "click: markers are the small non-green blobs on the backing once the "
           "subject is excluded using the Tier B keyed alpha. The largest one is so far "
           "out of focus that green bleeds through it and it still passes a greenness "
           "test, so it is caught on saturation instead — measured at 0.287 below the "
           "local backing against 0.182 for clean backing at p99.9, which is what sets "
           "the 0.20 threshold. Fifteen blobs per frame, 0.75% of the frame, all of "
           "them gone, the actor untouched. Recall is not perfect: one small mark at "
           "frame right survives.", "",
           "**C, one actor of two.** Selective removal — the man is taken out and the "
           "woman is left, which is the case that matters for dialogue coverage. SAM 2 "
           "kept them separate throughout despite them touching by the end of the shot. "
           "The bridge railing and the arch reconstruct well. The weakness is the "
           "vertical smudge where his head was, and it has a specific cause: he barely "
           "moves relative to the camera for 96 frames, so nothing in the clip ever "
           "reveals the building behind him. Compare demo A, where the pan does.", ""]
    return md


def failure_catalogue() -> list:
    return ["## Where it breaks", "",
            "**Moving stochastic texture.** The worst case measured. Water, foliage and "
            "crowds come back too calm: the fill carries about a third of the true "
            "motion, so it reads as a still patch sliding over a moving plate. Worse "
            "the longer the shot stays on it.", "",
            "**Large occlusions.** Doubling the hole from ~10% to ~20% of frame costs "
            "real accuracy even on an otherwise easy background, because there is less "
            "surrounding evidence per hole pixel. Expect degradation, not failure.", "",
            "**No parallax.** An object that holds still relative to the camera never "
            "reveals what is behind it, and no amount of temporal context helps: the "
            "inpainter is inventing, not recovering. This is the single best predictor "
            "of a bad fill, and it is visible in demo C. A camera move is the friend of "
            "removal, which is the opposite of the intuition.", "",
            "**Long holes.** The two failures above compound with shot length. Nothing "
            "here is scored past 96 frames, and the pipeline has not been tested on a "
            "shot where the object is occluded for hundreds of frames.", "",
            "**The memory ceiling is real and it is the binding constraint.** On this "
            "18 GB machine, 96 frames of 960x506 aborted the guard twice — once at "
            "465 MB reclaimable, once on 6,034 MB of swap growth — while 96 frames of "
            "960x400 ran fine. Demo B only completed at 48 frames, and then with "
            "502 MB of headroom against a 500 MB limit. It also cost 6.30 s/frame "
            "against 2.17 for demo A, because the smaller chunk that makes it fit also "
            "makes it slow. Removal is roughly 10x the cost of matting per frame and it "
            "is the stage that will stop a bigger job.", "",
            "The guard thresholds were not relaxed to make any of this pass.", ""]


def write_doc(scored: list) -> None:
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
    # The two moving clips land on the same ratio; that is the finding, so it is
    # derived from the scores here rather than typed in by hand.
    sta = [sc for sc in scored if sc["warp_err_gt"] <= 0.5]
    mov = [sc for sc in scored if sc["warp_err_gt"] > 0.5]
    worst = min(scored, key=lambda sc: sc["PSNR_hole"])
    md += ["## What the numbers say", ""]
    if sta:
        md += ["**Static backgrounds** (" + ", ".join(sc["label"] for sc in sta) +
               ") have a warp floor of exactly zero: the true background never moves, so "
               "every bit of motion in the fill is spurious. Measured: " +
               ", ".join("%.3f" % sc["warp_err_hole"] for sc in sta) + " (x1e3). That is "
               "low-amplitude flicker rather than drift - stable enough to cut with, but "
               "not frozen the way the plate under it is.", ""]
    if mov:
        rat = [100.0 * sc["warp_err_hole"] / sc["warp_err_gt"] for sc in mov]
        md += ["**Moving backgrounds** (" + ", ".join(sc["label"] for sc in mov) +
               ") fail the opposite way, and they fail by almost the same amount: the fill "
               "reproduces " + ", ".join("%.0f%%" % r for r in rat) + " of the true "
               "background motion. Under-movement, not over-movement. ProPainter's "
               "transformer regresses toward a temporally smooth answer, so moving water "
               "and passing architecture come back too calm.", ""]
        md += ["This is the trap in reading the temporal metric alone. The worst clip by "
               "PSNR is " + worst["label"] + " at %.1f dB" % worst["PSNR_hole"] +
               ", and its warp error still sits *below* the ground-truth floor "
               "(%.3f against %.3f)." % (worst["warp_err_hole"], worst["warp_err_gt"]) +
               " The fill is not noisy, it is too placid, and a stability score on its own "
               "would have called that a success. A fill that scores well on temporal "
               "stability and badly on PSNR is smoothing, not tracking; neither metric "
               "catches that by itself.", ""]

    md += demo_section()
    md += failure_catalogue()
    (ROOT / "docs" / "REMOVAL_BENCH.md").write_text("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", default=None)
    ap.add_argument("--dilate", type=int, default=12)
    ap.add_argument("--from-cache", action="store_true",
                    help="rebuild the doc from outputs/_removal/scores.json; the "
                         "inpainting is minutes per clip and the prose is not")
    args = ap.parse_args()

    if args.from_cache:
        cached = json.loads((ROOT / "outputs" / "_removal" / "scores.json").read_text())
        write_doc(list(cached.values()))
        print("-> docs/REMOVAL_BENCH.md (from cached scores)")
        return

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

    write_doc(scored)
    (ROOT / "outputs" / "_removal" / "scores.json").write_text(
        json.dumps(out_json, indent=2) + "\n")
    print("\n-> docs/REMOVAL_BENCH.md")


if __name__ == "__main__":
    main()
