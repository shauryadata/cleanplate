#!/usr/bin/env python3
"""RotoBench on Tier P: every method on every professional-truth clip.

    python scripts/run_rotobench.py                      # all methods, all Tier P clips
    python scripts/run_rotobench.py --method hairzoom2_960 --clip P01_08_3a

Each clip x method runs in its own worker process under the memory guard. That is
the only way the guard can stop an in-process model that is thrashing (it kills the
worker, as it kills ProPainter), it returns every byte of MPS memory between runs,
and one crash cannot take the rest of the night with it. Results already on disk are
skipped, so an interrupted run resumes. Two memory aborts stop the run: same error
twice, stop and report.

Where a clip also carries the in-house key (P01), the same prediction is scored a
second time against it, so the two references can be compared on identical output.
"""
from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import accuracy, bench, methods                       # noqa: E402
from cleanplate.memguard import MemoryGuard                           # noqa: E402
from cleanplate.paths import ROOT, rel                                # noqa: E402

OUT = ROOT / "outputs" / "_bench_p"
# Resource cap, added after the run stopped on two memory aborts, both the zoom pass on
# P03. The zoom methods re-matte the head box at native resolution, so their cost scales
# with that box: 0.25 and 0.28 MP completed (P01, P02), 0.48 MP did not (P03), on a
# machine that was already 13-16 GB into swap. Above this cap the job is recorded as
# "does not fit", not scored, and the zoom claims are judged on clips where every method
# ran. The methods themselves are untouched, so Task 4's claims are tested on Task 4's
# method.
ZOOM_METHODS = {"hairzoom_960", "hairzoom2_960", "cover_960", "cover2_960"}
ZOOM_BOX_CAP_MP = 0.30
METHODS = ["binary_960", "baseline_960", "matanyone2_960", "guided_960",
           "vitmatte_960", "hairzoom_960", "hairzoom2_960"]
# The coverage repair builds on hairzoom2, so it inherits the same size cap.
COVER = ["cover_960", "cover2_960"]


def zoom_box_mp(c) -> float:
    """Megapixels of the region the zoom pass would re-matte, from the reference."""
    gt = bench.load_alpha(c.alpha_dir)
    l, t, r, b = bench.hair_box(gt, pad=48)
    return (r - l) * (b - t) / 1e6


def worker(clip_name: str, mname: str) -> None:
    c = next(c for c in bench.clips("P") if c.name == clip_name)
    keep: list = []
    sc = bench.evaluate(mname, methods.make(mname), c, save_dir=OUT, keep=keep)
    inh = Path(c.alpha_dir).parent / "alpha_inhouse"
    if inh.is_dir():
        gt_in = bench.load_alpha(inh)
        gt = bench.load_alpha(c.alpha_dir)
        s2 = accuracy.score(keep[0], gt_in, mname, hair_box=bench.hair_box(gt))
        s2.update({"method": mname, "clip": c.name, "truth": "in-house key (Tier B settings)"})
        (OUT / mname / f"{c.name}__inhouse.json").write_text(json.dumps(s2, indent=2) + "\n")
    wf, bd, cv = sc["whole_frame"], sc["band"], sc["coverage"]
    print(f"   MAD {wf['MAD']:.2f}  band MAD {bd['MAD']:.1f}  dtSSD-n {wf['dtSSDn']}  "
          f"BF {wf['BF']:.3f}  dropout frames {cv['dropout_frames']}  "
          f"| {sc['seconds_per_frame']:.3f} s/f", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append")
    ap.add_argument("--clip", action="append")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(args.clip[0], args.method[0]); return

    clips = [c for c in bench.clips("P") if not args.clip or c.name in set(args.clip)]
    ms = args.method or METHODS
    jobs = []
    for c in clips:
        mp = zoom_box_mp(c)
        for m in ms:
            if (OUT / m / f"{c.name}.json").exists():
                continue
            if m in ZOOM_METHODS and mp > ZOOM_BOX_CAP_MP:
                (OUT / m).mkdir(parents=True, exist_ok=True)
                (OUT / m / f"{c.name}_TOOBIG.json").write_text(json.dumps(
                    {"method": m, "clip": c.name, "NOT_RUN": "zoom region "
                     f"{mp:.2f} MP exceeds the {ZOOM_BOX_CAP_MP} MP cap for this machine",
                     "zoom_box_mp": round(mp, 3)}, indent=2) + "\n")
                print(f"[rotobench] skip {m} on {c.name}: zoom region {mp:.2f} MP "
                      f"> {ZOOM_BOX_CAP_MP} MP cap (recorded, not scored)", flush=True)
                continue
            jobs.append((c.name, m))
    print(f"[rotobench] {len(clips)} clips x {len(ms)} methods; {len(jobs)} to run "
          f"({len(clips) * len(ms) - len(jobs)} already on disk)", flush=True)
    aborts, t_all = [], time.perf_counter()
    for k, (cn, m) in enumerate(jobs, 1):
        print(f"[rotobench] {k}/{len(jobs)} {m} on {cn}", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.Popen([sys.executable, __file__, "--worker", "--clip", cn,
                                 "--method", m], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        killed = None
        with MemoryGuard(f"{m} on {cn}") as g:
            while proc.poll() is None:
                time.sleep(2.0)
                if g.aborted:
                    killed = g.aborted
                    proc.send_signal(signal.SIGKILL)
                    break
        out = proc.stdout.read() if proc.stdout else ""
        lines = [l for l in out.splitlines() if l.strip()]
        if killed:
            aborts.append((cn, m, killed))
            rec = {"method": m, "clip": cn, "ABORTED": killed, "memory": g.stats()}
            (OUT / m).mkdir(parents=True, exist_ok=True)
            (OUT / m / f"{cn}_ABORTED.json").write_text(json.dumps(rec, indent=2) + "\n")
            print(f"   ABORTED by the memory guard: {killed}. {g.report()}", flush=True)
            if len(aborts) >= 2:
                print("[rotobench] second memory abort - stopping, as the brief requires. "
                      "Aborts: " + "; ".join(f"{a} {b}: {c}" for a, b, c in aborts))
                sys.exit(2)
            continue
        if proc.returncode != 0:
            print("   FAILED (exit %d):\n      " % proc.returncode
                  + "\n      ".join(lines[-8:]), flush=True)
            continue
        print("\n".join(l for l in lines if l.startswith("   ")) +
              f"   ({time.perf_counter() - t0:.0f}s; lowest available "
              f"{g.stats()['lowest_available_mb']:.0f} MB)", flush=True)
    print(f"[rotobench] done in {(time.perf_counter() - t_all) / 60:.0f} min -> {rel(OUT)}")


if __name__ == "__main__":
    main()
