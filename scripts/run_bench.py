#!/usr/bin/env python3
"""Score one or more matting methods on the truth clips.

    python scripts/run_bench.py --method baseline_960
    python scripts/run_bench.py --method baseline_960 --method binary_960 --tier a

Writes per-clip JSON under outputs/_bench/<method>/ and prints the tables.
"""
from __future__ import annotations

import os

# SAM 2 and MatAnyone both drive tqdm internally; the bench prints its own progress.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import accuracy, bench, methods                       # noqa: E402
from cleanplate.paths import ROOT, rel                                # noqa: E402

OUT = ROOT / "outputs" / "_bench"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", action="append", required=True)
    ap.add_argument("--tier", choices=["a", "b"], default=None)
    ap.add_argument("--clip", action="append", default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    cs = bench.clips(args.tier)
    if args.clip:
        cs = [c for c in cs if c.name in set(args.clip)]
    if not cs:
        sys.exit("no clips matched")
    print(f"[bench] {len(cs)} clip(s): {', '.join(c.name for c in cs)}\n")

    all_rows = []
    for mname in args.method:
        fn = methods.make(mname)
        rows = []
        for c in cs:
            t0 = time.perf_counter()
            print(f"[bench] {mname} on {c.name} ({c.tier}) ...", flush=True)
            try:
                sc = bench.evaluate(mname, fn, c, save_dir=args.out)
            except Exception as e:
                print(f"[bench]   FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            rows.append(sc)
            wf, hr = sc["whole_frame"], sc.get("hair_region", {})
            print(f"[bench]   MAD {wf['MAD']:.2f}  Grad {wf['Grad']:.2f}  "
                  f"dtSSD {wf['dtSSD']:.2f}  BF {wf['BF']:.3f}  |  "
                  f"hair MAD {hr.get('MAD', float('nan')):.2f}  "
                  f"| {sc['seconds_per_frame']:.3f} s/f  "
                  f"peak {sc['peak_rss_mb']:.0f}MB  ({time.perf_counter()-t0:.0f}s)",
                  flush=True)
        if rows:
            all_rows.append((mname, rows))
            (args.out / f"{mname}_summary.json").write_text(
                json.dumps({"method": mname,
                            "whole_frame": bench.mean_row(rows),
                            "hair_region": bench.mean_row(rows, "hair_region"),
                            "per_clip": rows}, indent=2) + "\n")

    print("\n=== whole frame, mean over clips ===")
    print(accuracy.table([bench.mean_row(r) for _, r in all_rows]))
    print("\n=== hair region only, mean over clips ===")
    print(accuracy.table([bench.mean_row(r, "hair_region") for _, r in all_rows],
                         region="hair_region"))
    print("\n=== cost ===")
    print(bench.cost_table([bench.mean_row(r) for _, r in all_rows]))
    print(f"\n[bench] details -> {rel(args.out)}")


if __name__ == "__main__":
    main()
