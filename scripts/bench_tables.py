#!/usr/bin/env python3
"""Turn the bench JSONs into the markdown tables that go in docs/METRICS.md.

    python scripts/bench_tables.py --method baseline_960 --method fullres_1920 ...
    python scripts/bench_tables.py --all --out docs/BENCH.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import accuracy, bench                                # noqa: E402
from cleanplate.paths import ROOT                                     # noqa: E402

BENCH = ROOT / "outputs" / "_bench"


def load(method: str) -> dict | None:
    p = BENCH / f"{method}_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def per_clip_table(summaries: list[dict], region: str, metric: str) -> str:
    clips = [c.name for c in bench.clips()]
    out = ["| Clip | tier | " + " | ".join(s["method"] for s in summaries) + " |",
           "|---|---|" + "---|" * len(summaries)]
    tiers = {c.name: c.tier for c in bench.clips()}
    for cn in clips:
        row = [cn, tiers.get(cn, "?")]
        vals = []
        for s in summaries:
            r = next((x for x in s["per_clip"] if x["clip"] == cn), None)
            vals.append(None if r is None else r[region][metric])
        ok = [v for v in vals if v is not None]
        best = min(ok) if metric != "BF" else (max(ok) if ok else None)
        for v in vals:
            if v is None:
                row.append("—")
            elif best is not None and abs(v - best) < 1e-9:
                row.append(f"**{v:.4g}**")
            else:
                row.append(f"{v:.4g}")
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--title", default="RotoBench")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    names = args.method
    if args.all or not names:
        names = sorted(p.stem.replace("_summary", "")
                       for p in BENCH.glob("*_summary.json"))
    summaries = [s for s in (load(n) for n in names) if s]
    if not summaries:
        sys.exit("no summaries found; run scripts/run_bench.py first")

    def rows(region: str) -> list[dict]:
        return [dict(bench.mean_row(s["per_clip"], region), label=s["method"])
                for s in summaries]

    parts = [f"# {args.title}", "",
             "Scored against reference alpha. Tier A clips have alpha that is exact by",
             "construction; tier B is a keyed reference and carries the keyer's own",
             "errors. Metric definitions and their validation are in",
             "`cleanplate/accuracy.py` and `tests/test_accuracy.py`.", "",
             "## Whole frame, mean over all clips", "",
             accuracy.table(rows("whole_frame")), "",
             "## Hair region only, mean over all clips", "",
             "The head bounding box, derived from the reference so it is identical for",
             "every method. This is where the argument is: a whole-frame average is",
             "dominated by the easy interior.", "",
             accuracy.table(rows("hair_region"), region="hair_region"), "",
             "## Cost", "", bench.cost_table(rows("whole_frame")), "",
             "## Per clip — hair-region MAD (lower better)", "",
             per_clip_table(summaries, "hair_region", "MAD"), "",
             "## Per clip — whole-frame MAD (lower better)", "",
             per_clip_table(summaries, "whole_frame", "MAD"), ""]
    md = "\n".join(parts)
    print(md)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
        print(f"\n-> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
