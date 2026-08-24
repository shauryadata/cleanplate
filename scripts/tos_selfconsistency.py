#!/usr/bin/env python3
"""Baseline vs the Task 4 winner on the three graded ToS shots.

There is no reference alpha for these, so this is self-consistency only: the Task 2
metrics, not accuracy. Reported separately from the truth tables for that reason.
"""
from __future__ import annotations

import os
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import methods, metrics, refine, track                 # noqa: E402
from cleanplate.ingest import resolve_frames_dir                       # noqa: E402
from cleanplate.paths import ROOT, peak_rss_mb                         # noqa: E402
from cleanplate.session import Prompt                                  # noqa: E402

SHOTS = ["walk", "dialogue", "hair"]
OUT = ROOT / "outputs" / "_tos_selfcheck"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for shot in SHOTS:
        p = Prompt.load(shot)
        if not p.total_clicks:
            print(f"  {shot}: no point.json, skipped"); continue
        masks, tst = track.track(shot, p, progress=None)
        fd = resolve_frames_dir(shot)
        out = {"binary (SAM 2)": masks}

        t0 = time.perf_counter()
        a_v1, r1 = refine.refine(shot, masks, model="matanyone", progress=None)
        out["baseline (MatAnyone v1)"] = a_v1
        t_v1 = r1["seconds_per_frame"]

        a_v2, r2 = refine.refine(shot, masks, model="matanyone2", progress=None)
        out["new default (MatAnyone 2)"] = a_v2
        t_v2 = r2["seconds_per_frame"]

        a_hq, z = methods.refine_hair_zoom(fd, a_v2, model="matanyone2")
        out["winner (zoom + MatAnyone 2)"] = a_hq
        t_hq = t_v2 + z["hairzoom_s_per_frame"]

        rows[shot] = {k: metrics.measure(v, k) for k, v in out.items()}
        rows[shot]["_cost"] = {"track_s_per_frame": tst["seconds_per_frame"],
                               "baseline_refine_s_per_frame": t_v1,
                               "default_refine_s_per_frame": t_v2,
                               "winner_refine_s_per_frame": round(t_hq, 4),
                               "peak_rss_mb": round(peak_rss_mb(), 1)}
        print(f"  {shot}: done ({tst['seconds_per_frame']} s/f track, "
              f"{t_v2} default, {t_hq:.3f} winner)")

    (OUT / "results.json").write_text(json.dumps(rows, indent=2, default=str) + "\n")
    md = ["# ToS shots — self-consistency (no reference alpha exists for these)", "",
          "The three graded Tears of Steel shots have no ground truth, so these are the",
          "Task 2 stability metrics only. Accuracy numbers live in BENCH.md.", ""]
    for shot, r in rows.items():
        md += [f"## {shot}", ""]
        md.append(metrics.table([r[k] for k in r if not k.startswith("_")]))
        c = r["_cost"]
        md += ["", f"Cost: track {c['track_s_per_frame']} s/frame, refine "
                   f"{c['default_refine_s_per_frame']} (default) / "
                   f"{c['winner_refine_s_per_frame']} (winner) s/frame, "
                   f"peak RSS {c['peak_rss_mb']:.0f} MB.", ""]
    (ROOT / "docs" / "TOS_SELFCHECK.md").write_text("\n".join(md))
    print(f"\n-> docs/TOS_SELFCHECK.md")


if __name__ == "__main__":
    main()
