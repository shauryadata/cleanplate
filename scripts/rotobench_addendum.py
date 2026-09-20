#!/usr/bin/env python3
"""POST-HOC addendum, clearly labelled as such: how much of P02's failure is the prompt?

    python scripts/rotobench_addendum.py

P02's registered oracle prompt put its "body" click on the actor's hand, which was
resting on his chest at frame 0. With both clicks on skin, SAM 2 selected skin and
hair and dropped the jacket for all 96 frames, so every method failed together. The
registered result stays in the standings - the prompts were frozen before the run and
that is the point. This addendum answers a different question, after the fact: with
ONE more click where the first prompt visibly failed, how much comes back?

The extra click is chosen by rule, from the reference and the registered result at
frame 0 only: the most interior point of the largest region of the key that the
registered mask missed. It is what a person would do next.

Written to outputs/_bench_p_addendum; never mixed into the registered standings.
"""
from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import bench, methods                                 # noqa: E402
from cleanplate.memguard import MemoryGuard                           # noqa: E402
from cleanplate.paths import ROOT, rel                                # noqa: E402

CLIP = "P02_04_1b"
OUT = ROOT / "outputs" / "_bench_p_addendum"
REG = ROOT / "outputs" / "_bench_p"
MS = ["binary_960", "matanyone2_960", "hairzoom2_960"]


def extra_click() -> list[int]:
    from scipy import ndimage
    gt = np.asarray(Image.open(ROOT / "truth" / CLIP / "alpha" / "00000.png")) > 127
    reg = np.asarray(Image.open(REG / "binary_960" / CLIP / "00000.png")) > 127
    miss = gt & ~reg
    lab, n = ndimage.label(miss)
    if not n:
        raise SystemExit("the registered mask missed nothing at frame 0")
    sizes = ndimage.sum(miss, lab, range(1, n + 1))
    big = lab == (1 + int(np.argmax(sizes)))
    d = ndimage.distance_transform_edt(np.pad(big, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return [int(x), int(y)]


def main() -> None:
    c = next(c for c in bench.clips("P") if c.name == CLIP)
    xy = extra_click()
    objs = [dict(c.oracle_objects[0], clicks=c.oracle_objects[0]["clicks"] + [xy])]
    amended = dataclasses.replace(c, oracle_objects=objs)
    print(f"[addendum] {CLIP}: registered clicks {c.oracle_objects[0]['clicks']}, "
          f"plus {xy} (largest missed region at frame 0)")
    rows = []
    for m in MS:
        with MemoryGuard(f"addendum {m}") as g:
            sc = bench.evaluate(m, methods.make(m), amended, save_dir=OUT)
        reg = json.loads((REG / m / f"{CLIP}.json").read_text())
        rows.append((m, reg, sc))
        print(f"   {m}: whole MAD {reg['whole_frame']['MAD']:.1f} -> "
              f"{sc['whole_frame']['MAD']:.1f}, band MAD {reg['band']['MAD']:.1f} -> "
              f"{sc['band']['MAD']:.1f}, dropout frames "
              f"{reg['coverage']['dropout_frames']} -> {sc['coverage']['dropout_frames']}"
              f"  ({g.stats()['lowest_available_mb']:.0f} MB lowest)", flush=True)
    (OUT / "summary.json").write_text(json.dumps(
        {"clip": CLIP, "post_hoc": True, "extra_click": xy,
         "rule": "most interior point of the largest key region the registered mask "
                 "missed at frame 0",
         "rows": [{"method": m, "registered": {k: r["whole_frame"][k] for k in ("MAD", "BF")}
                   | {"band_MAD": r["band"]["MAD"],
                      "dropout_frames": r["coverage"]["dropout_frames"]},
                   "amended": {k: a["whole_frame"][k] for k in ("MAD", "BF")}
                   | {"band_MAD": a["band"]["MAD"],
                      "dropout_frames": a["coverage"]["dropout_frames"]}}
                  for m, r, a in rows]}, indent=2) + "\n")
    print(f"[addendum] -> {rel(OUT)}")


if __name__ == "__main__":
    main()
