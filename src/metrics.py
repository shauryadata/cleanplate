#!/usr/bin/env python3
"""Measure matte quality, and compare runs.

Every number here is computed from the alpha sequence on disk. Nothing is estimated.

    # one run
    python src/metrics.py --run "walk v1 (binary)=outputs/walk/masks"

    # a comparison table
    python src/metrics.py \
        --run "walk v1 (binary)=outputs/walk/masks" \
        --run "walk v2 (refined)=outputs/walk_v2/alpha" \
        --table docs/METRICS.md

What is measured
    area change      frame-to-frame change in matte area, % of the previous frame.
                     Flicker. Subject motion contributes, so compare like with like.
    IoU              intersection-over-union of consecutive binarised mattes.
    perimeter CV     coefficient of variation of the boundary length. Boil: the edge
                     crawling frame to frame without the silhouette changing.
    softness         share of pixels strictly between 0 and 255. A binary matte scores
                     exactly 0; a real key has a band of fractional coverage at every edge.
    components       connected components in the binarised matte. Fragmentation, and
                     stray blobs detached from the subject.

Shape metrics binarise at alpha > 127 so a soft matte and a binary one are compared on
the same footing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
THRESH = 127


def load_alphas(d: Path) -> np.ndarray:
    if not d.is_dir():
        sys.exit(f"no such folder: {d}")
    ps = sorted((q for q in d.iterdir() if q.suffix.lower() == ".png"),
                key=lambda q: int(q.stem))
    if not ps:
        sys.exit(f"no PNGs in {d}")
    return np.stack([np.asarray(Image.open(p).convert("L")) for p in ps])


def perimeter(m: np.ndarray) -> int:
    """Boundary pixels: foreground pixels with a 4-neighbour outside the matte."""
    p = np.zeros_like(m)
    p[1:, :] |= m[1:, :] ^ m[:-1, :]
    p[:-1, :] |= m[1:, :] ^ m[:-1, :]
    p[:, 1:] |= m[:, 1:] ^ m[:, :-1]
    p[:, :-1] |= m[:, 1:] ^ m[:, :-1]
    return int((p & m).sum())


def measure(alphas: np.ndarray, label: str, source: Path) -> dict:
    n, h, w = alphas.shape
    hard = alphas > THRESH

    area = hard.reshape(n, -1).sum(1).astype(float)
    if (area == 0).any():
        empty = [int(i) for i in np.where(area == 0)[0]]
    else:
        empty = []
    safe = np.where(area[:-1] == 0, np.nan, area[:-1])
    d_pct = 100 * np.abs(np.diff(area)) / safe

    inter = (hard[:-1] & hard[1:]).reshape(n - 1, -1).sum(1).astype(float)
    union = (hard[:-1] | hard[1:]).reshape(n - 1, -1).sum(1).astype(float)
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    per = np.array([perimeter(m) for m in hard], dtype=float)
    # Normalise perimeter by sqrt(area) so a subject growing in frame does not read as boil.
    per_norm = per / np.sqrt(np.where(area == 0, np.nan, area))

    soft = ((alphas > 0) & (alphas < 255)).reshape(n, -1).mean(1)

    from scipy import ndimage
    comps = np.array([ndimage.label(m)[1] for m in hard], dtype=float)

    # Temporal L1 on the raw alpha: how much the key itself changes per frame.
    l1 = np.abs(alphas[1:].astype(float) - alphas[:-1].astype(float)).reshape(n - 1, -1).mean(1)

    return {
        "label": label,
        "source": str(source),
        "frames": int(n),
        "resolution": [int(w), int(h)],
        "binarise_threshold": THRESH,
        "empty_frames": empty,
        "area_change_pct": {"mean": round(float(np.nanmean(d_pct)), 3),
                            "max": round(float(np.nanmax(d_pct)), 3),
                            "argmax_frame": int(np.nanargmax(d_pct)) + 1},
        "iou_consecutive": {"mean": round(float(iou.mean()), 4),
                            "min": round(float(iou.min()), 4),
                            "argmin_frame": int(iou.argmin()) + 1},
        "perimeter": {"mean": round(float(per.mean()), 1),
                      "cv_pct": round(float(100 * per.std() / per.mean()), 3),
                      "norm_mean": round(float(np.nanmean(per_norm)), 4),
                      "norm_cv_pct": round(float(100 * np.nanstd(per_norm)
                                                 / np.nanmean(per_norm)), 3)},
        "softness": {"soft_pixel_fraction_mean": round(float(soft.mean()), 6),
                     "soft_pixel_fraction_max": round(float(soft.max()), 6),
                     "distinct_alpha_values": int(len(np.unique(alphas)))},
        "components": {"mean": round(float(comps.mean()), 3),
                       "max": int(comps.max())},
        "alpha_l1_change_mean": round(float(l1.mean()), 4),
        "area_fraction": {"mean": round(float((area / (h * w)).mean()), 5),
                          "min": round(float((area / (h * w)).min()), 5),
                          "max": round(float((area / (h * w)).max()), 5)},
    }


ROWS = [
    ("Flicker - mean frame-to-frame area change", "%",
     lambda m: m["area_change_pct"]["mean"], "lower"),
    ("Flicker - max frame-to-frame area change", "%",
     lambda m: m["area_change_pct"]["max"], "lower"),
    ("Consecutive-frame IoU - mean", "",
     lambda m: m["iou_consecutive"]["mean"], "higher"),
    ("Consecutive-frame IoU - min", "",
     lambda m: m["iou_consecutive"]["min"], "higher"),
    ("Boil - perimeter CV", "%", lambda m: m["perimeter"]["cv_pct"], "lower"),
    ("Boil - shape-normalised perimeter CV", "%",
     lambda m: m["perimeter"]["norm_cv_pct"], "lower"),
    ("Softness - pixels with 0 < alpha < 255", "% of frame",
     lambda m: 100 * m["softness"]["soft_pixel_fraction_mean"], "higher"),
    ("Softness - distinct alpha values", "",
     lambda m: m["softness"]["distinct_alpha_values"], "higher"),
    ("Fragmentation - mean connected components", "",
     lambda m: m["components"]["mean"], "lower"),
    ("Fragmentation - max connected components", "",
     lambda m: m["components"]["max"], "lower"),
]


def fmt(v: float) -> str:
    if isinstance(v, (int, np.integer)) or float(v).is_integer():
        return str(int(v))
    return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"


def table(measured: list[dict]) -> str:
    heads = [m["label"] for m in measured]
    out = ["| Metric | " + " | ".join(heads) + (" | Change |" if len(measured) == 2 else " |"),
           "|---|" + "---|" * (len(measured) + (1 if len(measured) == 2 else 0))]
    for name, unit, get, better in ROWS:
        vals = [float(get(m)) for m in measured]
        cells = [fmt(v) + (f" {unit}" if unit and unit != "" else "") for v in vals]
        row = f"| {name} | " + " | ".join(cells)
        if len(measured) == 2:
            a, b = vals
            if a == 0 and b == 0:
                verdict = "same"
            elif a == 0:
                verdict = f"0 -> {fmt(b)}" + ("  **better**" if better == "higher" else "  worse")
            else:
                pct = 100 * (b - a) / abs(a)
                improved = (pct < 0) if better == "lower" else (pct > 0)
                verdict = f"{pct:+.1f}%  " + ("**better**" if improved else
                                              ("same" if abs(pct) < 0.05 else "worse"))
            row += f" | {verdict} |"
        else:
            row += " |"
        out.append(row)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=PATH",
                    help="a run to measure (repeatable)")
    ap.add_argument("--table", type=Path, help="write a markdown comparison table here")
    ap.add_argument("--title", default="Matte quality", help="heading for the table file")
    ap.add_argument("--no-json", action="store_true",
                    help="do not write metrics.json beside each run")
    args = ap.parse_args()

    measured = []
    for spec in args.run:
        if "=" not in spec:
            sys.exit(f"--run needs LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        d = Path(path)
        if not d.is_absolute():
            d = (ROOT / d) if not d.exists() else d.resolve()
        m = measure(load_alphas(d), label.strip(), d.relative_to(ROOT)
                    if d.is_relative_to(ROOT) else d)
        measured.append(m)
        print(f"[metrics] {m['label']}: {m['frames']} frames  "
              f"flicker mean {m['area_change_pct']['mean']:.2f}%  "
              f"IoU min {m['iou_consecutive']['min']:.3f}  "
              f"soft {100 * m['softness']['soft_pixel_fraction_mean']:.3f}%  "
              f"components mean {m['components']['mean']:.2f}")
        if not args.no_json:
            out = d.parent / "metrics.json"
            out.write_text(json.dumps(m, indent=2) + "\n")
            print(f"[metrics]   -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")

    md = table(measured)
    print("\n" + md)
    if args.table:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        args.table.write_text(
            f"# {args.title}\n\n"
            "Generated by `src/metrics.py`. Shape metrics binarise at alpha > 127 so a soft\n"
            "matte and a binary one are compared on the same footing.\n\n"
            + md + "\n")
        print(f"\n[metrics] table -> {args.table}")


if __name__ == "__main__":
    main()
