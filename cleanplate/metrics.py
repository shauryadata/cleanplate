"""Matte quality, measured. Nothing here is estimated.

Shape metrics binarise at alpha > 127 so a soft matte and a binary one are compared
on the same footing.
"""
from __future__ import annotations

import numpy as np

THRESH = 127
SIGNIFICANT = 0.01      # a component counts once it reaches 1% of the largest


def _perimeter(m: np.ndarray) -> int:
    """Foreground pixels with a 4-neighbour outside the matte."""
    p = np.zeros_like(m)
    p[1:, :] |= m[1:, :] ^ m[:-1, :]
    p[:-1, :] |= m[1:, :] ^ m[:-1, :]
    p[:, 1:] |= m[:, 1:] ^ m[:, :-1]
    p[:, :-1] |= m[:, 1:] ^ m[:, :-1]
    return int((p & m).sum())


def measure(alphas: np.ndarray, label: str = "run") -> dict:
    """alphas: (N, H, W) uint8 0..255, or bool."""
    a8 = (alphas.astype(np.uint8) * 255) if alphas.dtype == bool else alphas
    n, h, w = a8.shape
    hard = a8 > THRESH

    area = hard.reshape(n, -1).sum(1).astype(float)
    prev = np.where(area[:-1] == 0, np.nan, area[:-1])
    d_pct = 100 * np.abs(np.diff(area)) / prev if n > 1 else np.array([np.nan])

    if n > 1:
        inter = (hard[:-1] & hard[1:]).reshape(n - 1, -1).sum(1).astype(float)
        union = (hard[:-1] | hard[1:]).reshape(n - 1, -1).sum(1).astype(float)
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        l1 = np.abs(a8[1:].astype(float) - a8[:-1].astype(float)
                    ).reshape(n - 1, -1).mean(1)
    else:
        iou = np.array([1.0]); l1 = np.array([0.0])

    per = np.array([_perimeter(m) for m in hard], dtype=float)
    per_norm = per / np.sqrt(np.where(area == 0, np.nan, area))
    soft = ((a8 > 0) & (a8 < 255)).reshape(n, -1).mean(1)

    from scipy import ndimage
    comps, comps_sig, specks = [], [], []
    for m in hard:
        lab, n_comp = ndimage.label(m)          # not `n` - that is the frame count
        comps.append(n_comp)
        if n_comp == 0:
            comps_sig.append(0); specks.append(0); continue
        sizes = ndimage.sum(m, lab, range(1, n_comp + 1))
        keep = sizes >= SIGNIFICANT * sizes.max()
        comps_sig.append(int(keep.sum()))
        specks.append(int(sizes[~keep].sum()))
    comps = np.array(comps, float); comps_sig = np.array(comps_sig, float)
    specks = np.array(specks, float)

    return {
        "label": label,
        "frames": int(n),
        "resolution": [int(w), int(h)],
        "binarise_threshold": THRESH,
        "empty_frames": [int(i) for i in np.where(area == 0)[0]],
        "area_change_pct": {"mean": round(float(np.nanmean(d_pct)), 3),
                            "max": round(float(np.nanmax(d_pct)), 3),
                            "argmax_frame": int(np.nanargmax(d_pct)) + 1},
        "iou_consecutive": {"mean": round(float(iou.mean()), 4),
                            "min": round(float(iou.min()), 4),
                            "argmin_frame": int(iou.argmin()) + 1},
        "perimeter": {"mean": round(float(per.mean()), 1),
                      "cv_pct": round(float(100 * per.std() / per.mean()), 3),
                      "norm_cv_pct": round(float(100 * np.nanstd(per_norm)
                                                 / np.nanmean(per_norm)), 3)},
        "softness": {"soft_pixel_fraction_mean": round(float(soft.mean()), 6),
                     "soft_pixel_fraction_max": round(float(soft.max()), 6),
                     "distinct_alpha_values": int(len(np.unique(a8)))},
        "components": {"mean": round(float(comps.mean()), 3), "max": int(comps.max()),
                       "significant_mean": round(float(comps_sig.mean()), 3),
                       "significant_max": int(comps_sig.max()),
                       "significant_threshold": SIGNIFICANT},
        "speck_pixels": {"mean": round(float(specks.mean()), 2), "max": int(specks.max())},
        "alpha_l1_change_mean": round(float(l1.mean()), 4),
        "area_fraction": {"mean": round(float((area / (h * w)).mean()), 5),
                          "min": round(float((area / (h * w)).min()), 5),
                          "max": round(float((area / (h * w)).max()), 5)},
    }


def per_frame_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two runs, frame by frame. 1.0 means that frame did not change.

    This is what makes SAM 2's global conditioning visible: a corrective click at
    frame N shows up as IoU < 1 on frames BEFORE N as well as after.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    ha = (a > THRESH) if a.dtype != bool else a
    hb = (b > THRESH) if b.dtype != bool else b
    n = len(ha)
    inter = (ha & hb).reshape(n, -1).sum(1).astype(float)
    union = (ha | hb).reshape(n, -1).sum(1).astype(float)
    return np.divide(inter, union, out=np.ones_like(inter), where=union > 0)


ROWS = [
    ("Flicker - mean frame-to-frame area change", "%",
     lambda m: m["area_change_pct"]["mean"], "lower"),
    ("Flicker - max frame-to-frame area change", "%",
     lambda m: m["area_change_pct"]["max"], "lower"),
    ("Consecutive-frame IoU - mean", "", lambda m: m["iou_consecutive"]["mean"], "higher"),
    ("Consecutive-frame IoU - min", "", lambda m: m["iou_consecutive"]["min"], "higher"),
    ("Boil - perimeter CV", "%", lambda m: m["perimeter"]["cv_pct"], "lower"),
    ("Boil - shape-normalised perimeter CV", "%",
     lambda m: m["perimeter"]["norm_cv_pct"], "lower"),
    ("Softness - pixels with 0 < alpha < 255", "% of frame",
     lambda m: 100 * m["softness"]["soft_pixel_fraction_mean"], "higher"),
    ("Softness - distinct alpha values", "",
     lambda m: m["softness"]["distinct_alpha_values"], "higher"),
    ("Fragmentation - mean significant components (>=1% of largest)", "",
     lambda m: m["components"]["significant_mean"], "lower"),
    ("Fragmentation - max significant components", "",
     lambda m: m["components"]["significant_max"], "lower"),
    ("Threshold noise - mean speck pixels", "px",
     lambda m: m["speck_pixels"]["mean"], "lower"),
]


def _fmt(v: float) -> str:
    if isinstance(v, (int, np.integer)) or float(v).is_integer():
        return str(int(v))
    return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"


def table(measured: list[dict]) -> str:
    """Markdown comparison table. With exactly two runs, adds a Change column."""
    if not measured:
        return "_no runs measured_"
    pair = len(measured) == 2
    heads = [m["label"] for m in measured]
    out = ["| Metric | " + " | ".join(heads) + (" | Change |" if pair else " |"),
           "|---|" + "---|" * (len(measured) + (1 if pair else 0))]
    for name, unit, get, better in ROWS:
        vals = [float(get(m)) for m in measured]
        cells = [_fmt(v) + (f" {unit}" if unit else "") for v in vals]
        row = f"| {name} | " + " | ".join(cells)
        if pair:
            a, b = vals
            if a == 0 and b == 0:
                verdict = "same"
            elif a == 0:
                verdict = f"0 -> {_fmt(b)}" + ("  **better**" if better == "higher" else "  worse")
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
