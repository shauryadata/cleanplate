"""RotoBench: run a matting method over the truth clips and score it.

A *method* is a callable that takes (clip, prompt, frames) and returns a uint8 alpha
sequence plus a stats dict. Everything else here is shared, so two methods differ only
in what they compute, never in how they are measured or prompted.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import accuracy
from .ingest import frame_paths, resolve_frames_dir
from .paths import ROOT, peak_rss_mb, rel
from .session import Prompt

TRUTH = ROOT / "truth"


@dataclass
class Clip:
    name: str
    tier: str
    frames_dir: Path
    alpha_dir: Path
    note: str = ""
    oracle_clicks: list | None = None

    @property
    def n(self) -> int:
        return len(frame_paths(self.frames_dir))


def clips(tier: str | None = None) -> list[Clip]:
    out = []
    for d in sorted(TRUTH.iterdir()):
        rp = d / "recipe.json"
        if not rp.exists():
            continue
        rec = json.loads(rp.read_text())
        t = rec.get("tier", "?")
        if tier and t.lower() != tier.lower():
            continue
        out.append(Clip(d.name, t, d / "frames", d / "alpha",
                        rec.get("recipe", {}).get("note", "") or rec.get("note", ""),
                        rec.get("oracle_clicks")))
    return out


def load_alpha(d: Path) -> np.ndarray:
    ps = sorted(d.glob("*.png"), key=lambda p: int(p.stem))
    return np.stack([np.asarray(Image.open(p).convert("L")) for p in ps])


def oracle_prompt(gt: np.ndarray, frame: int = 0,
                  override: list | None = None) -> Prompt:
    """The prompt every method under test receives. Identical for all of them.

    Default rule: one positive click at the most interior point of the reference alpha
    (the distance-transform peak). Deliberately simple and predictable - it is an
    *oracle*, derived from the answer, because the question is how good a matte a
    method makes, not how well a human guesses where to click.

    `override` lets a clip's recipe.json pin the clicks explicitly. That exists for one
    honest reason: A2 is a woman AND a child touching, so they are a single blob with a
    single interior peak, and one click can only ever return one of them. Rather than
    tune a heuristic until it happens to split them - which would be fitting the
    benchmark to the pipeline - the extra click is written down in the recipe, applies
    to every method equally, and is visible to anyone reading the file.
    """
    from scipy import ndimage
    p = Prompt()
    if override:
        for xy in override:
            p.add(frame, int(xy[0]), int(xy[1]), positive=True)
        return p
    m = gt[frame] > 127
    if not m.any():
        raise ValueError("reference alpha is empty on the prompt frame")
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    dist = ndimage.distance_transform_edt(m)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    p.add(frame, int(x), int(y), positive=True)
    return p


def hair_box(gt: np.ndarray, top_fraction: float = 0.30,
             pad: int = 12) -> tuple[int, int, int, int]:
    """Bounding box of the top slice of the subject: the head, hence the hair.

    Derived from the reference, so it is identical for every method. A whole-frame
    average is dominated by the easy interior; this is where the argument actually is.
    """
    m = gt.max(axis=0) > 127
    ys, xs = np.where(m)
    if len(ys) == 0:
        raise ValueError("reference alpha is empty")
    y0, y1 = int(ys.min()), int(ys.max())
    cut = int(y0 + (y1 - y0) * top_fraction)
    sel = m[y0:cut]
    xs2 = np.where(sel.any(axis=0))[0]
    H, W = m.shape
    return (max(0, int(xs2.min()) - pad), max(0, y0 - pad),
            min(W, int(xs2.max()) + pad), min(H, cut + pad))


def evaluate(method_name: str, fn, clip: Clip, save_dir: Path | None = None) -> dict:
    """Run one method on one clip and score it. Returns a result record."""
    gt = load_alpha(clip.alpha_dir)
    prompt = oracle_prompt(gt, override=clip.oracle_clicks)
    hb = hair_box(gt)

    t0 = time.perf_counter()
    alpha, stats = fn(clip, prompt)
    wall = time.perf_counter() - t0
    if alpha.shape != gt.shape:
        raise ValueError(f"{method_name} on {clip.name}: alpha {alpha.shape} "
                         f"!= truth {gt.shape}")

    sc = accuracy.score(alpha, gt, f"{method_name}", hair_box=hb)
    sc.update({
        "method": method_name, "clip": clip.name, "tier": clip.tier,
        "seconds_total": round(wall, 2),
        "seconds_per_frame": round(wall / max(len(gt), 1), 4),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "prompt": [prompt.frames[f].to_dict() for f in prompt.prompt_frames],
        "stage_stats": stats,
    })
    if save_dir:
        d = Path(save_dir) / method_name / clip.name
        d.mkdir(parents=True, exist_ok=True)
        for i, a in enumerate(alpha):
            Image.fromarray(a, mode="L").save(d / f"{i:05d}.png")
        (d.parent / f"{clip.name}.json").write_text(json.dumps(sc, indent=2) + "\n")
    return sc


def mean_row(results: list[dict], region: str = "whole_frame") -> dict:
    """Average a method's scores across clips, so a table can show one row per method."""
    if not results:
        return {}
    keys = ["MAD", "MSE", "Grad", "dtSSD", "BF"]
    out = {"label": results[0]["method"], region: {}}
    for k in keys:
        vals = [r[region][k] for r in results if region in r]
        out[region][k] = round(float(np.mean(vals)), 4) if vals else None
    out["seconds_per_frame"] = round(
        float(np.mean([r["seconds_per_frame"] for r in results])), 4)
    out["peak_rss_mb"] = round(float(np.max([r["peak_rss_mb"] for r in results])), 1)
    out["clips"] = len(results)
    return out


def cost_table(rows: list[dict]) -> str:
    out = ["| Method | s/frame | peak RSS | clips |", "|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['label']} | {r['seconds_per_frame']:.3f} | "
                   f"{r['peak_rss_mb']:.0f} MB | {r['clips']} |")
    return "\n".join(out)
