#!/usr/bin/env python3
"""Survey every shot in tearsofsteel-cleaned-exr and classify what it actually holds.

    python scripts/survey_pro.py            # -> docs/PRO_SURVEY.md + contact sheets

The directory is called "cleaned", and that name is misleading in both directions:
some shots are rig-removed full plates, others are the compositor's key with a real
alpha, and some keys are of the whole *set* (a window keyed out) rather than of an
actor. Nothing is assumed from the name; every shot is classified from its pixels.

Steps, all cached under datasets/tos_pro/_survey (gitignored, re-fetchable):
  1. list both archives and every shot's frame ranges (linear_hd and 4K linear)
  2. read the middle frame of each shot and classify its alpha
  3. for character keys, read start / middle / middle+1 / end of the candidate
     96-frame window and measure how much the matte moves
  4. write docs/PRO_SURVEY.md and contact sheets to outputs/_scout/
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import protruth as P                                   # noqa: E402
from cleanplate.paths import ROOT, rel                                 # noqa: E402

CACHE = ROOT / "datasets" / "tos_pro" / "_survey"
WIN = 96


def shots_in(url: str) -> list[str]:
    return sorted(set(re.findall(r'href="(\d\d_[0-9a-z_]+)/"', P.listing(url))))


def ranges(shot: str) -> dict:
    out = {"shot": shot}
    for key, url in (("hd", f"{P.CLEANED}/{shot}/linear_hd/"),
                     ("k4", f"{P.CLEANED}/{shot}/linear/"),
                     ("plate", f"{P.FOOTAGE}/{shot}/linear_hd/")):
        try:
            nums, digits = P.frame_numbers(url)
        except Exception:
            nums, digits = [], None
        out[key] = {"n": len(nums), "first": nums[0] if nums else None,
                    "last": nums[-1] if nums else None, "digits": digits,
                    "nums": nums}
    return out


def contiguous_window(nums: list[int], win: int = WIN) -> int | None:
    """Start of the contiguous run of `win` frames nearest the middle of the shot.

    Some shots have holes in their numbering, so first..last is not a promise that
    every frame exists. A window with a missing frame would silently shorten a clip.
    """
    have = set(nums)
    starts = [n for n in nums if all((n + k) in have for k in range(win))]
    if not starts:
        return None
    centre = (nums[0] + nums[-1] - win) / 2
    return min(starts, key=lambda n: abs(n - centre))


def source(r: dict) -> tuple[str, dict] | tuple[None, None]:
    """Prefer the 1920 linear_hd key; fall back to the 4K one."""
    if r["hd"]["n"]:
        return "linear_hd", r["hd"]
    if r["k4"]["n"]:
        return "linear", r["k4"]
    return None, None


def frame_path(shot: str, res: str, digits: int, n: int) -> Path:
    return CACHE / "frames" / shot / res / f"{n:0{digits}d}.exr"


def get(shot: str, res: str, digits: int, n: int) -> Path:
    return P.fetch(P.cleaned_url(shot, n, digits, res), frame_path(shot, res, digits, n))


def at_hd(a: np.ndarray) -> np.ndarray:
    """Everything is measured at 1920 wide so 4K and HD keys are comparable."""
    import cv2
    if a.shape[1] == 1920:
        return a
    return cv2.resize(a, (1920, round(a.shape[0] * 1920 / a.shape[1])),
                      interpolation=cv2.INTER_AREA)


def classify(rgb: np.ndarray, a: np.ndarray | None) -> dict:
    from scipy import ndimage
    if a is None:
        return {"class": "no alpha"}
    if a.min() > 0.99:
        return {"class": "constant alpha (clean plate)"}
    z = float((a <= 0.002).mean()); o = float((a >= 0.998).mean())
    soft = float(((a > 0.02) & (a < 0.98)).mean())
    lab, n = ndimage.label(a > 0.5)
    sizes = ndimage.sum(a > 0.5, lab, range(1, n + 1)) if n else np.array([])
    comps = int((sizes >= 0.01 * sizes.max()).sum()) if n else 0
    bg = rgb[a <= 0.002]
    black = float(np.percentile(np.abs(bg).max(axis=1), 99)) if len(bg) else None
    if z < 0.05 or o < 0.005:
        cls = "other"
    elif z < 0.40:
        # mostly foreground: the green screen is a window in a live set, and the
        # "foreground" is every lamp and desk in the room, not a subject
        cls = "set key"
    else:
        cls = "character key"
    return {"class": cls, "zero": round(z, 4), "one": round(o, 4),
            "soft": round(soft, 4), "components": comps,
            "bg_p99": None if black is None else round(black, 5),
            "premultiplied_over_black": black is not None and black < 0.05}


def motion(shot: str, res: str, digits: int, start: int) -> dict:
    """How much the matte moves across the candidate window. 1920-scale pixels."""
    from scipy import ndimage
    idx = [start, start + WIN // 2, start + WIN // 2 + 1, start + WIN - 1]
    al = {}
    for n in idx:
        _, a = P.read_exr(get(shot, res, digits, n))
        al[n] = at_hd(a)
    def c(a):
        return np.array(ndimage.center_of_mass(a > 0.5))
    s, m, m1, e = (al[n] for n in idx)
    step = float(np.abs(m1 - m).mean())                  # one-frame |d alpha|
    inter = ((s > 0.5) & (e > 0.5)).sum(); union = ((s > 0.5) | (e > 0.5)).sum()
    return {"window": [start, start + WIN - 1],
            "centroid_travel_px": round(float(np.linalg.norm(c(e) - c(s))), 1),
            "iou_start_end": round(float(inter / max(union, 1)), 3),
            "area_change_pct": round(100 * float(((e > .5).sum() - (s > .5).sum())
                                                 / max((s > .5).sum(), 1)), 1),
            "one_frame_dalpha_x1e3": round(1e3 * step, 3)}


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    cleaned = shots_in(P.CLEANED + "/"); footage = set(shots_in(P.FOOTAGE + "/"))
    print(f"[survey] {len(cleaned)} cleaned shots, {len(footage)} plate shots")
    with cf.ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(ranges, cleaned))
    for r in rows:
        r["has_plate"] = r["shot"] in footage and r["plate"]["n"] > 0

    def probe(r):
        res, rg = source(r)
        r["source"] = res
        if not res:
            r.update({"class": "empty"}); return r
        mid = rg["nums"][len(rg["nums"]) // 2]           # a frame that exists
        try:
            rgb, a = P.read_exr(get(r["shot"], res, rg["digits"], mid))
        except Exception as e:                              # record, do not crash
            r.update({"class": "unreadable", "error": str(e)[:200]}); return r
        r["resolution"] = [rgb.shape[1], rgb.shape[0]]
        r.update(classify(at_hd(rgb) if a is not None else rgb, None if a is None else at_hd(a)))
        r["mid_frame"] = mid
        return r
    with cf.ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(probe, rows))

    keys = [r for r in rows if r["class"] == "character key"]
    print(f"[survey] classes: " + ", ".join(
        f"{c}: {sum(r['class'] == c for r in rows)}" for c in
        sorted(set(r["class"] for r in rows))))

    def mot(r):
        res, rg = source(r)
        start = contiguous_window(rg["nums"])
        if start is not None:
            try:
                r["motion"] = motion(r["shot"], res, rg["digits"], start)
            except Exception as e:
                r["motion_error"] = str(e)[:200]
        return r
    with cf.ThreadPoolExecutor(4) as ex:
        list(ex.map(mot, keys))

    (CACHE / "survey.json").write_text(json.dumps(rows, indent=1))
    write_doc(rows)
    sheets(rows)


def write_doc(rows: list[dict]) -> None:
    by = lambda c: [r for r in rows if r["class"] == c]
    md = ["# Survey of `tearsofsteel-cleaned-exr`", "",
          "Generated by `scripts/survey_pro.py`. Every shot in the archive, classified "
          "from its pixels rather than its directory name. Source: "
          "media.xiph.org/tearsofsteel, (CC) Blender Foundation | mango.blender.org, "
          "CC BY 3.0.", "",
          "| Class | Shots | What it is |", "|---|---|---|"]
    desc = {"character key": "actor(s) over green screen, keyed out: the compositor's "
                             "matte in a real alpha channel. **Usable as truth.**",
            "set key": "a live set with a green window keyed out; every desk and lamp "
                       "is foreground. Not a subject matte.",
            "no alpha": "RGB only - a cleaned plate, no matte",
            "constant alpha (clean plate)": "alpha channel present but all ones",
            "other": "a matte that is neither mostly background nor mostly foreground",
            "empty": "no frames at 1920 or 4K",
            "unreadable": "listed, but the probe frame would not download or decode"}
    for c in ("character key", "set key", "other", "no alpha",
              "constant alpha (clean plate)", "empty", "unreadable"):
        if by(c):
            md.append(f"| {c} | {len(by(c))} | {desc[c]} |")
    md += ["", "## Character keys", "",
           "Motion is measured on the centred 96-frame window, at 1920 wide. "
           "`d alpha` is the mean one-frame change in the matte, the quantity dtSSD "
           "is built from.", "",
           "| Shot | Source | Frames | Plate | Components | Soft px | "
           "Centroid travel | IoU start/end | d alpha (x1e3) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(by("character key"), key=lambda r: r["shot"]):
        rg = r["hd"] if r["source"] == "linear_hd" else r["k4"]
        m = r.get("motion")
        md.append(f"| {r['shot']} | {r['source']} | {rg['n']} | "
                  f"{'yes' if r['has_plate'] else 'no'} | {r['components']} | "
                  f"{100 * r['soft']:.2f}% | "
                  + (f"{m['centroid_travel_px']} px | {m['iou_start_end']} | "
                     f"{m['one_frame_dalpha_x1e3']} |" if m else
                     "no contiguous 96-frame window | | |"))
    md += ["", "## Every shot", "",
           "| Shot | Class | 1920 frames | 4K frames | Plate frames |", "|---|---|---|---|---|"]
    for r in rows:
        f = lambda g: f"{g['first']}-{g['last']} ({g['n']})" if g["n"] else "-"
        md.append(f"| {r['shot']} | {r['class']} | {f(r['hd'])} | {f(r['k4'])} | "
                  f"{f(r['plate']) if r['has_plate'] else '-'} |")
    out = ROOT / "docs" / "PRO_SURVEY.md"
    out.write_text("\n".join(md) + "\n")
    print(f"[survey] -> {rel(out)}")


def sheets(rows: list[dict]) -> None:
    import cv2
    from PIL import Image, ImageDraw
    W, H = 300, 158
    cells = []
    for r in sorted(rows, key=lambda r: (r["class"] != "character key", r["shot"])):
        if not r.get("source") or r["class"] in ("unreadable", "empty"):
            continue
        rg = r["hd"] if r["source"] == "linear_hd" else r["k4"]
        rgb, a = P.read_exr(frame_path(r["shot"], r["source"], rg["digits"], r["mid_frame"]))
        t = cv2.resize(P.to_display(rgb), (W, H), interpolation=cv2.INTER_AREA)
        at = (np.full((H, W, 3), 40, np.uint8) if a is None else np.dstack(
            [cv2.resize((np.clip(a, 0, 1) * 255).astype(np.uint8), (W, H),
                        interpolation=cv2.INTER_AREA)] * 3))
        im = Image.fromarray(np.hstack([t, at])); g = ImageDraw.Draw(im)
        g.rectangle([0, 0, 2 * W, 14], fill=(0, 0, 0))
        g.text((4, 2), f"{r['shot']}  {r['class']}  {r['source']}", fill=(255, 255, 0))
        cells.append(np.asarray(im))
    while len(cells) % 3:
        cells.append(np.zeros_like(cells[0]))
    grid = np.vstack([np.hstack(cells[i:i + 3]) for i in range(0, len(cells), 3)])
    d = ROOT / "outputs" / "_scout"; d.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(grid), 6 * H):
        Image.fromarray(grid[k:k + 6 * H]).save(d / f"pro_survey_{k // (6 * H) + 1}.png")
    print(f"[survey] contact sheets -> {rel(d)}/pro_survey_*.png")


if __name__ == "__main__":
    main()
