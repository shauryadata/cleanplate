#!/usr/bin/env python3
"""Freeze the oracle prompt for every Tier P clip, before any method is run on it.

    python scripts/oracle_pro.py            # writes oracle_objects into each recipe
    python scripts/oracle_pro.py --render   # plus docs/img/rotobench_prompts.jpg

The rule (docs/SUBJECT_CONVENTION.md), applied to frame 0 of the reference only:

  * every significant component of the key (>= 0.5% of the frame) is a subject
  * a component that is one person gets ONE object with TWO clicks: the head (most
    interior point of the top quarter of the component) and the body (most interior
    point of the whole component). Task 4 showed a single click can land SAM 2 on
    "skin only"; two clicks describe a whole person
  * "most interior" is measured with the frame edge counted as boundary, so a torso
    cut off by the bottom of frame is not clicked on the frame edge (the Task 4
    oracle had that flaw; it did not matter for its clips, it would here)
  * where one connected blob holds several things - two people and a desk lamp - the
    recipe names the parts and pins one object per part (`oracle_parts`). Pinned by
    hand from frame 0, exactly as Task 4 did for A2 and A4, and recorded with the
    reason

Every method receives exactly these prompts. Git history shows they were committed
before the first method run on Tier P, which is the point of freezing them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate.paths import ROOT, rel                                 # noqa: E402

TRUTH = ROOT / "truth"
SIG = 0.005

# Hand-pinned parts, 1920x1012 coordinates on frame 0. Only where one connected
# component of the key holds several things a click cannot separate.
PARTS = {
    "P03_04_5k": {
        "why": "two people, a desk lamp, the desk and a monitor are one connected blob "
               "in the key; one object would pick one of them",
        "objects": [
            {"name": "man (curly hair)", "clicks": [[1373, 400], [1480, 800]]},
            {"name": "woman", "clicks": [[667, 373], [440, 640]]},
            {"name": "desk lamp", "clicks": [[747, 600]]},
            {"name": "desk", "clicks": [[400, 880]]},
            {"name": "monitor, right edge", "clicks": [[1853, 507]]},
        ]},
    "P09_04_5n": {
        "why": "stress: the man is one blob with a lamp, two equipment racks and three "
               "monitors; the body peak of the blob lands on a monitor",
        "objects": [
            {"name": "man", "clicks": [[1130, 330], [1000, 660]]},
            {"name": "desk lamp", "clicks": [[580, 510]]},
            {"name": "left rack", "clicks": [[600, 660]]},
            {"name": "front monitor", "clicks": [[1400, 880]]},
            {"name": "right equipment", "clicks": [[1560, 640]]},
            {"name": "left monitor", "clicks": [[280, 840]]},
        ]},
    "P10_07_1c": {
        "why": "stress: four kneeling people, cases and the floor platform are one blob",
        "objects": [
            {"name": "person, far left", "clicks": [[620, 688]]},
            {"name": "person, second left", "clicks": [[760, 648]]},
            {"name": "person, kneeling right", "clicks": [[1120, 708]]},
            {"name": "person, far right", "clicks": [[1220, 588]]},
            {"name": "floor platform", "clicks": [[960, 908]]},
            {"name": "equipment case, left", "clicks": [[480, 728]]},
            {"name": "boxes, right", "clicks": [[1400, 748]]},
        ]},
    "P11_04_5g": {
        "why": "stress: three people behind monitors and two runs of tables, one blob",
        "objects": [
            {"name": "person, left", "clicks": [[980, 576]]},
            {"name": "person, centre", "clicks": [[1060, 636]]},
            {"name": "person, right", "clicks": [[1440, 696]]},
            {"name": "monitors", "clicks": [[900, 756]]},
            {"name": "table, centre", "clicks": [[1120, 876]]},
            {"name": "table, right", "clicks": [[1700, 776]]},
        ]},
    "P12_04_3b": {
        "why": "the A4 case with professional truth: a held notepad, keyed with him",
        "objects": [
            {"name": "man", "clicks": [[1090, 430], [1090, 660]]},
            {"name": "notepad (held)", "clicks": [[990, 490]]},
        ]},
    "P13_07_3c": {
        "why": "the rope is keyed with him; a two-click person prompt would not cover it",
        "objects": [
            {"name": "soldier", "clicks": [[690, 148], [760, 508]]},
            {"name": "rope", "clicks": [[1096, 668]]},
        ]},
    "P14_07_3b": {
        "why": "the rope runs to the top of frame, so the head rule clicked the rope; "
               "the rifle is held and keyed",
        "objects": [
            {"name": "soldier", "clicks": [[1150, 556], [1200, 736]]},
            {"name": "rope", "clicks": [[900, 176], [820, 876]]},
            {"name": "rifle (held)", "clicks": [[1000, 736]]},
        ]},
}
SNAP = 24          # px: a pinned click is moved onto the key if it is this close


def interior_peak(m: np.ndarray) -> tuple[int, int]:
    """Most interior pixel, with everything outside the frame counted as background."""
    from scipy import ndimage
    pad = np.pad(m, 1)
    d = ndimage.distance_transform_edt(pad)[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return int(x), int(y)


def person_clicks(comp: np.ndarray) -> list[list[int]]:
    ys = np.where(comp.any(axis=1))[0]
    y0, y1 = int(ys.min()), int(ys.max())
    head = comp.copy()
    head[y0 + max(1, (y1 - y0) // 4):] = False
    return [list(interior_peak(head)), list(interior_peak(comp))]


def objects_for(d: Path) -> tuple[list, str]:
    from scipy import ndimage
    if d.name in PARTS:
        p = PARTS[d.name]
        fg = np.asarray(Image.open(d / "alpha" / "00000.png")) > 127
        dist, (iy, ix) = ndimage.distance_transform_edt(~fg, return_indices=True)
        objs = []
        for o in p["objects"]:
            clicks = []
            for x, y in o["clicks"]:
                if dist[y, x] > SNAP:
                    raise SystemExit(f"{d.name}: pinned click {x},{y} for {o['name']!r} "
                                     f"is {dist[y, x]:.0f} px from the key - re-pin it")
                clicks.append([int(ix[y, x]), int(iy[y, x])])
            objs.append({"name": o["name"], "clicks": clicks})
        return objs, "pinned parts: " + p["why"]
    a = np.asarray(Image.open(d / "alpha" / "00000.png")) > 127
    lab, n = ndimage.label(a)
    sizes = ndimage.sum(a, lab, range(1, n + 1)) if n else []
    objs = []
    for j, z in sorted(enumerate(sizes, start=1), key=lambda t: -t[1]):
        if z < SIG * a.size:
            continue
        objs.append({"name": f"component {len(objs) + 1}",
                     "clicks": person_clicks(lab == j)})
    return objs, "rule: per significant component, head + body interior peaks"


def render(clips: list[Path]) -> None:
    import cv2
    from PIL import ImageDraw
    cells = []
    for d in clips:
        rec = json.loads((d / "recipe.json").read_text())
        f = np.asarray(Image.open(d / "frames" / "00000.jpg").convert("RGB")).copy()
        a = np.asarray(Image.open(d / "alpha" / "00000.png"))
        cnt, _ = cv2.findContours((a > 127).astype(np.uint8), cv2.RETR_LIST,
                                  cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(f, cnt, -1, (60, 220, 255), 3)
        im = Image.fromarray(f); g = ImageDraw.Draw(im)
        for o in rec["oracle_objects"]:
            for x, y in o["clicks"]:
                g.ellipse([x - 16, y - 16, x + 16, y + 16], fill=(0, 230, 90),
                          outline=(0, 0, 0), width=4)
        im = im.resize((640, 337))
        g = ImageDraw.Draw(im)
        g.rectangle([0, 0, 640, 18], fill=(0, 0, 0))
        g.text((6, 3), f"{d.name}  ({len(rec['oracle_objects'])} object(s))",
               fill=(255, 255, 255))
        cells.append(np.asarray(im))
    while len(cells) % 3:
        cells.append(np.zeros_like(cells[0]))
    grid = np.vstack([np.hstack(cells[i:i + 3]) for i in range(0, len(cells), 3)])
    out = ROOT / "docs" / "img" / "rotobench_prompts.jpg"
    Image.fromarray(grid).save(out, quality=85)
    print(f"[oracle] {rel(out)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    clips = sorted(d for d in TRUTH.glob("P*") if (d / "alpha" / "00000.png").exists()
                   and "excluded" not in json.loads((d / "recipe.json").read_text()))
    for d in clips:
        rp = d / "recipe.json"
        rec = json.loads(rp.read_text())
        objs, how = objects_for(d)
        rec["oracle_objects"] = objs
        rec["oracle_rule"] = how
        rp.write_text(json.dumps(rec, indent=2) + "\n")
        print(f"[oracle] {d.name}: {len(objs)} object(s), "
              f"{sum(len(o['clicks']) for o in objs)} click(s) - {how[:60]}")
    if args.render:
        render(clips)


if __name__ == "__main__":
    main()
