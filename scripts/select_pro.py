#!/usr/bin/env python3
"""Pick each professional-truth shot's 96-frame window by a stated rule, not by eye.

    python scripts/select_pro.py            # -> datasets/tos_pro/_survey/windows.json

The rule: the contiguous 96-frame window nearest the centre of the shot in which
every significant foreground component descends from one present on the window's
first frame. That is what a click-based tool can be fairly asked for - the subject it
was shown at frame 0, followed through the shot. A second actor who walks in at frame
40 was never clicked, and scoring every method on missing him would measure prompting,
not matting. See docs/SUBJECT_CONVENTION.md.

"Significant" is at least 0.5% of the frame, so a stray floor marker does not count.
The check runs on every 8th frame of a candidate window; a component that enters and
leaves again inside eight frames is below what this can see, and the full-rate lineage
check in build_pro_truth.py is the one that must pass before a clip is scored.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import protruth as P                                   # noqa: E402
from cleanplate.paths import ROOT                                      # noqa: E402

CACHE = ROOT / "datasets" / "tos_pro" / "_survey"
WIN, STEP, SIG = 96, 8, 0.005

# (shot, axes it represents). Chosen from docs/PRO_SURVEY.md; exclusions and reasons
# are recorded there and in the subject convention.
CANDIDATES = [
    ("P01", "08_3a", ["backlit hair"]),
    ("P02", "04_1b", ["large motion", "turns away", "hair from behind"]),
    ("P03", "04_5k", ["curly hair", "two subjects", "keyed props (lamp, desk)"]),
    # P04 was 07_1b: excluded at build time. The held prop is held OUT of the key.
    ("P05", "09_1a", ["dark subject", "held object", "complex silhouette"]),
    ("P06", "08_4a", ["full body", "hands", "slow control"]),
    ("P07", "04_1a", ["small subject", "thin limbs"]),
    ("P08", "04_3d", ["full body", "outstretched arms"]),
    # Stress group: the key also holds set dressing in front of or around the subject
    # (monitors, a lamp, a platform, tables). Under the convention that is part of the
    # subject, so the oracle clicks it; these are reported apart from the core.
    ("P09", "04_5n", ["stress: keyed set dressing", "occlusion by props", "curly hair"]),
    ("P10", "07_1c", ["stress: keyed set dressing", "two subjects", "platform"]),
    ("P11", "04_5g", ["stress: keyed set dressing", "three subjects", "tables"]),
    # Short group: the brief's "held objects" and "fast motion" axes exist in the
    # archive only as keys shorter than 96 frames. They run at their full keyed length
    # (>= 60 frames) and are reported apart, so the core stays on 96-frame windows.
    ("P12", "04_3b", ["short", "held object (notepad): the A4 case"], "short"),
    ("P13", "07_3c", ["short", "fast motion", "motion blur", "rope"], "short"),
    ("P14", "07_3b", ["short", "held object (rifle)", "rope"], "short"),
]
# Considered and rejected, with the reason, so the set can be audited:
REJECTED = {
    "04_2c": "defocus axis; second actor enters at frame ~104 of 111, no clean window",
    "07_1b": "held prop (tracking-marker cylinder) is held OUT of the key: a deliberate "
             "hold-out, rule 7; also found only by the full-rate lineage check",
    "07_1f": "held prop is held out of the key (rule 7)",
    "07_1a": "extreme close-up whose key is mostly straight garbage-matte polygons",
    "04_2b": "a room set piece, not a subject",
    "04_1a": "alignment could not be proven (no plate frame matches the key)",
    "short": "02_3b 02_3d 04_5e 04_5h 04_5i 04_5j 04_5l 07_2b 07_2e 07_3a 07_3e 07_3f: "
             "fewer than 96 keyed frames and no axis the set lacks",
    "set keys": "24 shots whose foreground is a whole room with a window keyed out",
}


def comps(a: np.ndarray) -> list[np.ndarray]:
    from scipy import ndimage
    m = a > 0.5
    lab, n = ndimage.label(m)
    if not n:
        return []
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return [lab == (i + 1) for i, s in enumerate(sizes) if s >= SIG * m.size]


def lineage_ok(alphas: list[np.ndarray]) -> tuple[bool, int | None]:
    """True if every significant component overlaps (dilated) the previous subject."""
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    subject = np.zeros(alphas[0].shape, bool)
    for c in comps(alphas[0]):
        subject |= c
    for i, a in enumerate(alphas[1:], start=1):
        reach = cv2.dilate(subject.astype(np.uint8), k) > 0
        now = np.zeros_like(subject)
        for c in comps(a):
            if not (c & reach).any():
                return False, i
            now |= c
        subject = now
    return True, None


def load(shot: str, res: str, digits: int, n: int) -> np.ndarray:
    import cv2
    p = P.fetch(P.cleaned_url(shot, n, digits, res),
                CACHE / "frames" / shot / res / f"{n:0{digits}d}.exr")
    _, a = P.read_exr(p)
    if a.shape[1] != 1920:
        a = cv2.resize(a, (1920, 1012), interpolation=cv2.INTER_AREA)
    return a


def main() -> None:
    rows = {r["shot"]: r for r in json.loads((CACHE / "survey.json").read_text())}
    out = {}
    for cand in CANDIDATES:
        _pid, shot, axes = cand[:3]
        r = rows[shot]
        res = r["source"]
        rg = r["hd"] if res == "linear_hd" else r["k4"]
        have = set(rg["nums"])
        win = WIN
        if len(cand) > 3 and cand[3] == "short":
            # longest contiguous run, capped at 96 and floored at 60
            run = best = 1
            for a_, b_ in zip(rg["nums"], rg["nums"][1:]):
                run = run + 1 if b_ == a_ + 1 else 1; best = max(best, run)
            win = min(WIN, best)
            if win < 60:
                out[shot] = {"shot": shot, "axes": axes, "rejected": f"only {win} frames"}
                continue
        starts = [n for n in rg["nums"] if all((n + k) in have for k in range(win))]
        centre = (rg["nums"][0] + rg["nums"][-1] - win) / 2
        tried, queue = [], [min(starts, key=lambda n: abs(n - centre))]
        grid0 = rg["nums"][0]
        while queue and len(tried) < 6:
            s = queue.pop(0)
            # sample on one grid for the whole shot, so neighbouring windows share frames
            samples = sorted({s, s + win - 1} | {n for n in range(s, s + win)
                                                 if (n - grid0) % STEP == 0})
            alphas = [load(shot, res, rg["digits"], n) for n in samples]
            ok, bad = lineage_ok(alphas)
            tried.append({"start": s, "ok": ok,
                          "entrant_at": None if ok else samples[bad]})
            if ok:
                out[shot] = {"shot": shot, "axes": axes, "source": res,
                             "digits": rg["digits"], "window": [s, s + win - 1],
                             "centre_offset": round(s - centre, 1),
                             "components_at_start": len(comps(alphas[0])),
                             "tried": tried}
                break
            # Something entered between samples[bad-1] and samples[bad]. Either end the
            # window before it, or start it once the newcomer is already in shot.
            done = {t["start"] for t in tried}
            nxt = [c for c in (samples[bad - 1] - win + 1, samples[bad])
                   if c in set(starts) and c not in done]
            queue = sorted(set(queue) | set(nxt), key=lambda n: abs(n - centre))
        if shot not in out:
            out[shot] = {"shot": shot, "axes": axes, "rejected": "no window without a "
                         "late entrant among the six nearest the centre", "tried": tried}
        o = out[shot]
        print(f"[select] {shot:6s} " + (f"window {o['window'][0]}-{o['window'][1]} "
              f"({o['components_at_start']} subject component(s), "
              f"{len(o['tried'])} window(s) tried)" if "window" in o else
              f"REJECTED - {o['rejected']}: " + ", ".join(
                  f"{t['start']}->{t['entrant_at']}" for t in o["tried"])), flush=True)
    (CACHE / "windows.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
