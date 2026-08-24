#!/usr/bin/env python3
"""Checkpoint 0b: does a well-formed prompt fix the beard/jaw dropout on `hair`?

Metric: skin-toned plate pixels that sit inside the subject's filled silhouette but are
excluded by the matte. Motion-invariant, so the camera push-in does not confound it.
"""
from __future__ import annotations

import os
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json, sys
from pathlib import Path
import numpy as np
from scipy import ndimage
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import methods, refine, track                          # noqa: E402
from cleanplate.ingest import load_frame, n_frames                     # noqa: E402
from cleanplate.memguard import MemoryGuard                            # noqa: E402
from cleanplate.paths import ROOT                                      # noqa: E402
from cleanplate.session import Prompt                                  # noqa: E402

SHOT = "hair"


def skin(rgb):
    r, g, b = [rgb[..., i].astype(np.float32) for i in range(3)]
    y = .299*r + .587*g + .114*b
    cb = -.169*r - .331*g + .5*b + 128
    cr = .5*r - .419*g - .081*b + 128
    return (y > 50) & (cb > 77) & (cb < 135) & (cr > 133) & (cr < 180)


def excluded_skin(alpha: np.ndarray) -> np.ndarray:
    out = []
    for i, a8 in enumerate(alpha):
        a = a8 > 127
        hull = ndimage.binary_fill_holes(ndimage.binary_closing(a, np.ones((25, 25))))
        miss = ndimage.binary_opening(skin(load_frame(SHOT, i)) & hull & ~a,
                                      np.ones((3, 3)))
        out.append(int(miss.sum()))
    return np.array(out)


def main() -> None:
    n = n_frames(SHOT)
    their = Prompt.load(SHOT)                       # 1 click on the plaid shirt
    print(f"their prompt: {their.frames[0].positive}")

    # Well-formed: one click per region of the SUBJECT, placed on the plate by eye from
    # frame 0 and written down here so the workflow doc can quote them.
    wf = Prompt()
    for xy in [(660, 320), (628, 96), (640, 180), (700, 250)]:   # torso, hair, brow, cheek
        wf.add(0, *xy, True)
    print(f"well-formed prompt: {wf.frames[0].positive}")

    results = {}
    for label, prompt, model, hq in [
            ("their_run", their, "matanyone", False),
            ("wellformed_hq", wf, "matanyone2", True)]:
        with MemoryGuard(label) as g:
            masks, _ = track.track(SHOT, prompt, progress=None)
            g.check()
            alpha, _ = refine.refine(SHOT, masks, model=model, progress=None)
            g.check()
            if hq:
                from cleanplate.ingest import resolve_frames_dir
                alpha, _ = methods.refine_hair_zoom(resolve_frames_dir(SHOT), alpha,
                                                    model=model)
                g.check()
        es_bin = excluded_skin((masks.astype(np.uint8) * 255))
        es = excluded_skin(alpha)
        results[label] = {"excluded_skin_binary": es_bin.tolist(),
                          "excluded_skin_final": es.tolist(),
                          "guard": g.stats()}
        bad = int((es > 150).sum())
        print(f"  {label:14s} excluded-skin px: mean {es.mean():6.1f}  max {es.max():4d} "
              f"(f{int(es.argmax())})  frames>150px: {bad}/{n}")
        print(f"      SAM 2 stage alone: mean {es_bin.mean():6.1f}  max {es_bin.max():4d}")
        d = ROOT / "outputs" / "_userbug" / label
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"{SHOT}_alpha.npy", alpha)
    (ROOT / "outputs" / "_userbug" / "face_dropout.json").write_text(
        json.dumps(results, indent=2) + "\n")
    a = np.array(results["their_run"]["excluded_skin_final"])
    b = np.array(results["wellformed_hq"]["excluded_skin_final"])
    print(f"\n  mean excluded skin {a.mean():.1f} -> {b.mean():.1f} px "
          f"({100*(b.mean()-a.mean())/max(a.mean(),1e-9):+.1f}%)")
    print(f"  frames over 150 px: {int((a>150).sum())} -> {int((b>150).sum())}")


if __name__ == "__main__":
    main()
