"""Accuracy of a matte against a reference alpha.

The self-consistency metrics in `metrics.py` say whether a matte is stable. These say
whether it is *right*. Both are needed: a matte can be perfectly stable and perfectly
wrong.

Definitions are stated explicitly because the literature is not consistent about
scaling. Alpha is always in [0, 1] internally; reported numbers are scaled as noted so
they read at a sensible magnitude, the same way the video-matting papers report them.

    MAD    mean |pred - gt|, x1e3
    MSE    mean (pred - gt)^2, x1e3
    Grad   gradient error. Alpha is convolved with first-order Gaussian derivatives
           (sigma=1.4, after Rhemann et al. 2009); the squared difference of the
           gradient magnitudes is summed and divided by pixel count, x1e3.
    dtSSD  temporal accuracy. Per consecutive frame pair, the alpha time-derivative of
           the prediction is compared with that of the reference:
           sqrt(mean((dpred - dgt)^2)), averaged over pairs, x1e2. Penalises flicker
           that the reference does not have, and equally penalises being too static
           when the reference moves.
    BF     boundary F-measure on the binarised matte (alpha > 0.5), DAVIS-style:
           boundary pixels match if within `tol` px of a reference boundary pixel.
           tol defaults to 0.8% of the image diagonal. Reported 0..1, higher better.

Added in Task 6, alongside - never replacing - the columns above, so history stays
comparable:

    dtSSD-n  dtSSD divided by the reference's own motion, M = mean over pairs of
             sqrt(mean(dgt^2)). A matte that never moves has dpred = 0, so its dtSSD
             equals M exactly and dtSSD-n = 1.0 on every shot, fast or slow. Plain dtSSD
             let a frozen matte score well whenever the reference barely moved; this
             cannot. 0 is perfect, 1 is "no better than standing still", above 1 is
             flicker the reference does not have. M is reported too.
    band     every metric again, restricted to the reference's transition band: pixels
             with 0.02 < alpha < 0.98 plus the alpha=0.5 contour, dilated by 0.5% of the
             diagonal. This is where mattes differ; it replaces nothing, and unlike the
             hair box it follows detail wherever it is.
    coverage misses deep inside the subject: pixels at least 4 px inside the
             reference's opaque core where the prediction is below 0.5. Edge disagreement
             cannot reach them, so this counts genuine dropouts - the failure a real user
             reported as "part of the face drops out" - and nothing else.

Every metric also has a *hair-region* variant: identical maths restricted to a supplied
bounding box, because a whole-frame average is dominated by the easy interior and hides
exactly the thing this project is trying to fix.
"""
from __future__ import annotations

import numpy as np


def _as01(a: np.ndarray) -> np.ndarray:
    """Accept bool, uint8 0..255 or float; always return float32 in [0, 1]."""
    if a.dtype == bool:
        return a.astype(np.float32)
    a = a.astype(np.float32)
    if a.max() > 1.0001:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def _crop(a: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
    if box is None:
        return a
    l, t, r, b = box
    return a[..., t:b, l:r]


# ---------------------------------------------------------------- per-frame
def mad(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.abs(_as01(pred) - _as01(gt)).mean() * 1e3)


def mse(pred: np.ndarray, gt: np.ndarray) -> float:
    d = _as01(pred) - _as01(gt)
    return float((d * d).mean() * 1e3)


def _grad_mag(x: np.ndarray, sigma: float = 1.4) -> np.ndarray:
    """Gradient magnitude after first-order Gaussian derivatives (Rhemann et al.)."""
    from scipy.ndimage import gaussian_filter
    gx = gaussian_filter(x, sigma, order=(0, 1))
    gy = gaussian_filter(x, sigma, order=(1, 0))
    return np.sqrt(gx * gx + gy * gy)


def grad_error(pred: np.ndarray, gt: np.ndarray, sigma: float = 1.4) -> float:
    """Gradient error: penalises getting the *structure* of the edge wrong."""
    p, g = _as01(pred), _as01(gt)
    d = _grad_mag(p, sigma) - _grad_mag(g, sigma)
    return float((d * d).sum() / d.size * 1e3)


def boundary_f(pred: np.ndarray, gt: np.ndarray, tol: float | None = None,
               thresh: float = 0.5) -> float:
    """DAVIS-style boundary F-measure on the binarised matte."""
    from scipy.ndimage import distance_transform_edt
    p = _as01(pred) > thresh
    g = _as01(gt) > thresh
    h, w = p.shape[-2:]
    if tol is None:
        tol = 0.008 * np.hypot(h, w)
    r = max(1, int(round(tol)))

    def boundary(m: np.ndarray) -> np.ndarray:
        e = np.zeros_like(m)
        e[1:, :] |= m[1:, :] ^ m[:-1, :]
        e[:-1, :] |= m[1:, :] ^ m[:-1, :]
        e[:, 1:] |= m[:, 1:] ^ m[:, :-1]
        e[:, :-1] |= m[:, 1:] ^ m[:, :-1]
        return e & m

    bp, bg = boundary(p), boundary(g)
    if not bp.any() and not bg.any():
        return 1.0                      # both empty: trivially in agreement
    if not bp.any() or not bg.any():
        return 0.0
    # One Euclidean distance transform beats r iterations of binary dilation: it is
    # the correct disk-shaped tolerance rather than an octagon, and at 1920x1080 with
    # r ~ 18 the iterated version dominated the whole benchmark's runtime.
    d_to_pred = distance_transform_edt(~bp)
    d_to_gt = distance_transform_edt(~bg)
    precision = float((bp & (d_to_gt <= r)).sum()) / max(int(bp.sum()), 1)
    recall = float((bg & (d_to_pred <= r)).sum()) / max(int(bg.sum()), 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def band_mask(gt: np.ndarray, frac: float = 0.005) -> np.ndarray:
    """Transition band of one reference frame: soft pixels plus the 0.5 contour,
    dilated by `frac` of the image diagonal (about 11 px at 1920x1012)."""
    import cv2
    g = _as01(gt)
    soft = (g > 0.02) & (g < 0.98)
    hard = (g > 0.5).astype(np.uint8)
    edge = (cv2.morphologyEx(hard, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0)
    r = max(1, int(round(frac * float(np.hypot(*g.shape)))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    return cv2.dilate((soft | edge).astype(np.uint8), k) > 0


def soft_depth(alpha: np.ndarray, box: tuple[int, int, int, int] | None = None) -> float:
    """Median distance (px) from each soft pixel to the nearest opaque one.

    The Task 4 measure behind the "truth edges are about 2 px deep" claim, made a
    function so the claim can be re-tested against any reference. NaN if no soft px.
    """
    from scipy import ndimage
    a = _crop(_as01(alpha), box)
    soft = (a > 0.02) & (a < 0.98)
    if not soft.any() or not (a >= 0.98).any():
        return float("nan")
    d = ndimage.distance_transform_edt(~(a >= 0.98))
    return float(np.median(d[soft]))


# ---------------------------------------------------------------- sequence
def dtssd(pred: np.ndarray, gt: np.ndarray) -> float:
    """Temporal accuracy over a whole sequence: (N, H, W) in, one number out."""
    p, g = _as01(pred), _as01(gt)
    if len(p) < 2:
        return 0.0
    dp = np.diff(p, axis=0)
    dg = np.diff(g, axis=0)
    d = dp - dg
    per_pair = np.sqrt((d * d).reshape(len(d), -1).mean(axis=1))
    return float(per_pair.mean() * 1e2)


def dtssd_parts(pred: np.ndarray, gt: np.ndarray,
                masks: np.ndarray | None = None) -> tuple[float, float]:
    """(dtSSD, M): the temporal error and the reference's own motion, both x1e2.

    With `masks` (N, H, W), each pair is measured over mask[t] | mask[t+1] only.
    """
    p, g = _as01(pred), _as01(gt)
    if len(p) < 2:
        return 0.0, 0.0
    errs, mots = [], []
    for t in range(len(p) - 1):
        dp, dg = p[t + 1] - p[t], g[t + 1] - g[t]
        if masks is not None:
            m = masks[t] | masks[t + 1]
            if not m.any():
                continue
            dp, dg = dp[m], dg[m]
        errs.append(np.sqrt(np.mean((dp - dg) ** 2)))
        mots.append(np.sqrt(np.mean(dg ** 2)))
    if not errs:
        return 0.0, 0.0
    return float(np.mean(errs) * 1e2), float(np.mean(mots) * 1e2)


def dtssd_norm(pred: np.ndarray, gt: np.ndarray,
               masks: np.ndarray | None = None) -> float:
    """dtSSD / M. 1.0 is exactly what a frozen matte scores. NaN if M == 0."""
    e, m = dtssd_parts(pred, gt, masks)
    return float(e / m) if m > 0 else float("nan")


def coverage(pred: np.ndarray, gt: np.ndarray, depth: int = 4,
             dropout_frac: float = 0.0005) -> dict:
    """Interior misses (dropouts) and exterior false fill, per sequence.

    miss:  pred < 0.5 at least `depth` px inside the reference's opaque core
    false: pred >= 0.5 at least `depth` px outside the reference's clear background
    A frame is a dropout frame when its misses exceed `dropout_frac` of the core.
    """
    import cv2
    p, g = _as01(pred), _as01(gt)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * depth + 1,) * 2)
    miss, false, frac = [], [], []
    for pp, gg in zip(p, g):
        core = cv2.erode((gg >= 0.98).astype(np.uint8), k) > 0
        clear = cv2.erode((gg <= 0.02).astype(np.uint8), k) > 0
        m = int((core & (pp < 0.5)).sum())
        miss.append(m)
        false.append(int((clear & (pp >= 0.5)).sum()))
        frac.append(m / max(int(core.sum()), 1))
    miss, frac = np.array(miss), np.array(frac)
    return {"miss_px_mean": round(float(miss.mean()), 1),
            "miss_px_p95": round(float(np.percentile(miss, 95)), 1),
            "miss_px_max": int(miss.max()),
            "dropout_frames": int((frac > dropout_frac).sum()),
            "worst_frame": int(np.argmax(miss)),
            "false_px_mean": round(float(np.mean(false)), 1)}


def score(pred: np.ndarray, gt: np.ndarray, label: str = "run",
          hair_box: tuple[int, int, int, int] | None = None,
          ignore: np.ndarray | None = None) -> dict:
    """Score a whole sequence. pred/gt are (N, H, W).

    hair_box is (left, top, right, bottom) in pixels; when given, every metric is also
    computed inside that box alone.

    ignore (N, H, W bool) marks pixels that are not the subject under the convention
    (docs/SUBJECT_CONVENTION.md) - a flag edge the key also holds, say. They are
    neutralised by setting the prediction equal to the reference there, so they add
    no error to any metric and cannot move a boundary. The same rule applies to every
    method.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")
    p, g = _as01(pred), _as01(gt)
    if ignore is not None:
        p = np.where(ignore, g, p)
    n = len(p)

    def block(pp: np.ndarray, gg: np.ndarray) -> dict:
        return {
            "MAD": round(float(np.mean([mad(a, b) for a, b in zip(pp, gg)])), 4),
            "MSE": round(float(np.mean([mse(a, b) for a, b in zip(pp, gg)])), 4),
            "Grad": round(float(np.mean([grad_error(a, b) for a, b in zip(pp, gg)])), 4),
            "dtSSD": round(dtssd(pp, gg), 4),
            "BF": round(float(np.mean([boundary_f(a, b) for a, b in zip(pp, gg)])), 4),
        }

    out = {"label": label, "frames": int(n),
           "resolution": [int(p.shape[2]), int(p.shape[1])],
           "whole_frame": block(p, g)}
    e, m = dtssd_parts(p, g)
    out["whole_frame"]["dtSSDn"] = round(e / m, 4) if m > 0 else None
    out["whole_frame"]["motion"] = round(m, 4)
    if hair_box is not None:
        out["hair_box"] = list(hair_box)
        out["hair_region"] = block(_crop(p, hair_box), _crop(g, hair_box))
        e, m = dtssd_parts(_crop(p, hair_box), _crop(g, hair_box))
        out["hair_region"]["dtSSDn"] = round(e / m, 4) if m > 0 else None
        out["hair_region"]["motion"] = round(m, 4)
    bands = np.stack([band_mask(gg) for gg in g])
    gm = [np.abs(_grad_mag(pp) - _grad_mag(gg)) for pp, gg in zip(p, g)]
    e, m = dtssd_parts(p, g, bands)
    out["band"] = {
        "MAD": round(float(np.mean([np.abs(pp - gg)[b].mean() * 1e3
                                    for pp, gg, b in zip(p, g, bands) if b.any()])), 4),
        "MSE": round(float(np.mean([((pp - gg) ** 2)[b].mean() * 1e3
                                    for pp, gg, b in zip(p, g, bands) if b.any()])), 4),
        "Grad": round(float(np.mean([(d ** 2)[b].mean() * 1e3
                                     for d, b in zip(gm, bands) if b.any()])), 4),
        "dtSSD": round(e, 4),
        "dtSSDn": round(e / m, 4) if m > 0 else None,
        "motion": round(m, 4),
        "BF": out["whole_frame"]["BF"],
        "band_fraction": round(float(bands.mean()), 5)}
    out["coverage"] = coverage(p, g)
    return out


ROWS = [
    ("MAD (x1e3, lower better)", "MAD", "lower"),
    ("MSE (x1e3, lower better)", "MSE", "lower"),
    ("Grad (x1e3, lower better)", "Grad", "lower"),
    ("dtSSD (x1e2, lower better)", "dtSSD", "lower"),
    ("dtSSD-n (1.0 = frozen matte, lower better)", "dtSSDn", "lower"),
    ("Boundary F (0-1, higher better)", "BF", "higher"),
]


def table(scores: list[dict], region: str = "whole_frame") -> str:
    """Markdown table of several scored runs, best value in each row marked."""
    if not scores:
        return "_nothing scored_"
    heads = [s["label"] for s in scores]
    out = ["| Metric | " + " | ".join(heads) + " |",
           "|---|" + "---|" * len(scores)]
    for name, key, better in ROWS:
        vals = [s.get(region, {}).get(key) for s in scores]
        ok = [v for v in vals if v is not None]
        best = (min(ok) if better == "lower" else max(ok)) if ok else None
        cells = []
        for v in vals:
            if v is None:
                cells.append("—")
            elif best is not None and abs(v - best) < 1e-9:
                cells.append(f"**{v:.4g}**")
            else:
                cells.append(f"{v:.4g}")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


# ------------------------------------------------------- removal / inpainting
def psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Peak signal-to-noise ratio in dB. With a mask, only inside it.

    For removal the whole-frame number is meaningless — most of the frame is untouched
    and scores infinity. The hole is the only part under test.
    """
    p = pred.astype(np.float64)
    g = gt.astype(np.float64)
    if mask is not None:
        m = mask.astype(bool)
        if m.ndim == 2:
            m = m[..., None]
        m = np.broadcast_to(m, p.shape)
        if not m.any():
            return float("nan")
        mse_v = float(((p - g) ** 2)[m].mean())
    else:
        mse_v = float(((p - g) ** 2).mean())
    if mse_v <= 1e-12:
        return 100.0
    return float(10.0 * np.log10(255.0 ** 2 / mse_v))


def ssim(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Structural similarity. With a mask, averaged over the masked pixels only.

    Computed on the full frame then averaged inside the mask, because SSIM needs a
    spatial neighbourhood and cropping to a ragged hole would corrupt the windows.
    """
    from skimage.metrics import structural_similarity
    _, s_map = structural_similarity(gt, pred, channel_axis=-1, data_range=255,
                                     full=True)
    s_map = s_map.mean(axis=-1)
    if mask is None:
        return float(s_map.mean())
    m = mask.astype(bool)
    return float(s_map[m].mean()) if m.any() else float("nan")


def warping_error(frames: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Temporal warping error: how well frame t-1, warped by optical flow, predicts t.

    Flow is estimated on the *result*, so a fill that flickers or crawls scores badly
    even when every individual frame looks plausible. Reported x1e3. Lower is better.
    """
    import cv2
    if len(frames) < 2:
        return 0.0
    errs = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for i in range(1, len(frames)):
        cur_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = cur_gray.shape
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        warped = cv2.remap(frames[i - 1], gx + flow[..., 0], gy + flow[..., 1],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        d = (warped.astype(np.float32) - frames[i].astype(np.float32)) / 255.0
        d = (d ** 2).mean(axis=-1)
        if mask is not None:
            m = (mask[i] if mask.ndim == 3 else mask).astype(bool)
            errs.append(float(d[m].mean()) if m.any() else 0.0)
        else:
            errs.append(float(d.mean()))
        prev_gray = cur_gray
    return float(np.mean(errs) * 1e3)


def score_removal(pred: np.ndarray, gt: np.ndarray, holes: np.ndarray,
                  label: str = "run") -> dict:
    """Score a removal against the background it was supposed to reveal."""
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")
    hm = holes > 127 if holes.dtype != bool else holes
    return {
        "label": label, "frames": int(len(pred)),
        "resolution": [int(pred.shape[2]), int(pred.shape[1])],
        "hole_area_fraction": round(float(hm.mean()), 5),
        "PSNR_hole": round(float(np.mean([psnr(p, g, m)
                                          for p, g, m in zip(pred, gt, hm)])), 3),
        "SSIM_hole": round(float(np.mean([ssim(p, g, m)
                                          for p, g, m in zip(pred, gt, hm)])), 5),
        "PSNR_frame": round(float(np.mean([psnr(p, g) for p, g in zip(pred, gt)])), 3),
        "warp_err_hole": round(warping_error(pred, hm), 4),
        "warp_err_gt": round(warping_error(gt, hm), 4),
    }


REMOVAL_ROWS = [
    ("PSNR in the hole (dB, higher better)", "PSNR_hole", "higher"),
    ("SSIM in the hole (higher better)", "SSIM_hole", "higher"),
    ("PSNR whole frame (dB, higher better)", "PSNR_frame", "higher"),
    ("Temporal warp error in the hole (x1e3, lower better)", "warp_err_hole", "lower"),
    ("  same metric on the true background (reference floor)", "warp_err_gt", "lower"),
    ("Hole area (fraction of frame)", "hole_area_fraction", "lower"),
]


def removal_table(scores: list[dict]) -> str:
    if not scores:
        return "_nothing scored_"
    out = ["| Metric | " + " | ".join(s["label"] for s in scores) + " |",
           "|---|" + "---|" * len(scores)]
    for name, key, better in REMOVAL_ROWS:
        vals = [s.get(key) for s in scores]
        ok = [v for v in vals if v is not None]
        best = (max(ok) if better == "higher" else min(ok)) if ok else None
        cells = []
        for v in vals:
            if v is None:
                cells.append("—")
            elif best is not None and abs(v - best) < 1e-9 and not name.startswith("  "):
                cells.append(f"**{v:.5g}**")
            else:
                cells.append(f"{v:.5g}")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)
