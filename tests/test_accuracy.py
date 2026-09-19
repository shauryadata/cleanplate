#!/usr/bin/env python3
"""Validate every accuracy metric on cases whose answer is known in advance.

A metric you have not checked is a number you cannot cite. Run:  python tests/test_accuracy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import accuracy as A                                   # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def disc(h: int, w: int, cy: float, cx: float, r: float, soft: float = 0.0) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(yy - cy, xx - cx)
    if soft <= 0:
        return (d <= r).astype(np.float32)
    return np.clip((r - d) / soft + 0.5, 0, 1).astype(np.float32)


def main() -> None:
    H = W = 128
    rng = np.random.default_rng(0)

    print("\n1. identity: a matte scored against itself must be perfect")
    gt = np.stack([disc(H, W, 64, 64 + i * 0.5, 30, soft=2) for i in range(8)])
    s = A.score(gt, gt, "identity")["whole_frame"]
    check("MAD == 0", s["MAD"] == 0, f"{s['MAD']}")
    check("MSE == 0", s["MSE"] == 0, f"{s['MSE']}")
    check("Grad == 0", s["Grad"] == 0, f"{s['Grad']}")
    check("dtSSD == 0", s["dtSSD"] == 0, f"{s['dtSSD']}")
    check("BF == 1", s["BF"] == 1.0, f"{s['BF']}")

    print("\n2. MAD has a hand-computable value")
    a = np.zeros((1, 10, 10), np.float32)
    b = np.zeros((1, 10, 10), np.float32); b[0, :, :5] = 1.0     # half the pixels differ by 1
    check("MAD == 500 (half the pixels off by 1.0, x1e3)", abs(A.mad(a[0], b[0]) - 500) < 1e-6,
          f"{A.mad(a[0], b[0])}")
    check("MSE == 500 as well when the error is 0 or 1",
          abs(A.mse(a[0], b[0]) - 500) < 1e-6, f"{A.mse(a[0], b[0])}")
    c = np.full((10, 10), 0.5, np.float32)
    check("MAD == 500 for a constant 0.5 error", abs(A.mad(a[0], c) - 500) < 1e-6)
    check("MSE == 250 for a constant 0.5 error (0.25 x 1e3)",
          abs(A.mse(a[0], c) - 250) < 1e-6, f"{A.mse(a[0], c)}")

    print("\n3. scale invariance: uint8 0..255 and float 0..1 must score the same")
    g8 = (gt * 255).round().astype(np.uint8)
    s8 = A.score(g8, gt, "uint8-vs-float")["whole_frame"]
    check("MAD ~ 0 across dtypes", s8["MAD"] < 2.0, f"{s8['MAD']}")

    print("\n4. worse predictions must score worse (monotonicity)")
    base = disc(H, W, 64, 64, 30, soft=2)
    prev = -1.0
    for shift in (0, 1, 3, 8, 20):
        p = disc(H, W, 64, 64 + shift, 30, soft=2)
        m = A.mad(p, base)
        check(f"MAD increases with a {shift}px shift", m >= prev - 1e-9, f"MAD={m:.2f}")
        prev = m

    print("\n5. boundary F: tolerant of a tiny shift, intolerant of a big one")
    bf1 = A.boundary_f(disc(H, W, 64, 65, 30), base)
    bf20 = A.boundary_f(disc(H, W, 64, 84, 30), base)
    check("BF high for a 1px shift", bf1 > 0.9, f"{bf1:.3f}")
    check("BF low for a 20px shift", bf20 < 0.3, f"{bf20:.3f}")
    check("BF(1px) > BF(20px)", bf1 > bf20)
    check("BF == 1 when both mattes are empty",
          A.boundary_f(np.zeros((H, W)), np.zeros((H, W))) == 1.0)
    check("BF == 0 when one matte is empty",
          A.boundary_f(np.zeros((H, W)), base) == 0.0)

    print("\n6. Grad: sensitive to edge softness even when MAD is small")
    hard = disc(H, W, 64, 64, 30, soft=0)
    soft = disc(H, W, 64, 64, 30, soft=6)
    g_hs = A.grad_error(hard, soft)
    check("Grad(hard vs soft) > 0", g_hs > 0, f"{g_hs:.3f}")
    check("Grad(soft vs soft) == 0", A.grad_error(soft, soft) == 0)
    # Grad and MAD live on different scales, so their magnitudes are not comparable.
    # The claim worth testing is relative: for a change that alters the EDGE PROFILE
    # but barely moves the silhouette, Grad should react proportionally more than MAD.
    ref = disc(H, W, 64, 64, 30, soft=2)
    softer = disc(H, W, 64, 64, 30, soft=8)          # same silhouette, softer edge
    shifted = disc(H, W, 64, 65, 30, soft=2)         # same edge profile, moved 1px
    r_grad = A.grad_error(softer, ref) / max(A.grad_error(shifted, ref), 1e-9)
    r_mad = A.mad(softer, ref) / max(A.mad(shifted, ref), 1e-9)
    check("Grad reacts to an edge-profile change relatively more than MAD does",
          r_grad > r_mad, f"Grad ratio={r_grad:.2f} vs MAD ratio={r_mad:.2f}")

    print("\n7. dtSSD: catches flicker the reference does not have")
    still = np.stack([base] * 8)
    flick = np.stack([base if i % 2 == 0 else disc(H, W, 64, 64, 26, soft=2)
                      for i in range(8)])
    check("dtSSD == 0 for a still matte vs itself", A.dtssd(still, still) == 0)
    d_fl = A.dtssd(flick, still)
    check("dtSSD > 0 when the prediction flickers", d_fl > 0, f"{d_fl:.3f}")
    moving = np.stack([disc(H, W, 64, 64 + i, 30, soft=2) for i in range(8)])
    check("dtSSD > 0 when the prediction is too static for a moving reference",
          A.dtssd(still, moving) > 0, f"{A.dtssd(still, moving):.3f}")
    check("dtSSD is symmetric", abs(A.dtssd(flick, still) - A.dtssd(still, flick)) < 1e-9)

    print("\n8. hair-region breakdown must differ from the whole frame")
    pred = base.copy()[None].repeat(4, 0)
    truth = np.stack([disc(H, W, 64, 64, 30, soft=2)] * 4)
    pred[:, :40, :] = 0                      # break only the top strip
    sc = A.score(pred, truth, "region", hair_box=(0, 0, W, 40))
    check("hair-region MAD > whole-frame MAD when only that region is wrong",
          sc["hair_region"]["MAD"] > sc["whole_frame"]["MAD"],
          f"hair={sc['hair_region']['MAD']:.1f} whole={sc['whole_frame']['MAD']:.1f}")
    sc2 = A.score(truth, truth, "clean", hair_box=(0, 0, W, 40))
    check("hair-region == 0 on a perfect match", sc2["hair_region"]["MAD"] == 0)

    print("\n9. shape mismatch must raise, not silently broadcast")
    try:
        A.score(np.zeros((4, 8, 8)), np.zeros((4, 8, 9)))
        check("raises on shape mismatch", False)
    except ValueError:
        check("raises on shape mismatch", True)


    # ------------------------------------------------------------------ Task 6
    rng = np.random.default_rng(6)
    N, H, W = 24, 96, 128

    def moving(speed: float, soft: float = 3.0) -> np.ndarray:
        return np.stack([disc(H, W, 48, 30 + speed * t, 18, soft) for t in range(N)])

    print("\n10. dtSSD-n: a frozen matte scores exactly 1.0, whatever the shot's speed")
    for sp in (0.25, 2.0):
        ref = moving(sp)
        frozen = np.repeat(ref[:1], N, axis=0)
        check(f"frozen matte -> dtSSD-n == 1.0 at speed {sp} px/frame",
              abs(A.dtssd_norm(frozen, ref) - 1.0) < 1e-6, f"{A.dtssd_norm(frozen, ref):.6f}")
    check("perfect matte -> dtSSD-n == 0", A.dtssd_norm(moving(1.0), moving(1.0)) == 0.0)

    print("\n11. the gap dtSSD-n closes: freezing on a slow shot is nearly free in a mean")
    # Within one shot dtSSD-n is dtSSD over a constant, so it cannot reorder methods
    # there. The gap is in aggregation: a slow shot's dtSSD is small whatever you do,
    # so a method that simply freezes on it loses almost nothing in a cross-shot mean
    # dominated by fast shots. (A first draft of this test asserted a within-shot
    # reversal, which is mathematically impossible - kept here as the lesson.)
    slow, fast = moving(0.25), moving(3.0)
    def boil(ref, sd):
        return np.clip(ref + rng.normal(0, sd, ref.shape) * (ref > 0.01), 0, 1)
    A_ = (np.repeat(slow[:1], N, axis=0), boil(fast, 0.01))   # frozen on slow, good on fast
    B_ = (boil(slow, 0.01), boil(fast, 0.04))                  # tracks both, noisier on fast
    mean_d = lambda m: (A.dtssd(m[0], slow) + A.dtssd(m[1], fast)) / 2
    mean_n = lambda m: (A.dtssd_norm(m[0], slow) + A.dtssd_norm(m[1], fast)) / 2
    check("mean plain dtSSD ranks the method that froze FIRST (the known gap)",
          mean_d(A_) < mean_d(B_), f"froze {mean_d(A_):.3f} < tracked {mean_d(B_):.3f}")
    check("mean dtSSD-n exposes the freeze and ranks it last",
          mean_n(A_) > mean_n(B_), f"froze {mean_n(A_):.3f} > tracked {mean_n(B_):.3f}")
    check("and says why: the frozen shot scores exactly 1.0",
          abs(A.dtssd_norm(A_[0], slow) - 1.0) < 1e-6)
    fast = moving(3.0)
    frozen_fast = np.repeat(fast[:1], N, axis=0)
    honest_fast = boil(fast, 0.02)
    check("on a fast shot the tracking matte beats frozen on dtSSD-n",
          A.dtssd_norm(honest_fast, fast) < A.dtssd_norm(frozen_fast, fast),
          f"{A.dtssd_norm(honest_fast, fast):.3f} < {A.dtssd_norm(frozen_fast, fast):.3f}")

    print("\n12. a one-frame lag - the Tier P off-by-one, had it gone unnoticed")
    ref = moving(2.0)
    lag = np.concatenate([ref[:1], ref[:-1]])
    sc = A.score(lag, ref, "lag")
    check("a matte one frame late is not free: MAD > 0 and dtSSD-n > 0",
          sc["whole_frame"]["MAD"] > 0 and sc["whole_frame"]["dtSSDn"] > 0,
          f"MAD {sc['whole_frame']['MAD']:.2f}, dtSSD-n {sc['whole_frame']['dtSSDn']:.3f}")

    print("\n13. ignore: error inside an ignored region costs nothing, outside it does")
    ref = moving(1.0)
    bad = ref.copy(); bad[:, 0:20, 100:120] = 1.0       # a blob the key also holds
    ig = np.zeros(ref.shape, bool); ig[:, 0:20, 100:120] = True
    check("ignored error -> MAD == 0", A.score(bad, ref, "x", ignore=ig)["whole_frame"]["MAD"] == 0)
    check("the same error un-ignored -> MAD > 0", A.score(bad, ref, "x")["whole_frame"]["MAD"] > 0)

    print("\n14. coverage: a hole in the subject is a dropout, an edge shift is not")
    ref = np.stack([disc(H, W, 48, 64, 30, 0.0)] * N)
    holed = ref.copy(); holed[10:14, 40:52, 58:70] = 0.0  # 4 frames, 12x12 hole, deep inside
    shifted = np.stack([disc(H, W, 48, 66, 30, 0.0)] * N) # whole matte 2 px off
    ch, cs = A.coverage(holed, ref), A.coverage(shifted, ref)
    check("interior hole -> 4 dropout frames", ch["dropout_frames"] == 4, str(ch["dropout_frames"]))
    check("2 px edge shift -> 0 misses (edge error is not coverage)", cs["miss_px_max"] == 0,
          str(cs["miss_px_max"]))

    print("\n15. band: detail error shows in the band, interior error does not")
    ref = np.stack([disc(H, W, 48, 64, 30, 6.0)] * N)
    edge_err = np.stack([disc(H, W, 48, 64, 30, 1.0)] * N)   # hard where truth is soft
    sc = A.score(edge_err, ref, "edge")
    check("band MAD > whole-frame MAD for an edge-only error",
          sc["band"]["MAD"] > sc["whole_frame"]["MAD"],
          f"band {sc['band']['MAD']:.1f} vs whole {sc['whole_frame']['MAD']:.1f}")

    print("\n16. soft depth: recovers a known ramp width")
    for ramp in (2.0, 6.0):
        a = disc(200, 200, 100, 100, 60, ramp)
        dd = A.soft_depth(a)
        check(f"ramp {ramp:.0f} px -> median depth about {ramp / 2:.0f} px",
              abs(dd - ramp / 2) <= 1.0, f"{dd:.2f}")

    print("\n" + ("ALL ACCURACY METRICS VALIDATED" if not FAILS
                  else f"{len(FAILS)} FAILURES: {FAILS}"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
