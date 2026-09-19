#!/usr/bin/env python3
"""RotoBench Tier P: standings, the claims ledger, and the evidence behind each verdict.

    python scripts/rotobench_report.py        # -> docs/ROTOBENCH_RESULTS.md (+ json)

Reads outputs/_bench_p (written by scripts/run_rotobench.py) and the Tier P truth.

THE DECISION RULES BELOW WERE WRITTEN AND COMMITTED BEFORE ANY TIER P RESULT WAS
READ. Each prior conclusion gets a verdict by a threshold fixed in advance - HOLDS,
REVERSES, PARTIAL, or NOT RE-TESTED with the reason - so no verdict can have been
chosen to fit the numbers. Where a threshold turns out to be badly placed, the report
says so rather than moving it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate import accuracy, bench, metrics                       # noqa: E402
from cleanplate.paths import ROOT, rel                                # noqa: E402

OUT = ROOT / "outputs" / "_bench_p"
METHODS = ["binary_960", "baseline_960", "matanyone2_960", "guided_960",
           "vitmatte_960", "hairzoom_960", "hairzoom2_960"]
LABEL = {"binary_960": "binary (SAM 2 only)", "baseline_960": "MatAnyone v1",
         "matanyone2_960": "MatAnyone 2", "guided_960": "guided filter",
         "vitmatte_960": "trimap + ViTMatte", "hairzoom_960": "hairzoom (v1)",
         "hairzoom2_960": "hairzoom2 (MA2)"}


# ------------------------------------------------------------------ loading
def results() -> dict:
    """{method: {clip: record}} for every finished, non-aborted job."""
    out = {}
    for m in METHODS + ["cover_960", "cover2_960"]:
        d = OUT / m
        if not d.is_dir():
            continue
        for j in d.glob("P*.json"):
            if "__" in j.stem or j.stem.endswith("_ABORTED"):
                continue
            out.setdefault(m, {})[j.stem] = json.loads(j.read_text())
    return out


def groups() -> dict:
    g = {"core": [], "stress": [], "short": []}
    for c in bench.clips("P"):
        g.setdefault(c.group, []).append(c.name)
    g["all"] = sorted(sum((v for k, v in g.items()), []))
    return g


def pred(m: str, clip: str) -> np.ndarray:
    d = OUT / m / clip
    return np.stack([np.asarray(Image.open(p)) for p in
                     sorted(d.glob("*.png"), key=lambda q: int(q.stem))])


def truth(clip: str, sub: str = "alpha") -> np.ndarray:
    return bench.load_alpha(ROOT / "truth" / clip / sub)


def mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


# ------------------------------------------------------------------ standings
def standings(R: dict, clips: list[str], methods: list[str]) -> list[dict]:
    rows = []
    for m in methods:
        rs = [R[m][c] for c in clips if c in R.get(m, {})]
        if not rs:
            continue
        rows.append({
            "method": m, "clips": len(rs),
            "MAD": mean(r["whole_frame"]["MAD"] for r in rs),
            "band_MAD": mean(r["band"]["MAD"] for r in rs),
            "band_MAD_median": float(np.median([r["band"]["MAD"] for r in rs])),
            "band_Grad": mean(r["band"]["Grad"] for r in rs),
            "hair_MAD": mean(r["hair_region"]["MAD"] for r in rs),
            "BF": mean(r["whole_frame"]["BF"] for r in rs),
            "hair_BF": mean(r["hair_region"]["BF"] for r in rs),
            "dtSSD": mean(r["whole_frame"]["dtSSD"] for r in rs),
            "dtSSDn": mean(r["whole_frame"]["dtSSDn"] for r in rs),
            "dropout_frames": mean(r["coverage"]["dropout_frames"] for r in rs),
            "miss_px": mean(r["coverage"]["miss_px_mean"] for r in rs),
            "s_per_frame": mean(r["seconds_per_frame"] for r in rs),
        })
    # mean rank on band MAD, clip by clip: one clip cannot dominate a rank
    for c in clips:
        have = [(R[m][c]["band"]["MAD"], m) for m in methods if c in R.get(m, {})]
        for k, (_, m) in enumerate(sorted(have), start=1):
            for r in rows:
                if r["method"] == m:
                    r.setdefault("_ranks", []).append(k)
    for r in rows:
        r["mean_rank"] = mean(r.pop("_ranks", []))
    return sorted(rows, key=lambda r: r["mean_rank"])


def standings_md(rows: list[dict]) -> str:
    cols = [("mean rank", "mean_rank", "{:.2f}", "lower"),
            ("band MAD", "band_MAD", "{:.1f}", "lower"),
            ("band MAD median", "band_MAD_median", "{:.1f}", "lower"),
            ("hair MAD", "hair_MAD", "{:.2f}", "lower"),
            ("MAD", "MAD", "{:.2f}", "lower"),
            ("BF", "BF", "{:.3f}", "higher"),
            ("dtSSD-n", "dtSSDn", "{:.3f}", "lower"),
            ("dropout frames", "dropout_frames", "{:.1f}", "lower"),
            ("s/frame", "s_per_frame", "{:.2f}", "lower")]
    head = "| Method | " + " | ".join(c[0] for c in cols) + " |"
    out = [head, "|---|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for _, k, f, better in cols:
            vals = [x[k] for x in rows if np.isfinite(x[k])]
            best = (min(vals) if better == "lower" else max(vals)) if vals else None
            v = r[k]
            s = f.format(v) if np.isfinite(v) else "-"
            cells.append(f"**{s}**" if best is not None and np.isfinite(v)
                         and abs(v - best) < 1e-12 else s)
        out.append(f"| {LABEL.get(r['method'], r['method'])} | " + " | ".join(cells) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------ claims
def soft_depth_at_mid(a: np.ndarray, gt: np.ndarray) -> float:
    """The Task 4 measure: median soft depth in the hair box, at the middle frame."""
    i = len(gt) // 2
    return accuracy.soft_depth(a[i], bench.hair_box(gt))


def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)


def ledger(R: dict, G: dict) -> list[dict]:
    core = G["core"]
    S = {r["method"]: r for r in standings(R, core, METHODS)}
    L = []

    def add(cid, src, claim, test, measured, verdict, note=""):
        L.append({"id": cid, "source": src, "claim": claim, "test": test,
                  "measured": measured, "verdict": verdict, "note": note})

    # C1 - the 2 px edge-depth claim: the REFERENCE's own depth
    d_truth = {c: soft_depth_at_mid(truth(c), truth(c)) for c in core}
    med = float(np.nanmedian(list(d_truth.values())))
    add("C1", "Task 4", "Reference hair edges are about 2 px deep (median 2.0-2.8 px on "
        "Tier A/B).", "Median soft depth of the professional key in the hair box, middle "
        "frame, core clips. HOLDS if <= 3.0 px, REVERSES if >= 4.0 px.",
        f"{med:.2f} px (per clip: " + ", ".join(f"{c[:3]} {v:.1f}" for c, v in
                                                d_truth.items()) + ")",
        "HOLDS" if med <= 3.0 else "REVERSES" if med >= 4.0 else "PARTIAL")

    # C2 - baseline too soft, not too hard
    if "baseline_960" in R:
        soft, hard, rows = 0, 0, []
        for c in core:
            if c not in R["baseline_960"]:
                continue
            db = soft_depth_at_mid(pred("baseline_960", c), truth(c))
            rows.append(f"{c[:3]} {db:.1f} vs {d_truth[c]:.1f}")
            soft += db >= d_truth[c] + 1.0
            hard += db <= d_truth[c] - 1.0
        n = len(rows)
        add("C2", "Task 4 (reversing Task 2)", "The baseline matte is too SOFT (4-5 px "
            "against a 2 px truth), not too hard.",
            "Baseline soft depth vs the professional key's, middle frame, core. HOLDS if "
            "the baseline is >= 1 px softer on a majority; REVERSES if >= 1 px harder on "
            "a majority.", "; ".join(rows),
            "HOLDS" if soft > n / 2 else "REVERSES" if hard > n / 2 else "PARTIAL")

    # C3 - the hairzoom2 win
    if all(m in S for m in ("hairzoom2_960", "baseline_960")):
        best = min(S.values(), key=lambda r: r["hair_MAD"])
        gain = 100 * (S["hairzoom2_960"]["hair_MAD"] / S["baseline_960"]["hair_MAD"] - 1)
        v = ("HOLDS" if best["method"] == "hairzoom2_960" and gain < 0 else
             "PARTIAL" if gain < 0 else "REVERSES")
        add("C3", "Task 4", "hairzoom2 wins the hair region: MAD -37.6% vs the baseline, "
            "with the best MSE, Grad and dtSSD.",
            "Mean hair-region MAD over core clips. HOLDS if hairzoom2 is lowest of all 7 "
            "and beats the baseline; PARTIAL if it beats the baseline only; REVERSES if "
            "not.", f"hairzoom2 {S['hairzoom2_960']['hair_MAD']:.2f} vs baseline "
            f"{S['baseline_960']['hair_MAD']:.2f} ({gain:+.1f}%); lowest: "
            f"{LABEL[best['method']]} {best['hair_MAD']:.2f}", v)

    # C4 - zoom + MA2 beats zoom + v1 on >= 4 of 5, cheaper
    if all(m in R for m in ("hairzoom2_960", "hairzoom_960")):
        keys = ["MAD", "MSE", "Grad", "dtSSD", "BF"]
        wins = 0
        for k in keys:
            a = mean(R["hairzoom2_960"][c]["hair_region"][k] for c in core if c in R["hairzoom2_960"])
            b = mean(R["hairzoom_960"][c]["hair_region"][k] for c in core if c in R["hairzoom_960"])
            wins += (a > b) if k == "BF" else (a < b)
        cheaper = S["hairzoom2_960"]["s_per_frame"] < S["hairzoom_960"]["s_per_frame"]
        add("C4", "Task 4", "Zoom with MatAnyone 2 beats zoom with v1 on 4 of 5 hair "
            "metrics, and costs less.", "Hair-region means over core; count of metrics "
            "won, and s/frame. HOLDS if >= 4 wins and cheaper.",
            f"{wins}/5 won; {S['hairzoom2_960']['s_per_frame']:.2f} vs "
            f"{S['hairzoom_960']['s_per_frame']:.2f} s/frame",
            "HOLDS" if wins >= 4 and cheaper else "PARTIAL" if wins >= 3 else "REVERSES")

    # C5 - MatAnyone 2 beats v1 on every whole-frame metric (the app default)
    if all(m in R for m in ("matanyone2_960", "baseline_960")):
        keys = ["MAD", "MSE", "Grad", "dtSSD", "BF"]
        wins = 0
        for k in keys:
            a = mean(R["matanyone2_960"][c]["whole_frame"][k] for c in core if c in R["matanyone2_960"])
            b = mean(R["baseline_960"][c]["whole_frame"][k] for c in core if c in R["baseline_960"])
            wins += (a > b) if k == "BF" else (a < b)
        add("C5", "Task 4 (app default)", "MatAnyone 2 beats v1 on every whole-frame "
            "metric at essentially the same cost - the reason it is the app default.",
            "Whole-frame means over core. HOLDS if 5/5, PARTIAL if 3-4, REVERSES if <= 2.",
            f"{wins}/5 won", "HOLDS" if wins == 5 else "PARTIAL" if wins >= 3 else "REVERSES")

    # C6 - guided filter worse than nothing on hair
    if all(m in S for m in ("guided_960", "baseline_960")):
        g, b = S["guided_960"]["hair_MAD"], S["baseline_960"]["hair_MAD"]
        add("C6", "Task 4", "The guided filter is worse than doing nothing on hair.",
            "Hair MAD, guided vs baseline, core. HOLDS if guided is worse.",
            f"guided {g:.2f} vs baseline {b:.2f}", "HOLDS" if g > b else "REVERSES")

    # C7 - ViTMatte best hair boundary-F
    if "vitmatte_960" in S:
        best = max(S.values(), key=lambda r: r["hair_BF"])
        add("C7", "Task 4", "Trimap + ViTMatte has the best hair boundary-F.",
            "Mean hair-region BF over core. HOLDS if ViTMatte is highest.",
            f"ViTMatte {S['vitmatte_960']['hair_BF']:.4f}; highest: "
            f"{LABEL[best['method']]} {best['hair_BF']:.4f}",
            "HOLDS" if best["method"] == "vitmatte_960" else "REVERSES")

    # C8, C9 - B1: the same actor and hair, now with professional truth (P01)
    p01 = "P01_08_3a"
    if all(p01 in R.get(m, {}) for m in ("hairzoom2_960", "baseline_960")):
        hz, bl = (R[m][p01]["hair_region"]["MAD"] for m in ("hairzoom2_960", "baseline_960"))
        gain = 100 * (hz / bl - 1)
        add("C8", "Task 4", "On real backlit hair (B1) the winner gains about half what "
            "synthetic clips suggest: -19.7% against -44%.",
            "hairzoom2 vs baseline hair MAD on P01 (08_3a: the same actor and hair as B1, "
            "professional key). HOLDS if the gain is between -10% and -30%.",
            f"{gain:+.1f}% ({bl:.2f} -> {hz:.2f})",
            "HOLDS" if -30 <= gain <= -10 else "REVERSES" if gain > -5 else "PARTIAL")
        inh = {m: json.loads((OUT / m / f"{p01}__inhouse.json").read_text())
               for m in ("hairzoom2_960", "baseline_960")
               if (OUT / m / f"{p01}__inhouse.json").exists()}
        note = ""
        if len(inh) == 2:
            g2 = 100 * (inh["hairzoom2_960"]["hair_region"]["MAD"]
                        / inh["baseline_960"]["hair_region"]["MAD"] - 1)
            note = f"Same predictions against the in-house key: {g2:+.1f}%."
        add("C9", "Task 5", "HQ (MatAnyone 2 + zoom) recovers about 12% of hair MAD on "
            "B1 against the baseline.",
            "The same comparison on P01 with the professional key. HOLDS if the gain is "
            "between -6% and -18%.", f"{gain:+.1f}%",
            "HOLDS" if -18 <= gain <= -6 else "REVERSES" if gain > -3 else "PARTIAL", note)

    # C10 - self-consistency cannot rank accuracy
    sc = self_consistency(R, core)
    if sc:
        ms = [m for m in METHODS if m in sc]
        rho = spearman([sc[m]["iou_consecutive"] for m in ms],
                       [-S[m]["band_MAD"] for m in ms])
        frozen_top = sc.get("frozen", {}).get("iou_consecutive", 0) >= max(
            sc[m]["iou_consecutive"] for m in ms)
        add("C10", "Task 4", "Self-consistency cannot tell you which matte is right.",
            "Spearman correlation, across the 7 methods on core clips, between "
            "consecutive-frame IoU (self-consistency) and band MAD (truth). HOLDS if "
            "|rho| < 0.5, or if a frozen matte - wrong by construction - tops "
            "self-consistency.", f"rho = {rho:+.2f}; frozen control tops IoU: "
            f"{'yes' if frozen_top else 'no'}",
            "HOLDS" if abs(rho) < 0.5 or frozen_top else "REVERSES")

    # C11 - dropout is segmentation, matting does not cause it
    if all(m in R for m in ("binary_960", "baseline_960", "matanyone2_960")):
        worse = []
        for c in G["all"]:
            if all(c in R[m] for m in ("binary_960", "baseline_960", "matanyone2_960")):
                b = R["binary_960"][c]["coverage"]["dropout_frames"]
                r = max(R[m][c]["coverage"]["dropout_frames"]
                        for m in ("baseline_960", "matanyone2_960"))
                worse.append(r > b)
        n = len(worse)
        add("C11", "Task 5", "Dropouts are a segmentation failure; the matting stage does "
            "not cause them.", "Dropout frames (interior misses >= 4 px inside the key's "
            "core), SAM 2 alone vs after MatAnyone v1/v2, every clip. HOLDS if refinement "
            "adds dropout frames on at most a third of clips.",
            f"refinement adds dropout frames on {sum(worse)}/{n} clips",
            "HOLDS" if sum(worse) <= n / 3 else "PARTIAL" if sum(worse) <= 2 * n / 3
            else "REVERSES")

    # C12 - the in-house key as a ranking reference
    rs = inhouse_agreement(R)
    if rs:
        add("C12", "Task 4 (implicit)", "A simple keyer is a good enough reference to "
            "rank methods (Tier B was built on that assumption).",
            "P01, identical predictions scored against the professional key and against "
            "the in-house key with the Tier B settings. Spearman rho of the 7-method "
            "hair-MAD ranking. HOLDS if rho >= 0.8.",
            f"rho = {rs['rho']:+.2f}; key-vs-key MAD {rs['key_vs_key_MAD']:.2f}, "
            f"soft px {100 * rs['soft_pro']:.2f}% (pro) vs {100 * rs['soft_in']:.2f}% "
            f"(in-house)", "HOLDS" if rs["rho"] >= 0.8 else "REVERSES")

    add("C13", "Task 5", "Better clicks do not fix the face dropout; a corrective click "
        "made it worse.", "Would need prompt variations on Tier P.", "-",
        "NOT RE-TESTED", "Tier P prompts are frozen oracle prompts by design; varying "
        "them to test this would break the pre-registration that makes the standings "
        "credible. It stays a Task 5 finding on the user's own shot.")
    return L


def self_consistency(R: dict, clips: list[str]) -> dict:
    """Mean consecutive-frame IoU and flicker per method, plus a frozen control.

    Measured at 960 wide (a stability measure; resolution does not change it), which
    keeps this pass to minutes. The control is hairzoom2's frame 0 repeated: a matte
    that never moves, as self-consistent as a matte can be, and wrong whenever the
    subject moves.
    """
    import cv2
    cache = OUT / "_selfconsistency.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out: dict = {}
    for m in METHODS + ["frozen"]:
        src = "hairzoom2_960" if m == "frozen" else m
        vals, flick, acc = [], [], []
        for c in clips:
            if c not in R.get(src, {}):
                continue
            a = pred(src, c)
            if m == "frozen":
                a = np.repeat(a[:1], len(a), axis=0)
                gt = truth(c)
                ig = bench.load_ignore(ROOT / "truth" / c / "ignore", len(gt), gt.shape[1:])
                acc.append(accuracy.score(a, gt, "frozen", ignore=ig)["band"]["MAD"])
            small = np.stack([cv2.resize(x, (960, 506), interpolation=cv2.INTER_AREA)
                              for x in a])
            mm = metrics.measure(small, m)
            vals.append(mm["iou_consecutive"]["mean"])
            flick.append(float(mm["alpha_l1_change_mean"]))
        if vals:
            out[m] = {"iou_consecutive": mean(vals), "alpha_l1_change": mean(flick),
                      "clips": len(vals)}
            if acc:
                out[m]["band_MAD"] = mean(acc)
    cache.write_text(json.dumps(out, indent=2))
    return out


def inhouse_agreement(R: dict) -> dict | None:
    p01 = "P01_08_3a"
    pairs = []
    for m in METHODS:
        j = OUT / m / f"{p01}__inhouse.json"
        if j.exists() and p01 in R.get(m, {}):
            pairs.append((R[m][p01]["hair_region"]["MAD"],
                          json.loads(j.read_text())["hair_region"]["MAD"]))
    if len(pairs) < 4:
        return None
    pro, inh = truth(p01), truth(p01, "alpha_inhouse")
    kk = accuracy.score(inh, pro, "in-house vs pro")
    soft = lambda a: float(((a > 5) & (a < 250)).mean())
    return {"rho": spearman([p for p, _ in pairs], [q for _, q in pairs]),
            "key_vs_key_MAD": kk["whole_frame"]["MAD"],
            "key_vs_key_band_MAD": kk["band"]["MAD"],
            "soft_pro": soft(pro), "soft_in": soft(inh),
            "depth_pro": soft_depth_at_mid(pro, pro), "depth_in": soft_depth_at_mid(inh, pro)}


# ------------------------------------------------------------------ checkpoint 4
COVER = ["cover_960", "cover2_960"]
COVER_BASE = "hairzoom2_960"          # the app's default output since Task 5

# Integration rule for the coverage repair, fixed BEFORE any cover_* result existed.
# It ships behind an app toggle only if, over all Tier P clips:
RULE = {"dropout_frames_cut_pct": 25.0,   # dropout frames fall by at least this much
        "recovered_frac_min": 0.50,        # at least half the base's dropout px recovered
        "core_band_mad_worse_pct": 2.0,    # core band MAD no more than this much worse
        "false_per_recovered_max": 0.10}   # false fill added <= 10% of coverage recovered


def dropout_breakdown(R: dict, G: dict) -> dict:
    """Where the base matte dropped out, how much did each repair bring back, and at
    what cost elsewhere. D = pixels >= 4 px inside the key's opaque core where the
    BASE is below 0.5, per frame."""
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    out = {}
    for m in COVER:
        if m not in R:
            continue
        tot_d = tot_rec = tot_false = tot_alpha = 0.0
        per = {}
        for c in G["all"]:
            if c not in R[m] or c not in R.get(COVER_BASE, {}):
                continue
            gt, base, x = truth(c), pred(COVER_BASE, c), pred(m, c)
            d_n = rec = fal = asum = 0.0
            for g, b, y in zip(gt, base, x):
                core = cv2.erode((g >= 250).astype(np.uint8), k) > 0
                clear = cv2.erode((g <= 5).astype(np.uint8), k) > 0
                D = core & (b < 128)
                d_n += D.sum(); rec += (D & (y >= 128)).sum(); asum += y[D].sum() / 255.0
                fal += (clear & (y >= 128) & (b < 128)).sum()
            per[c] = {"dropout_px": int(d_n), "recovered_px": int(rec),
                      "false_px_added": int(fal)}
            tot_d += d_n; tot_rec += rec; tot_false += fal; tot_alpha += asum
        df_b = sum(R[COVER_BASE][c]["coverage"]["dropout_frames"] for c in per)
        df_m = sum(R[m][c]["coverage"]["dropout_frames"] for c in per)
        core = [c for c in G["core"] if c in R[m]]
        bm_b = mean(R[COVER_BASE][c]["band"]["MAD"] for c in core)
        bm_m = mean(R[m][c]["band"]["MAD"] for c in core)
        res = {"clips": len(per), "base_dropout_px": int(tot_d),
               "recovered_px": int(tot_rec),
               "recovered_frac": tot_rec / tot_d if tot_d else float("nan"),
               "mean_alpha_in_dropouts": tot_alpha / tot_d if tot_d else float("nan"),
               "false_px_added": int(tot_false),
               "dropout_frames_base": int(df_b), "dropout_frames": int(df_m),
               "dropout_frames_cut_pct": 100 * (1 - df_m / df_b) if df_b else float("nan"),
               "core_band_mad_change_pct": 100 * (bm_m / bm_b - 1),
               "per_clip": per}
        checks = {
            "dropout frames cut >= 25%": res["dropout_frames_cut_pct"] >= RULE["dropout_frames_cut_pct"],
            "recovered >= 50% of base dropouts": res["recovered_frac"] >= RULE["recovered_frac_min"],
            "core band MAD worse by <= 2%": res["core_band_mad_change_pct"] <= RULE["core_band_mad_worse_pct"],
            "false fill <= 10% of recovered": (tot_false <= RULE["false_per_recovered_max"] * tot_rec)
                                              if tot_rec else False}
        res["checks"] = checks
        res["integrate"] = all(checks.values())
        out[m] = res
    return out


def main() -> None:
    R, G = results(), groups()
    L = ledger(R, G)
    md = ["# RotoBench Tier P - results", "",
          "Generated by `scripts/rotobench_report.py` from `outputs/_bench_p`. Decision "
          "rules for the claims ledger were committed before any result was read.", ""]
    for g in ("core", "stress", "short", "all"):
        rows = standings(R, G[g], METHODS + [m for m in COVER if m in R])
        if rows:
            md += [f"## Standings - {g} ({len(G[g])} clips)", "", standings_md(rows), ""]
    md += ["## Claims ledger", "", "| # | From | Claim | Test | Measured | Verdict |",
           "|---|---|---|---|---|---|"]
    for c in L:
        md.append(f"| {c['id']} | {c['source']} | {c['claim']} | {c['test']} | "
                  f"{c['measured']}{(' ' + c['note']) if c['note'] else ''} | "
                  f"**{c['verdict']}** |")
    DB = dropout_breakdown(R, G)
    if DB:
        md += ["", "## Checkpoint 4 - the coverage repair", "",
               "Base: " + COVER_BASE + ". Dropout pixels D: >= 4 px inside the key's opaque "
               "core where the base is below 0.5. Rule fixed before any result: "
               + json.dumps(RULE), "",
               "| Repair | base dropout px | recovered | mean alpha in D | false fill added "
               "| dropout frames | core band MAD | integrate? |", "|---|---|---|---|---|---|---|---|"]
        for m, r in DB.items():
            md.append(f"| {m} | {r['base_dropout_px']:,} | {100 * r['recovered_frac']:.1f}% "
                      f"| {r['mean_alpha_in_dropouts']:.2f} | {r['false_px_added']:,} "
                      f"| {r['dropout_frames_base']} -> {r['dropout_frames']} "
                      f"({r['dropout_frames_cut_pct']:+.0f}% cut) "
                      f"| {r['core_band_mad_change_pct']:+.1f}% | "
                      f"{'**yes**' if r['integrate'] else 'no: ' + ', '.join(k for k, v in r['checks'].items() if not v)} |")
        (OUT / "dropout_breakdown.json").write_text(json.dumps(DB, indent=2, default=float))
    (ROOT / "docs" / "ROTOBENCH_RESULTS.md").write_text("\n".join(md) + "\n")
    (OUT / "ledger.json").write_text(json.dumps(L, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
