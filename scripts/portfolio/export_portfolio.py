#!/usr/bin/env python3
"""Export CleanPlate's demo assets for the portfolio site (revamp Gate 1).

    .venv/bin/python scripts/portfolio/export_portfolio.py --portfolio ~/Documents/Portfolio_v2
    .venv/bin/python scripts/portfolio/export_portfolio.py --portfolio ... --skip-render

Writes into the portfolio:

  public/demos/cleanplate/
    pair.webm, pair.mp4              the walk plate (left half) and the comp over Mars
                                     (right half) side by side in one file, 8 s: one
                                     decoder, so the portfolio's compare slider can never
                                     drift (revamp Gate 4B)
    pair-poster.jpg                  the first frame of both halves
    stills/1-click.jpg               the click, on frame 0
    stills/2-mask.png                SAM 2's binary mask, frame 0
    stills/3-matte.png               the high-quality soft alpha, frame 0
    stills/4-comp.jpg                the comp, frame 0
    stills/2-mask-detail.png, stills/3-matte-detail.png
                                     the head region of 2 and 3, where the edge differs
    hair-pair.webm/.mp4, hair-pair-poster.jpg
                                     the app's two refine passes on RotoBench clip P01
                                     (backlit hair), standard left and HQ right, 4 s
  content/data/
    rotobench.json                   Tier P standings and the stability-vs-truth table,
                                     computed by scripts/rotobench_report.py's functions
    cleanplate.demo.json             what was rendered, how, how long it took, credits

The demo shot is the README's walk shot (Tears of Steel at 270 s) run for 8 s instead of
4. The take is one continuous Steadicam move from 257.9 s to past 278 s, so nothing is cut
or looped to reach the length. The clicks are the published walk clicks
(shots/walk/point.json) scaled to 1280 px, plus corrections on frames 96, 105 and 150 for the
half the 4 s shot never reached (EXTRA_PROMPT). The matte is RotoBench's best method,
hairzoom2_960: SAM 2 and MatAnyone 2 at 960 px, the hair zoom on the head, the alpha
upscaled to the 1280 px plate, with MatAnyone re-anchored as the jacket back comes into
view (REANCHOR). The comp is src/comp_preview.py with harmonize v0 and the
background track, over the Mars panorama.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cleanplate import compose                                        # noqa: E402
from cleanplate.paths import out_dir, rel                             # noqa: E402

SHOT = "portfolio_walk"
START, SECONDS, WIDTH, FPS = 270.0, 8.0, 1280, 24
PUBLISHED_SHOT = "walk"               # the clicks come from here, scaled
BG = ROOT / "datasets" / "backgrounds" / "mars_curiosity_360_pano.jpg"
# The background has to pan with the plate: about 3,400 px in x and 170 in y over this
# shot at 1280 px. The widest band of the panorama that is free of stitching seams (flat
# black) and left of the rover, at the aspect that travel needs, found by search: 1870x260.
BG_CROP = "0,160,1870,420"
HAIR_CLIP = "P01_08_3a"
HAIR_HEAD = (300, 40, 1180, 600)      # scripts/reel_prep.py's head crop, 1920x1012 space
HAIR_BACKDROP = (18, 21, 28)          # scripts/reel_prep.py's neutral card
HAIR_PASSES = {"standard": "matanyone2_960", "hq": "hairzoom2_960"}
# The 8 s run needs corrections the 4 s shot never did. From about frame 90 the camera
# orbit shows the back of the jacket (a grey shoulder panel and a yellow print) that no
# conditioning frame described, and SAM 2 leaves it out: coverage, not edges
# (docs/KNOWN_ISSUES.md). Two frames, each describing the whole subject as the README asks:
# head, shirt, sleeve, shoulder panel, print. In 1280 px coordinates, like the rest.
EXTRA_PROMPT = {
    96: {"positive": [(540, 100), (540, 330), (480, 330), (465, 225), (435, 400)],
         "negative": []},
    105: {"positive": [(530, 120), (550, 350), (480, 330), (450, 228), (422, 400)],
          "negative": []},
    150: {"positive": [(650, 130), (665, 380), (600, 380), (575, 260), (535, 420)],
          "negative": []},
}
# MatAnyone anchors on frame 0 and propagates its own memory, so a corrective click never
# reaches it on its own (docs/DECISIONS.md, "First-frame-only anchoring"). Re-injecting the
# corrected SAM 2 mask is refine_matte.py's --reanchor. The jacket back comes into view
# from about frame 88, so re-anchoring only on the correction frames dropped it in at
# frame 105 as a 21% jump in matte area. Re-anchoring at 90 and 100 as well still let it
# thin out over 92 to 98, where SAM 2 itself lost the panel; the frame-96 correction holds it.
REANCHOR = [90, 96, 100, 105, 150]
KEEP_GREEN = (61, 220, 132)           # the app's "keep" click colour
MAX_LOOP_BYTES = 2_500_000            # portfolio brief: hard cap per loop


def run(cmd: list[str]) -> str:
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    p = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    if p.returncode:
        sys.stderr.write(p.stderr)
        raise SystemExit(f"failed: {cmd[1] if len(cmd) > 1 else cmd[0]}")
    return p.stdout


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                          text=True).stdout.strip()


# ------------------------------------------------------------------ render
def render() -> dict:
    import shutil

    from cleanplate import methods, refine, runs
    from cleanplate.ingest import frame_size
    from cleanplate.session import Prompt

    if out_dir(SHOT).exists():
        shutil.rmtree(out_dir(SHOT))
    py = sys.executable
    run([py, "src/extract_shot.py", "--name", SHOT, "--start", str(START),
         "--duration", str(SECONDS), "--width", str(WIDTH), "--force"])
    w, h = frame_size(SHOT)

    pub = json.loads((ROOT / "shots" / PUBLISHED_SHOT / "point.json").read_text())
    k = w / pub["image_size"][0]
    prompt = Prompt()
    for fp in pub["prompts"]:
        for x, y in fp["positive"]:
            prompt.add(fp["frame"], round(x * k), round(y * k), True)
        for x, y in fp["negative"]:
            prompt.add(fp["frame"], round(x * k), round(y * k), False)
    for f, clicks in EXTRA_PROMPT.items():
        for x, y in clicks["positive"]:
            prompt.add(f, x, y, True)
        for x, y in clicks["negative"]:
            prompt.add(f, x, y, False)
    prompt.save(SHOT, image_size=(w, h),
                note=f"Frames 0 and 67: the published walk clicks "
                     f"(shots/{PUBLISHED_SHOT}/point.json) scaled by {k:.4f} to {w} px. The "
                     "matte runs at 960 px, where they map back to the published clicks "
                     "exactly; same start frame, so the frame-67 correction lands on the "
                     "same picture. Frames 96, 105 and 150: corrections for the second half, "
                     "which the 4 s shot never reached. The orbit shows the back of the "
                     "jacket and SAM 2 left it out, so each frame describes the whole "
                     "subject: head, shirt, sleeve, shoulder panel, print. MatAnyone is "
                     f"re-anchored on frames {REANCHOR} so the panel grows in.")

    # RotoBench's best method, hairzoom2_960, step by step as methods.hair_zoom runs it:
    # SAM 2 and MatAnyone 2 at 960 px, the alpha upscaled to the plate (bilinear, the way
    # RotoBench scores it), then the hair zoom on the full-resolution head. The one
    # addition is REANCHOR. Tracking at the plate's own 1280 px is not an option: SAM 2's
    # multi-mask head picks a different granularity at a different resolution
    # (methods.fullres_matte), and here the same click selected only the T-shirt.
    frames = ROOT / "shots" / SHOT / "frames"
    t0 = time.perf_counter()
    fdir, scale = methods._scaled_frames(frames, 960)
    masks960, tstats = methods._cached_track(fdir, methods._scale_prompt(prompt, scale), "auto")
    alpha960, rstats = refine.refine(fdir, masks960, model="matanyone2", reanchor=REANCHOR)
    alphas = methods._upscale(alpha960, (w, h))
    alphas, zstats = methods.refine_hair_zoom(frames, alphas, zoom=2.0, model="matanyone2")
    masks = methods._upscale(masks960.astype(np.uint8) * 255, (w, h)) > 127
    stats = {"method": "hairzoom2_960 + reanchor", "width": 960, "scale": round(scale, 4),
             "track_s_per_frame": tstats["seconds_per_frame"], "track_device": tstats["device"],
             "track_cached": bool(tstats.get("cached")),
             "refine_model": rstats["model"], "refine_device": rstats["device"],
             "refine_s_per_frame": rstats["seconds_per_frame"],
             "reanchor_frames": rstats["reanchor_frames"], **zstats,
             "total_s": round(time.perf_counter() - t0, 1)}
    runs.save_masks(SHOT, masks)
    runs.save_alpha(SHOT, alphas, stats)
    print(f"[matte] hairzoom2_960 + reanchor {REANCHOR} on {len(alphas)} frames in "
          f"{stats['total_s']:.0f} s: track {stats['track_s_per_frame']:.3f}, refine "
          f"{stats['refine_s_per_frame']:.3f}, hair zoom {stats['hairzoom_s_per_frame']:.3f} "
          "s/frame")

    run([py, "src/export_rgba.py", "--shot", SHOT, "--out-name", SHOT,
         "--alpha", rel(out_dir(SHOT) / "alpha"), "--despill", "green"])
    comp_log = run([py, "src/comp_preview.py", "--shot", SHOT, "--out-name", SHOT,
                    "--bg-image", rel(BG), "--bg-crop", BG_CROP, "--track-bg"])
    (out_dir(SHOT) / "comp_log.txt").write_text(comp_log)
    return {}


# ------------------------------------------------------------------ encode
def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-count_frames", "-show_entries",
                          "stream=width,height,nb_read_frames:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    return {"width": s["width"], "height": s["height"],
            "frames": int(s["nb_read_frames"]),
            "seconds": round(float(j["format"]["duration"]), 3)}


def encode(patterns: list[Path], out_base: Path) -> list[dict]:
    """WebM VP9 (crf 33, constant quality, two-pass) and MP4 H.264 (crf 28, faststart),
    no audio. Two or more frame sequences are stacked side by side (hstack) into one file,
    so a player decodes them as one stream and the halves can never drift apart."""
    src: list[str] = []
    for pattern in patterns:
        src += ["-framerate", str(FPS), "-start_number", "0", "-i", str(pattern)]
    n = len(patterns)
    # Square pixels: the plate's JPEG frames carry an 801:800 sample aspect from the
    # source scale, which would make the halves meet a pixel or two off the middle.
    # Limited (tv) range: the JPEG frames are full range, and a full-range VP9 stream
    # fails in Chrome's macOS hardware decoder (VideoToolbox, "video decode error"),
    # so the pair froze on its poster there. The other loops are limited range too.
    vf = ["-filter_complex",
          ("".join(f"[{i}:v]" for i in range(n)) + f"hstack=inputs={n}," if n > 1 else "")
          + "setsar=1,scale=out_range=tv,format=yuv420p"]
    # cpu-used 0 (the slowest, best-compressing effort) at the same crf keeps the 2560 px
    # pair under the cap without lowering quality.
    vp9 = ["-c:v", "libvpx-vp9", "-crf", "33", "-b:v", "0", "-pix_fmt", "yuv420p", "-color_range", "tv",
           "-row-mt", "1", "-deadline", "good", "-cpu-used", "0"]
    made = []
    out = out_base.with_suffix(".webm")
    log = out_base.with_name(f".{out_base.name}-vp9")
    for p in ("1", "2"):
        dest = ["-f", "null", "-"] if p == "1" else [str(out)]
        subprocess.run(["ffmpeg", "-v", "error", "-y", *src, *vf, "-an", *vp9, "-pass", p,
                        "-passlogfile", str(log), *dest], check=True)
    for f in out_base.parent.glob(f".{out_base.name}-vp9*"):
        f.unlink()
    mp4 = out_base.with_suffix(".mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", *src, *vf, "-an", "-c:v", "libx264",
                    "-crf", "28", "-preset", "slow", "-profile:v", "high", "-pix_fmt",
                    "yuv420p", "-color_range", "tv", "-movflags", "+faststart", str(mp4)],
                   check=True)
    for f in (out, mp4):
        size = f.stat().st_size
        if size > MAX_LOOP_BYTES:
            raise SystemExit(f"{f.name} is {size:,} bytes, over the 2.5 MB loop cap")
        made.append({"file": f.name, "bytes": size, **probe(f)})
    return made


def side_by_side(*imgs: Image.Image) -> Image.Image:
    """The first frames of a pair, side by side like the loop: a poster for it."""
    out = Image.new("RGB", (sum(i.width for i in imgs), max(i.height for i in imgs)))
    x = 0
    for img in imgs:
        out.paste(img.convert("RGB"), (x, 0))
        x += img.width
    return out


def save_jpg(img: Image.Image, path: Path, q: int = 85) -> None:
    img.convert("RGB").save(path, quality=q, optimize=True, progressive=True)


def click_marker(img: Image.Image, xy: tuple[int, int]) -> Image.Image:
    """Keep-click marker: dark halo, white ring, green dot. Drawn 4x and downsampled."""
    s = 4
    big = img.convert("RGB").resize((img.width * s, img.height * s), Image.LANCZOS)
    layer = Image.new("RGBA", big.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x, y = xy[0] * s, xy[1] * s
    for rad, fill in ((17, (0, 0, 0, 90)), (13, (255, 255, 255, 255)), (9, KEEP_GREEN + (255,))):
        d.ellipse([x - rad * s, y - rad * s, x + rad * s, y + rad * s], fill=fill)
    big = Image.alpha_composite(big.convert("RGBA"), layer)
    return big.resize(img.size, Image.LANCZOS)


def head_box(alpha: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    """The pipeline's own hair region on this frame, widened to a 4:3 detail crop."""
    from cleanplate import methods
    l, t, r, b = methods._hair_box_from_alpha(alpha[None])
    cx, cy = (l + r) / 2, (t + b) / 2
    bw = max(r - l, (b - t) * 4 / 3) * 1.15
    bh = bw * 3 / 4
    l = int(max(0, min(w - bw, cx - bw / 2))); t = int(max(0, min(h - bh, cy - bh / 2)))
    return l, t, int(l + bw), int(t + bh)


def export(portfolio: Path) -> None:
    demos = portfolio / "public" / "demos" / "cleanplate"
    data = portfolio / "content" / "data"
    (demos / "stills").mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    out = out_dir(SHOT)
    frames_dir = ROOT / "shots" / SHOT / "frames"
    n = len(list(frames_dir.glob("*.jpg")))
    n_comp = len(list((out / "comp_image").glob("*.png")))
    if n != round(SECONDS * FPS) or n_comp != n:
        raise SystemExit(f"expected {round(SECONDS * FPS)} plate and comp frames, "
                         f"found {n} and {n_comp}: run without --skip-render")

    # Revamp Gate 4B: the plate and the comp in one file, plate on the left. The two
    # used to be separate loops kept in step by re-seeking, which drifted on iPhones.
    for old in ("plate.webm", "plate.mp4", "comp.webm", "comp.mp4", "plate-poster.jpg",
                "poster.jpg", "hair-standard.webm", "hair-standard.mp4", "hair-hq.webm",
                "hair-hq.mp4", "hair-poster.jpg"):
        (demos / old).unlink(missing_ok=True)
    assets = encode([frames_dir / "%05d.jpg", out / "comp_image" / "%05d.png"], demos / "pair")
    if {(a["width"], a["frames"]) for a in assets} != {(2 * WIDTH, n)}:
        raise SystemExit(f"the pair should be {2 * WIDTH} px wide and {n} frames: {assets}")

    plate0 = Image.open(frames_dir / "00000.jpg").convert("RGB")
    comp0 = Image.open(out / "comp_image" / "00000.png").convert("RGB")
    save_jpg(side_by_side(plate0, comp0), demos / "pair-poster.jpg")

    # Four stages, one frame: the click, SAM 2's mask, the soft alpha, the comp.
    prompt = json.loads((ROOT / "shots" / SHOT / "point.json").read_text())
    f0 = next(p for p in prompt["prompts"] if p["frame"] == 0)
    still = click_marker(plate0, tuple(f0["positive"][0]))
    save_jpg(still, demos / "stills" / "1-click.jpg")
    mask0 = Image.open(out / "masks" / "00000.png").convert("L")
    alpha0 = Image.open(out / "alpha" / "00000.png").convert("L")
    mask0.save(demos / "stills" / "2-mask.png", optimize=True)
    alpha0.save(demos / "stills" / "3-matte.png", optimize=True)
    save_jpg(comp0, demos / "stills" / "4-comp.jpg")
    box = head_box(np.asarray(alpha0), *alpha0.size)
    mask0.crop(box).save(demos / "stills" / "2-mask-detail.png", optimize=True)
    alpha0.crop(box).save(demos / "stills" / "3-matte-detail.png", optimize=True)

    # The app's refine toggle on backlit hair: standard pass against the HQ pass, both
    # despilled over a neutral card exactly as scripts/reel_prep.py builds the reel's.
    hair_frames = sorted((ROOT / "truth" / HAIR_CLIP / "frames").glob("*.jpg"),
                         key=lambda p: int(p.stem))
    tmp = out / "_hair_portfolio"
    for name, method in HAIR_PASSES.items():
        d = tmp / name
        d.mkdir(parents=True, exist_ok=True)
        apaths = sorted((ROOT / "outputs" / "_bench_p" / method / HAIR_CLIP).glob("*.png"),
                        key=lambda p: int(p.stem))
        if len(apaths) != len(hair_frames):
            raise SystemExit(f"{method}/{HAIR_CLIP}: {len(apaths)} alphas, "
                             f"{len(hair_frames)} frames")
        l, t, r, b = HAIR_HEAD
        for i, (fp, ap) in enumerate(zip(hair_frames, apaths)):
            rgb = np.asarray(Image.open(fp).convert("RGB"))[t:b, l:r]
            al = np.asarray(Image.open(ap).convert("L"))[t:b, l:r]
            rgb, _ = compose.despill_green(rgb, al)
            bg = np.zeros_like(rgb); bg[:] = HAIR_BACKDROP
            Image.fromarray(compose.over(rgb, al, bg)).save(d / f"{i:05d}.png")
    # Standard on the left, HQ on the right, one file (Gate 4B).
    assets += encode([tmp / "standard" / "%05d.png", tmp / "hq" / "%05d.png"],
                     demos / "hair-pair")
    save_jpg(side_by_side(Image.open(tmp / "standard" / "00000.png"),
                          Image.open(tmp / "hq" / "00000.png")), demos / "hair-pair-poster.jpg")

    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain").splitlines()
    write_rotobench(data / "rotobench.json", commit)
    write_demo_json(data / "cleanplate.demo.json", demos, assets, commit, box)

    print(f"\nwrote {rel(demos) if demos.is_relative_to(ROOT) else demos}")
    for p in sorted(demos.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(demos)}  {p.stat().st_size:,} bytes")
    total = sum(p.stat().st_size for p in demos.rglob("*") if p.is_file())
    print(f"  total {total / 1e6:.2f} MB")
    print(f"wrote {data / 'rotobench.json'} and {data / 'cleanplate.demo.json'}")
    print(f"source commit: {commit}" + (f" (uncommitted changes: {dirty})" if dirty else ""))


# ------------------------------------------------------------------ data
def write_rotobench(path: Path, commit: str) -> None:
    import rotobench_report as rb

    R, G = rb.results(), rb.groups()
    core = rb.paired(R, rb.METHODS, G["core"])
    methods = rb.METHODS + [m for m in rb.COVER if m in R]
    rows = rb.standings(R, core, methods)
    keep = ["clips", "mean_rank", "band_MAD", "band_MAD_median", "hair_MAD", "MAD", "BF",
            "dtSSDn", "dropout_frames", "s_per_frame"]
    standings = [{"id": r["method"], "method": rb.LABEL.get(r["method"], r["method"]),
                  "kind": "coverage_repair" if r["method"] in rb.COVER else "matting",
                  **{k: round(float(r[k]), 4) for k in keep}} for r in rows]

    # Same construction as rotobench_report.selfcheck_table (docs/ROTOBENCH.md).
    sc = rb.self_consistency(R, G["core"])
    S = {r["method"]: r for r in rb.standings(R, core, rb.METHODS)}
    pairs = [(m, sc[m]["iou_consecutive"], S[m]["band_MAD"]) for m in rb.METHODS
             if m in sc and m in S]
    by_self = {m: i for i, (m, _, _) in enumerate(sorted(pairs, key=lambda p: -p[1]), 1)}
    by_truth = {m: i for i, (m, _, _) in enumerate(sorted(pairs, key=lambda p: p[2]), 1)}
    stability = [{"id": m, "method": rb.LABEL.get(m, m),
                  "iou_consecutive": round(iou, 4), "band_MAD": round(bm, 1),
                  "rank_self_consistency": by_self[m], "rank_truth": by_truth[m]}
                 for m, iou, bm in sorted(pairs, key=lambda p: -p[1])]
    frozen = sc.get("frozen")
    rho = rb.spearman([p[1] for p in pairs], [-p[2] for p in pairs])

    ledger = json.loads((rb.OUT / "ledger.json").read_text())
    verdicts: dict[str, int] = {}
    for c in ledger:
        verdicts[c["verdict"]] = verdicts.get(c["verdict"], 0) + 1
    c3 = next(c for c in ledger if c["id"] == "C3")

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "repo": "github.com/shauryadata/cleanplate",
            "commit": commit,
            "published_in": ["docs/ROTOBENCH.md", "docs/ROTOBENCH_RESULTS.md"],
            "computed_by": "scripts/rotobench_report.py (results, paired, standings, "
                           "self_consistency) over outputs/_bench_p",
        },
        "benchmark": {
            "name": "RotoBench v1",
            "tier": "P",
            "reference": "The Tears of Steel compositing team's own keys, each proven "
                         "frame-aligned with its plate before scoring. A professional "
                         "key, still a key: not hand-painted roto or rendered truth.",
            "clips": {"total": len(G["all"]), "core": len(G["core"]),
                      "stress": len(G["stress"]), "short": len(G["short"])},
            "working_width_px": 960,
            "scored_at": "1920x1012",
            "prompts": "Oracle prompts derived from each reference's first frame, "
                       "committed before any method ran.",
        },
        "metrics": {
            "band_MAD": "Mean absolute alpha error (x10^3) in the reference's transition "
                        "band, where mattes differ. Lower is better.",
            "hair_MAD": "Mean absolute alpha error (x10^3) in the top 30% of the "
                        "subject's box. Lower is better.",
            "MAD": "Mean absolute alpha error (x10^3), whole frame. Lower is better.",
            "BF": "Boundary F-measure on the binarised matte. Higher is better.",
            "dtSSDn": "Temporal error divided by the reference's own motion; a frozen "
                      "matte scores 1.0. Lower is better.",
            "dropout_frames": "Frames losing more than 0.05% of the reference's opaque "
                              "core. Lower is better.",
            "s_per_frame": "Apple M3 Pro, 18 GB, MPS.",
            "mean_rank": "Mean rank of band MAD across clips, so no single clip decides "
                         "the order.",
        },
        "standings": {
            "group": "core",
            "clips": len(core),
            "note": f"Core clips where every method ran ({len(core)} of "
                    f"{len(G['core'])}; the hair zoom's size cap excluded the rest).",
            "rows": standings,
            "coverage_repair_note": "cover_960 and cover2_960 are Task 6 coverage repairs, "
                                    "scored against a rule fixed before they ran. They "
                                    "recover 0.4% and 1.9% of the missing coverage; "
                                    "neither was integrated.",
        },
        "stability_vs_truth": {
            "note": "Consecutive-frame IoU measures self-consistency, not accuracy. The "
                    "frozen control is the best method's frame 0 repeated: perfectly "
                    "stable and wrong.",
            "rows": stability,
            "frozen_control": {"iou_consecutive": round(frozen["iou_consecutive"], 4),
                               "band_MAD": round(frozen["band_MAD"], 1)} if frozen else None,
            "spearman_rho": round(rho, 2),
        },
        "claims_ledger": {
            "total": len(ledger),
            "verdicts": verdicts,
            "hair_claim": {"id": c3["id"], "claim": c3["claim"],
                           "measured": c3["measured"], "verdict": c3["verdict"]},
        },
        "footnote": "Scored against Tier P keys: the Tears of Steel compositing team's own "
                    "keys (Blender Foundation, CC BY 3.0), frame-aligned per clip. "
                    f"{len(core)} core clips where every method ran.",
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"[rotobench] {len(standings)} standings rows over {len(core)} core clips, "
          f"{len(stability)} stability rows, rho {rho:+.2f}, ledger {verdicts}")


def write_demo_json(path: Path, demos: Path, assets: list[dict], commit: str,
                    detail_box: tuple[int, int, int, int]) -> None:
    out = out_dir(SHOT)
    shot = json.loads((ROOT / "shots" / SHOT / "shot.json").read_text())
    prompt = json.loads((ROOT / "shots" / SHOT / "point.json").read_text())
    st = json.loads((out / "refine.json").read_text())     # the method's own stats
    log = (out / "comp_log.txt").read_text() if (out / "comp_log.txt").exists() else ""
    pan = re.search(r"plate moves (-?\d+)\.\.(-?\d+) px in x, (-?\d+)\.\.(-?\d+) in y", log)
    harm = re.search(r"harmonize (\w+) @([\d.]+): gain (\[[^\]]*\]) offset (\[[^\]]*\]) "
                     r"grain \+([\d.]+)", log)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"repo": "github.com/shauryadata/cleanplate", "commit": commit,
                   "script": "scripts/portfolio/export_portfolio.py"},
        "shot": {
            "film": "Tears of Steel",
            "start_s": float(shot["start"]), "seconds": shot["duration_s"],
            "frames": shot["frame_count"], "fps": shot["fps"],
            "resolution": shot["resolution"],
            "note": "The README's walk shot run for 8 s instead of 4. One continuous "
                    "Steadicam take: no cut and no loop inside the clip.",
        },
        "prompt": {"frames": prompt["prompts"], "note": prompt.get("_note")},
        "method": "hairzoom2_960 (cleanplate.methods), RotoBench's best on Tier P, with "
                  f"MatAnyone re-anchored on frames {st['reanchor_frames']}",
        "pipeline": [
            {"stage": "track", "model": "SAM 2.1 hiera-small", "width_px": st["width"],
             "device": st["track_device"], "s_per_frame": round(st["track_s_per_frame"], 3)},
            {"stage": "refine", "model": st["refine_model"], "width_px": st["width"],
             "device": st["refine_device"], "s_per_frame": round(st["refine_s_per_frame"], 3)},
            {"stage": "hair zoom (high quality)",
             "model": f"{st['hairzoom_model']} at {st['zoom']:g}x on the head",
             "s_per_frame": round(st["hairzoom_s_per_frame"], 3)},
            {"stage": "upscale", "model": f"alpha to {WIDTH} px, bilinear, as RotoBench "
                                          "scores it"},
            {"stage": "despill + RGBA", "model": "cleanplate.compose (green)"},
            {"stage": "comp", "model": "harmonize v0 + background track",
             "harmonize": ({"mode": harm[1], "strength": float(harm[2]),
                            "gain": json.loads(harm[3]), "offset": json.loads(harm[4]),
                            "grain_sigma": float(harm[5])} if harm else None),
             "background_pan_px": ({"x": [int(pan[1]), int(pan[2])],
                                    "y": [int(pan[3]), int(pan[4])]} if pan else None),
             "background_crop": [int(v) for v in BG_CROP.split(",")]},
        ],
        "assets": {
            "loops": assets,
            "posters": ["pair-poster.jpg", "hair-pair-poster.jpg"],
            "pair": {"files": ["pair.webm", "pair.mp4"],
                     "left": "the plate", "right": "the comp", "half_width_px": WIDTH},
            "stills": {"frame": 0, "files": ["stills/1-click.jpg", "stills/2-mask.png",
                                             "stills/3-matte.png", "stills/4-comp.jpg"],
                       "detail_files": ["stills/2-mask-detail.png",
                                        "stills/3-matte-detail.png"],
                       "detail_box": list(detail_box)},
            "hair_toggle": {"clip": f"RotoBench {HAIR_CLIP} (backlit hair)",
                            "files": ["hair-pair.webm", "hair-pair.mp4"],
                            "left": "standard", "right": "hq",
                            "standard": f"{HAIR_PASSES['standard']}: MatAnyone 2, the "
                                        "app's default refine",
                            "hq": f"{HAIR_PASSES['hq']}: MatAnyone 2 + hair zoom, the "
                                  "app's high-quality toggle",
                            "backdrop_rgb": list(HAIR_BACKDROP)},
        },
        "credits": [
            "Tears of Steel footage: (CC) Blender Foundation | mango.blender.org, CC BY 3.0.",
            "Mars background: NASA/JPL-Caltech, Curiosity 360-degree panorama, public "
            "domain. NASA does not endorse this project.",
            "SAM 2.1 (Meta, Apache 2.0) for tracking; MatAnyone 2 (S-Lab License 1.0, "
            "non-commercial use only) for the soft matte. CleanPlate's own code is MIT.",
        ],
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--portfolio", required=True, type=Path)
    ap.add_argument("--skip-render", action="store_true",
                    help="re-encode and re-export from the existing render")
    a = ap.parse_args()
    if not a.skip_render:
        render()
    export(a.portfolio.expanduser())


if __name__ == "__main__":
    main()
