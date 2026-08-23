#!/usr/bin/env python3
"""CleanPlate - local web app.

    python app.py

Runs entirely on your machine. Nothing is uploaded: no share link, no telemetry, no
network calls at runtime. The server binds to loopback only. The one deliberate
omission is the Image component's default "share" button, which posts to Hugging
Face Spaces Discussions.
"""
from __future__ import annotations

import argparse
import os

# Disable analytics before gradio is imported, as well as via the Blocks argument.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# SAM 2 and MatAnyone both drive tqdm internally; the app has its own progress UI,
# so the console bars are pure noise.
os.environ.setdefault("TQDM_DISABLE", "1")

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cleanplate import __version__, compose, ingest, metrics, paths, refine, runs, \
    track, viewer                                                       # noqa: E402
from cleanplate.session import Prompt                                    # noqa: E402

DEMO_SHOT = "walk"
CHANGED_IOU = 0.99            # below this, a frame counts as changed by a re-run
VIEW_MODES = ["Plate", "Matte overlay", "Matte", "RGBA on checkerboard", "Comp"]
THEME = gr.themes.Base(primary_hue="emerald", neutral_hue="slate")

CSS = """
:root { --cp-dim:#8b93a7; --cp-line:#262b36; }
.gradio-container { max-width: 1620px !important; }
#cp-head { display:flex; align-items:baseline; gap:.75rem; flex-wrap:wrap;
           padding:.2rem 0 .6rem; border-bottom:1px solid var(--cp-line);
           margin-bottom:.9rem; }
#cp-head h1 { font-size:1.35rem; font-weight:650; margin:0; letter-spacing:-.01em; }
#cp-head .tag { font-size:.78rem; color:var(--cp-dim); }
#cp-head .badge { font-size:.7rem; letter-spacing:.06em; text-transform:uppercase;
                  border:1px solid #2c6e49; color:#7ee2a8; background:#12241a;
                  padding:.16rem .5rem; border-radius:999px; margin-left:auto; }
.cp-hint { color:var(--cp-dim); font-size:.82rem; line-height:1.45; }
.cp-warn { border-left:3px solid #d97706; padding:.55rem .8rem; background:#241c0c;
           border-radius:4px; font-size:.85rem; line-height:1.5; }
.cp-bad  { border-left:3px solid #b91c1c; padding:.55rem .8rem; background:#2a1414;
           border-radius:4px; font-size:.85rem; line-height:1.5; }
.cp-ok   { border-left:3px solid #2c6e49; padding:.55rem .8rem; background:#12241a;
           border-radius:4px; font-size:.85rem; line-height:1.5; }
.cp-cached { border-left:3px solid #6366f1; padding:.55rem .8rem; background:#181a2e;
             border-radius:4px; font-size:.85rem; }
/* six tabs must fit without an overflow menu - the Changes tab is the point of
   the app and must not hide behind a "..." */
button.tab-nav-button, .tab-nav button { padding-left:.55rem !important;
    padding-right:.55rem !important; font-size:.86rem !important; }
.cp-scroll { max-height: 340px; overflow-y: auto; }
.cp-scroll table { font-size:.8rem; }
footer { display:none !important; }
"""


# ------------------------------------------------------------------ state
def blank_state() -> dict:
    return {"shot": None, "prompt": Prompt(), "masks": None, "alphas": None,
            "prev_masks": None, "ran_prompt_frames": [], "track_stats": None,
            "refine_stats": None, "rgba": None, "comp": None, "timings": [],
            "last_iou": None, "correction_frame": None, "despilled": 0}


def active_alpha(st: dict) -> np.ndarray | None:
    """Whatever the current best matte is: refined if present, else binary."""
    if st.get("alphas") is not None:
        return st["alphas"]
    if st.get("masks") is not None:
        return (st["masks"].astype(np.uint8) * 255)
    return None


def need_shot(st: dict) -> None:
    if not st or not st.get("shot"):
        raise gr.Error("Load a shot first.")


# ------------------------------------------------------------------ rendering
def render(st: dict, idx: int, mode: str, bg_choice: str, bg_colour: str,
           bg_image) -> np.ndarray | None:
    if not st or not st.get("shot"):
        return None
    idx = int(idx)
    rgb = ingest.load_frame(st["shot"], idx)
    alpha = active_alpha(st)
    a = alpha[idx] if alpha is not None else None

    if mode == "Plate" or a is None:
        out = rgb
    elif mode == "Matte overlay":
        out = viewer.overlay(rgb, a)
    elif mode == "Matte":
        out = viewer.matte_view(a)
    elif mode == "RGBA on checkerboard":
        src = st["rgba"][idx][..., :3] if st.get("rgba") is not None else rgb
        out = viewer.checker_view(src, a)
    elif mode == "Comp":
        src = st["rgba"][idx][..., :3] if st.get("rgba") is not None else rgb
        out = compose.over(src, a, background_plate(st, bg_choice, bg_colour, bg_image))
    else:
        out = rgb

    if mode in ("Plate", "Matte overlay"):
        e = st["prompt"].frames.get(idx)
        if e:
            out = viewer.draw_points(out, e.positive, e.negative)
    return out


def background_plate(st: dict, bg_choice: str, bg_colour: str, bg_image) -> np.ndarray:
    w, h = ingest.frame_size(st["shot"])
    if bg_choice == "Image" and bg_image is not None:
        from PIL import Image as PILImage
        im = bg_image if isinstance(bg_image, PILImage.Image) else PILImage.fromarray(
            np.asarray(bg_image))
        return compose.cover_fit(im.convert("RGB"), (w, h))
    try:
        rgb = compose.hex_rgb(bg_colour or "#D4571E")
    except ValueError:
        rgb = (212, 87, 30)
    return compose.solid((w, h), rgb)


def points_table(st: dict) -> list[list]:
    rows = []
    for f in st["prompt"].prompt_frames:
        e = st["prompt"].frames[f]
        for x, y in e.positive:
            rows.append([f, "keep", x, y])
        for x, y in e.negative:
            rows.append([f, "exclude", x, y])
    return rows


def prompt_summary(st: dict) -> str:
    p = st["prompt"]
    if not p.total_clicks:
        return "<div class='cp-hint'>No points yet. Click the subject on any frame.</div>"
    lonely = p.negative_only_frames()
    bits = [f"**{p.total_clicks}** click(s) across frame(s) `{p.prompt_frames}`"]
    if lonely:
        bits.append(
            f"<div class='cp-bad'><b>Blocked:</b> frame(s) {lonely} have only "
            "<i>exclude</i> points. SAM&nbsp;2 reads a conditioning frame with no keep "
            "point as “object absent” and will blank that frame. Add a keep click on "
            "the subject there.</div>")
    if len(p.prompt_frames) > 1:
        bits.append(
            "<div class='cp-hint'>Corrective frames are <b>global</b>: SAM&nbsp;2 "
            "attends to every prompt frame at every timestep, so a click on a later "
            "frame can change earlier ones. The Corrections tab shows exactly what "
            "moved.</div>")
    return "  \n".join(bits)


def timings_md(st: dict) -> str:
    if not st.get("timings"):
        return "<div class='cp-hint'>No stages run yet.</div>"
    rows = ["| Stage | Device | s/frame | total | MPS fallback ops |", "|---|---|---|---|---|"]
    for t in st["timings"]:
        rows.append(f"| {t['stage']} | `{t['device']}` | {t['s_per_frame']:.3f} | "
                    f"{t['total_s']:.1f}s | {t['fallback'] or 'none'} |")
    return "\n".join(rows)


# ------------------------------------------------------------------ callbacks
def load_shot(shot: str, st: dict, progress=gr.Progress()):
    if not shot or shot not in paths.list_shots():
        raise gr.Error("Pick a shot that exists under shots/.")

    # A fresh clone has shot.json but no frames (footage is gitignored). Rebuild the
    # cut from the recorded ffmpeg parameters so the app works without the CLI.
    if not paths.is_built(shot):
        meta_path = paths.shot_dir(shot) / "shot.json"
        if not meta_path.exists():
            raise gr.Error(f"Shot {shot!r} has no frames and no shot.json to rebuild "
                           "from.")
        meta = json.loads(meta_path.read_text())
        if not paths.SOURCE_MOVIE.exists():
            raise gr.Error(
                "The source movie is missing, so this shot cannot be built.\n"
                "Run  ./scripts/download.sh footage  in the repo, then click Load again."
                f"\n(Expected at {paths.rel(paths.SOURCE_MOVIE)}; "
                "(CC) Blender Foundation | mango.blender.org)")
        progress(0.2, desc=f"cutting {shot} out of the source movie with ffmpeg")
        try:
            ingest.extract_shot(shot, meta["start"], meta.get("duration_s", 4.0),
                                width=meta.get("resolution", [960])[0],
                                fps=meta.get("fps", 24.0), force=True)
        except Exception as e:
            raise gr.Error(f"Could not build shot {shot!r}: {e}")

    n = ingest.n_frames(shot)
    st = blank_state()
    st["shot"] = shot
    st["prompt"] = Prompt.load(shot)
    meta = {}
    sj = paths.shot_dir(shot) / "shot.json"
    if sj.exists():
        meta = json.loads(sj.read_text())
    w, h = ingest.frame_size(shot)
    info = [f"**{shot}** · {n} frames · {w}×{h}"]
    if meta:
        info.append(f"cut at t={meta.get('start')} for {meta.get('duration_s')}s "
                    f"@ {meta.get('fps')}fps")
        if meta.get("credit"):
            info.append(f"_{meta['credit']}_")
    msg = (f"Loaded **{shot}** — {n} frames. "
           + (f"Restored {st['prompt'].total_clicks} saved click(s) on frames "
              f"{st['prompt'].prompt_frames} from `point.json`."
              if st["prompt"].total_clicks else "No saved clicks yet."))
    return (st, render(st, 0, "Plate", "Solid colour", "#D4571E", None),
            gr.update(maximum=n - 1, value=0, interactive=True),
            "  \n".join(info), msg, points_table(st), prompt_summary(st),
            gr.update(value="Plate"))


def on_view_change(idx, st, mode, bgc, bgcol, bgimg):
    return render(st, idx, mode, bgc, bgcol, bgimg)


def on_click(st: dict, idx: float, mode: str, click_mode: str,
             bgc, bgcol, bgimg, evt: gr.SelectData):
    if not st or not st.get("shot"):
        raise gr.Error("Load a shot first.")
    if mode not in ("Plate", "Matte overlay"):
        raise gr.Error("Switch the view to Plate or Matte overlay to place points — "
                       "the other views are not in plate coordinates.")
    x, y = int(evt.index[0]), int(evt.index[1])
    st["prompt"].add(int(idx), x, y, positive=click_mode.startswith("Keep"))
    return (st, render(st, idx, mode, bgc, bgcol, bgimg),
            points_table(st), prompt_summary(st),
            f"{'Keep' if click_mode.startswith('Keep') else 'Exclude'} point at "
            f"({x}, {y}) on frame {int(idx)}.")


def on_undo(st, idx, mode, bgc, bgcol, bgimg):
    need_shot(st)
    p = st["prompt"].undo(int(idx))
    msg = f"Removed {p} from frame {int(idx)}." if p else \
        f"No points on frame {int(idx)} to undo."
    return (st, render(st, idx, mode, bgc, bgcol, bgimg),
            points_table(st), prompt_summary(st), msg)


def on_clear_frame(st, idx, mode, bgc, bgcol, bgimg):
    need_shot(st)
    st["prompt"].clear_frame(int(idx))
    return (st, render(st, idx, mode, bgc, bgcol, bgimg),
            points_table(st), prompt_summary(st), f"Cleared frame {int(idx)}.")


def on_clear_all(st, idx, mode, bgc, bgcol, bgimg):
    need_shot(st)
    st["prompt"].clear()
    return (st, render(st, idx, mode, bgc, bgcol, bgimg),
            points_table(st), prompt_summary(st), "Cleared every point.")


def run_track(st: dict, idx, mode, bgc, bgcol, bgimg, progress=gr.Progress()):
    need_shot(st)
    try:
        track.validate_prompt(st["prompt"], ingest.n_frames(st["shot"]))
    except track.PromptError as e:
        raise gr.Error(str(e))

    prev_masks = st.get("masks")
    prev_frames = list(st.get("ran_prompt_frames") or [])

    def cb(frac: float, msg: str) -> None:
        progress(min(max(frac, 0.0), 1.0), desc=msg)

    t0 = time.perf_counter()
    try:
        masks, stats = track.track(st["shot"], st["prompt"], progress=cb)
    except (FileNotFoundError, RuntimeError) as e:
        raise gr.Error(str(e))
    wall = time.perf_counter() - t0

    st["masks"] = masks
    st["alphas"] = None                     # a new track invalidates the old refine
    st["rgba"] = None
    st["track_stats"] = stats
    st["prev_masks"] = prev_masks
    st["timings"] = [t for t in st["timings"] if t["stage"] != "Track (SAM 2)"]
    st["timings"].insert(0, {"stage": "Track (SAM 2)", "device": stats["device"],
                             "s_per_frame": stats["seconds_per_frame"],
                             "total_s": wall,
                             "fallback": ", ".join(stats["mps_fallback_ops"])})

    now_frames = st["prompt"].prompt_frames
    new_frames = sorted(set(now_frames) - set(prev_frames))
    st["correction_frame"] = min(new_frames) if (prev_masks is not None and new_frames) \
        else None
    st["ran_prompt_frames"] = now_frames

    chart, report = correction_report(st)
    msg = (f"Tracked {len(masks)} frames in {stats['propagate_s']}s "
           f"({stats['seconds_per_frame']:.3f} s/frame, {stats['fps']} fps) on "
           f"`{stats['device']}`. MPS fallback ops: "
           f"{stats['mps_fallback_ops'] or 'none'}.")
    return (st, render(st, idx, mode, bgc, bgcol, bgimg), msg,
            gr.update(value="Matte overlay"), chart, report, timings_md(st))


def correction_report(st: dict):
    """Per-frame IoU against the previous run, with the retroactive check."""
    prev, cur = st.get("prev_masks"), st.get("masks")
    if prev is None or cur is None:
        return None, ("<div class='cp-hint'>Run Track twice to compare. After a re-run "
                      "this panel shows, frame by frame, exactly what your new points "
                      "changed — including frames <i>before</i> the frame you clicked."
                      "</div>")
    if prev.shape != cur.shape:
        return None, "<div class='cp-warn'>Shot changed between runs; nothing to compare.</div>"

    iou = metrics.per_frame_iou(prev, cur)
    st["last_iou"] = iou
    cf = st.get("correction_frame")
    changed = np.where(iou < CHANGED_IOU)[0]
    chart = viewer.iou_chart(iou, cf, CHANGED_IOU)

    lines = [f"**{len(changed)}** of {len(iou)} frames changed "
             f"(IoU &lt; {CHANGED_IOU}). Mean IoU {iou.mean():.4f}, "
             f"min {iou.min():.4f} at frame {int(iou.argmin())}."]
    if cf is None:
        lines.append("<div class='cp-hint'>No new prompt frame in this run, so there is "
                     "no correction frame to measure against.</div>")
    else:
        retro = [int(i) for i in changed if i < cf]
        if retro:
            worst = int(min(retro, key=lambda i: iou[i]))
            lines.append(
                f"<div class='cp-warn'><b>Retroactive change: {len(retro)} frame(s) "
                f"before your click on frame {cf} moved.</b><br/>"
                f"Range f{min(retro)}–f{max(retro)}, worst f{worst} at IoU "
                f"{iou[worst]:.3f}.<br/>SAM&nbsp;2 puts every conditioning frame in the "
                "memory bank at <i>every</i> timestep, so a corrective click redefines "
                "the object for the whole shot. If those earlier frames got worse, the "
                "click on frame "
                f"{cf} probably does not describe the whole subject — add keep points "
                "covering its other parts on that frame.</div>")
        else:
            lines.append(
                f"<div class='cp-ok'>No retroactive change: every frame that moved is at "
                f"or after your click on frame {cf}.</div>")
    return chart, "\n\n".join(lines)


def run_refine(st: dict, warmup, dilate, erode, idx, mode, bgc, bgcol, bgimg,
               progress=gr.Progress()):
    need_shot(st)
    if st.get("masks") is None:
        raise gr.Error("Run Track first — refinement needs a binary mask to anchor on.")
    ok, why = refine.available()
    if not ok:
        raise gr.Error(why)

    def cb(frac: float, msg: str) -> None:
        progress(min(max(frac, 0.0), 1.0), desc=msg)

    t0 = time.perf_counter()
    try:
        alphas, stats = refine.refine(st["shot"], st["masks"], warmup=int(warmup),
                                      dilate=int(dilate), erode=int(erode), progress=cb)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        raise gr.Error(str(e))
    wall = time.perf_counter() - t0

    st["alphas"] = alphas
    st["rgba"] = None
    st["refine_stats"] = stats
    st["timings"] = [t for t in st["timings"] if t["stage"] != "Refine (MatAnyone)"]
    st["timings"].append({"stage": "Refine (MatAnyone)", "device": stats["device"],
                          "s_per_frame": stats["seconds_per_frame"], "total_s": wall,
                          "fallback": ", ".join(stats["mps_fallback_ops"])})
    msg = (f"Refined {len(alphas)} frames in {stats['refine_s']}s "
           f"({stats['seconds_per_frame']:.3f} s/frame, {stats['fps']} fps) on "
           f"`{stats['device']}`. Soft pixels: "
           f"{100 * stats['soft_pixel_fraction']['mean']:.3f}% of frame "
           f"(binary matte is exactly 0%).")
    return (st, render(st, idx, mode, bgc, bgcol, bgimg), msg,
            gr.update(value="RGBA on checkerboard"), timings_md(st))


def run_despill(st: dict, do_despill, strength, band, idx, mode, bgc, bgcol, bgimg,
                progress=gr.Progress()):
    need_shot(st)
    alpha = active_alpha(st)
    if alpha is None:
        raise gr.Error("Run Track first.")
    t0 = time.perf_counter()
    out, total = [], 0
    n = len(alpha)
    for i in range(n):
        rgb = ingest.load_frame(st["shot"], i)
        if do_despill:
            rgb, k = compose.despill_green(rgb, alpha[i], float(strength), int(band))
            total += k
        out.append(compose.to_rgba(rgb, alpha[i]))
        progress((i + 1) / n, desc=f"building RGBA {i + 1}/{n}")
    wall = time.perf_counter() - t0
    st["rgba"] = np.stack(out)
    st["despilled"] = total
    st["timings"] = [t for t in st["timings"] if t["stage"] != "RGBA + despill"]
    st["timings"].append({"stage": "RGBA + despill", "device": "cpu",
                          "s_per_frame": wall / n, "total_s": wall, "fallback": ""})
    msg = (f"Built {n} RGBA frames in {wall:.1f}s. "
           + (f"Green despill altered {total} pixels "
              f"({total / n:.0f}/frame, inside a ±{int(band)}px edge band)."
              if do_despill else "Despill off."))
    return (st, render(st, idx, mode, bgc, bgcol, bgimg), msg,
            gr.update(value="RGBA on checkerboard"), timings_md(st))


def metrics_table(st: dict) -> str:
    if st.get("masks") is None:
        return "<div class='cp-hint'>Run Track to get numbers.</div>"
    runs_ = [metrics.measure(st["masks"], "binary (SAM 2)")]
    if st.get("alphas") is not None:
        runs_.append(metrics.measure(st["alphas"], "refined (MatAnyone)"))
    return metrics.table(runs_)


# ------------------------------------------------------------------ export
def _staging() -> Path:
    d = Path(tempfile.gettempdir()) / "cleanplate_exports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def export_bundle(st: dict, what: str, fps: float, bgc, bgcol, bgimg,
                  progress=gr.Progress()):
    need_shot(st)
    alpha = active_alpha(st)
    if alpha is None:
        raise gr.Error("Run Track first — there is nothing to export yet.")
    stamp = f"{st['shot']}_{int(time.time())}"
    stage = _staging() / stamp
    stage.mkdir(parents=True, exist_ok=True)

    if what == "Matte PNG sequence (zip)":
        progress(0.3, desc="writing mattes")
        d = stage / "matte"
        compose.write_sequence(list(alpha), d, mode="L")
        out = runs.zip_dir(d, _staging() / f"{stamp}_matte.zip", f"{st['shot']}_matte")
    elif what == "RGBA PNG sequence (zip)":
        if st.get("rgba") is None:
            raise gr.Error("Build the RGBA first, on the Refine & despill tab.")
        progress(0.3, desc="writing RGBA")
        d = stage / "rgba"
        compose.write_sequence(list(st["rgba"]), d, mode="RGBA")
        out = runs.zip_dir(d, _staging() / f"{stamp}_rgba.zip", f"{st['shot']}_rgba")
    elif what == "metrics.json":
        payload = {"shot": st["shot"],
                   "track": st.get("track_stats"), "refine": st.get("refine_stats"),
                   "binary": metrics.measure(st["masks"], "binary (SAM 2)")}
        if st.get("alphas") is not None:
            payload["refined"] = metrics.measure(st["alphas"], "refined (MatAnyone)")
        out = _staging() / f"{stamp}_metrics.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
    elif what in ("comp.mp4", "matte.mp4", "side_by_side.mp4"):
        n = len(alpha)
        plate = background_plate(st, bgc, bgcol, bgimg)
        frames_dir = stage / "vid"
        frames_dir.mkdir(parents=True, exist_ok=True)
        from PIL import Image as PILImage
        for i in range(n):
            src = st["rgba"][i][..., :3] if st.get("rgba") is not None \
                else ingest.load_frame(st["shot"], i)
            if what == "matte.mp4":
                fr = viewer.matte_view(alpha[i])
            elif what == "comp.mp4":
                fr = compose.over(src, alpha[i], plate)
            else:
                orig = ingest.load_frame(st["shot"], i)
                fr = np.hstack([orig, viewer.matte_view(alpha[i]),
                                compose.over(src, alpha[i], plate)])
            PILImage.fromarray(fr).save(frames_dir / f"{i:05d}.png")
            progress((i + 1) / n * 0.8, desc=f"rendering {i + 1}/{n}")
        progress(0.9, desc="encoding with ffmpeg")
        out = compose.encode(frames_dir, _staging() / f"{stamp}_{what}", float(fps))
    else:
        raise gr.Error(f"unknown export {what!r}")

    shutil.rmtree(stage, ignore_errors=True)
    size = out.stat().st_size / 1e6
    return gr.update(value=str(out), visible=True), \
        f"Ready: `{out.name}` ({size:.1f} MB). Click to save."


def save_session(st: dict):
    need_shot(st)
    if not st["prompt"].total_clicks:
        raise gr.Error("No points to save.")
    w, h = ingest.frame_size(st["shot"])
    p = st["prompt"].save(st["shot"], (w, h),
                          note="Written by the CleanPlate app. Reload with the "
                               "Load button, or run src/track_matte.py --shot "
                               f"{st['shot']} to reproduce from the CLI.")
    return (f"<div class='cp-ok'>Saved <code>{paths.rel(p)}</code> — "
            f"{st['prompt'].total_clicks} click(s) on frames "
            f"{st['prompt'].prompt_frames}. This run is now reproducible: "
            f"<code>python src/track_matte.py --shot {st['shot']}</code>.</div>")


def reload_session(st: dict, idx, mode, bgc, bgcol, bgimg):
    need_shot(st)
    st["prompt"] = Prompt.load(st["shot"])
    return (st, render(st, idx, mode, bgc, bgcol, bgimg), points_table(st),
            prompt_summary(st),
            f"<div class='cp-ok'>Reloaded {st['prompt'].total_clicks} click(s) from "
            f"<code>shots/{st['shot']}/point.json</code>.</div>")


# ------------------------------------------------------------------ UI
def build() -> gr.Blocks:
    ok_refine, why_refine = refine.available()
    shots = paths.list_shots()

    with gr.Blocks(title="CleanPlate", analytics_enabled=False) as demo:
        st = gr.State(blank_state())

        gr.HTML("<div id='cp-head'><h1>CleanPlate</h1>"
                "<span class='tag'>click once, get a production matte · "
                f"v{__version__}</span>"
                "<span class='badge'>runs locally · nothing uploaded</span></div>")

        with gr.Row():
            # ---------------- viewer
            with gr.Column(scale=5):
                view_mode = gr.Radio(VIEW_MODES, value="Plate", label="View",
                                     interactive=True)
                viewer_img = gr.Image(label="Frame", type="numpy", height=470,
                                      format="png", interactive=False,
                                      buttons=["download", "fullscreen"])
                frame_slider = gr.Slider(0, 1, value=0, step=1, label="Frame",
                                         interactive=False)
                status = gr.Markdown("")

            # ---------------- controls
            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.Tab("Shot"):
                        shot_dd = gr.Dropdown(shots or ["(none found)"],
                                              value=DEMO_SHOT if DEMO_SHOT in shots
                                              else (shots[0] if shots else None),
                                              label="Available shots")
                        load_btn = gr.Button(f"Load demo shot ({DEMO_SHOT})",
                                             variant="primary")
                        info_md = gr.Markdown("_No shot loaded._")
                        gr.Markdown(
                            "<div class='cp-hint'>Shots are folders of numbered JPEGs "
                            "under <code>shots/</code>. Cut a new one with "
                            "<code>src/extract_shot.py</code>. Nothing leaves this "
                            "machine.</div>")

                    with gr.Tab("Prompt"):
                        click_mode = gr.Radio(["Keep (subject)", "Exclude (not subject)"],
                                              value="Keep (subject)",
                                              label="What does a click mean?")
                        gr.Markdown("<div class='cp-hint'>Click on the image to place a "
                                    "point on the <b>current frame</b>. Use View = Plate "
                                    "or Matte overlay.</div>")
                        with gr.Row():
                            undo_btn = gr.Button("Undo last", size="sm")
                            clearf_btn = gr.Button("Clear frame", size="sm")
                            clear_btn = gr.Button("Clear all", size="sm")
                        pts_df = gr.Dataframe(headers=["frame", "kind", "x", "y"],
                                              datatype=["number", "str", "number",
                                                        "number"],
                                              label="Points", interactive=False,
                                              row_count=(0, "dynamic"))
                        prompt_md = gr.Markdown("")
                        track_btn = gr.Button("Run Track (SAM 2)", variant="primary")

                    with gr.Tab("Refine"):
                        gr.Markdown(
                            "<div class='cp-hint'>MatAnyone re-solves the boundary as a "
                            "matting problem, turning the binary cutout into fractional "
                            "alpha.</div>" if ok_refine else
                            f"<div class='cp-warn'>Refine unavailable — {why_refine}"
                            "</div>")
                        with gr.Row():
                            warmup_n = gr.Slider(0, 20, value=10, step=1, label="Warmup")
                            dilate_n = gr.Slider(0, 20, value=10, step=1, label="Dilate")
                            erode_n = gr.Slider(0, 20, value=10, step=1, label="Erode")
                        refine_btn = gr.Button("Run Refine (MatAnyone)",
                                               variant="primary", interactive=ok_refine)
                        gr.Markdown("---")
                        despill_cb = gr.Checkbox(True, label="Green despill in the edge band")
                        with gr.Row():
                            despill_s = gr.Slider(0, 1, value=1.0, step=0.05,
                                                  label="Strength")
                            despill_b = gr.Slider(0, 6, value=2, step=1, label="Band px")
                        rgba_btn = gr.Button("Build RGBA")

                    with gr.Tab("Backdrop"):
                        bg_choice = gr.Radio(["Solid colour", "Image"],
                                             value="Solid colour", label="Comp over")
                        bg_colour = gr.ColorPicker("#D4571E", label="Colour")
                        gr.Markdown("<div class='cp-hint'>A warm flat colour is the edge "
                                    "stress test: cool halos from foliage or pavement "
                                    "show up instantly.</div>")
                        bg_image = gr.Image(label="Background plate", type="pil",
                                            height=170, buttons=["fullscreen"])

                    with gr.Tab("Changes"):
                        gr.Markdown("### What did your last re-run change?")
                        iou_img = gr.Image(label="Per-frame IoU vs previous run",
                                           type="numpy", height=300, interactive=False,
                                           buttons=["fullscreen"])
                        corr_md = gr.Markdown(
                            "<div class='cp-hint'>Run Track twice to compare.</div>")

                    with gr.Tab("Export"):
                        export_what = gr.Dropdown(
                            ["Matte PNG sequence (zip)", "RGBA PNG sequence (zip)",
                             "comp.mp4", "matte.mp4", "side_by_side.mp4",
                             "metrics.json"],
                            value="Matte PNG sequence (zip)", label="Export")
                        export_fps = gr.Slider(1, 60, value=24, step=1, label="fps (video)")
                        export_btn = gr.Button("Prepare export", variant="primary")
                        export_file = gr.File(label="Download", visible=False)
                        export_md = gr.Markdown("")
                        gr.Markdown("---\n### Session")
                        with gr.Row():
                            save_btn = gr.Button("Save point.json", size="sm")
                            reload_btn = gr.Button("Reload point.json", size="sm")
                        session_md = gr.Markdown("")
                        gr.Markdown("### Timings, this session")
                        timings_out = gr.Markdown(
                            "<div class='cp-hint'>No stages run yet.</div>")
                        with gr.Accordion("Metrics", open=False):
                            metrics_md = gr.Markdown(
                                "<div class='cp-hint'>Run Track to get numbers.</div>",
                                elem_classes=["cp-scroll"])
                            refresh_btn = gr.Button("Refresh metrics", size="sm")

        # -------------------------------------------------- wiring
        view_inputs = [frame_slider, st, view_mode, bg_choice, bg_colour, bg_image]
        ctx = [frame_slider, view_mode, bg_choice, bg_colour, bg_image]

        load_btn.click(load_shot, [shot_dd, st],
                       [st, viewer_img, frame_slider, info_md, status, pts_df,
                        prompt_md, view_mode])
        for comp in (frame_slider, view_mode, bg_choice, bg_colour, bg_image):
            comp.change(on_view_change, view_inputs, viewer_img, show_progress="hidden")

        viewer_img.select(on_click, [st, frame_slider, view_mode, click_mode,
                                     bg_choice, bg_colour, bg_image],
                          [st, viewer_img, pts_df, prompt_md, status])
        undo_btn.click(on_undo, [st, *ctx], [st, viewer_img, pts_df, prompt_md, status])
        clearf_btn.click(on_clear_frame, [st, *ctx],
                         [st, viewer_img, pts_df, prompt_md, status])
        clear_btn.click(on_clear_all, [st, *ctx],
                        [st, viewer_img, pts_df, prompt_md, status])

        track_btn.click(run_track, [st, *ctx],
                        [st, viewer_img, status, view_mode, iou_img, corr_md,
                         timings_out]).then(metrics_table, st, metrics_md)
        refine_btn.click(run_refine, [st, warmup_n, dilate_n, erode_n, *ctx],
                         [st, viewer_img, status, view_mode, timings_out]
                         ).then(metrics_table, st, metrics_md)
        rgba_btn.click(run_despill, [st, despill_cb, despill_s, despill_b, *ctx],
                       [st, viewer_img, status, view_mode, timings_out])

        refresh_btn.click(metrics_table, st, metrics_md)
        export_btn.click(export_bundle,
                         [st, export_what, export_fps, bg_choice, bg_colour, bg_image],
                         [export_file, export_md])
        save_btn.click(save_session, st, session_md)
        reload_btn.click(reload_session, [st, *ctx],
                         [st, viewer_img, pts_df, prompt_md, session_md])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1",
                    help="loopback by default; nothing is exposed off this machine")
    ap.add_argument("--open", action="store_true", help="open a browser window")
    args = ap.parse_args()

    print(f"\n  CleanPlate v{__version__} — local only, no uploads, no telemetry")
    print(f"  http://{args.host}:{args.port}\n")
    build().launch(server_name=args.host, server_port=args.port, share=False,
                   inbrowser=args.open, quiet=True, css=CSS, theme=THEME,
                   show_error=True, pwa=False, footer_links=[])


if __name__ == "__main__":
    main()
