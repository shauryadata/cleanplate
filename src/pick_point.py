#!/usr/bin/env python3
"""Open a frame and print the pixel you click. The CLI twin of the app's canvas.

Left click  = positive point (this is the thing I want)
Right click = negative point (this is NOT the thing I want)
u / backspace = undo the last point on this frame
Enter or closing the window = finish

Clicks are stored per frame in shots/<name>/point.json and merged across sessions, so
you can come back and add a corrective click on a later frame without losing the
first-frame prompt:

    python src/pick_point.py --shot walk              # positive click on frame 0
    python src/pick_point.py --shot walk --frame 67   # right-click the intruder

A corrective frame must also carry a positive click on the subject: SAM 2 reads a
conditioning frame with only negative points as "object absent" and blanks it.
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from cleanplate.ingest import frame_paths
from cleanplate.paths import rel, shot_dir
from cleanplate.session import Prompt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--replace", action="store_true",
                    help="drop every existing prompt instead of merging this frame in")
    args = ap.parse_args()

    frames = frame_paths(args.shot)
    if not frames:
        sys.exit(f"no frames for shot {args.shot!r}. Run src/extract_shot.py first.")
    if not (0 <= args.frame < len(frames)):
        sys.exit(f"frame {args.frame} outside 0-{len(frames) - 1}")
    path = frames[args.frame]

    import matplotlib
    if matplotlib.get_backend().lower() in ("agg", "pdf", "ps", "svg", "template"):
        sys.exit(f"matplotlib is using the non-interactive "
                 f"'{matplotlib.get_backend()}' backend, so no window can open. "
                 "On macOS try MPLBACKEND=macosx.")
    import matplotlib.pyplot as plt
    from PIL import Image

    img = Image.open(path)
    w, h = img.size
    prompt = Prompt() if args.replace else Prompt.load(args.shot)
    if not args.replace:
        prompt.clear_frame(args.frame)      # this session re-clicks this frame
    existing = [f for f in prompt.prompt_frames]

    fig, ax = plt.subplots(figsize=(min(16, w / 90), min(10, h / 90)))
    ax.imshow(img)
    ax.set_title(f"{args.shot}  frame {args.frame}   ({w}x{h})\n"
                 "left-click = keep · right-click = exclude · u = undo · close when done",
                 fontsize=10)
    fig.tight_layout()
    artists: list = []

    def redraw() -> None:
        while artists:
            artists.pop().remove()
        e = prompt.frames.get(args.frame)
        if not e:
            fig.canvas.draw_idle()
            return
        for pts, colour in ((e.positive, "#22c55e"), (e.negative, "#ef4444")):
            for x, y in pts:
                artists.append(ax.scatter([x], [y], s=180, marker="*", c=colour,
                                          edgecolors="black", linewidths=1.2, zorder=5))
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        if event.inaxes is not ax or event.xdata is None or event.button not in (1, 3):
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        prompt.add(args.frame, x, y, positive=event.button == 1)
        print(f"  {'+' if event.button == 1 else '-'} ({x}, {y})", flush=True)
        redraw()

    def on_key(event) -> None:
        if event.key in ("u", "backspace"):
            p = prompt.undo(args.frame)
            if p:
                print(f"  undo {p}", flush=True)
                redraw()
        elif event.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    print(f"[pick] {rel(path)}  ({w}x{h})")
    if existing:
        print(f"[pick] existing prompt frames kept: {existing}")
    print("[pick] left-click the actor. Right-click to exclude. Close when done.")
    plt.show()

    e = prompt.frames.get(args.frame)
    if not e:
        print("[pick] no points clicked on this frame - nothing to do.")
        return

    print(f"\n[pick] {len(e.positive)} positive, {len(e.negative)} negative "
          f"on frame {args.frame}")
    cmd = ["python src/track_matte.py", f"--shot {args.shot}"]
    for x, y in e.positive:
        cmd.append(f"--point {x},{y}" if args.frame == 0 else f"--at {args.frame}:{x},{y}:+")
    for x, y in e.negative:
        cmd.append(f"--neg {x},{y}" if args.frame == 0 else f"--at {args.frame}:{x},{y}:-")
    print("\n" + " \\\n    ".join(cmd) + "\n")

    lonely = prompt.negative_only_frames()
    if lonely:
        print(f"[pick] WARNING: frame(s) {lonely} have only negative points. SAM 2 will "
              "blank those frames. Add a positive click on the subject there.",
              file=sys.stderr)
    if args.no_save:
        return
    out = prompt.save(args.shot, (w, h))
    print(f"[pick] saved -> {rel(out)}  "
          f"({len(prompt.prompt_frames)} prompt frame(s): {prompt.prompt_frames})")
    print(f"[pick] track_matte.py --shot {args.shot} will read it automatically.")


if __name__ == "__main__":
    main()
