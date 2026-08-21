#!/usr/bin/env python3
"""Open a frame and print the pixel you click.

Left click  = positive point (this is the thing I want)
Right click = negative point (this is NOT the thing I want)
u / backspace = undo the last point
Enter or closing the window = finish

Prints a ready-to-paste ``track_matte.py`` invocation and, unless --no-save,
writes the points to ``shots/<name>/point.json`` so track_matte can pick them
up with no arguments at all.

    python src/pick_point.py --shot walk
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot", required=True, help="shot name under shots/")
    ap.add_argument("--frame", type=int, default=0, help="frame index to click on (default 0)")
    ap.add_argument("--no-save", action="store_true", help="print only, do not write point.json")
    args = ap.parse_args()

    frames_dir = ROOT / "shots" / args.shot / "frames"
    if not frames_dir.is_dir():
        sys.exit(f"no such shot: {frames_dir}\nRun src/extract_shot.py first.")
    frame_path = frames_dir / f"{args.frame:05d}.jpg"
    if not frame_path.exists():
        matches = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        if not matches:
            sys.exit(f"no frames in {frames_dir}")
        frame_path = matches[args.frame]

    import matplotlib
    if matplotlib.get_backend().lower() in ("agg", "pdf", "ps", "svg", "template"):
        sys.exit(f"matplotlib is using the non-interactive '{matplotlib.get_backend()}' "
                 "backend, so no window can open. On macOS the 'macosx' backend should "
                 "be available; try MPLBACKEND=macosx.")
    import matplotlib.pyplot as plt
    from PIL import Image

    img = Image.open(frame_path)
    w, h = img.size
    points: list[tuple[int, int, int]] = []   # (x, y, label)

    fig, ax = plt.subplots(figsize=(min(16, w / 90), min(10, h / 90)))
    ax.imshow(img)
    ax.set_title(f"{args.shot}  frame {args.frame}   ({w}x{h})\n"
                 "left-click = keep · right-click = exclude · u = undo · close when done",
                 fontsize=10)
    ax.set_xlabel("click the actor")
    fig.tight_layout()

    artists: list = []

    def redraw() -> None:
        while artists:
            artists.pop().remove()
        for x, y, lab in points:
            colour = "#22c55e" if lab == 1 else "#ef4444"
            artists.append(ax.scatter([x], [y], s=180, marker="*",
                                      c=colour, edgecolors="black", linewidths=1.2,
                                      zorder=5))
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        if event.inaxes is not ax or event.xdata is None:
            return
        if event.button not in (1, 3):
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        label = 1 if event.button == 1 else 0
        points.append((x, y, label))
        print(f"  {'+' if label else '-'} ({x}, {y})", flush=True)
        redraw()

    def on_key(event) -> None:
        if event.key in ("u", "backspace") and points:
            print(f"  undo {points.pop()[:2]}", flush=True)
            redraw()
        elif event.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print(f"[pick] {frame_path.relative_to(ROOT)}  ({w}x{h})")
    print("[pick] left-click the actor. Right-click to exclude. Close the window when done.")
    plt.show()

    if not points:
        print("[pick] no points clicked - nothing to do.")
        return

    pos = [(x, y) for x, y, lab in points if lab == 1]
    neg = [(x, y) for x, y, lab in points if lab == 0]

    print(f"\n[pick] {len(pos)} positive, {len(neg)} negative")
    cmd = ["python src/track_matte.py", f"--shot {args.shot}"]
    for x, y in pos:
        cmd.append(f"--point {x},{y}")
    for x, y in neg:
        cmd.append(f"--neg {x},{y}")
    if args.frame:
        cmd.append(f"--prompt-frame {args.frame}")
    print("\n" + " \\\n    ".join(cmd) + "\n")

    if not args.no_save:
        out = ROOT / "shots" / args.shot / "point.json"
        out.write_text(json.dumps({
            "shot": args.shot,
            "prompt_frame": args.frame,
            "image_size": [w, h],
            "positive": pos,
            "negative": neg,
        }, indent=2) + "\n")
        print(f"[pick] saved -> {out.relative_to(ROOT)}")
        print(f"[pick] track_matte.py --shot {args.shot} will read it automatically.")


if __name__ == "__main__":
    main()
