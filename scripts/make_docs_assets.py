#!/usr/bin/env python3
"""Drive the running app with Playwright and capture the README screenshots and GIF.

Every image is a real screenshot of the real app - nothing is mocked up. Run the app
first, then point this at it:

    python app.py --port 7860 &
    python scripts/make_docs_assets.py --url http://127.0.0.1:7860

Writes docs/img/*.png and docs/img/walkthrough.gif.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate.paths import ROOT                                    # noqa: E402

OUT = ROOT / "docs" / "img"
VIEWPORT = {"width": 1500, "height": 1000}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--track-timeout", type=int, default=240,
                    help="seconds to allow for a Run Track pass")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    shots: list[Path] = []

    def snap(name: str, note: str = "") -> Path:
        p = OUT / f"{name}.png"
        page.screenshot(path=str(p))
        shots.append(p)
        print(f"  [shot] {p.relative_to(ROOT)}  {note}")
        return p

    def wait_status(fragment: str, timeout: int) -> None:
        """Block until the status line contains `fragment`."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if fragment.lower() in page.inner_text("body").lower():
                return
            page.wait_for_timeout(1000)
        raise TimeoutError(f"never saw {fragment!r} within {timeout}s")

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        print(f"[docs] driving {args.url}")
        page.goto(args.url, wait_until="load")
        page.wait_for_timeout(1500)

        snap("01_empty", "cold start")

        # --- load the demo shot
        page.get_by_role("button", name="Load demo shot").click()
        wait_status("96 frames", 300)
        page.wait_for_timeout(1200)
        snap("02_loaded", "demo shot loaded")

        # --- prompt tab: place a real click, so the shot shows the interaction
        page.get_by_role("tab", name="Prompt").click()
        page.wait_for_timeout(700)
        # Clear frame 0 first: the shot restores a saved click there, and adding a
        # second one a pixel away would change the matte (see the near-click warning).
        page.get_by_role("button", name="Clear frame").click()
        page.wait_for_timeout(800)
        img = page.locator("#cp-viewer img").first
        box = img.bounding_box()
        if box:
            # the actor's chest, in plate coordinates, mapped onto the rendered image
            page.mouse.click(box["x"] + box["width"] * 360 / 960,
                             box["y"] + box["height"] * 200 / 400)
            page.wait_for_timeout(1500)
        snap("03_prompt", "click prompting, marker placed")

        # --- track
        page.get_by_role("button", name="Run Track").click()
        page.wait_for_timeout(6000)
        snap("04_tracking", "progress during the SAM 2 pass")
        wait_status("tracked 96 frames", args.track_timeout)
        page.wait_for_timeout(1500)
        snap("05_matte", "matte overlay")

        # --- refine
        page.get_by_role("tab", name="Refine").click()
        page.wait_for_timeout(500)
        page.get_by_role("button", name="Run Refine").click()
        wait_status("refined 96 frames", 300)
        page.wait_for_timeout(1200)
        snap("06_refined", "soft alpha on the checkerboard")

        page.get_by_role("button", name="Build RGBA").click()
        wait_status("built 96 rgba frames", 180)
        page.wait_for_timeout(800)
        page.get_by_role("radio", name="Comp").click()
        page.wait_for_timeout(1200)
        snap("07_comp", "comp over the flat-colour stress test")

        # --- export
        page.get_by_role("tab", name="Export").click()
        page.wait_for_timeout(800)
        snap("08_export", "export panel and per-stage timings")

        browser.close()

    # --- keep the repo light: captured at 2x for crispness, stored at 1600px
    from PIL import Image
    for shot in shots:
        im = Image.open(shot).convert("RGB")
        if im.width > 1600:
            im = im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS)
        im.save(shot, optimize=True)

    # --- GIF from the captured states
    frames = [Image.open(p).convert("RGB").resize((900, 600), Image.LANCZOS)
              for p in shots]
    gif = OUT / "walkthrough.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=1600,
                   loop=0, optimize=True)
    print(f"[docs] {gif.relative_to(ROOT)}  ({gif.stat().st_size / 1e6:.1f} MB, "
          f"{len(frames)} frames)")


if __name__ == "__main__":
    main()
