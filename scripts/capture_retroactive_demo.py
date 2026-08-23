#!/usr/bin/env python3
"""Reproduce the f067 jacket incident in the app and screenshot the warning.

Task 2 found that SAM 2 attends to every conditioning frame at every timestep, so an
under-specified corrective click at frame 67 redefined the object for the whole shot
and dropped the jacket from frame 24 onward. Task 3 turned that into UI. This script
drives the app to make it happen again, on purpose, and captures the evidence:

    python app.py --port 7860 &
    python scripts/capture_retroactive_demo.py --url http://127.0.0.1:7860

Writes docs/img/09_changes.png.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleanplate.paths import ROOT                                   # noqa: E402

OUT = ROOT / "docs" / "img"
# plate coordinates: the chest at f0; then the under-specified f67 correction that
# broke it in Task 2 - one keep on the t-shirt plus one exclude on the pavement.
F0_KEEP = (360, 200)
F67_KEEP = (410, 230)
F67_EXCLUDE = (321, 379)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 1000},
                                device_scale_factor=2)

        def wait_for(fragment: str, timeout: int) -> None:
            end = time.time() + timeout
            while time.time() < end:
                if fragment.lower() in page.inner_text("body").lower():
                    return
                page.wait_for_timeout(1000)
            raise TimeoutError(f"never saw {fragment!r}")

        def click_plate(xy: tuple[int, int]) -> None:
            box = page.locator("#cp-viewer img").first.bounding_box()
            page.mouse.click(box["x"] + box["width"] * xy[0] / 960,
                             box["y"] + box["height"] * xy[1] / 400)
            page.wait_for_timeout(1200)

        print(f"[demo] driving {args.url}")
        page.goto(args.url, wait_until="load")
        page.wait_for_timeout(1500)
        page.get_by_role("button", name="Load demo shot").click()
        wait_for("96 frames", 300)
        page.wait_for_timeout(1000)

        # --- baseline: one click on the actor, nothing else
        page.get_by_role("tab", name="Prompt").click()
        page.wait_for_timeout(500)
        page.get_by_role("button", name="Clear all").click()
        page.wait_for_timeout(600)
        click_plate(F0_KEEP)
        print("[demo] baseline: 1 keep click on frame 0 -> Run Track")
        page.get_by_role("button", name="Run Track").click()
        wait_for("tracked 96 frames", args.timeout)
        page.wait_for_timeout(1500)

        # --- the under-specified correction on frame 67
        num = page.locator("#cp-frame input[type=number]").first
        num.fill("67")
        num.press("Enter")
        page.wait_for_timeout(1500)
        page.get_by_role("tab", name="Prompt").click()
        page.wait_for_timeout(500)
        page.get_by_role("radio", name="Keep (subject)").click()
        click_plate(F67_KEEP)
        page.get_by_role("radio", name="Exclude (not subject)").click()
        click_plate(F67_EXCLUDE)
        print("[demo] correction: keep on the t-shirt + exclude on the pavement, "
              "frame 67 -> Run Track")
        page.get_by_role("button", name="Run Track").click()
        wait_for("tracked 96 frames", args.timeout)
        page.wait_for_timeout(2000)

        page.get_by_role("tab", name="Changes").click()
        page.wait_for_timeout(1500)
        out = OUT / "09_changes.png"
        page.screenshot(path=str(out))
        body = page.inner_text("body")
        line = next((l for l in body.splitlines() if "Retroactive change" in l), None)
        print(f"[demo] {out.relative_to(ROOT)}")
        print(f"[demo] warning text: {line or 'NOT FOUND - the app did not flag it'}")
        browser.close()
        if not line:
            sys.exit(1)


if __name__ == "__main__":
    main()
