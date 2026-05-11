"""Take a screenshot of the Gradio review UI for the README.

Assumes the review server is already running at 127.0.0.1:7860.
"""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "review_ui.png"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        # Gradio uses websockets, so wait_until="networkidle" hangs.
        # domcontentloaded is enough; we sleep a bit afterwards to let
        # the framework hydrate the audio player + buttons.
        page.goto("http://127.0.0.1:7860", wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(5000)
        page.screenshot(path=str(OUT), full_page=False)
        browser.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
