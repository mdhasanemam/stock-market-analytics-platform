"""Capture README screenshots from a RUNNING instance of the app.

    pip install playwright
    python -m playwright install chromium
    # terminal 1:  python -m uvicorn backend.app.main:app
    # terminal 2:  python scripts/capture_screenshots.py            (optional: --base http://127.0.0.1:8000)

Writes PNGs to docs/screenshots/. Works with DATA_PROVIDER=csv (offline demo) or yfinance (live).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--symbol", default="AAPL")
    a = ap.parse_args()
    base, sym = a.base.rstrip("/"), a.symbol.upper()
    OUT.mkdir(parents=True, exist_ok=True)

    def shot(page, name: str, full: bool = False) -> None:
        page.wait_for_timeout(600)                       # let Plotly finish animating
        page.screenshot(path=str(OUT / f"{name}.png"), full_page=full)
        print("saved", OUT / f"{name}.png")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        ctx.add_init_script("try{localStorage.setItem('theme','dark')}catch(e){}")
        page = ctx.new_page()

        page.goto(f"{base}/")
        page.wait_for_selector("#featured .stock-card", timeout=60000)
        shot(page, "01-home")

        page.goto(f"{base}/dashboard?symbol={sym}")
        page.wait_for_selector("#chart.js-plotly-plot", timeout=60000)
        for sel in ("sma20", "sma50", "bollinger"):
            page.check(f'.ov[value="{sel}"]')
        for sel in ("rsi", "macd"):
            page.check(f'.pn[value="{sel}"]')
        shot(page, "02-dashboard", full=True)
        page.click("#themeToggle")                       # light theme variant
        shot(page, "03-dashboard-light")
        page.click("#themeToggle")

        page.goto(f"{base}/predict?symbol={sym}")
        page.wait_for_selector("#model option", state="attached", timeout=30000)
        page.click("#run")
        page.wait_for_selector("#chart.js-plotly-plot", timeout=180000)
        shot(page, "04-predict", full=True)

        page.goto(f"{base}/compare?symbol={sym}")
        page.wait_for_selector(".mb", state="attached", timeout=30000)
        page.click("#run")
        page.wait_for_selector("#lines.js-plotly-plot", timeout=300000)
        shot(page, "05-compare", full=True)

        for s in (sym, "MSFT"):                          # best-effort: ignore "already on watchlist"
            page.request.post(f"{base}/api/watchlist", data={"symbol": s})
        page.goto(f"{base}/watchlist")
        page.wait_for_selector("#list .stock-card", timeout=60000)
        shot(page, "06-watchlist")

        page.goto(f"{base}/history")
        page.wait_for_selector("#list tbody tr", timeout=30000)
        shot(page, "07-history")
        browser.close()


if __name__ == "__main__":
    main()
