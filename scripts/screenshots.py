"""Take the README screenshots of the arXiv map with headless Chromium.

Serve web/ first, then run this against it:

    python -m http.server 8765 --bind 127.0.0.1 -d web &
    uv run --no-project --with playwright --with pillow python scripts/screenshots.py http://127.0.0.1:8765

Views are set through window.atlas (the page exposes it), so the shots are repeatable.
"""
import io
import os
import sys

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "images")
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"


def save(png_bytes, name, max_kb=400):
    """PNG if it fits under max_kb, else the best JPEG quality that fits. A 256-colour palette was tried
    and shifts the legend's category colours, so it is not used."""
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, name)
    for old in (stem + ".png", stem + ".jpg"):
        if os.path.exists(old):
            os.remove(old)
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    path = stem + ".png"
    im.save(path, optimize=True)
    if os.path.getsize(path) > max_kb * 1024:
        os.remove(path)
        path = stem + ".jpg"
        for q in (90, 85, 80, 75, 70):
            im.save(path, quality=q, optimize=True, progressive=True)
            if os.path.getsize(path) <= max_kb * 1024:
                break
    print(path, os.path.getsize(path) // 1024, "KB")


def set_view(page, cx, cy, zoom):
    page.evaluate("([cx, cy, z]) => { const a = window.atlas; a.view = {cx, cy, zoom: z}; a.dirty = true; }",
                  [cx, cy, zoom])
    page.wait_for_timeout(600)


def hover_near(page, sx, sy):
    """Move the mouse onto the visible paper nearest to (sx, sy) so the tooltip shows."""
    pos = page.evaluate("""([sx, sy]) => {
        const a = window.atlas;
        for (let r = 0; r < 200; r += 4) {
            for (let k = 0; k < 16; k++) {
                const x = sx + r * Math.cos(k * Math.PI / 8), y = sy + r * Math.sin(k * Math.PI / 8);
                const i = a.nearest(x * a.dpr, y * a.dpr, 3 * a.dpr);
                if (i >= 0) return [x, y];
            }
        }
        return null;
    }""", [sx, sy])
    if pos:
        page.mouse.move(*pos)
        page.wait_for_selector("#tip", state="visible", timeout=10000)
        page.wait_for_timeout(400)
    return pos


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"])
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.on("console", lambda m: print("console:", m.text) if m.type == "error" else None)
        page.goto(f"{BASE}/?data=arxiv")
        page.wait_for_function("window.atlas && window.atlas.labelsByLevel", timeout=120000)
        page.wait_for_timeout(1500)
        print("renderer:", page.evaluate("(() => { const gl = window.atlas.gl;"
                                         " const e = gl.getExtension('WEBGL_debug_renderer_info');"
                                         " return e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : '?'; })()"))

        # 1. astrophysics at the finest label level, with a paper's tooltip
        set_view(page, 0.235, 0.80, 14)
        hover_near(page, 860, 470)
        save(page.screenshot(), "astro_zoom_hover")
        page.mouse.move(5, 895)
        page.wait_for_timeout(300)

        # 2. one category on its own (alt-click in the legend), centred on its main cluster
        set_view(page, 0.56, 0.5, 2.2)
        page.locator(".cat", has_text="quantitative biology").click(modifiers=["Alt"])
        page.wait_for_timeout(800)
        save(page.screenshot(), "filter_qbio")
        page.reload()
        page.wait_for_function("window.atlas && window.atlas.labelsByLevel", timeout=120000)
        page.wait_for_timeout(1500)

        # 3. the year slider set to 2005: only papers up to that year are drawn
        page.evaluate("""() => { const yr = document.getElementById('year');
            yr.value = 2005; yr.dispatchEvent(new Event('input')); }""")
        set_view(page, 0.5, 0.5, 1)
        save(page.screenshot(), "year_2005")

        browser.close()


if __name__ == "__main__":
    main()
