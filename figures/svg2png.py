#!/usr/bin/env python
"""Rasterize an SVG to PNG with headless Chrome. The manuscript includes PNGs, so run this after
hand-editing a figure SVG (the make_figure.py scripts redraw from data and would discard the edits).

    python svg2png.py fig1_modes/fig1.svg                      # -> fig1_modes/fig1.png, 4200 px wide
    python svg2png.py in.svg out.png --width 2200 --transparent
"""
import argparse
import math
import os
import re
import subprocess
import sys
import tempfile

CHROME = os.environ.get("CHROME", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
PUB_WIDTH_PX = 4200          # width of every publication raster in this repo


def render(svg_path, png_path, width_px=PUB_WIDTH_PX, transparent=False, chrome=CHROME):
    """Render svg_path to png_path at width_px, preserving the viewBox aspect ratio."""
    if not os.path.exists(chrome):
        raise SystemExit("Chrome not found at %s (set the CHROME env var)" % chrome)

    # Chrome resolves a relative --screenshot path against its own cwd, so always pass an absolute one.
    png_path = os.path.abspath(png_path)
    stale = os.path.getmtime(png_path) if os.path.exists(png_path) else None

    svg = open(svg_path, encoding="utf-8").read()
    vb = re.search(r'viewBox="\s*([\d.eE+-]+)[,\s]+([\d.eE+-]+)[,\s]+([\d.eE+-]+)[,\s]+([\d.eE+-]+)"', svg)
    if not vb:
        raise SystemExit("%s has no viewBox; cannot determine its aspect ratio" % svg_path)
    w, h = float(vb.group(3)), float(vb.group(4))

    # Pin the viewport to the viewBox so the screenshot is exactly the artboard.
    svg = re.sub(r'(<svg\b[^>]*?)\swidth="[^"]*"', r"\1", svg, count=1)
    svg = re.sub(r'(<svg\b[^>]*?)\sheight="[^"]*"', r"\1", svg, count=1)
    svg = re.sub(r"<svg\b", '<svg width="%.2fpx" height="%.2fpx"' % (w, h), svg, count=1)

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as fh:
        fh.write('<!doctype html><meta charset="utf-8">'
                 "<style>html,body{margin:0;padding:0;overflow:hidden}svg{display:block}</style>"
                 "<body>%s</body>" % svg)
        html = fh.name
    try:
        subprocess.run([chrome, "--headless", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=%.4f" % (width_px / w),
                        "--screenshot=%s" % png_path,
                        "--window-size=%d,%d" % (math.ceil(w) + 1, math.ceil(h) + 1),
                        "--default-background-color=%s" % ("00000000" if transparent else "FFFFFFFF"),
                        html], capture_output=True, timeout=180)
    finally:
        os.remove(html)

    if not os.path.exists(png_path) or os.path.getmtime(png_path) == stale:
        raise SystemExit("Chrome wrote no output for %s (expected %s)" % (svg_path, png_path))
    return png_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("svg")
    ap.add_argument("png", nargs="?", help="default: the SVG path with a .png suffix")
    ap.add_argument("--width", type=int, default=PUB_WIDTH_PX, help="output width in px (default 4200)")
    ap.add_argument("--transparent", action="store_true", help="keep the background transparent")
    a = ap.parse_args(argv)

    out = a.png or os.path.splitext(a.svg)[0] + ".png"
    render(a.svg, out, a.width, a.transparent)
    size = os.path.getsize(out)
    print("wrote %s (%.0f kB)" % (out, size / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
