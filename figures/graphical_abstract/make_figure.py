#!/usr/bin/env python3
"""Figure 0 (graphical abstract): the ACS/JCIM TOC graphic, built from the aligned overlay renders
that render_overlays.py writes to data/. Emits candidates/ACS_<pair>_<style>.png (the manuscript
includes this PNG) plus a flattened, Illustrator-editable .svg.

    python make_figure.py
"""
from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.font_manager import FontProperties

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "candidates")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))   # figures/svgtools.py
import svgtools                                                  # noqa: E402

plt.rcParams["font.family"] = ["DejaVu Sans", "Arial"]
plt.rcParams["svg.fonttype"] = "none"   # keep text as live <text> (editable), not glyph paths
MONO = FontProperties(family=["Consolas", "DejaVu Sans Mono", "monospace"])

INK, SUB = "#1f2933", "#6b7480"
TEAL = "#c0e6c2"                      # reference molecule (fig 1B green)
PAPER = "#ffffff"
AMBER = "#f0a500"                     # speed accent
PURPLE = "#8a63c4"                    # agent accent
GREEN = "#3f9e57"

ACS_W, ACS_H = 6.5, 3.5               # 2x the 3.25 x 1.75 in ACS TOC box; aspect 1.857:1


def dot(ax, x, y, color, size=150, ec="white", lw=1.4, z=4):
    """Round marker (scatter stays circular regardless of axes aspect)."""
    ax.scatter([x], [y], s=size, c=color, edgecolors=ec, linewidths=lw,
               transform=ax.transAxes, zorder=z, clip_on=False)


def star(ax, x, y, r, color, ar, alpha=1.0, z=7):
    """A 4-point sparkle star, aspect-corrected so it stays symmetric."""
    pts = []
    for k in range(8):
        ang = np.pi / 2 + k * np.pi / 4
        rr = r if k % 2 == 0 else r * 0.34
        pts.append((x + rr * np.cos(ang), y + rr * ar * np.sin(ang)))
    ax.add_patch(Polygon(pts, closed=True, transform=ax.transAxes, fc=color,
                 ec="none", zorder=z, alpha=alpha))


def load_overlay(pair, style):
    """data/overlay_<pair>_<style>.png (written by render_overlays.py), tight-cropped."""
    a = mpimg.imread(os.path.join(DATA, f"overlay_{pair}_{style}.png"))
    if a.shape[-1] == 4:
        ys, xs = np.where(a[..., 3] > 0.02)
        if len(xs):
            m = 30
            a = a[max(ys.min() - m, 0):min(ys.max() + m, a.shape[0]),
                  max(xs.min() - m, 0):min(xs.max() + m, a.shape[1])]
    return a


def base_axes(figsize):
    fig = plt.figure(figsize=figsize, facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return fig, ax


def place_overlay(fig, img, x, y, w, h):
    ax = fig.add_axes([x, y, w, h]); ax.set_axis_off()
    ax.imshow(img); ax.set_aspect("equal")


def save(fig, name, dpi=200):
    svg = os.path.join(OUT, name + ".svg")
    if not svgtools.guard_svg(svg):
        plt.close(fig)             # finished by hand: keep it (--force to regenerate)
        return
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=dpi, facecolor=PAPER)
    fig.savefig(svg, facecolor=PAPER)
    plt.close(fig)
    print("  candidates/%s.png / .svg" % name)
    svgtools.flatten(svg)          # flat + ungrouped, text stays live


def candidate_ACS(pair, style):
    """ACS/JCIM TOC graphic: hero overlay plus three value-prop lines. The final box is
    3.25 x 1.75 in at >=300 dpi, so fonts are sized to stay >=~7 pt after down-scaling."""
    fig, ax = base_axes((ACS_W, ACS_H))
    ar = ACS_W / ACS_H
    ax.add_patch(FancyBboxPatch((0.012, 0.04), 0.976, 0.92, transform=ax.transAxes,
                 boxstyle="round,pad=0,rounding_size=0.035", fc="#fbfcfd",
                 ec="#eef1f4", lw=0.9, zorder=0))
    ax.text(0.04, 0.90, "fast_shepherd_score", transform=ax.transAxes, ha="left",
            va="center", fontsize=19, fontweight="bold", color=INK, fontproperties=MONO)
    ax.text(0.04, 0.815, "GPU-accelerated 3D molecular alignment & scoring",
            transform=ax.transAxes, ha="left", va="center", fontsize=12, color=SUB)
    img = load_overlay(pair, style)
    place_overlay(fig, img, 0.01, 0.10, 0.44, 0.62)
    ax.text(0.235, 0.075, "aspirin ▸ query, aligned in 3D", transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color=TEAL, fontweight="bold")

    xi, xt = 0.515, 0.565
    HEAD, SUB_ = 18, 12
    ax.text(xi, 0.44, "⚡", transform=ax.transAxes, ha="center", va="center", fontsize=21)
    ax.text(xt, 0.455, "Fast", transform=ax.transAxes, ha="left", va="center",
            fontsize=HEAD, fontweight="bold", color=AMBER)
    ax.text(xt, 0.385, "up to 1M alignments/second/GPU", transform=ax.transAxes, ha="left",
            va="center", fontsize=SUB_, color=SUB)
    for k, c in enumerate(["#8a94a0", "#d0554c", GREEN]):
        dot(ax, xi - 0.016 + k * 0.016, 0.66, c, size=85)
    ax.text(xt, 0.68, "Versatile", transform=ax.transAxes, ha="left", va="center",
            fontsize=HEAD, fontweight="bold", color=GREEN)
    ax.text(xt, 0.61, "18 scoring modes (12 new)", transform=ax.transAxes,
            ha="left", va="center", fontsize=SUB_, color=SUB)
    ax.add_patch(FancyBboxPatch((0.485, 0.115), 0.485, 0.16, transform=ax.transAxes,
                 boxstyle="round,pad=0,rounding_size=0.028", fc="#f4eefc",
                 ec=PURPLE, lw=1.4, zorder=1))
    star(ax, xi, 0.205, 0.024, PURPLE, ar, z=5)
    ax.text(xt, 0.225, "Extensible", transform=ax.transAxes, ha="left", va="center",
            fontsize=HEAD, fontweight="bold", color=PURPLE, zorder=5)
    ax.text(xt, 0.155, "add your own — via an agent", transform=ax.transAxes,
            ha="left", va="center", fontsize=SUB_, color=PURPLE, zorder=5)
    save(fig, f"ACS_{pair}_{style}", dpi=600)


def main():
    print("composing the graphical abstract ...")
    candidate_ACS("methylsalicylate", "stick")
    print("done ->", os.path.relpath(OUT, HERE))


if __name__ == "__main__":
    main()
