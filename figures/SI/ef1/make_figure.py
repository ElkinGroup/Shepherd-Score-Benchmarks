#!/usr/bin/env python
"""SI figure: EF1% counterpart of main-text Figure 2B, so the matched fss-versus-comparator
enrichment comparison can be read at a glance instead of out of the per-target tables.

Reuses enrichment/panels.draw_panel_B with metric="ef1"; identical bars, colours, SEM and
paired-t brackets, with the random-ranking baseline at EF1% = 1 instead of ROC-AUC = 0.5.

Run:  python make_figure.py        -> si_ef1.svg + si_ef1.png (the SI includes the PNG)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "enrichment"))

import matplotlib                      # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt        # noqa: E402
import panels                          # noqa: E402


def main():
    matplotlib.rcParams["font.family"] = "Arial"
    matplotlib.rcParams["svg.hashsalt"] = "si-ef1"
    fig, ax = plt.subplots(figsize=(13.0, 5.2))
    panels.draw_panel_B(ax, panels._xs_A(), metric="ef1")
    fig.tight_layout()
    for ext, kw in (("svg", {"metadata": {"Date": None}}), ("png", {"dpi": 300})):
        out = os.path.join(HERE, f"si_ef1.{ext}")
        fig.savefig(out, facecolor="white", **kw)
        if ext == "svg":
            svg = Path(out)
            svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                           encoding="utf-8")
    print("wrote", os.path.join(HERE, "si_ef1.png"))


if __name__ == "__main__":
    main()
