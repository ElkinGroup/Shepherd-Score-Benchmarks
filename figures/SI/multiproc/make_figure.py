#!/usr/bin/env python
"""SI figure: fss ``vol`` screening across CPU worker processes and across GPUs, drawn with
throughput's draw_multiproc from benchmarks/throughput/results/P6_cpu.json and P6_gpu.json.

Run:  python make_figure.py        -> si_multiproc.svg + si_multiproc.png (the SI includes the PNG)
"""
from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "throughput"))

import matplotlib                      # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt        # noqa: E402
import panels_layout as p2             # noqa: E402  (sets the fig2 rcParams as a side effect)


def main():
    matplotlib.rcParams["font.family"] = "Arial"
    matplotlib.rcParams["svg.hashsalt"] = "si-multiproc"
    fig, (ax_cpu, ax_gpu) = plt.subplots(1, 2, figsize=(8.0, 3.1))
    data = Path(HERE).parents[2] / 'benchmarks/publication_data/figure4_parallel.csv'
    rows = list(csv.DictReader(data.open()))
    def cells(family, key):
        return {r['task_id']: dict(aligns_per_sec=float(r['rate']), size=int(r['n']),
                valid=True, **{key:int(r['count'])}) for r in rows if r['family']==family}
    # The one-core in-process reference (P6_cpu_reference.json) is quoted in the SI text, not drawn.
    p2.draw_multiproc(ax_cpu, cells("cpu", "cores"), "cores", "alignments / second", max_key=64)
    gpu_cells = cells("gpu", "gpus")
    gpu_sizes = {c["size"] for c in gpu_cells.values()}
    if len(gpu_sizes) != 1:
        raise ValueError("GPU scaling cells must use the same library size")
    gpu_size = gpu_sizes.pop()
    gpu_exponent = int(math.log10(gpu_size))
    gpu_label = f"$10^{gpu_exponent}$" if gpu_size == 10**gpu_exponent else f"{gpu_size:,}"
    p2.draw_multiproc(ax_gpu, gpu_cells, "gpus", "alignments / second", max_key=4)
    ax_cpu.set_title("A    CPU worker processes  (N = $10^4$, vol)", loc="left", fontsize=10.5,
                     fontweight="bold", pad=10)
    ax_gpu.set_title(f"B    GPUs, one process each  (N = {gpu_label}, vol)", loc="left",
                     fontsize=10.5, fontweight="bold", pad=10)
    fig.tight_layout(w_pad=3.0)
    for ext, kw in (("svg", {"metadata": {"Date": None}}), ("png", {"dpi": 300})):
        out = os.path.join(HERE, f"si_multiproc.{ext}")
        fig.savefig(out, facecolor="white", **kw)
        if ext == "svg":
            svg = Path(out)
            svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                           encoding="utf-8")
        print("wrote", os.path.relpath(out))


if __name__ == "__main__":
    main()
