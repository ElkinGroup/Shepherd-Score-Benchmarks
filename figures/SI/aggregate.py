#!/usr/bin/env python3
"""Collate the SI accuracy campaign (accuracy/data/si_enrichment_*.json) and the selected paper speed
cells into si_table.tex, si_summary.json and si_modes.svg/png.

    python aggregate.py
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
from dudez.benchmark import validate_publication_inputs

HERE = os.path.dirname(os.path.abspath(__file__))
ACC_DIR = str(Path(HERE).parents[1] / 'benchmarks/accuracy/data')
FIG3_DIR = str(Path(HERE).parents[1] / 'benchmarks/enrichment/data')
# Throughput comes from the main fig2 harness (P3 pairwise cells) so it is measured the same way
# as the main figure's.
_FIG2_RESULTS = str(Path(HERE).parents[1] / 'benchmarks/throughput/results')

SI_MODES = ["vol_pharm", "vol_atomtype", "vol_mr", "surf_tversky", "surf_esp_tversky",
            "vol_and_surf_esp_tversky", "vol_color_tversky", "vol_lipo_tversky", "pharm_tversky",
            "vol_fukui"]
# The main-text mode each Tversky variant differs from only by the reduction. The other four modes
# add a channel of their own and have no like-for-like counterpart, so they carry no delta.
TANIMOTO_FORM = {"surf_tversky": "surf", "surf_esp_tversky": "surf_esp",
                 "vol_and_surf_esp_tversky": "vol_and_surf_esp",
                 "vol_color_tversky": "vol_color", "vol_lipo_tversky": "vol_lipo",
                 "pharm_tversky": "pharm"}
PRETTY = {"vol_pharm": "vol_pharm", "vol_atomtype": "vol_atomtype", "vol_mr": "vol_mr",
          "surf_tversky": "surf_tversky", "surf_esp_tversky": "surf_esp_tversky",
          "vol_and_surf_esp_tversky": "vol_and_surf_esp_tversky",
          "vol_color_tversky": "vol_color_tversky", "vol_lipo_tversky": "vol_lipo_tversky",
          "pharm_tversky": "pharm_tversky", "vol_fukui": "vol_fukui"}


def _collect(pattern, key):
    """mode -> list of per-target mean AUC / EF1 / hit rate over the JSONs matching `pattern`."""
    auc, ef1, hit = {}, {}, {}
    for path in sorted(glob.glob(pattern)):
        d = json.load(open(path))
        for mode, blk in d.get("results", {}).items():
            m = blk.get("metrics", {})
            if "auc" in m:
                auc.setdefault(mode, []).append(m["auc"]["mean"])
            if "ef1" in m:
                ef1.setdefault(mode, []).append(m["ef1"]["mean"])
            if "hit_rate" in m:
                hit.setdefault(mode, []).append(m["hit_rate"]["mean"])
    return auc, ef1, hit


def main():
    source = Path(HERE).parents[1] / 'benchmarks/publication_data/enrichment.json'
    data = json.loads(source.read_text())['per_target']
    def collect(modes):
        return tuple({m:[r[m][k] for r in data.values()] for m in modes} for k in ('auc','ef1','hit_rate'))
    si_auc, si_ef1, si_hit = collect(SI_MODES)
    base_auc, base_ef1, base_hit = collect(sorted({m for r in data.values() for m in r}))

    def _mean(d, k):
        return float(np.nanmean(d[k])) if d.get(k) else float("nan")

    n_targets = max((len(v) for v in si_auc.values()), default=0)
    summary = {"n_targets": n_targets, "modes": {}}
    for mode in SI_MODES:
        b = TANIMOTO_FORM.get(mode)
        summary["modes"][mode] = {
            "auc": _mean(si_auc, mode), "ef1": _mean(si_ef1, mode),
            "hit_rate": _mean(si_hit, mode),
            "n": len(si_auc.get(mode, [])),
            "tanimoto_form": b,
            "tanimoto_auc": _mean(base_auc, b) if b else float("nan"),
            "tanimoto_ef1": _mean(base_ef1, b) if b else float("nan"),
            "tanimoto_hit_rate": _mean(base_hit, b) if b else float("nan"),
            "delta_auc": (_mean(si_auc, mode) - _mean(base_auc, b)) if b else float("nan"),
        }

    speed = {}
    speed_n_pairs = None
    from refresh_throughput import source, render
    for cell in source.pairwise():
        speed[cell['mode']] = cell['aligns_per_sec']
        speed_n_pairs = cell['n']
    for mode in SI_MODES:
        summary["modes"][mode]["aligns_per_sec"] = speed.get(mode, float("nan"))
    summary["speed_n_pairs"] = speed_n_pairs

    with open(os.path.join(HERE, "si_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _write_table(summary)
    render(summary)
    print(f"aggregated {n_targets} targets; wrote si_summary.json, si_table.tex, si_modes.svg/png")
    for mode in SI_MODES:
        m = summary["modes"][mode]
        vs = (f" (Tanimoto form {m['tanimoto_form']} {m['tanimoto_auc']:.3f}, d={m['delta_auc']:+.3f})"
              if m["tanimoto_form"] else "")
        print(f"  {mode:26} AUC={m['auc']:.3f}{vs}  {m['aligns_per_sec']:.0f} aligns/s")


def _write_table(summary):
    def esc(s):
        return s.replace('_', chr(92) + '_')

    def num(v, fmt, pending="\\emph{pending}"):
        return format(v, fmt) if (v == v) else pending    # NaN -> pending

    rows = []
    for mode in SI_MODES:
        m = summary["modes"][mode]
        if m["tanimoto_form"]:
            vs = f"{num(m['tanimoto_auc'], '.3f')} & {num(m['delta_auc'], '+.3f')}"
        else:
            vs = "-- & --"
        rows.append(
            f"\\texttt{{{esc(PRETTY[mode])}}} & {num(m['auc'], '.3f')} & {vs} & "
            f"{num(m['ef1'], '.1f')} & {num(m['hit_rate'], '.3f')} \\\\")
    body = "\n".join(rows)
    tex = (
        "% Auto-generated by SI/aggregate.py -- do not edit by hand.\n"
        "\\begin{tabular}{lcccrr}\n\\toprule\n"
        "Additional mode & ROC-AUC & Tanimoto ROC-AUC & $\\Delta$ & EF1\\% & hit rate \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(HERE, "si_table.tex"), "w") as f:
        f.write(tex)


def _plot(summary, si_auc, base_auc):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["svg.hashsalt"] = "shepherd-score-si"
    matplotlib.rcParams["font.family"] = "Arial"
    import matplotlib.pyplot as plt

    has_speed = any(summary["modes"][m]["aligns_per_sec"] == summary["modes"][m]["aligns_per_sec"]
                    for m in SI_MODES)

    def _draw_speed(ax):
        modes = [m for m in SI_MODES if summary["modes"][m]["aligns_per_sec"] ==
                 summary["modes"][m]["aligns_per_sec"]]
        x = np.arange(len(modes))
        aps = [summary["modes"][m]["aligns_per_sec"] for m in modes]
        ax.barh(x, aps, height=.7, color="#5a9367", edgecolor="#333", linewidth=.6)
        ax.set_yticks(x); ax.set_yticklabels(modes, fontsize=10.5)
        ax.set_xlabel("alignments / second", fontsize=11.5)
        ax.set_title("Pairwise alignment throughput", loc="left", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlim(0, max(aps)*1.23)
        ax.tick_params(axis='x', labelsize=10)
        ax.spines[['top','right']].set_visible(False)
        for yy, value in zip(x, aps):
            ax.text(value+max(aps)*.018, yy, f'{value:,.0f}', va='center', fontsize=10.5)
        ax.invert_yaxis()

    if has_speed:
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        _draw_speed(ax)
    else:
        fig, ax = plt.subplots(figsize=(10, 4)); ax.axis("off")
        ax.text(0.5, 0.5, "SI benchmark campaign pending\n(regenerate with figures/SI/aggregate.py)",
                ha="center", va="center", fontsize=15, color="#888")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "si_modes.svg"), bbox_inches="tight", metadata={"Date": None})
    svg_path = Path(HERE) / "si_modes.svg"
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(os.path.join(HERE, "si_modes.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
