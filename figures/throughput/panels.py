#!/usr/bin/env python
"""Speed-figure panel drawers, each into a matplotlib Axes from ./results/*.json (every bar comes
from a results file). make_figure.py composes the figure; draw_prep here is the shared panel C.
Running this file directly writes one PNG per panel."""
from __future__ import annotations
import glob
import json, os, sys
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["svg.fonttype"] = "none"          # keeps text editable in the SVG
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["axes.labelsize"] = 11.5
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "..", "benchmarks", "throughput", "results"))   # a composer may repoint this
OUT = HERE

INK, SUB, HAIR, GREY = "#1b1b1b", "#5b636b", "#dfe3e6", "#b9bfc6"
MODE_COLOR = {"vol": "#5554ae", "vol_esp": "#7a2b73", "vol_color": "#a52f2f",
              "surf": "#1a619f", "surf_esp": "#006d72", "vol_and_surf_esp": "#007d00", "pharm": "#ff8b00",
              "vol_lipo": "#d35d7f"}                              # match fig3
# A missing (fss, mode) cell draws nothing. Family names are organisational only.
TOOL_LABEL = {"roshambo2": "ROSHAMBO2", "phesa": "PheSA", "shaep": "ShaEP", "pharao": "Pharao",
              "rdkit": "RDKit"}
FAMILIES = [("volumetric shape", ["vol", "vol_esp"]), ("shape + lipophilicity", ["vol_lipo"]),
            ("shape + color", ["vol_color"]), ("shape + ESP", ["vol_and_surf_esp"]),
            ("surface shape", ["surf_esp", "surf"]), ("pharmacophore", ["pharm"])]


# Stamped onto each loaded cell by the renderer (never written back): which results file and panel
# it came from. Cell keys do not encode the panel, so cells from two files can collide.
SRC, PANEL = "_source_file", "_source_panel"
# A cell that fired one of these is surfaced, never dropped.
SPREAD_FLAGS = ("spread_flagged", "build_spread_flagged", "prep_spread_flagged")


def cells(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return {}
    doc = json.load(open(p))
    cs = doc.get("cells", {})
    panel = (doc.get("meta") or {}).get("panel")
    for c in cs.values():
        if isinstance(c, dict):
            c[SRC], c[PANEL] = name, panel
    return cs


REGIONA_FILES = ["REGIONA_%s.json" % t for t in ("fss", "roshambo2", "pharao", "phesa", "shaep")]


def cells_many(names):
    """The cells of several results files as one dict; a key seen in two files is an error."""
    out = {}
    for name in names:
        for k, c in cells(name).items():
            if k in out:
                raise ValueError("cell %r is in two of %s" % (k, names))
            out[k] = c
    return out


def meta(name):
    p = os.path.join(DATA, name)
    return json.load(open(p)).get("meta", {}) if os.path.exists(p) else {}


def flagged(c):
    """Which of a cell's spread tripwires fired (older cells predate the field)."""
    return [k for k in SPREAD_FLAGS if c.get(k)]


def note_flagged(c, where):
    """Print a cell's fired tripwires to stderr, naming where it is used."""
    fl = flagged(c)
    if fl:
        sys.stderr.write(
            "fig2 WARNING: %s uses a cell that flagged itself -- %s | %s | %s | spread %s%% | %s\n"
            % (where, c.get(SRC), c.get("mode"), c.get("size"), c.get("spread_pct"), ", ".join(fl)))
    return bool(fl)


def one_source(points, what):
    """The single results file every cell in ``points`` came from; raise if they disagree."""
    by_src = {}
    for c in points:
        by_src.setdefault(c.get(SRC), []).append(c)
    if len(by_src) > 1:
        detail = "; ".join(
            "%s: %s" % (src, ", ".join("N=%s -> %s" % (c.get("size"), c.get("aligns_per_sec"))
                                       for c in sorted(cs, key=lambda c: c.get("size") or 0)))
            for src, cs in sorted(by_src.items(), key=lambda kv: str(kv[0])))
        raise ValueError(
            "%s mixes %d results files: %s. Every point of one curve must come from ONE panel -- "
            "cell keys do not carry the panel, so a merge of two files silently substitutes one "
            "file's cell for another's at the sizes they share." % (what, len(by_src), detail))
    return next(iter(by_src))


def _fmt(v):
    if v >= 10000: return f"{v/1000:.0f}k"
    if v >= 1000:  return f"{v/1000:.1f}k"
    if v >= 10:    return f"{v:.0f}"
    return f"{v:.1f}"


def _title(ax, text, pad=6):
    ax.set_title(text, fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=pad)


# --- panel C: library preparation --------------------------------------------
P4_SCHEMA = 2                    # older schemas are refused; see p4_construction.py
# Panel C is conformers prepared per hour on one core (3600e6 / us_per_conformer), one prep_molecule
# cell of P4.json per row; every row is the total from SMILES and the dashed reference is the bare
# RDKit embed. Two rows fold in the measured store write (_PREP_PLUS_INDEX).
_PREP_ROWS = [  # (tool, cell mode, label line 1, label line 2, colour key)
    ("existing",  "embed",    "RDKit embed",            "ETKDGv3 + MMFF94",        None),
    ("roshambo2", "embed",    "ROSHAMBO2",              "+ PCA frame, H5 store",   "tool"),
    ("pharao",    "embed",    "ShaEP / PheSA / Pharao", "",                        "tool"),
    ("fss",       "vol",      "vol, vol_lipo",          "volume profile, store",   "vol"),
    ("fss",       "pharm",    "vol_color, pharm",       "+ pharmacophores",        "pharm"),
    ("fss",       "vol_esp",  "vol_esp",                "+ xTB charges",           "vol_esp"),
    ("fss",       "surf",     "surf",                   "+ surface",               "surf"),
    ("fss",       "surf_esp", "surf_esp, vol_and_surf_esp", "+ surface, xTB charges", "surf_esp"),
]
# Row labels are drawn left of the axes in a column this many axes widths wide (the panel's left
# 45% against the axes' 55%); panel_C() lays its own figure out the same way.
PREP_LABEL_COL = 0.45 / 0.55
PREP_LABEL_FS, PREP_SUB_FS, PREP_VALUE_FS, PREP_TICK_FS, PREP_XLABEL_FS = 7.2, 6.3, 7.0, 7.4, 9.0
PREP_BAR_H = 0.56
# (tool, prep mode) -> the build_index cell (tool, INDEX_MODE) of REGIONA_<tool>.json whose
# us_per_mol_indexed is added to the row.
_PREP_PLUS_INDEX = {("roshambo2", "embed"): "roshambo2", ("fss", "vol"): "fss"}
INDEX_MODE = "vol"


def _us(v):
    if v < 1e3:
        return f"{v:.0f} µs"
    if v < 1e6:
        return f"{v / 1e3:.0f} ms"
    return f"{v / 1e6:.1f} s"


def _pick(cs, workload, tool, mode=None):
    """Largest-N cell matching (workload, tool, mode) that passed its own validity tripwires."""
    best = None
    for c in cs.values():
        if c.get("workload") != workload or c.get("tool") != tool:
            continue
        if mode is not None and c.get("mode") != mode:
            continue
        if workload == "prep_molecule":
            if not c.get("prep_valid") or c.get("prep_spread_flagged"):
                continue
        # an unstaged build measured pool shard I/O, not the tool
        elif not (c.get("build_valid") and c.get("build_staged")) or c.get("build_spread_flagged"):
            continue
        if best is None or (c.get("size") or 0) > (best.get("size") or 0):
            best = c
    return best


def prep_rows(cs, *, index_cells=None, panel_meta=None):
    """Panel C's rows, ``[(label, sub, rate, facecolour, edgecolour)]`` in _PREP_ROWS order; rate is
    conformers per hour on one core. ``index_cells`` (the REGIONA files) supplies the store write
    _PREP_PLUS_INDEX folds in."""
    if (panel_meta or {}).get("schema", 1) < P4_SCHEMA:
        raise ValueError("results/P4.json is the legacy schema, whose competitor bars timed "
                         "single-character SMILES and whose fss bars bypassed the adapters. "
                         "Re-run p4_construction.py; this panel will not render it.")
    rows = []                                             # (label, sub, rate, mfc, mec)
    for tool, mode, label, sub, ckey in _PREP_ROWS:
        c = _pick(cs, "prep_molecule", tool, mode)
        if not (c and c.get("us_per_conformer")):
            print("  panel C: no prep_molecule cell for %s/%s -- row skipped" % (tool, mode))
            continue
        note_flagged(c, "panel C prep row %s/%s" % (tool, mode))
        us = c["us_per_conformer"]
        itool = _PREP_PLUS_INDEX.get((tool, mode))
        if itool:
            ic = _pick(index_cells or {}, "build_index", itool, INDEX_MODE)
            if not (ic and ic.get("us_per_mol_indexed")):
                raise ValueError("panel C row %s/%s folds in the %s store write, but no valid "
                                 "build_index cell was found in %s" % (tool, mode, itool, REGIONA_FILES))
            note_flagged(ic, "panel C store write for %s" % itool)
            us += ic["us_per_mol_indexed"]
        rate = 3600e6 / us                                # conformers per hour
        if ckey is None:
            mfc, mec = GREY, GREY
        elif ckey == "tool":
            mfc, mec = "white", INK
        else:
            mfc = mec = MODE_COLOR[ckey]
        rows.append((label, sub, rate, mfc, mec))
    if not rows:
        raise ValueError("results/P4.json carries no valid prep_molecule cells")
    return rows


def draw_prep(ax, cs, title, *, index_cells=None, panel_meta=None):
    """Panel C: one horizontal bar per prep_rows row on a linear conformers-per-hour axis, the RDKit
    embed as a dashed reference."""
    rows = prep_rows(cs, index_cells=index_cells, panel_meta=panel_meta)
    embed = next((r[2] for r in rows if r[0] == "RDKit embed"), None)

    ys = [-float(i) for i in range(len(rows))]
    tr = ax.get_yaxis_transform()                         # x in axes fractions, y in data
    for yy, (label, sub, rate, mfc, mec) in zip(ys, rows):
        ax.barh(yy, rate, height=PREP_BAR_H, color=mfc, edgecolor=mec, lw=0.9, zorder=3)
        ax.text(-0.03, yy + 0.06, label, transform=tr, ha="right", va="baseline",
                fontsize=PREP_LABEL_FS, color=INK)
        ax.text(-0.03, yy - 0.13, sub, transform=tr, ha="right", va="top",
                fontsize=PREP_SUB_FS, color=SUB)
        # the printed rate sits right of the bar, or right of the embed line when it would straddle it
        ref = embed or rate
        xt = (max(rate, ref) if -0.22 * ref < rate - ref < 0.1 * ref else rate) + ref * 0.03
        ax.text(xt, yy, "{:,.0f} /h".format(rate), va="center",
                fontsize=PREP_VALUE_FS, color=INK, zorder=4)
    if embed:
        ax.axvline(embed, color=SUB, lw=0.8, ls=(0, (3, 2)), zorder=2)
    # a hairline between the existing tools and the fss rows
    n_top = sum(1 for r in rows if not r[0].startswith(("vol", "surf")))
    if 0 < n_top < len(rows):
        ax.plot([-PREP_LABEL_COL, 1.0], [-(n_top - 0.5)] * 2, transform=tr, color=HAIR, lw=0.7,
                clip_on=False, zorder=1)

    xmax = max(r[2] for r in rows) * 1.42                 # room for the printed rate
    ax.set_xlim(0, xmax)
    ax.set_ylim(ys[-1] - 0.6, 0.6)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", which="both", length=2.5, pad=1.5, labelsize=PREP_TICK_FS)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%dk" % (v / 1e3) if v else "0"))
    ax.set_xlabel("conformers prepared / hour", fontsize=PREP_XLABEL_FS)
    _title(ax, title, pad=18)

    print("  panel C rates (conformers/hour, one core): "
          + ", ".join("%s=%.0f" % (r[0], r[2]) for r in rows))
    if embed:
        print("  panel C, 10^7 conformers: embed %.0f CPU-h; " % (1e7 / embed)
              + ", ".join("%s %.0f CPU-h" % (r[0].split(",")[0], 1e7 / r[2])
                          for r in rows if r[3] not in ("white", GREY)))


def mark_flagged(ax, xs, ys, fl):
    """Report on stdout the points whose cell flagged its own spread; they are drawn, not ringed."""
    pts = [(x, y) for x, y, f in zip(xs, ys, fl) if f]
    for x, y in pts:
        print("  spread-flagged (drawn, not ringed): N=%g at %.0f aligns/s" % (x, y))


def build_curves(cs, which):
    """``{mode: [(size, aligns_per_sec, flagged), ...]}`` for one file's screen cells; each mode must come from one file."""
    d = {}
    for c in cs.values():
        if c.get("workload") != "screen" or "aligns_per_sec" not in c or not c.get("valid", True):
            continue
        d.setdefault(c["mode"], []).append(c)
    out = {}
    for m, pts in d.items():
        where = "the %s %s scaling curve" % (which, m)
        one_source(pts, where)
        for c in pts:
            note_flagged(c, where)
        out[m] = sorted((c["size"], c["aligns_per_sec"], bool(flagged(c))) for c in pts)
    return out


def draw_scaling(ax, gpu, cpu, title):
    g, cp = build_curves(gpu, "GPU"), build_curves(cpu, "CPU")
    for _, ms in FAMILIES:
        for m in ms:
            if m in g:
                xs, ys, fl = zip(*g[m]); ax.plot(xs, ys, "-", color=MODE_COLOR[m], lw=2.1, marker="o", ms=3.6, label=m)
                mark_flagged(ax, xs, ys, fl)
            if m in cp:
                xs, ys, fl = zip(*cp[m]); ax.plot(xs, ys, "--", color=MODE_COLOR[m], lw=1.5, marker="s", ms=3.0, alpha=0.85)
                mark_flagged(ax, xs, ys, fl)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("library size N", fontsize=12.5); ax.set_ylabel("aligns / sec", fontsize=12.5)
    ax.spines[["top", "right"]].set_visible(False)
    # legend at the right edge: 7 modes in two columns, the GPU/CPU title above, surf centred below
    from matplotlib.lines import Line2D
    keyed = ["vol", "vol_esp", "vol_lipo", "vol_color",
             "vol_and_surf_esp", "surf_esp", "pharm"]
    h6 = [Line2D([0], [0], color=MODE_COLOR[m], lw=2.1, label=m) for m in keyed]
    leg1 = ax.legend(handles=h6, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 0.11),
                     frameon=False, handlelength=0.9, handletextpad=0.4, columnspacing=0.9,
                     labelspacing=0.35, fontsize=9)
    ax.add_artist(leg1)
    ax.figure.canvas.draw()                       # realise handle positions
    inv = ax.transAxes.inverted()
    hc = []                                        # (centre_x, centre_y, half_width) of each colour handle
    for h in leg1.legend_handles:
        e = h.get_window_extent()
        (x0, y0), (x1, y1) = inv.transform((e.x0, e.y0)), inv.transform((e.x1, e.y1))
        hc.append(((x0 + x1) / 2, (y0 + y1) / 2, abs(x1 - x0) / 2))
    col_cx = sorted({round(c[0], 4) for c in hc})                 # the two column handle centres
    row_cy = sorted({round(c[1], 4) for c in hc}, reverse=True)   # rows, top -> bottom
    halfw = max(c[2] for c in hc)
    surf_cx = 0.5 * (col_cx[0] + col_cx[-1])
    surf_y = row_cy[-1] - (row_cy[0] - row_cy[1])                 # one row-gap below the bottom row
    ax.add_line(Line2D([surf_cx - halfw, surf_cx + halfw], [surf_y, surf_y], color=MODE_COLOR["surf"],
                       lw=2.1, transform=ax.transAxes, clip_on=False))
    ax.text(surf_cx + halfw + 0.008, surf_y, "surf", transform=ax.transAxes, ha="left", va="center", fontsize=9)
    bb = leg1.get_window_extent().transformed(inv)                # title centred over the key block
    ax.text(0.5 * (bb.x0 + bb.x1), bb.y1 + 0.02, "GPU —  /  CPU --", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=10)
    _title(ax, title)


def draw_multiproc(ax, cs, keyfield, unit_label, note=None, max_key=None):
    rows = sorted((c for c in cs.values() if keyfield in c and "aligns_per_sec" in c
                   and (max_key is None or c[keyfield] <= max_key)), key=lambda c: c[keyfield])
    compute = any("aligns_per_sec_compute" in c for c in rows)
    rates = [c.get("aligns_per_sec_compute", c["aligns_per_sec"]) for c in rows]
    base = rates[0] or 1.0
    y = np.arange(len(rows))
    ax.barh(y, rates, height=0.62, color=MODE_COLOR["vol"], edgecolor="#333333", linewidth=0.5)
    for yi, r in zip(y, rates):
        ax.text(r, yi, f"  {_fmt(r)}/s\n  {r/base:.1f}x", va="center", fontsize=9.4, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([str(c[keyfield]) for c in rows], fontsize=11)
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v/1000:g}k"))
    ax.set_xlabel(unit_label, fontsize=11.5)
    ax.set_ylabel("CPU cores" if keyfield == "cores" else "GPUs", fontsize=11.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.42)
    if note:
        ax.text(0.97, 0.12, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, style="italic", color=SUB)


def _save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", os.path.basename(p))


def panel_C():
    fig = plt.figure(figsize=(3.44, 3.68))               # the panel's size in the composed figure
    ax = fig.add_axes([0.47, 0.13, 0.50, 0.74])          # left 45% of the panel = row-label column
    draw_prep(ax, cells("P4.json"), "Library preparation cost",
              index_cells=cells_many(REGIONA_FILES), panel_meta=meta("P4.json"))
    _save(fig, "panel_C.png")


def panel_E():
    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    draw_scaling(ax, cells("P5.h200.json"), cells("P5_cpu.json"),
                 "Scaling of CPU and GPU screening alignment speed with library size")
    _save(fig, "panel_E.png")


def panel_F():
    # CPU cores + GPUs subplots share one title
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    draw_multiproc(ax1, cells("P6_cpu.json"), "cores", "aligns / sec", note="bandwidth\nlimited", max_key=16)
    draw_multiproc(ax2, cells("P6_gpu.json"), "gpus", "aligns / sec", max_key=4)
    fig.suptitle("Multiprocessing scaling across CPU cores and GPUs, mode = vol",
                 fontsize=12.5, fontweight="bold", color=INK, x=0.015, ha="left", y=1.0)
    fig.subplots_adjust(wspace=0.55, top=0.85, bottom=0.19, left=0.09, right=0.98)
    _save(fig, "panel_F.png")


if __name__ == "__main__":
    for fn in (panel_C, panel_E, panel_F):
        fn()
