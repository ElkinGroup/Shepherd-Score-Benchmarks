#!/usr/bin/env python
"""Panel drawers for the composed speed figure (make_figure.py): draw_scaling, draw_vbar (vertical
bars, linear speed axis), draw_multiproc (SI) and draw_prep (delegates to panels.py). Panel rectangles
live in make_figure.LAYOUT; running this file directly writes one PNG per panel."""
from __future__ import annotations
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Palette, families, results readers and draw_prep are taken from panels.py, not copied. Importing
# it also selects the Agg backend, so this import stays above pyplot.
import panels as _sib
import paper_results as results
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import offset_copy
import numpy as np

DATA = _sib.DATA           # one results directory, in the harness folder that produces it
OUT = HERE

# Value labels printed on the bars/points.
VALUE_FONTSIZE = 10.0            # comparison bars (draw_vbar)
VALUE_FONTSIZE_SMALL = 9.6       # the small multiproc panels (draw_multiproc), two lines per bar
# Mode key in the scaling panel: size and anchor (axes fractions, bottom-right corner) are pinned
# so refreshed data cannot reflow it; the header is right-anchored at a pinned spot.
LEGEND_FONTSIZE = 8.5
LEGEND_ANCHOR = (0.995, 0.012)
LEGEND_HEADER_XY = (0.985, 0.215)
LEGEND_HEADER_FONTSIZE = 10.0
INK, SUB, HAIR, GREY = _sib.INK, _sib.SUB, _sib.HAIR, _sib.GREY
MODE_COLOR, FAMILIES, TOOL_LABEL = _sib.MODE_COLOR, _sib.FAMILIES, _sib.TOOL_LABEL
cells, meta, _fmt = _sib.cells, _sib.meta, _sib._fmt
cells_many, REGIONA_FILES = _sib.cells_many, _sib.REGIONA_FILES

# competitor -> fss modes it is grouped under; iteration order sets the per-mode bar order
# (vol -> PheSA, ShaEP; vol_and_surf_esp -> ShaEP; pharm -> Pharao, PheSA)
COMP_MODES_B = {"pharao": ["pharm"], "phesa": ["vol", "pharm"], "shaep": ["vol", "vol_and_surf_esp"],
                "rdkit": ["vol", "vol_color"], "espsim": ["vol_and_surf_esp"]}
# GPU screening/pairwise competitors
COMP_MODES_AC = {"roshambo2": ["vol", "vol_color"]}
# What each competitor bar is, in the tool's own terms, keyed by the fss mode it stands beside:
# ROSHAMBO2's shape Tanimoto and its combined shape+color score, ShaEP's shape-only and
# shape+ESP scores, PheSA's shape and pharmacophore settings. A tool that appears once (Pharao,
# pharmacophores only) keeps its bare name.
COMP_LABEL = {("roshambo2", "vol"): "ROSHAMBO2 shape", ("roshambo2", "vol_color"): "ROSHAMBO2 combo",
              ("shaep", "vol"): "ShaEP shape", ("shaep", "vol_and_surf_esp"): "ShaEP shape+ESP",
              ("phesa", "vol"): "PheSA shape", ("phesa", "pharm"): "PheSA pharm",
              ("rdkit", "vol"): "RDKit shape", ("rdkit", "vol_color"): "RDKit shape+color",
              ("espsim", "vol_and_surf_esp"): "ESP-Sim ESP",
              ("pharao", "pharm"): "Pharao"}


def by_tool_mode(cs, size=None):
    best = {}
    for c in cs.values():
        if "aligns_per_sec" not in c or not c.get("valid", True):
            continue
        if size is not None and c.get("size") != size:
            continue
        k = (c["tool"], c["mode"])
        if k not in best or c["aligns_per_sec"] > best[k]:
            _sib.note_flagged(c, "a comparison bar")
            best[k] = c["aligns_per_sec"]
    return best


def _merged(*names):
    """One cell dict from several results files, refusing any key present in two of them."""
    m = {}
    for n in names:
        cs = cells(n)
        clash = sorted(set(m) & set(cs))
        if clash:
            raise ValueError(
                "%s collides with the files merged before it on %d cell(s): %s. Keeping only "
                "one of each, silently, is how a panel reports a number no run produced. Read "
                "one file, or give the panel a source-aware key."
                % (n, len(clash), ", ".join(clash[:4])))
        m.update(cs)
    return m


def _ktick(v):
    return "0" if v == 0 else (f"{v/1000:.0f}k" if v >= 1000 else f"{v:g}")


def _title(ax, text, pad=6):
    """Draw a panel title unless the caller pins its own (make_figure passes None)."""
    if text:
        ax.set_title(text, fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=pad)


def plateau(name):
    """CPU comparison bars are read at the largest N each mode reached: ``{('fss', mode): (size, rate)}``,
    selected by size and never by rate (a cell whose optimisation stopped early reads faster)."""
    best = {}
    for c in cells(name).values():
        if c.get("tool") != "fss" or "aligns_per_sec" not in c or not c.get("valid", True):
            continue
        if c.get("size") is None:
            continue
        k = c["mode"]
        cand = (c["size"], c["aligns_per_sec"])   # size first; a tie on size takes the faster repeat
        if k not in best or cand > best[k]:
            _sib.note_flagged(c, "a CPU plateau bar")
            best[k] = cand
    return {("fss", m): v for m, v in best.items()}


def _collect_bars(cs, comp_modes, size, fss_from, fss_size=None):
    """Family-ordered (value, colour, hollow, mode, tool) bars: fss at ``fss_size`` in ``fss_from``
    (or its plateau when ``fss_size`` is None), each followed by its hollow competitors."""
    if fss_from is not None:
        cs = {k: c for k, c in cs.items() if c.get("tool") != "fss"}
    tm = by_tool_mode(cs, size=size)
    if fss_from is not None and fss_size is not None:
        for c in _sib.cells(fss_from).values():
            if (c.get("tool") == "fss" and c.get("workload") == "screen"
                    and c.get("size") == fss_size and c.get("aligns_per_sec")):
                _sib.note_flagged(c, "a comparison bar (fss at N=%d)" % fss_size)
                tm[("fss", c["mode"])] = c["aligns_per_sec"]
    elif fss_from is not None:
        for k, (_, val) in plateau(fss_from).items():
            tm[k] = val
    bars = []
    for fam, modes in FAMILIES:
        for m in modes:
            if ("fss", m) in tm:
                bars.append((tm[("fss", m)], MODE_COLOR[m], False, m, "fss"))
            for t, cm in comp_modes.items():
                if m in cm and (t, m) in tm:
                    bars.append((tm[(t, m)], MODE_COLOR[m], True, m, t))
    return bars


def _nice_ticks(maxval, n=4):
    """Linear ticks covering [0, maxval] in ~n intervals on the 1/2/2.5/5/10 ladder."""
    if maxval <= 0:
        return [0], 1.0
    ticks = [t for t in MaxNLocator(nbins=n, steps=[1, 2, 2.5, 5, 10]).tick_values(0, maxval)
             if 0 <= t <= maxval * 1.0001]
    return ticks, (ticks[1] - ticks[0] if len(ticks) > 1 else maxval)


def draw_vbar(ax, cs, comp_modes, title, size, fss_from=None, fss_size=None):
    """Family-grouped vertical bars on a linear speed axis: fss solid, competitors hollow."""
    bars = _collect_bars(cs, comp_modes, size, fss_from, fss_size)
    xs, x, prev = [], 0.0, None
    for b in bars:
        if prev is not None and b[3] != prev:
            x += 0.85
        xs.append(x); x += 1.0; prev = b[3]
    maxval = max(b[0] for b in bars)
    ticks, step = _nice_ticks(maxval)
    top = max(ticks[-1], maxval) * 1.16          # headroom for the value labels above the bars
    for xi, (val, col, hollow, m, t) in zip(xs, bars):
        if hollow:
            ax.bar(xi, val, width=1.0, facecolor="none", edgecolor=col, linewidth=1.6)
        else:
            ax.bar(xi, val, width=1.0, facecolor=col, edgecolor="none")
        ax.text(xi, val + top * 0.012, _fmt(val), va="bottom", ha="center", rotation=90,
                fontsize=VALUE_FONTSIZE, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([m if t == "fss" else COMP_LABEL.get((t, m), TOOL_LABEL[t])
                        for (_, _, _, m, t) in bars],
                       fontsize=10.0, color=INK, rotation=35, rotation_mode="anchor",
                       ha="right", va="center")
    ax.set_ylim(0, top)
    ax.set_yticks([t for t in ticks if t <= top])
    ax.set_yticklabels([_ktick(t) for t in ticks if t <= top])
    ax.set_xlim(min(xs) - 0.75, max(xs) + 0.75)
    ax.set_ylabel("aligns / sec", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    _title(ax, title)


def draw_prep(ax, cs, title, *, index_cells=None, panel_meta=None):
    """Panel C, delegated to panels.draw_prep."""
    return _sib.draw_prep(ax, cs, title, index_cells=index_cells, panel_meta=panel_meta)


def draw_scaling(ax, gpu, cpu, title, legend=True):
    g, cp = _sib.build_curves(gpu, "GPU"), _sib.build_curves(cpu, "CPU")
    for _, ms in FAMILIES:
        for m in ms:
            if m in g:
                xs, ys, fl = zip(*g[m]); ax.plot(xs, ys, "-", color=MODE_COLOR[m], lw=2.1, marker="o", ms=3.6, label=m)
                _sib.mark_flagged(ax, xs, ys, fl)
            if m in cp:
                xs, ys, fl = zip(*cp[m]); ax.plot(xs, ys, "--", color=MODE_COLOR[m], lw=1.5, marker="s", ms=3.0, alpha=0.85)
                _sib.mark_flagged(ax, xs, ys, fl)
    ax.set_xscale("log"); ax.set_yscale("log")
    _, top = ax.get_ylim()
    _, right = ax.get_xlim()
    ax.set_xlim(10, right)
    ax.set_ylim(1, top)             # include the sub-10 CPU curve inside the logarithmic axes
    ax.set_xlabel("library size N", fontsize=12.5); ax.set_ylabel("aligns / sec", fontsize=12.5)
    ax.spines[["top", "right"]].set_visible(False)
    _title(ax, title)
    if not legend:
        return
    from matplotlib.lines import Line2D
    # eight modes over two columns, filled column-major
    keyed = ["vol", "vol_esp", "vol_lipo", "vol_color",
             "vol_and_surf_esp", "surf_esp", "pharm", "surf"]
    handles = [Line2D([0], [0], color=MODE_COLOR[m], lw=2.1, label=m) for m in keyed]
    leg1 = ax.legend(handles=handles, ncol=2, loc="lower right", bbox_to_anchor=LEGEND_ANCHOR,
                     frameon=False, handlelength=0.9, handletextpad=0.4, columnspacing=0.9,
                     labelspacing=0.35, fontsize=LEGEND_FONTSIZE)
    ax.add_artist(leg1)
    ax.text(*LEGEND_HEADER_XY, "GPU —  /  CPU --", transform=ax.transAxes,
            ha="right", va="baseline", fontsize=LEGEND_HEADER_FONTSIZE)


def draw_multiproc(ax, cs, keyfield, unit_label, note=None, max_key=None, reference=None):
    """Vertical bars, one per core/GPU count ascending, rate and speed-up printed above each."""
    rows = sorted((c for c in cs.values() if keyfield in c and "aligns_per_sec" in c
                   and (max_key is None or c[keyfield] <= max_key)), key=lambda c: c[keyfield])
    for c in rows:
        _sib.note_flagged(c, "a %s scaling bar" % keyfield)
    # steady-state wall-clock rate; aligns_per_sec_compute subtracts the per-call fixed cost
    rates = [c["aligns_per_sec"] for c in rows]
    base = rates[0] or 1.0
    x = np.arange(len(rows))
    ax.bar(x, rates, width=0.62, color=MODE_COLOR["vol"], edgecolor="#333333", linewidth=0.5)
    ticks, _ = _nice_ticks(max(rates), n=3)
    top = max(max(ticks), max(rates)) * 1.30          # headroom for the two-line labels above the bars
    ref_rate = reference[0] if reference else None
    for xi, r in zip(x, rates):
        # the first column's label clears the reference dash when there is one
        y = max(r, ref_rate) if (ref_rate and xi == 0) else r
        ax.text(xi, y + top * 0.02, f"{_fmt(r)}/s\n{r/base:.2f}x", ha="center", va="bottom",
                fontsize=VALUE_FONTSIZE_SMALL, color=INK, linespacing=1.15)
    ax.set_xticks(x); ax.set_xticklabels([str(c[keyfield]) for c in rows], fontsize=10)
    ax.set_ylim(0, top)
    ax.set_yticks([t for t in ticks if t <= top])
    ax.set_yticklabels([_ktick(t) for t in ticks if t <= top], fontsize=9.5)
    ax.set_xlim(-0.62, len(rows) - 0.38)
    ax.set_ylabel(unit_label, fontsize=10.5)
    ax.set_xlabel("CPU cores" if keyfield == "cores" else "GPUs", fontsize=10.5, labelpad=2)
    ax.spines[["top", "right"]].set_visible(False)
    if note:                                          # over the short bars, left of the tall ones
        ax.text(0.02, 0.97, note, transform=ax.transAxes, ha="left", va="top", fontsize=8.0,
                style="italic", color=SUB)
    if reference:
        # (rate, tag): a short dash across the one-core column with a one-letter tag beside it
        r, tag = reference
        ax.plot([-0.31, 0.31], [r, r], color=INK, lw=1.8, solid_capstyle="butt", zorder=4)
        ax.text(0.36, r, tag, ha="left", va="center", fontsize=8.0, style="italic", color=INK,
                zorder=5)


def _save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", os.path.basename(p))


def panel_scaling():
    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    draw_scaling(ax, results.as_cells(results.gpu_ladder()), results.as_cells(results.cpu_ladder()),
                 "Scaling with library size (screening mode)")
    _save(fig, "panel_scaling.png")


def panel_screening():
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    draw_vbar(ax, results.as_cells(results.screen_comparison()), COMP_MODES_AC,
              "GPU screening\n(N=10M, L40S)", size=10000000)
    _save(fig, "panel_gpu_screening.png")


def panel_cpu_screening():
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    draw_vbar(ax, results.as_cells(results.cpu_comparison()), COMP_MODES_B,
              "CPU screening\n(single core, N=1,000)", size=1000)
    _save(fig, "panel_cpu_screening.png")


def panel_pairwise():
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    draw_vbar(ax, results.as_cells(results.pair_comparison()), COMP_MODES_AC,
              "GPU pairwise\n(N=100k, L40S)", size=100000)
    _save(fig, "panel_gpu_pairwise.png")


# --- panel D comparison charts (make_figure.py) --------------------------------------------------
NL_NAME_FS, NL_VALUE_FS, NL_TICK_FS = 8.5, 9.75, 12.5
NL_FSS_DOT_PT, NL_COMP_DOT_PT = 11.0, 9.0   # stacked variant: fss sphere / competitor ring
NL_BALL_FS = 10.0                           # the speed printed (rotated) above each fss sphere
NL_RING_FS = 6.5                            # the number inside each competitor ring
NL_LIST_PITCH = 24.0                        # row pitch of the numbered competitor list, points
NL_LIST_INDENT = 14.0                       # list text starts this far right of its number (up to 2 digits)
NL_LIST_TITLE = "External Tool"            # heading over each numbered list (must fit the column)
NL_DOT_PT = 6.0               # marker diameter, points
NL_LABEL_DX = 12.0            # gap from the line to the first label character, points
NL_BAND = "#c9ced3"           # foot band = the full range of the panel to the left


def _nl_tick(v):
    if v == 0:
        return "0"
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


def numberline_top(bars, n=5):
    """Axis maximum on the 1/2/2.5/5/10 ladder for a set of bars: (ticks, top)."""
    maxval = max(b[0] for b in bars)
    ticks, step = _nice_ticks(maxval, n=n)
    top = ticks[-1] if ticks[-1] >= maxval else ticks[-1] + step
    return [t for t in ticks if t <= top] + ([top] if top not in ticks else []), top


NL_COL_PT = 11.0              # column pitch of the stacked (lollipop) variant, points
NL_STEM = "#d5d9dd"


def draw_numberline_stacked(ax, bars, prev_top=None, tool_only=False, start_number=1, draw_list=True):
    """The stacked variant: one thin stem per fss mode (FAMILIES order, NL_COL_PT apart), the mode's
    solid dot at its speed and its competitors as hollow dots on the same stem. The axis carries
    three ticks only (0, the previous panel's top, this panel's top); fss modes are identified by
    colour (the shared key), competitors by a number inside the ring, counting up from
    ``start_number`` in stem-then-speed order. With ``draw_list`` the numbers are keyed by an
    in-panel list (tool over setting; ``tool_only`` drops the setting line) at the top right of the
    panel; either way, returns ``(top, entries)`` with ``entries`` the same
    ``(number, tool, setting, colour)`` rows in list order, for a caller who draws its own list
    (e.g. one shared list for several panels) when ``draw_list=False``."""
    ticks, top = numberline_top(bars)
    fig = ax.figure
    x0, y0, w, h = ax.get_position().bounds
    h_pt = h * fig.get_figheight() * 72.0
    pt2val = top / h_pt
    modes = [m for _, ms in FAMILIES for m in ms if any(b[3] == m for b in bars)]
    col = {m: 6.0 + i * NL_COL_PT for i, m in enumerate(modes)}
    if draw_list:
        x_lab = col[modes[-1]] + NL_FSS_DOT_PT / 2 + 5.5  # the numbered competitor list
        ax.set_xlim(-30, 150)
    else:
        ax.set_xlim(-30, col[modes[-1]] + 15.0)           # no list: just room for the stems
    ax.set_ylim(0, top)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_position(("data", 0))
    ax.spines["left"].set_linewidth(1.1)
    # three ticks, labelled by hand so the previous-top label can clear the 0 label; the previous
    # chart's axis is drawn as a foot band only when it is smaller than this one's
    show_prev = bool(prev_top) and prev_top < top
    tick_vals = [0.0] + ([prev_top] if show_prev else []) + [top]
    ax.set_yticks(tick_vals)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=3, pad=4)
    for v in tick_vals:
        y_lab = v
        if show_prev and v == prev_top and v / pt2val < 0.68 * NL_TICK_FS:
            y_lab = 0.68 * NL_TICK_FS * pt2val            # sits just above the 0 label
        ax.text(-5.0, y_lab, _nl_tick(v), ha="right", va="center", fontsize=NL_TICK_FS, color=INK,
                clip_on=False)
    if show_prev:
        ax.add_patch(plt.Rectangle((-2.6, 0), 5.2, prev_top, facecolor=NL_BAND, edgecolor="none",
                                   zorder=1.5, clip_on=False))
    ax.plot([0, col[modes[-1]] + 9.0], [0, 0], color=INK, lw=1.1, zorder=1, clip_on=False,
            solid_capstyle="butt")                       # x axis line

    # stems (lowest competitor -> top of the group; baseline -> ball when there is no competitor)
    for m in modes:
        vals = [b[0] for b in bars if b[3] == m]
        comp = [b[0] for b in bars if b[3] == m and b[2]]
        lo = min(comp) if comp else 0.0
        ax.plot([col[m], col[m]], [lo, max(vals)], color=MODE_COLOR[m], lw=2.0, zorder=2, clip_on=False,
                solid_capstyle="butt")
    # competitors are numbered by stem (left to right), then by speed (top to bottom) within a
    # stem, counting up from start_number; the number sits inside the ring
    comp = [i for i, b in enumerate(bars) if b[2]]
    order = sorted(comp, key=lambda i: (modes.index(bars[i][3]), -bars[i][0]))
    number = {i: k + start_number for k, i in enumerate(order)}
    for i, (val, c, hollow, m, t) in enumerate(bars):
        if hollow:
            ax.plot([col[m]], [val], "o", ms=NL_COMP_DOT_PT, mfc="white", mec=c, mew=1.2, zorder=5,
                    clip_on=False)
            ax.text(col[m], val, str(number[i]), ha="center", va="center", fontsize=NL_RING_FS,
                    fontweight="bold", color=c, zorder=6, clip_on=False)
        else:
            ax.plot([col[m]], [val], "o", ms=NL_FSS_DOT_PT, mfc=c, mec="white", mew=0.6, zorder=4,
                    clip_on=False)
            ax.text(col[m], val + (NL_FSS_DOT_PT / 2 + 2.0) * pt2val, _fmt(val), rotation=90,
                    ha="center", va="bottom", fontsize=NL_BALL_FS, color=INK, zorder=6, clip_on=False)

    def split(i):
        val, c, hollow, m, t = bars[i]
        tool = TOOL_LABEL[t]
        full = COMP_LABEL.get((t, m), tool)
        setting = full[len(tool):].strip() if full.startswith(tool) else ""
        return tool, ("" if tool_only else setting)

    entries = [(number[i],) + split(i) + (bars[i][1],) for i in order]

    if draw_list:
        # the in-panel list: tool over setting, stacked from the top of the panel, no leaders
        y_pt = 2.0                                    # from the top of the axes, in points
        ax.text(x_lab, top - y_pt * pt2val, NL_LIST_TITLE, ha="left", va="top",
                fontsize=NL_VALUE_FS - 1.0, fontweight="bold", color=INK, zorder=6, clip_on=False)
        y_pt += NL_VALUE_FS * 1.6
        for i in order:
            val, c, hollow, m, t = bars[i]
            tool, setting = split(i)
            y = top - y_pt * pt2val
            ax.text(x_lab, y, "%d" % number[i], ha="left", va="top", fontsize=NL_VALUE_FS,
                    fontweight="bold", color=c, zorder=6, clip_on=False)
            ax.text(x_lab + NL_LIST_INDENT, y, tool, ha="left", va="top", fontsize=NL_VALUE_FS,
                    color=INK, zorder=6, clip_on=False)
            if setting:
                st = ax.text(x_lab + NL_LIST_INDENT, y - NL_VALUE_FS * 1.15 * pt2val, setting,
                             ha="left", va="top", fontsize=NL_VALUE_FS - 1.0, color=SUB, zorder=6,
                             clip_on=False)
                st._fig2_keep_font = True             # "pharm" here is a setting, not a mode name
                y_pt += NL_LIST_PITCH
            else:
                y_pt += NL_LIST_PITCH - NL_VALUE_FS * 1.0
    return top, entries


PREPV_LABEL_FS, PREPV_SUB_FS, PREPV_VALUE_FS, PREPV_TICK_FS = 9.5, 8.5, 9.5, 10.0
PREPV_FSS_FILL = _sib.GREY       # every fss bar is grey; the mode is the swatch below
PREPV_REF_EDGE = _sib.SUB        # the bare RDKit embed: white bar, dashed grey outline
PREPV_SWATCH_PT = 6.5            # side of the mode-colour square under each fss bar


def draw_prep_vertical(ax, cs, *, index_cells=None, panel_meta=None):
    """Panel C with the bars running vertically: one bar per prep row, its two-line label rotated
    90 degrees under the axis, the rate printed upright-rotated above the bar, the RDKit embed as a
    dashed reference line and a hairline between the existing tools and the fss rows."""
    rows = _sib.prep_rows(cs, index_cells=index_cells, panel_meta=panel_meta)
    embed = next((r[2] for r in rows if r[0] == "RDKit embed"), None)
    xs = list(range(len(rows)))
    top = max(r[2] for r in rows) * 1.32                 # headroom for the rotated rate
    tr = ax.get_xaxis_transform()                        # x in data, y in axes fractions
    for x, (label, sub, rate, mfc, mec) in zip(xs, rows):
        modes = [m for m in label.split(", ") if m in MODE_COLOR]      # the fss rows
        if modes:
            mfc = mec = PREPV_FSS_FILL
        ls = "-"
        if label == "RDKit embed":                    # the reference: hollow, dashed
            mfc, mec, ls = "white", PREPV_REF_EDGE, (0, (2.5, 1.5))
        ax.bar(x, rate, width=0.62, color=mfc, edgecolor=mec, lw=0.9, ls=ls, zorder=3)
        ax.text(x, rate + top * 0.025, "%.0fk" % (rate / 1e3), rotation=90, ha="center",
                va="bottom", fontsize=PREPV_VALUE_FS, color=INK, zorder=4)
        # one mode-colour square per mode under the axis, then the label hangs below them
        for k, m in enumerate(modes):
            ax.scatter([x + (k - (len(modes) - 1) / 2.0) * 0.24], [-0.045], marker="s",
                       s=PREPV_SWATCH_PT ** 2, c=MODE_COLOR[m], edgecolors="none", transform=tr,
                       clip_on=False, zorder=4)
        y_lab = -0.13
        ax.text(x, y_lab, label, rotation=45, ha="right", va="top", rotation_mode="anchor",
                transform=tr, fontsize=PREPV_LABEL_FS, color=INK, clip_on=False)
        if sub:                                       # a parallel line, offset 8.5 pt down-right
            ax.text(x, y_lab, sub, rotation=45, ha="right", va="top", rotation_mode="anchor",
                    transform=offset_copy(tr, fig=ax.figure, x=8.6, y=-8.6, units="points"),
                    fontsize=PREPV_SUB_FS, color=SUB, clip_on=False)
    n_top = sum(1 for r in rows if not r[0].startswith(("vol", "surf")))
    if 0 < n_top < len(rows):
        ax.plot([n_top - 0.5] * 2, [-0.20, 1.0], transform=tr, color=_sib.HAIR, lw=0.7,
                clip_on=False, zorder=1)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, top)
    ax.set_xticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color(_sib.GREY)
    ax.tick_params(axis="y", which="both", length=2.5, pad=1.5, labelsize=PREPV_TICK_FS)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[2, 4, 5, 10]))
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: "%dk" % (v / 1e3) if v else "0"))
    ax.set_ylabel("conformers / hour", fontsize=11.5)


# --- panel D as bar charts (make_figure.py --bars): same axes conventions as the stem charts ------
BAR_GAP_PT, GROUP_GAP_PT, BAR_MAX_W_PT = 1.5, 4.5, 10.0
BAR_SPAN_PT = 146.0             # the bars fill this much of the axes, from the axis line
BAR_XLABEL_FS = 7.5             # competitor names under the axis, rotated 90 degrees


def draw_bars_stacked(ax, bars, prev_top=None):
    """Panel D as grouped bars: per mode (FAMILIES order) the fss bar in the mode colour with its
    speed above, then its competitors as hollow bars named under the axis (rotated). The bars
    spread over BAR_SPAN_PT. Axis conventions (three ticks, foot band, x axis line) match
    draw_numberline_stacked; there is no list, so the chart has no x-axis title either."""
    ticks, top = numberline_top(bars)
    fig = ax.figure
    x0, y0, w, h = ax.get_position().bounds
    h_pt = h * fig.get_figheight() * 72.0
    pt2val = top / h_pt
    modes = [m for _, ms in FAMILIES for m in ms if any(b[3] == m for b in bars)]
    ax.set_xlim(-30, 150)
    ax.set_ylim(0, top)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_position(("data", 0))
    ax.spines["left"].set_linewidth(1.1)
    # the previous chart's axis only fits as a foot band when it's smaller than this one's
    show_prev = bool(prev_top) and prev_top < top
    tick_vals = [0.0] + ([prev_top] if show_prev else []) + [top]
    ax.set_yticks(tick_vals)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=3, pad=4)
    for v in tick_vals:
        y_lab = v
        if show_prev and v == prev_top and v / pt2val < 0.68 * NL_TICK_FS:
            y_lab = 0.68 * NL_TICK_FS * pt2val
        ax.text(-5.0, y_lab, _nl_tick(v), ha="right", va="center", fontsize=NL_TICK_FS, color=INK,
                clip_on=False)
    if show_prev:
        ax.add_patch(plt.Rectangle((-2.6, 0), 5.2, prev_top, facecolor=NL_BAND, edgecolor="none",
                                   zorder=1.5, clip_on=False))

    # bar width from the count: n bars in n_groups groups fill BAR_SPAN_PT (capped)
    n, ng = len(bars), len(modes)
    bw = min(BAR_MAX_W_PT, (BAR_SPAN_PT - 4.0 - (n - ng) * BAR_GAP_PT - (ng - 1) * GROUP_GAP_PT) / n)
    xs, x = {}, 4.0
    for m in modes:
        group = [i for i, b in enumerate(bars) if b[3] == m and not b[2]] +                 sorted([i for i, b in enumerate(bars) if b[3] == m and b[2]], key=lambda i: -bars[i][0])
        for k, i in enumerate(group):
            xs[i] = x + bw / 2
            x += bw + (BAR_GAP_PT if k < len(group) - 1 else 0.0)
        x += GROUP_GAP_PT
    x_end = x - GROUP_GAP_PT
    ax._fig2_no_xlabel = True
    ax.plot([0, x_end + 4.0], [0, 0], color=INK, lw=1.1, zorder=1, clip_on=False, solid_capstyle="butt")
    for i, (val, c, hollow, m, t) in enumerate(bars):
        if hollow:
            ax.bar(xs[i], val, width=bw, facecolor="white", edgecolor=c, linewidth=1.2, zorder=3)
            ax.text(xs[i], -4.0 * pt2val, COMP_LABEL.get((t, m), TOOL_LABEL[t]), rotation=90,
                    ha="center", va="top", fontsize=BAR_XLABEL_FS, color=INK, zorder=6, clip_on=False)
        else:
            ax.bar(xs[i], val, width=bw, facecolor=c, edgecolor="none", zorder=4)
            ax.text(xs[i], val + 2.5 * pt2val, _fmt(val), rotation=90, ha="center", va="bottom",
                    fontsize=NL_BALL_FS, color=INK, zorder=6, clip_on=False)
    return top


if __name__ == "__main__":
    for fn in (panel_scaling, panel_screening, panel_cpu_screening, panel_pairwise):
        fn()
