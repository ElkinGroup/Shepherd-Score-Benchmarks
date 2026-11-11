#!/usr/bin/env python
"""Figure 3 panels B (metric bars + SEM + paired-t brackets) and C (per-target winner map), drawn
into matplotlib Axes for make_figure.py from ./data via enrichment_data.py. Panel B takes the
metric as an argument so the SI EF1%% figure reuses it."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import enrichment_data as mf                          # noqa: E402

targets = mf.load_targets()
names_alpha = sorted(targets)
present = [m for m in mf.METHODS if any(m[0] in targets[t] for t in names_alpha)]
present_fss = [m for m in present if not m[3]]

# tag -> axis label
CLEAN_LABEL = {
    "surf": "surf", "surf_esp": "surf_esp", "vol_and_surf_esp": "vol_and_surf_esp",
    "shaep_esp": "ShaEP ESP", "vol_esp": "vol_esp", "vol": "vol",
    "roshambo2_shape": "ROSHAMBO2", "shaep_shape": "ShaEP shape",
    "vol_color": "vol_color", "roshambo2_combo": "ROSHAMBO2 combo",
    "rdkit_shape": "RDKit shape", "rdkit_color": "RDKit color",
    "pharm": "pharm", "pharao": "Pharao", "phesa_pharm": "PheSA",
    "vol_lipo": "vol_lipo",
    "espsim_esp": "ESP-Sim ESP",
}
# competitor tag -> the fss mode it is compared against (outline colour)
COMP_TO_FSS = {comp: fss for fss, comp, _ in mf.COMPARISONS}
SEP_BEFORE = set(mf.SEPARATE) | {"surf_esp"}


from matplotlib.legend_handler import HandlerBase   # noqa: E402


class _HandlerErrorbar(HandlerBase):
    """Legend icon: a capped error bar."""
    def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
        xc = xd + w * 0.5
        cap = w * 0.17
        y0, y1 = yd + h * 0.02, yd + h * 0.98
        parts = [Line2D([xc, xc], [y0, y1]),
                 Line2D([xc - cap, xc + cap], [y1, y1]),
                 Line2D([xc - cap, xc + cap], [y0, y0])]
        for p in parts:
            p.set(color="#222222", lw=2.0, solid_capstyle="butt")
            p.set_transform(trans)
        return parts


class _HandlerBracket(HandlerBase):
    """Legend icon: a significance bracket."""
    def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
        x0, x1 = xd + w * 0.10, xd + w * 0.90
        y0, y1 = yd + h * 0.12, yd + h * 0.82
        line = Line2D([x0, x0, x1, x1], [y0, y1, y1, y0], color="#222222", lw=2.0, solid_capstyle="butt")
        line.set_transform(trans)
        return [line]


def values(tag, metric):
    return np.array([targets[t][tag][metric] for t in names_alpha if tag in targets[t]], dtype=float)


def _xs_A():
    """Bar x positions: one group per fss mode, its competitors attached, a fixed gap between groups."""
    pitch, group_gap = 1.0, 0.6
    xs, x = [], 0.0
    for i, (tag, _, family, ext) in enumerate(present):
        if i > 0 and not ext:
            x += group_gap
        xs.append(x); x += pitch
    return np.array(xs)


#  metric -> (bar baseline = the value a random ranking is expected to give, axis label)
METRIC_BASELINE = {"auc": (0.5, "ROC-AUC"), "ef1": (1.0, "EF1%")}


def draw_panel_B(ax, xs_A, ytop=None, metric="auc"):
    """Mean bars + SEM + paired-t brackets for ``metric``; ``ytop`` pins the y-axis maximum."""
    floor, metric_label = METRIC_BASELINE[metric]
    x_of, top_of = {}, {}
    for xi, (tag, _, family, ext) in zip(xs_A, present):
        v = values(tag, metric)
        mean = float(v.mean())
        sem = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        if ext:                                     # hollow, outlined in the matched fss mode's colour
            edge = mf.MODE_COLOR.get(COMP_TO_FSS.get(tag), "#333333")
            ax.bar(xi, mean - floor, bottom=floor, width=0.92, color="white",
                   edgecolor=edge, linewidth=1.8, zorder=2)
        else:
            ax.bar(xi, mean - floor, bottom=floor, width=0.92,
                   color=mf.MODE_COLOR.get(tag, "#888888"), edgecolor="#333333", linewidth=0.6, zorder=2)
        ax.errorbar(xi, mean, yerr=sem, fmt="none", ecolor="#222222", elinewidth=1.9,
                    capsize=4.5, capthick=1.9, zorder=4)
        x_of[tag], top_of[tag] = xi, mean + sem
    ax.axhline(floor, color="#888888", lw=1.1, ls="--", zorder=3)

    if ytop is not None:
        ax.set_ylim(floor, ytop)
    else:
        top = ax.get_ylim()[1]
        ax.set_ylim(floor, floor + (top - floor) * 1.22)

    # significance brackets
    lo, hi = ax.get_ylim()
    span = hi - lo
    pad, tick = 0.033 * span, 0.011 * span
    highest = hi
    # Every bracket in a comparison group starts from one baseline, the tallest bar the group
    # spans, so the stack clears the bars by the same margin and steps by a constant amount.
    # Anchoring each bracket to its own pair instead makes the spacing follow bar heights.
    group_floor = {}
    for tag_a, tag_b, _level in mf.COMPARISONS:
        if tag_a in top_of and tag_b in top_of:
            group_floor[tag_a] = max(group_floor.get(tag_a, 0.0), top_of[tag_a], top_of[tag_b])
    for tag_a, tag_b, level in mf.COMPARISONS:
        if tag_a not in x_of or tag_b not in x_of:
            continue
        res = mf.paired_test(targets, names_alpha, tag_a, tag_b, metric)
        if res is None:
            continue
        _n, _delta, p = res
        x1, x2 = x_of[tag_a], x_of[tag_b]
        y = group_floor[tag_a] + pad * (1.0 + 2.6 * level)
        ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick],
                lw=1.8, color="#222222", zorder=5, clip_on=False)
        mark = mf._stars(p)
        ax.text((x1 + x2) / 2.0, y + 0.12 * tick, mark, ha="center", va="bottom",
                fontsize=9 if mark == "n.s." else 12,
                style="italic" if mark == "n.s." else "normal",
                color="#333333", zorder=5, clip_on=False)
        highest = max(highest, y + 3.2 * tick)
    if ytop is None and highest > hi:
        ax.set_ylim(lo, highest)

    ax.set_xticks(xs_A)
    ax.set_xticklabels([CLEAN_LABEL.get(m[0], m[1]) for m in present],
                       rotation=40, ha="right", fontsize=13)
    ax.set_xlim(xs_A[0] - 0.7, xs_A[-1] + 0.7)
    ax.set_ylabel(f"{metric_label} ({len(names_alpha)} target avg)", fontsize=17)
    ax.tick_params(axis="y", labelsize=14)
    ax.spines[["top", "right"]].set_visible(False)


def draw_panel_C(ax):
    """Per-target top-3 fss-mode bubble map."""
    fss_tags  = [tag for tag, *_ in present_fss]
    n_fss     = len(present_fss)
    n_targets = len(names_alpha)
    RANK_SIZE = {1: 320, 2: 145, 3: 62}

    def fss_ranks(tname):
        aucs = [(mi, targets[tname].get(tag, {}).get("auc", np.nan)) for mi, tag in enumerate(fss_tags)]
        valid = sorted([(mi, v) for mi, v in aucs if not np.isnan(v)], key=lambda x: -x[1])
        return {mi: rank + 1 for rank, (mi, _) in enumerate(valid[:3])}

    def winner_idx(tname):
        for mi, rank in fss_ranks(tname).items():
            if rank == 1:
                return mi
        return n_fss

    names_sorted = sorted(names_alpha, key=lambda t: (winner_idx(t), t))

    for mi in range(n_fss):
        y = n_fss - 1 - mi
        ax.plot([-0.5, n_targets - 0.5], [y, y], "-", color="#eeeeee", lw=0.8, zorder=1)

    for rank, size in RANK_SIZE.items():
        xs, ys, cs = [], [], []
        for ti, tname in enumerate(names_sorted):
            ranks = fss_ranks(tname)
            for mi, (mtag, _, family, _) in enumerate(present_fss):
                if ranks.get(mi) == rank:
                    xs.append(ti); ys.append(n_fss - 1 - mi)
                    cs.append(mf.MODE_COLOR.get(mtag, mf.FAMILY_COLOR.get(family, "#888888")))
        ax.scatter(xs, ys, s=size, c=cs, edgecolors="white", linewidths=0.6, zorder=3)

    ax.set_xlim(-0.7, n_targets - 0.3)
    ax.set_ylim(-0.7, n_fss + 0.5)                 # headroom above the top row for the key
    ax.set_xticks(range(n_targets))
    ax.set_xticklabels(names_sorted, rotation=35, rotation_mode="anchor", ha="right", fontsize=11)
    ax.set_yticks(range(n_fss))
    ax.set_yticklabels([CLEAN_LABEL.get(present_fss[n_fss - 1 - i][0], present_fss[n_fss - 1 - i][1])
                        for i in range(n_fss)], fontsize=14)
    ax.tick_params(length=0)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

    import matplotlib.lines as mlines
    rank_handles = [
        mlines.Line2D([], [], marker="o", linestyle="none", markersize=np.sqrt(320)/1.6,
                      markerfacecolor="#8a8a8a", markeredgecolor="white", label="1st"),
        mlines.Line2D([], [], marker="o", linestyle="none", markersize=np.sqrt(145)/1.6,
                      markerfacecolor="#8a8a8a", markeredgecolor="white", label="2nd"),
        mlines.Line2D([], [], marker="o", linestyle="none", markersize=np.sqrt(62)/1.6,
                      markerfacecolor="#8a8a8a", markeredgecolor="white", label="3rd"),
    ]
    ax.legend(handles=rank_handles, title="rank among ShEPhERD-score modes (top-3 by ROC-AUC)",
              title_fontsize=10, fontsize=11, ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.0),
              frameon=True, facecolor="white", framealpha=0.92, edgecolor="#cccccc",
              handlelength=1.0, columnspacing=1.5, handletextpad=0.5, borderpad=0.6)
