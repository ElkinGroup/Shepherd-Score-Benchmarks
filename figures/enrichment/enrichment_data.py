#!/usr/bin/env python3
# figures/enrichment/enrichment_data.py -- the data layer for Figure 3: loads the per-target results
# from ./data and owns the method list, comparisons, colours and the paired t-test. panels.py draws.

import glob, json, os
import numpy as np
from scipy import stats as _stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "..", "benchmarks", "publication_data"))     # make_figure.py may repoint this

# (tag, label, family, external). Order drives panel B bar order and panel C row order; each fss
# mode is followed by the external tool(s) it is compared against. Both ShaEP rows come from one
# run (_ext_shaep_<T>.json); phesa_pharm comes from _ext_phesapharm_<T>.json.
METHODS = [
    ("vol",                      "fss vol",                "volumetric shape", False),
    ("roshambo2_shape",          "ROSHAMBO2",              "volumetric shape", True),
    ("shaep_shape",              "ShaEP shape",            "volumetric shape", True),
    ("rdkit_shape",              "RDKit shape",            "volumetric shape", True),
    ("vol_esp",                  "fss vol+esp",            "volumetric shape",      False),
    ("vol_lipo",                 "fss vol_lipo",           "shape + lipophilicity", False),
    ("vol_color",                "fss vol+color",          "shape + color",         False),
    ("roshambo2_combo",          "ROSHAMBO2 combo",        "shape + color",         True),
    ("rdkit_color",              "RDKit color",            "shape + color",         True),
    ("vol_and_surf_esp",         "fss vol_and_surf_esp",   "shape + ESP",           False),
    ("shaep_esp",                "ShaEP ESP",              "shape + ESP",           True),
    ("espsim_esp",               "ESP-Sim ESP",            "shape + ESP",           True),
    ("surf_esp",                 "fss surf_esp",           "surface shape",         False),
    ("surf",                     "fss surf",               "surface shape",         False),
    ("pharm",                    "fss pharm",              "pharmacophore",         False),
    ("pharao",                   "Pharao",                 "pharmacophore",         True),
    ("phesa_pharm",              "PheSA pharm",            "pharmacophore",         True),
]

# Head-to-head brackets over panel B: (fss mode, external tool doing equivalent work, stacking
# level). Two-sided paired t-test over the targets both methods scored.
COMPARISONS = [
    ("vol_and_surf_esp", "shaep_esp",       0),
    ("vol_and_surf_esp", "espsim_esp",      1),
    ("vol",              "roshambo2_shape", 0),
    ("vol",              "shaep_shape",     1),
    ("vol",              "rdkit_shape",     2),
    ("vol_color",        "roshambo2_combo", 0),
    ("vol_color",        "rdkit_color",     1),
    ("pharm",            "pharao",          0),
    ("pharm",            "phesa_pharm",     1),
]

P_STRONG, P_WEAK = 0.005, 0.05


def _stars(p):
    """Significance label for a p-value (matches the legend key)."""
    if p < P_STRONG:
        return "**"
    if p < P_WEAK:
        return "*"
    return "n.s."


def paired_test(targets, names, tag_a, tag_b, metric="auc"):
    """(n_paired, mean delta, p) for a two-sided paired t-test over shared targets; None if < 2."""
    both = [t for t in names if tag_a in targets[t] and tag_b in targets[t]]
    if len(both) < 2:
        return None
    a = np.array([targets[t][tag_a][metric] for t in both], dtype=float)
    b = np.array([targets[t][tag_b][metric] for t in both], dtype=float)
    p = float(_stats.ttest_rel(a, b).pvalue)
    return len(both), float((a - b).mean()), p

# per-mode colours (lab_palette.ase + ff8b00)
MODE_COLOR = {
    "surf":                    "#1a619f",   # blue
    "surf_esp":                "#006d72",   # teal
    "vol_and_surf_esp":        "#007d00",   # green
    "vol_esp":                 "#7a2b73",   # magenta
    "vol":                     "#5554ae",   # periwinkle
    "vol_color":               "#a52f2f",   # red
    "pharm":                   "#ff8b00",   # orange
    "vol_lipo":                "#d35d7f",   # rose-pink
}

# family colour: outline of hollow external-tool bars, and the fallback
FAMILY_COLOR = {
    "surface shape":         "#1a619f",
    "shape + ESP":           "#007d00",
    "volumetric shape":      "#5554ae",
    "shape + color":         "#a52f2f",
    "shape + lipophilicity": "#d75c63",
    "pharmacophore":         "#ff8b00",
}

SEPARATE = {"vol_esp"}


def load_targets():
    """Per-target metrics from the single assembled publication campaign."""
    from pathlib import Path
    path = Path(DATA) / 'enrichment.json'
    return json.loads(path.read_text())['per_target']
