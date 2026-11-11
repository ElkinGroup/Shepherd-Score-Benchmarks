#!/usr/bin/env python3
"""DUDE-Z conformer generation, shared input validation, and retrieval metrics."""

import numpy as np

# fixed protocol settings (recorded into every output JSON)
SEED = 42               # conformer generation seed
N_CONFORMERS = 50       # conformers per library compound
SURF_POINTS = 200       # surface points per molecule
ALPHA = 0.81            # Gaussian width for the shape kernels (ShEPhERD default)


# ---- metrics (tie-aware) ---------------------------------------------------------------------
def roc_auc(labels, scores):
    """Tie-aware ROC-AUC via the Mann-Whitney rank identity (equals sklearn.roc_auc_score)."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")          # ascending; stable for ties
    ranks = np.empty(len(s), float)
    i = 0
    while i < len(s):                                # average the ranks of tied scores
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0  # 1-indexed, tie-averaged
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def enrichment_factor(labels, scores, frac=0.01):
    """Enrichment factor at the top ``frac`` of the ranked list (1.0 = random)."""
    order = np.argsort(-np.asarray(scores), kind="mergesort")   # best score first
    y = np.asarray(labels)[order]
    top_n = max(1, int(round(frac * len(y))))
    base_rate = y.mean()
    return float(y[:top_n].mean() / base_rate) if base_rate > 0 else float("nan")


def hit_rate(labels, scores, frac=0.01):
    """EF divided by the largest EF the target could yield (perfect ranking = 1.0 on every target)."""
    n = len(labels)
    n_act = int(np.sum(labels))
    top_n = max(1, int(round(frac * n)))
    if n_act == 0 or n == 0:
        return float("nan")
    ef = enrichment_factor(labels, scores, frac)
    ef_max = (min(top_n, n_act) / top_n) / (n_act / n)          # perfect ranking's EF
    return float(ef / ef_max) if ef_max > 0 else float("nan")


def evaluate(labels, scores):
    """The metrics block written into every result JSON."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("labels and scores must be nonempty vectors of equal length")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("labels must be binary and scores must be finite")
    return {"auc": {"mean": roc_auc(labels, scores)},
            "ef1": {"mean": enrichment_factor(labels, scores, 0.01)},
            "hit_rate": {"mean": hit_rate(labels, scores, 0.01)}}


# ---- inputs ----------------------------------------------------------------------------------
def read_smiles(path):
    """One SMILES per line ('SMILES [id]'); returns the list of SMILES."""
    out = []
    for line in open(path):
        parts = line.split()
        if parts:
            out.append(parts[0])
    return out


def load_query(target_dir):
    """The target's crystal-ligand query (xtal-lig.sdf from prepare_queries.py) as an RDKit Mol."""
    import os
    from rdkit import Chem
    path = os.path.join(target_dir, "xtal-lig.sdf")
    mol = next(iter(Chem.SDMolSupplier(path, removeHs=False)), None)
    if mol is None:
        raise ValueError(f"could not read crystal-ligand query: {path}")
    return mol


def ensemble_conformers(smiles, n_conformers=N_CONFORMERS):
    """SMILES -> up to ``n_conformers`` single-conformer RDKit Mols (ETKDG + MMFF, seeded); [] on failure."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = SEED
    params.pruneRmsThresh = 0.5          # drop near-duplicate conformers
    if AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params) == 0:
        return []
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
    except Exception:
        pass                              # unparameterisable by MMFF -> keep the ETKDG geometry
    Chem.SanitizeMol(mol)  # Restore RDKit aromaticity after MMFF atom typing.
    out = []
    for conf in mol.GetConformers():
        single = Chem.Mol(mol)
        single.RemoveAllConformers()
        single.AddConformer(conf, assignId=True)
        out.append(single)
    return out


# ---- library construction --------------------------------------------------------------------
def _build_compound(arg):
    """Every usable conformer of one compound, featurised (pool worker; the featuriser and
    n_conformers travel in the argument because spawn workers re-import this module)."""
    smiles, n_conformers, featurise = arg
    out = []
    for rd in ensemble_conformers(smiles, n_conformers):
        item = featurise(rd)
        if item is not None:                 # a conformer whose featurisation failed is dropped
            out.append(item)
    return out


def build_library(smiles_list, jobs, featurise, n_conformers=N_CONFORMERS):
    """``(items, owner, kept)``: the flat list of featurised conformers, the compound index of each,
    and the input indices of the compounds that kept at least one conformer. ``featurise(rd)`` must
    be module-level (it is pickled to the workers) and returns None to drop a conformer."""
    args = [(smi, n_conformers, featurise) for smi in smiles_list]
    if jobs and jobs > 1:
        import multiprocessing as mp
        # spawn, not fork: the parent imports torch, and forking an OpenMP/CUDA process deadlocks
        with mp.get_context("spawn").Pool(jobs) as pool:
            per_compound = pool.map(_build_compound, args, chunksize=4)
    else:
        per_compound = [_build_compound(a) for a in args]

    items, owner, kept = [], [], []
    for i, confs in enumerate(per_compound):
        if not confs:
            continue
        kept.append(i)
        idx = len(kept) - 1
        items.extend(confs)
        owner.extend([idx] * len(confs))
    return items, np.array(owner, dtype=int), kept


def best_per_compound(scores, conformer_owner, n_compounds):
    """Per-compound scores: each compound's best conformer score."""
    out = np.full(n_compounds, -np.inf)
    np.maximum.at(out, np.asarray(conformer_owner), np.asarray(scores, dtype=float))
    return out


# ---- shared SDF inputs ----------------------------------------------------------------------
def shared_library_paths(sdf_dir, target):
    """(library sdf, query sdf, meta json) for a target's exported library."""
    import os
    return (os.path.join(sdf_dir, "%s.sdf" % target),
            os.path.join(sdf_dir, "%s_query.sdf" % target),
            os.path.join(sdf_dir, "%s_meta.json" % target))


def read_meta(sdf_dir, target):
    """The export's index: conformer names in file order, each conformer's compound, the counts."""
    import json
    _, _, path = shared_library_paths(sdf_dir, target)
    with open(path) as fh:
        meta = json.load(fh)
    for key in ("names", "owner", "n_actives", "n_compounds"):
        if key not in meta:
            raise ValueError("%s: missing %r" % (path, key))
    if len(meta["names"]) != len(meta["owner"]):
        raise ValueError("%s: %d names but %d owner entries"
                         % (path, len(meta["names"]), len(meta["owner"])))
    if len(set(meta['names'])) != len(meta['names']):
        raise ValueError(f'{path}: duplicate conformer names')
    if any(not isinstance(i, int) or i < 0 for i in meta['owner']):
        raise ValueError(f'{path}: owner indices must be nonnegative integers')
    if not 0 <= meta["n_actives"] <= meta["n_compounds"]:
        raise ValueError("%s: n_actives=%r out of range for n_compounds=%r"
                         % (path, meta["n_actives"], meta["n_compounds"]))
    if meta["owner"] and max(meta["owner"]) >= meta["n_compounds"]:
        raise ValueError("%s: owner index %d >= n_compounds %d"
                         % (path, max(meta["owner"]), meta["n_compounds"]))
    if set(meta['owner']) != set(range(meta['n_compounds'])):
        raise ValueError(f'{path}: compound without conformers')
    return meta


def read_shared_library(sdf_dir, target):
    """``(mols, query, meta)``: the exported molecules, in ``meta['names']`` order.

    Any disagreement between the SDF and the meta index is an error. A silent skip here would put a
    pass back on its own library, which is the failure this function exists to prevent.
    """
    from rdkit import Chem
    lib_path, query_path, _ = shared_library_paths(sdf_dir, target)
    meta = read_meta(sdf_dir, target)
    mols = list(Chem.SDMolSupplier(lib_path, removeHs=False))
    if len(mols) != len(meta["names"]):
        raise ValueError("%s: %d records but meta lists %d conformers"
                         % (lib_path, len(mols), len(meta["names"])))
    unreadable = [i for i, m in enumerate(mols) if m is None]
    if unreadable:
        raise ValueError("%s: %d records did not parse (first at index %d)"
                         % (lib_path, len(unreadable), unreadable[0]))
    names = [m.GetProp("_Name") for m in mols]
    if names != list(meta["names"]):
        first = next(i for i, (a, b) in enumerate(zip(names, meta["names"])) if a != b)
        raise ValueError("%s: record order disagrees with meta at index %d (%r vs %r)"
                         % (lib_path, first, names[first], meta["names"][first]))
    query = next(iter(Chem.SDMolSupplier(query_path, removeHs=False)), None)
    if query is None:
        raise ValueError("%s: could not read the exported query" % query_path)
    return mols, query, meta


def labels_from_meta(meta):
    """The compound labels implied by the export: actives first, then decoys."""
    return np.array([1] * meta["n_actives"] + [0] * (meta["n_compounds"] - meta["n_actives"]))


def evaluate_from_meta(meta, scores):
    """Metrics from per-conformer scores given in ``meta['names']`` order.

    ``scores`` may carry ``-inf`` for a conformer a method could not score; the compound still
    ranks on its remaining conformers. A compound with no finite conformer is an error, because
    dropping it would change the ranked set and break comparability with the main text.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(meta["owner"]),):
        raise ValueError("expected %d conformer scores, got %s"
                         % (len(meta["owner"]), scores.shape))
    best = best_per_compound(scores, np.asarray(meta["owner"], dtype=int), meta["n_compounds"])
    dead = np.flatnonzero(~np.isfinite(best))
    if len(dead):
        raise ValueError("%d compounds have no scoreable conformer (first: index %d); the ranked "
                         "set would no longer match the main text" % (len(dead), dead[0]))
    return {"metrics": evaluate(labels_from_meta(meta), best)}
