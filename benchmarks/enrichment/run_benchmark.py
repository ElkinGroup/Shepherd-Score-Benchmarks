#!/usr/bin/env python3
"""Score a DUDE-Z target from a frozen benchmark store."""
import argparse
import json
import os
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comparators import drivers as _drv, paths as _cpaths
from dudez import protocol as P
from dudez import benchmark as B

RUNNER_SHA256 = B.digest(__file__)
DRIVERS_SHA256 = B.digest(_drv.__file__)


def run_fss(args):
    B.score(args.store, args.target, args.out_dir, args.fss_modes or B.MAIN_MODES)


def run_roshambo2(args):
    meta = json.load(open(os.path.join(args.sdf_dir, f"{args.target}_meta.json")))
    lib_sdf = os.path.join(args.sdf_dir, f"{args.target}.sdf")
    qsdf = os.path.join(args.sdf_dir, f"{args.target}_query.sdf")
    names = meta["names"]

    # vol -> tanimoto_shape; vol_color -> tanimoto_combination (drivers sets the required optim_mode)
    def run_mode(mode):
        out = os.path.join(args.work_dir, f"_rosh_{args.target}_{mode}")
        scores, _ = _drv.score("roshambo2", mode, qsdf, lib_sdf, names, out)
        return _ext_eval(meta, scores)

    results = {"roshambo2_shape": run_mode("vol"), "roshambo2_combo": run_mode("vol_color")}
    _write(args, f"_ext_roshambo2_{args.target}.json", {"target": args.target, "results": results})


def run_pharao(args):
    meta = json.load(open(os.path.join(args.sdf_dir, f"{args.target}_meta.json")))
    lib_sdf = os.path.join(args.sdf_dir, f"{args.target}.sdf")
    qsdf = os.path.join(args.sdf_dir, f"{args.target}_query.sdf")
    names = meta["names"]
    tab = os.path.join(args.work_dir, f"_alignit_{args.target}.tab")
    scores, missing = _drv.score("align_it", "pharm", qsdf, lib_sdf, names, tab)
    _write(args, f"_ext_pharao_{args.target}.json",
           {"target": args.target, "results": {"pharao": _ext_eval(meta, scores, missing)}})


def run_shaep(args):
    meta = json.load(open(os.path.join(args.sdf_dir, f"{args.target}_meta.json")))
    lib_sdf = os.path.join(args.sdf_dir, f"{args.target}.sdf")
    qsdf = os.path.join(args.sdf_dir, f"{args.target}_query.sdf")
    names = meta["names"]

    # ShaEP needs "y" on stdin (overwrite prompt), a memory cap and library chunking (its field
    # grids can exceed 24 GB); one run yields both the shape and the shape+ESP column.
    chunk_dir = os.path.join(args.work_dir, f"_shaep_chunks_{args.target}")
    os.makedirs(chunk_dir, exist_ok=True)
    chunks = _split_sdf(lib_sdf, args.shaep_chunk, chunk_dir)

    shape, comb = {}, {}
    for ci, chunk in enumerate(chunks):
        out_txt = os.path.join(chunk_dir, f"_out_{ci}.txt")
        sp = _drv.Spec(
            tool="shaep", mode="vol", out_path=_Path(out_txt),
            argv=_drv.shaep_argv(queries=[qsdf], input_file=chunk, output_file=out_txt, threads=1),
            env=_cpaths.tool_env("shaep"),
            stdin=_drv.SHAEP_STDIN,
            timeout=_drv.SHAEP_TIMEOUT, mem_cap_gb=_drv.SHAEP_MEM_CAP_GB)
        _drv.run(sp)
        for d in _drv.parse_shaep(out_txt, "shape", query_names=["query"]).values():
            shape.update(d)
        for d in _drv.parse_shaep(out_txt, "combined", query_names=["query"]).values():
            comb.update(d)

    def metrics_for(by_name, tag):
        miss = [n for n in names if n not in by_name]
        if miss:
            raise ValueError(f'ShaEP/{tag}: {len(miss)} missing conformer scores')
        return _ext_eval(meta, [by_name[n] for n in names])

    results = {"shaep_shape": metrics_for(shape, "shape"),
               "shaep_esp": metrics_for(comb, "esp")}
    _write(args, f"_ext_shaep_{args.target}.json", {"target": args.target, "results": results})


# Two rigid single-conformer modes differing only in ppWeight: phesa_shape (0.0) and phesa_pharm
# (1.0, directional pharmacophore). No colour pass: PheSA's colour is directional, vol_color's is not.
def run_phesa(args):
    meta = json.load(open(os.path.join(args.sdf_dir, f"{args.target}_meta.json")))
    lib_sdf = os.path.join(args.sdf_dir, f"{args.target}.sdf")
    qsdf = os.path.join(args.sdf_dir, f"{args.target}_query.sdf")
    names = meta["names"]

    def run_mode(mode):
        tab = os.path.join(args.work_dir, f"_phesa_{args.target}_{mode}.tab")
        scores, missing = _drv.score("phesa", mode, qsdf, lib_sdf, names, tab,
                               threads=args.phesa_threads)
        if missing:
            raise ValueError(f'PheSA/{mode}: {len(missing)} missing conformer scores')
        return _ext_eval(meta, scores)

    # two files: enrichment_data.py reads phesa_pharm from _ext_phesapharm_<T>.json
    _write(args, f"_ext_phesa_{args.target}.json",
           {"target": args.target, "results": {"phesa_shape": run_mode("vol")}})
    _write(args, f"_ext_phesapharm_{args.target}.json",
           {"target": args.target, "results": {"phesa_pharm": run_mode("pharm")}})


def run_espsim(args):
    """Score every exported conformer with ESP-Sim: ``espsim_shape`` (``GetShapeSim``), ``espsim_esp``
    (``GetEspSim``, Carbo metric, signed) and ``espsim_combo`` (their sum, formed here). Alignment is
    RDKit's ``GetCrippenO3A``, shared with throughput. Charges are MMFF94 with Gasteiger fallback."""
    from rdkit import Chem
    from comparators.espsim_protocol import prepare, align_and_score
    import espsim

    meta = json.load(open(os.path.join(args.sdf_dir, f"{args.target}_meta.json")))
    lib_sdf = os.path.join(args.sdf_dir, f"{args.target}.sdf")
    qsdf = os.path.join(args.sdf_dir, f"{args.target}_query.sdf")
    names = meta["names"]

    query = next(iter(Chem.SDMolSupplier(qsdf, removeHs=False, sanitize=True)))
    if query is None:
        raise SystemExit(f"[{args.target}] could not read the query from {qsdf}")
    query_atoms = query.GetNumAtoms()
    reference = prepare(query)
    query = reference[0]
    q_added = query.GetNumAtoms() != query_atoms
    print(f"[{args.target}] query {'protonated' if q_added else 'already had H'}, "
          f"{query.GetNumAtoms()} atoms", flush=True)

    shape, esp, added = [], [], 0
    for mol in Chem.SDMolSupplier(lib_sdf, removeHs=False, sanitize=True):
        if mol is None:
            raise ValueError(f'{args.target}: unreadable library conformer')
        fit = prepare(mol)
        added += fit[0].GetNumAtoms() != mol.GetNumAtoms()
        prb, score = align_and_score(espsim, fit, reference)
        shape.append(float(espsim.GetShapeSim(prb, query)))
        esp.append(score)
    for tag, vals in (("shape", shape), ("esp", esp)):
        if len(vals) != len(names):
            raise SystemExit(f"[{args.target}] {tag}: scored {len(vals)} conformers, "
                             f"{len(names)} exported")
    if added:
        print(f"  [{args.target}] {added} library conformer(s) needed hydrogens", flush=True)

    combo = [s + e for s, e in zip(shape, esp)]
    results = {"espsim_shape": _ext_eval(meta, shape),
               "espsim_esp": _ext_eval(meta, esp),
               "espsim_combo": _ext_eval(meta, combo)}
    for tag, r in results.items():
        m = r["metrics"]
        print(f"  {tag:14} AUC={m['auc']['mean']:.3f} EF1%={m['ef1']['mean']:.1f} "
              f"hit={m['hit_rate']['mean']:.3f}", flush=True)
    _write(args, f"_ext_espsim_{args.target}.json", {"target": args.target, "results": results})


def run_rdkit(args):
    """Score every exported conformer with RDKit rdShapeAlign: ``rdkit_shape`` (shape Tanimoto of the
    shape-only alignment) and ``rdkit_color`` (shape + color of the shape-plus-color alignment, the
    TanimotoCombo convention pairing with fss vol_color and ROSHAMBO2 combo), at the module's default optimizer."""
    from rdkit import Chem
    from rdkit.Chem import rdShapeAlign as sa

    meta = json.load(open(os.path.join(args.sdf_dir, f"{args.target}_meta.json")))
    lib_sdf = os.path.join(args.sdf_dir, f"{args.target}.sdf")
    qsdf = os.path.join(args.sdf_dir, f"{args.target}_query.sdf")
    names = meta["names"]

    query = next(iter(Chem.SDMolSupplier(qsdf, removeHs=False, sanitize=True)))
    if query is None:
        raise SystemExit(f"[{args.target}] could not read the query from {qsdf}")

    def prepare(mol, colors):
        """RDKit 2024.09 took ``useColors`` as a bool; 2025.03 and later take ShapeInputOptions."""
        opts_cls = getattr(sa, "ShapeInputOptions", None)
        if opts_cls is None:
            return sa.PrepareConformer(mol, -1, colors)
        opts = opts_cls()
        opts.useColors = colors
        return sa.PrepareConformer(mol, -1, opts)

    def run_mode(mode):
        colors = mode == "vol_color"
        ref = prepare(Chem.Mol(query), colors)
        scores = []
        for mol in Chem.SDMolSupplier(lib_sdf, removeHs=False, sanitize=True):
            if mol is None:
                raise ValueError(f'{args.target}: unreadable library conformer')
            shape, color = sa.AlignMol(ref, Chem.Mol(mol), -1, useColors=colors)
            scores.append(float(shape + color) if colors else float(shape))
        if len(scores) != len(names):
            raise SystemExit(f"[{args.target}] {mode}: scored {len(scores)} conformers, "
                             f"{len(names)} exported")
        return _ext_eval(meta, scores)

    results = {"rdkit_shape": run_mode("vol"), "rdkit_color": run_mode("vol_color")}
    for tag, r in results.items():
        m = r["metrics"]
        print(f"  {tag:14} AUC={m['auc']['mean']:.3f} EF1%={m['ef1']['mean']:.1f} "
              f"hit={m['hit_rate']['mean']:.3f}", flush=True)
    _write(args, f"_ext_rdkit_{args.target}.json", {"target": args.target, "results": results})


def _split_sdf(sdf_path, chunk_size, out_dir):
    """Split a multi-record SDF into chunk-sized files (records end in a '$$$$' line)."""
    records, buf = [], []
    for line in open(sdf_path):
        buf.append(line)
        if line.rstrip() == "$$$$":
            records.append("".join(buf))
            buf = []
    paths = []
    for i in range(0, len(records), chunk_size):
        p = os.path.join(out_dir, f"_chunk_{i // chunk_size}.sdf")
        open(p, "w").write("".join(records[i:i + chunk_size]))
        paths.append(p)
    return paths


def _ext_eval(meta, scores, missing=()):
    """Metrics for an external tool from its per-conformer scores in ``meta['names']`` order."""
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('external tool returned a nonfinite conformer score')
    return dict(P.evaluate_from_meta(meta, scores), _scores=scores,
                missing_names=list(missing), missing_score_policy='zero' if missing else None)


def _write(args, fname, payload):
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, fname)
    for mode, result in payload['results'].items():
        scores = result.pop('_scores')
        score_path = os.path.join(args.out_dir, f'{args.target}.{mode}.npy')
        with open(score_path + '.tmp', 'wb') as f:
            np.save(f, scores)
        os.replace(score_path + '.tmp', score_path)
        result.update(n_scored=len(scores), scores_sha256=B.digest(score_path))
    payload["benchmark_id"] = args.manifest["benchmark_id"]
    payload['runner_sha256'] = RUNNER_SHA256
    payload['drivers_sha256'] = DRIVERS_SHA256
    if args.what == 'espsim':
        from comparators import espsim_protocol
        payload['espsim_protocol_sha256'] = B.digest(espsim_protocol.__file__)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", required=True,
                    choices=["fss", "roshambo2", "pharao", "shaep", "phesa", "rdkit", "espsim"],
                    help="which pass to run. fss + roshambo2 run in their own conda env; "
                         "pharao/shaep/phesa run in any numpy env (e.g. fss) -- their binaries come "
                         "from COMPARATORS_HOME. phesa additionally needs a JDK-17 java: set "
                         "JAVA_BIN=<jdk17-env>/bin/java (else paths.py picks system Java 8 and fails).")
    ap.add_argument("--target", required=True, help="target name, e.g. ACES")
    ap.add_argument("--store", required=True, help="frozen DUDE-Z benchmark root")
    ap.add_argument("--fss-modes", nargs="+", choices=B.MAIN_MODES)
    ap.add_argument("--out-dir", default="data", help="where the result JSONs are written")
    ap.add_argument("--shaep-chunk", type=int, default=150,
                    help="ShaEP library chunk size -- bounds its in-RAM field grids (shaep pass)")
    ap.add_argument("--phesa-threads", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)),
                    help="threads for PheSA descriptor build + scoring (phesa pass)")
    args = ap.parse_args()
    args.sdf_dir = os.path.join(args.store, args.target)
    args.work_dir = os.path.join(args.out_dir, "work", args.target, args.what)
    os.makedirs(args.work_dir, exist_ok=True)
    if args.what != "fss":
        # Verify the same input identity before an external tool reads the SDFs.
        args.manifest = B.verify(args.store, args.target)
    {"fss": run_fss, "roshambo2": run_roshambo2, "pharao": run_pharao,
     "shaep": run_shaep, "phesa": run_phesa, "rdkit": run_rdkit,
     "espsim": run_espsim}[args.what](args)


if __name__ == "__main__":
    main()
