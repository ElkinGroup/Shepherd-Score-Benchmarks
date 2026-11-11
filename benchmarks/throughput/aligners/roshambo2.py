"""ROSHAMBO2 adapter: GPU shape ("vol") and shape + pharmacophore color ("vol_color"), heavy-atom.
Runs in-process in its own env (no torch) at ROSHAMBO2's own default start_mode; do not raise it.
Pairwise is emulated via the full |Q|x|D| cross product (recorded per row as ``cross_overlays``)."""
from __future__ import annotations

import os

import numpy as np

from .base import Aligner, Job, MolHolder

# compute() kwargs (start_mode, optim_mode, combination_param, optimizer_settings) come from
# comparators/drivers.roshambo2_kwargs(), shared with fig3
from comparators import drivers as _drv

#: Unpickling alias: cached pools may reference ``aligners.roshambo2._MolHolder``.
_MolHolder = MolHolder

# RDKit drops SetProp properties (including _Name, which ROSHAMBO2 requires) when pickling by
# default, and the pool cache pickles the built mols.
try:
    from rdkit import Chem as _Chem
    _Chem.SetDefaultPickleProperties(_Chem.PropertyPickleOptions.AllProps)
except Exception:
    pass

# H5 group size for the screen store (ROSHAMBO2's own default is 1,000,000) and its execution
# mode ("in_memory" preloads the library; "eager" streams one group at a time). Both change the
# measured rate and are recorded per cell.
MAX_MOLS_PER_GROUP = int(os.environ.get("ROSH_MAX_MOLS_PER_GROUP", 1_000_000))
DATA_MODE = os.environ.get("ROSH_DATA_MODE", "in_memory")


def _gpu_name():
    """GPU name via nvidia-smi (the roshambo2 env has no torch)."""
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "cpu"


def _mean_score(res, use_color):
    """Mean score over a roshambo2 results dict ({query_name: DataFrame}), flattened over every per-query table; column preference mirrors drivers.roshambo2_column()."""
    prefs = (["tanimoto_combination", "tanimoto_combo_legacy", "tanimoto_shape"]
             if use_color else ["tanimoto_shape"])
    vals = []
    for _qkey, df in (res or {}).items():
        col = next((c for c in prefs if c in df.columns), None)
        if col is None:
            continue
        v = np.asarray(df[col].values, dtype=float)
        if col == "tanimoto_combo_legacy":
            v = v / 2.0                            # 0-2 -> 0-1
        vals.extend(float(x) for x in v if np.isfinite(x))
    return float(np.mean(vals)) if vals else 0.0


class Roshambo2Aligner(Aligner):
    name = "roshambo2"
    supports_screen = True

    def __init__(self, backend: str = "cuda", seed: int = 42):
        if backend != "cuda":
            raise ValueError(f"roshambo2 backend must be 'cuda', got {backend!r}")
        self.backend = backend
        self.seed = seed
        self._store_h5 = None         # on-disk H5 path for the screen path
        self._store_n = 0             # library size of the on-disk H5 store

    # --- capability + identity -------------------------------------------------
    @property
    def supported_modes(self) -> set:
        return {"vol", "vol_color"}

    def device_label(self) -> str:
        return _gpu_name()

    def atom_basis(self, mode: str) -> str:
        # ROSHAMBO2 calls Chem.RemoveHs() before building any Gaussian
        return "heavy"

    # --- PREP: featurize SMILES -> picklable holders (untimed) ------------------
    def build(self, smiles: list) -> list:
        """Flat conformer-ensemble list, ``N_CONFORMERS`` holders per surviving compound (same layout as the fss adapter)."""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from workloads import N_CONFORMERS

        mols = []
        for i, smi in enumerate(smiles):
            try:
                m = Chem.MolFromSmiles(smi)
                if m is None:
                    continue
                m = Chem.AddHs(m)
                p = AllChem.ETKDGv3()
                p.randomSeed = self.seed
                p.pruneRmsThresh = -1.0        # full ensemble; pruning would vary the count
                cids = AllChem.EmbedMultipleConfs(m, numConfs=N_CONFORMERS, params=p)
                if len(cids) < N_CONFORMERS:
                    continue                   # drop the compound, never a short block
                try:
                    AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=200)
                except Exception:
                    pass
            except Exception:
                continue
            for k, cid in enumerate(list(cids)[:N_CONFORMERS]):
                rd = Chem.Mol(m, confId=int(cid))
                # unique _Name per conformer (ROSHAMBO2 rejects duplicates)
                name = f"mol{i}_c{k}"
                rd.SetProp("_Name", name)
                # heavy-atom positions: ROSHAMBO2's shape is heavy-atom
                pos = np.asarray(Chem.RemoveHs(rd).GetConformer().GetPositions(), float)
                mols.append(MolHolder(rd, pos, name))
        return mols

    # --- PAIRWISE: emulated full cross-product via Roshambo2.compute() ----------
    def prepare_pairwise(self, mols: list, index_pairs: list, mode: str) -> Job:
        from rdkit import Chem
        from roshambo2 import Roshambo2

        use_color = (mode == "vol_color")
        # distinct query/dataset indices -> the |Q|x|D| block ROSHAMBO2 always computes;
        # n stays len(index_pairs)
        Q = sorted({i for i, _ in index_pairs})
        D = sorted({j for _, j in index_pairs})
        # rename by pool index on copies: build-time names are not unique across the parallel
        # build's chunks, and ROSHAMBO2 rejects duplicate _Names
        def _named(idx, prefix):
            m = Chem.Mol(mols[idx].mol)
            m.SetProp("_Name", f"{prefix}{idx}")
            return m
        qmols = [_named(i, "q") for i in Q]
        dmols = [_named(j, "d") for j in D]

        cross_overlays = len(Q) * len(D)
        if len(set(index_pairs)) != cross_overlays or len(index_pairs) != cross_overlays:
            raise ValueError('ROSHAMBO2 pairwise requires the exact full query/library product')

        # construct in setup (untimed); time only compute()
        r = Roshambo2(qmols, dmols, color=use_color)
        holder = {}

        _kw = {**_drv.roshambo2_kwargs(mode),
               "keep_order": True, "reduce_over_conformers": False}

        def run():
            holder["res"] = r.compute(**_kw)
            if sum(len(df) for df in holder["res"].values()) != cross_overlays:
                raise ValueError('ROSHAMBO2 returned an incomplete pairwise result')

        def score():
            return _mean_score(holder.get("res"), use_color)

        return Job(len(index_pairs), run, score, extra={"cross_overlays": cross_overlays})

    # --- SCREEN: on-disk H5 index ----------------------------------------------
    def build_store(self, mols: list, lib_indices: list, path: str, modes=None) -> None:
        """Write the library to ``<path>/lib.h5`` via prepare_from_rdkitmols (no SDF), chunked by MAX_MOLS_PER_GROUP; color features only when ``modes`` includes vol_color. Untimed."""
        from rdkit import Chem
        from roshambo2.prepare import prepare_from_rdkitmols

        use_color = (modes is not None) and ("vol_color" in modes)
        h5 = os.path.join(path, "lib.h5")

        # Copy one chunk at a time so peak RSS is bounded by a chunk, not the library; only the
        # surviving (k, idx) pairs are held, as two int64 arrays.
        import numpy as _np
        _ks = _np.fromiter((k for k, idx in enumerate(lib_indices) if mols[idx] is not None),
                           dtype=_np.int64)
        _idxs = _np.fromiter((idx for idx in lib_indices if mols[idx] is not None),
                             dtype=_np.int64)
        written = int(_ks.size)

        # prepare crashes when the count divides max_mols_per_group evenly
        mmg = MAX_MOLS_PER_GROUP
        while written and written % mmg == 0:
            mmg += 1
        if os.path.exists(h5):
            os.remove(h5)                         # save_to_h5 appends; never merge into a stale file
        # prepare_from_rdkitmols defaults n_cpus to the node's core count, not the cgroup's; pass
        # the SLURM allocation when it is known.
        _ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or None
        for _s in range(0, written, mmg):
            _chunk = []
            for _j in range(_s, min(_s + mmg, written)):
                m = Chem.Mol(mols[int(_idxs[_j])].mol)
                m.SetProp("_Name", f"lib{int(_ks[_j])}")   # unique _Name per library position
                _chunk.append(m)
            _data = prepare_from_rdkitmols(_chunk, color=use_color, quiet=True, n_cpus=_ncpu)
            _data.save_to_h5(h5, append=(_s > 0), with_smiles=True)
            del _chunk, _data                     # bound peak at one chunk, not the library
        self._store_h5 = h5 if written else None
        self._store_n = written                   # library size (== screen denominator n)

    def region_a_boundary(self) -> dict:
        """Where this tool featurizes: inside build_store (prepare_from_rdkitmols), unlike fss."""
        return {
            "input_stage": "embedded",
            "includes_featurization": True,
            "pool_stage": "MolHolder(embedded RDKit mol) -- no descriptors",
            "build_stage": "Roshambo2Mol featurization (heavy-atom COM + PCA frame, plus colour "
                           "dummies when color=True) inside prepare_from_rdkitmols, then H5 write",
            "source": "build_store -> roshambo2.prepare.prepare_from_rdkitmols",
        }

    def index_count(self, path: str) -> int:
        """Molecules in the H5 summed over groups, read from the file (not ``self._store_n``) so a partial write is detected."""
        import h5py
        with h5py.File(os.path.join(path, "lib.h5"), "r") as fh:
            return int(sum(g["r_n"].shape[0] for g in fh.values()))

    def prepare_screen(self, store_path: str, query: object, mode: str, top_k: int) -> Job:
        from roshambo2 import Roshambo2

        use_color = (mode == "vol_color")
        qmol = query.mol if isinstance(query, MolHolder) else query

        # ROSHAMBO2 requires a name; keep the frozen query unchanged.
        from rdkit import Chem
        qmol = Chem.Mol(qmol)
        if not qmol.HasProp("_Name") or not qmol.GetProp("_Name"):
            qmol.SetProp("_Name", "query")

        # resolve the on-disk H5: the store directory, else the path build_store stashed
        h5 = None
        if store_path:
            cand = os.path.join(store_path, "lib.h5")
            if os.path.exists(cand):
                h5 = cand
        if h5 is None and self._store_h5 and os.path.exists(self._store_h5):
            h5 = self._store_h5
        if h5 is None:
            raise RuntimeError("roshambo2 screen has no H5 store (build_store not run first?)")

        # n from the file: a cached store skips build_store, leaving _store_n = 0
        n = self.index_count(os.path.dirname(h5) if os.path.isfile(h5) else h5)
        if not n:
            n = self._store_n
        if not n:
            raise RuntimeError(
                "roshambo2 screen denominator is 0 -- the H5 store at %r reports no molecules; "
                "refusing to emit a cell whose throughput would be 0.0 a/s" % (h5,))
        holder = {}
        r = Roshambo2([qmol], h5, color=use_color, data_mode=DATA_MODE)

        # max_results=top_k matches what fss is asked for (keep_order=True would marshal the whole
        # library); reduce_over_conformers stays False because n counts conformers
        _kw = {**_drv.roshambo2_kwargs(mode),
               "keep_order": False, "max_results": int(top_k),
               "reduce_over_conformers": False}

        def run():
            holder["res"] = r.compute(**_kw)

        def score():
            return _mean_score(holder.get("res"), use_color)

        return Job(n, run, score)

    # sync(): inherited no-op; ROSHAMBO2 exposes no device barrier, so compute() is assumed
    # host-synchronous. This has not been confirmed against its CUDA stream handling.
    # reset(): inherited no-op; prepare_* construct a fresh Roshambo2() per cell.
