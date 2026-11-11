"""align-it (Pharao) adapter: CPU pharmacophore overlap via the ``align_it`` binary; one mode, pharm.
Pairwise issues one subprocess per distinct reference; screen runs one query vs the library SDF.
argv and parser come from comparators.drivers; never pass --rankBy or --scoreOnly."""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np

from .base import Job, SubprocessAligner, count_sdf

from comparators import drivers as _drv, paths as _cpaths


# align_it scores file: no header, tab-sep, 11 fields; field[2]=dbase mol name, field[-3]=TANIMOTO
_parse_alignit = _drv.parse_align_it


class PharaoAligner(SubprocessAligner):
    name = "pharao"
    supports_screen = True

    def __init__(self, backend: str = "cpu", seed: int = 42):
        if backend != "cpu":
            raise ValueError(f"pharao backend must be cpu, got {backend!r}")
        self.backend = backend
        self.seed = seed
        # align_it needs its own lib dir and OpenBabel on LD_LIBRARY_PATH; tool_env builds that
        # once here, not inside the timed run()
        self._env = _cpaths.tool_env("align_it")
        self._dir = self._workdir()

    # --- capability + identity -------------------------------------------------
    @property
    def supported_modes(self) -> set:
        return {"pharm"}

    def device_label(self) -> str:
        return f"CPU (align_it, {self.threads} thr)"

    def atom_basis(self, mode: str) -> str:
        # a pharmacophore-feature point set, not atoms
        return "pharm"

    #: align-it keys its output by SDF title, so its molecules are named at build()
    name_mols = True

    # --- PAIRWISE: one align_it call per distinct reference ---------------------
    def prepare_pairwise(self, mols: list, index_pairs: list, mode: str) -> Job:
        assert mode == "pharm", f"pharao only supports 'pharm', got {mode!r}"

        # group fit indices by reference index
        by_ref = {}
        for i, j in index_pairs:
            by_ref.setdefault(i, []).append(j)

        # one reference SDF + one db SDF per distinct ref; each db record's title names its pair
        # jobs: list of (ref_sdf, db_sdf, tab_path, [(pair_name, (i, j)), ...])
        jobs = []
        d = self._dir
        for k, (i, fits) in enumerate(by_ref.items()):
            qsdf = os.path.join(d, f"ref_{k}.sdf")
            self._write_sdf([mols[i].mol], qsdf)

            db = os.path.join(d, f"db_{k}.sdf")
            db_mols, db_names, row_pairs = [], [], []
            for r, j in enumerate(fits):
                pair_name = f"p{k}_{r}"             # unique db name per pair
                db_mols.append(mols[j].mol)
                db_names.append(pair_name)
                row_pairs.append((pair_name, (i, j)))
            self._write_sdf(db_mols, db, names=db_names)

            tab = os.path.join(d, f"_alignit_{k}.tab")
            jobs.append((qsdf, db, tab, row_pairs))

        env = self._env

        # every argv is built here (untimed); run() only execs. No --scoreOnly, --rankBy or --best
        # (see comparators.drivers.align_it_argv).
        cmds = [_drv.align_it_argv(reference=qsdf, dbase=db, scores=tab)
                for qsdf, db, tab, _rp in jobs]

        def run():
            for cmd in cmds:
                self._run(cmd, env=env)             # raises on nonzero exit

        def score():
            vals = []
            for _qsdf, _db, tab, row_pairs in jobs:
                if not os.path.exists(tab):
                    continue
                scores = _parse_alignit(tab)        # {db_name: pharm_tanimoto}
                for pair_name, _ij in row_pairs:
                    if pair_name in scores:
                        vals.append(scores[pair_name])
            return np.mean(vals) if vals else 0.0

        return Job(len(index_pairs), run, score)

    # --- SCREEN: one align_it call, query vs <store_path>/library.sdf (build_store inherited)
    def prepare_screen(self, store_path: str, query: object, mode: str, top_k: int = 0) -> Job:
        assert mode == "pharm", f"pharao only supports 'pharm', got {mode!r}"
        lib_sdf = os.path.join(store_path, "library.sdf")
        n_lib = count_sdf(lib_sdf)                   # screen throughput denominator
        cell = tempfile.mkdtemp(prefix="screencell_", dir=self._dir)
        ref_sdf = os.path.join(cell, "ref.sdf")
        self._write_sdf([query.mol], ref_sdf, names=["ref"])
        tab = os.path.join(cell, "scores.tab")
        env = self._env
        cmd = _drv.align_it_argv(reference=ref_sdf, dbase=lib_sdf, scores=tab)

        def run():
            self._run(cmd, env=env)                 # raises on nonzero exit

        def score():
            scores = _parse_alignit(tab)            # {db_name: pharm_tanimoto}
            return float(np.mean(list(scores.values()))) if scores else 0.0

        return Job(n_lib, run, score)

    def __del__(self):
        try:
            shutil.rmtree(self._dir, ignore_errors=True)
        except Exception:
            pass
