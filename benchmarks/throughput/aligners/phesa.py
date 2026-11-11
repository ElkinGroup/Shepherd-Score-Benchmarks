"""PheSA adapter: CPU Gaussian shape + pharmacophore overlay via the PheSAScreen Java harness
(aligners/phesa_java, openchemlib). Modes: vol (ppWeight 0) and pharm (ppWeight 1), rigid single-
conformer alignment; one JVM call per cell, so JVM start-up sits inside the timed region."""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np

from .base import Job, SubprocessAligner, count_sdf

from comparators import drivers as _drv, paths as _cpaths


class PheSAAligner(SubprocessAligner):
    name = "phesa"
    supports_screen = True

    def __init__(self, backend: str = "cpu", seed: int = 42):
        if backend != "cpu":
            raise ValueError(f"phesa backend must be 'cpu', got {backend!r}")
        self.backend = backend
        self.seed = seed
        # jar/classdir/java resolve through comparators.paths: PHESA_JAR / PHESA_CLASSDIR / JAVA_BIN
        # override; java otherwise comes from PATH, JAVA_HOME, then the jdk17 conda env
        self.jar = str(_cpaths.phesa_jar())
        self.classdir = str(_cpaths.phesa_classdir())
        self.java = _cpaths.java_bin()
        # the harness would grab all cores; --threads is always passed (default 1, PHESA_THREADS overrides)
        self.threads = int(os.environ.get("PHESA_THREADS", self.threads))
        # one workdir; stale per-cell dirs are purged at the next prepare_*
        self.workdir = self._workdir()

    # --- capability + identity -------------------------------------------------
    @property
    def supported_modes(self) -> set:
        # not vol_color: PheSA's color is directional, fss/ROSHAMBO2's is directionless
        return {"vol", "pharm"}

    def device_label(self) -> str:
        return f"CPU (phesa, {self.threads} thr)"

    def atom_basis(self, mode: str) -> str:
        # openchemlib MolecularVolume skips H ("hydrogens don't contribute to the molecular volume"),
        # though explicit H must still be present in the input SDF. Same basis as ROSHAMBO2; ShaEP is all-atom.
        return "heavy"

    def _cp(self):
        return f"{self.jar}{os.pathsep}{self.classdir}"

    # --- PAIRWISE: one JVM call, pairs addressed by record order ----------------
    def prepare_pairwise(self, mols: list, index_pairs: list, mode: str) -> Job:
        assert mode in self.supported_modes, f"phesa cannot run {mode!r}"
        # the harness addresses pairs by record order in the SDFs
        q_idx = sorted({i for i, _ in index_pairs})
        d_idx = sorted({j for _, j in index_pairs})
        q_pos = {orig: k for k, orig in enumerate(q_idx)}
        d_pos = {orig: k for k, orig in enumerate(d_idx)}

        cell = tempfile.mkdtemp(prefix="cell_", dir=self.workdir)
        q_sdf = os.path.join(cell, "q.sdf")
        d_sdf = os.path.join(cell, "d.sdf")
        self._write_sdf([mols[i].mol for i in q_idx], q_sdf,
                        names=[f"m{k}" for k in range(len(q_idx))])
        self._write_sdf([mols[j].mol for j in d_idx], d_sdf,
                        names=[f"m{k}" for k in range(len(d_idx))])
        pairs_tsv = os.path.join(cell, "pairs.tsv")
        with open(pairs_tsv, "w") as fh:
            for i, j in index_pairs:
                fh.write(f"{q_pos[i]}\t{d_pos[j]}\n")
        out_tab = os.path.join(cell, "out.tab")

        # argv from comparators (always passes --ppweight explicitly)
        cmd = _drv.phesa_pairs_argv(mode=mode, queries=q_sdf, dbase=d_sdf, pairs=pairs_tsv,
                                    scores=out_tab, threads=self.threads)

        # one JVM call: cold-start + the descriptor build are inside the timed region
        def run():
            self._run(cmd)                          # raises on nonzero exit

        def score():
            return _mean_score(out_tab, col=2)      # pairs output: qi  di  sim

        self._clean_old_cells(keep=cell)
        return Job(len(index_pairs), run, score)

    # --- SCREEN: one JVM call, the reference vs <store_path>/library.sdf (build_store inherited)
    def prepare_screen(self, store_path: str, query: object, mode: str, top_k: int = 0) -> Job:
        assert mode in self.supported_modes, f"phesa cannot run {mode!r}"
        lib_sdf = os.path.join(store_path, "library.sdf")
        n_lib = count_sdf(lib_sdf)
        cell = tempfile.mkdtemp(prefix="screencell_", dir=self.workdir)
        q_sdf = os.path.join(cell, "q.sdf")
        self._write_sdf([query.mol], q_sdf, names=["m0"])
        out_tab = os.path.join(cell, "out.tab")
        cmd = _drv.phesa_screen_argv(mode=mode, reference=q_sdf, dbase=lib_sdf,
                                     scores=out_tab, threads=self.threads)

        def run():
            self._run(cmd)                          # raises on nonzero exit

        def score():
            return _mean_score(out_tab, col=1)      # screen output: name  sim

        self._clean_old_cells(keep=cell)
        return Job(n_lib, run, score)

    def _clean_old_cells(self, keep=None):
        """Drop prior per-cell scratch (sequential in-process cells)."""
        for d in os.listdir(self.workdir):
            if d.startswith(("cell_", "screencell_")):
                full = os.path.join(self.workdir, d)
                if full != keep:
                    shutil.rmtree(full, ignore_errors=True)

    def __del__(self):
        try:
            shutil.rmtree(self.workdir, ignore_errors=True)
        except Exception:
            pass


def _mean_score(path, col):
    """Mean of column ``col`` (0-based) of a tab-separated scores file; 0.0 on a missing/empty file."""
    if not os.path.exists(path):
        return 0.0
    vals = []
    with open(path) as fh:
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if len(p) <= col:
                continue
            try:
                vals.append(float(p[col]))
            except ValueError:
                continue
    return float(np.mean(vals)) if vals else 0.0
