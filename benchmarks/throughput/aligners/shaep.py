"""ShaEP adapter: CPU atom-volume shape + its own ESP field via the ``shaep`` binary (all-atom).
Modes: vol reads the shape column; vol_esp and vol_and_surf_esp read the 0.5*(shape+ESP) combined
column; the timed run is identical for every mode. argv, parser and guards come from comparators.drivers."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np

from .base import Job, SubprocessAligner, count_sdf

from comparators import drivers as _drv


# ShaEP holds every molecule's field grid in RAM, so the screen library is fed in chunks (score-neutral).
SHAEP_CHUNK = 5000

# score() reads the combined 0.5*(shape+ESP) column for these modes, else the shape column.
_ESP_MODES = {"vol_esp", "vol_and_surf_esp"}


class ShaepAligner(SubprocessAligner):
    name = "shaep"
    supports_screen = True

    def __init__(self, backend: str = "cpu", seed: int = 42):
        if backend != "cpu":
            raise ValueError(f"shaep backend must be 'cpu', got {backend!r}")
        self.backend = backend
        self.seed = seed
        # SHAEP_THREADS overrides the single-thread default; the binary path resolves in
        # drivers.shaep_argv (SHAEP_BIN overrides)
        self.threads = int(os.environ.get("SHAEP_THREADS", self.threads))
        # one workdir; per-cell scratch is purged by _clean_cells() at the next prepare_*
        self.workdir = self._workdir()

    # --- capability + identity -------------------------------------------------
    @property
    def supported_modes(self) -> set:
        # no surface-point channel, so surf/surf_esp are not advertised
        return {"vol", "vol_esp", "vol_and_surf_esp"}

    def device_label(self) -> str:
        return f"CPU (shaep, {self.threads} thr)"

    def atom_basis(self, mode: str) -> str:
        # shape + ESP grids over every atom including explicit H (ROSHAMBO2 and PheSA are heavy-atom)
        return "all"

    def _shaep(self, cmd, what):
        """One ShaEP call. The SHAEP_STDIN answer (ShaEP blocks on an overwrite prompt otherwise),
        SHAEP_TIMEOUT and the SHAEP_MEM_CAP_GB address-space cap are all load-bearing."""
        try:
            return self._run(cmd, input=_drv.SHAEP_STDIN, timeout=_drv.SHAEP_TIMEOUT,
                             preexec_fn=_drv._preexec(_drv.SHAEP_MEM_CAP_GB))
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"shaep {what} timeout >{_drv.SHAEP_TIMEOUT:.0f}s "
                               f"(pathological molecule)")

    # --- PAIRWISE: one multi-query shaep call ----------------------------------
    def prepare_pairwise(self, mols: list, index_pairs: list, mode: str) -> Job:
        self._clean_cells()             # sequential in-process cells: drop the last cell's scratch
        # cell-unique titles (q0.., d0..) so ShaEP's by-title output maps back unambiguously
        q_idx = sorted({i for i, _ in index_pairs})
        d_idx = sorted({j for _, j in index_pairs})
        q_name = {orig: f"q{k}" for k, orig in enumerate(q_idx)}
        d_name = {orig: f"d{k}" for k, orig in enumerate(d_idx)}
        q_names = [q_name[i] for i in q_idx]

        celldir = tempfile.mkdtemp(prefix="cell_", dir=self.workdir)
        db_sdf = os.path.join(celldir, "db.sdf")
        self._write_sdf([mols[j].mol for j in d_idx], db_sdf,
                        names=[d_name[j] for j in d_idx])
        out_txt = os.path.join(celldir, "out.txt")

        qpaths = []                                  # one --query <file> per query mol
        for i in q_idx:
            qp = os.path.join(celldir, f"{q_name[i]}.sdf")
            self._write_sdf([mols[i].mol], qp, names=[q_name[i]])
            qpaths.append(qp)
        # -j parallelises across queries
        cmd = _drv.shaep_argv(queries=qpaths, input_file=db_sdf, output_file=out_txt,
                              threads=min(len(q_idx), self.threads))

        # the (i, j) pairs to average, by the cell-unique names ShaEP reports
        want = [(q_name[i], d_name[j]) for i, j in index_pairs]

        def run():
            self._shaep(cmd, "pairwise")

        def score():
            return _score_from_out(out_txt, q_names, want, mode)

        return Job(len(index_pairs), run, score)

    # --- SCREEN: query vs <store_path>/library.sdf (build_store inherited) -----
    def prepare_screen(self, store_path: str, query: object, mode: str, top_k: int = 0) -> Job:
        self._clean_cells()             # sequential in-process cells: drop the last cell's scratch
        lib_sdf = os.path.join(store_path, "library.sdf")
        q_name = "q0"
        celldir = tempfile.mkdtemp(prefix="screencell_", dir=self.workdir)
        q_sdf = os.path.join(celldir, "query.sdf")
        self._write_sdf([query.mol], q_sdf, names=[q_name])
        # bound ShaEP's in-RAM field grids: chunk the library (score-neutral)
        chunks = _drv._split_sdf(lib_sdf, SHAEP_CHUNK, celldir)
        n_lib = count_sdf(lib_sdf)
        plan = []                                    # (argv, out_path) per chunk
        for ci, ch in enumerate(chunks):
            out_txt = os.path.join(celldir, f"out_{ci}.txt")
            plan.append((_drv.shaep_argv(queries=[q_sdf], input_file=ch, output_file=out_txt,
                                         threads=self.threads), out_txt))

        def run():
            for cmd, _out_txt in plan:
                self._shaep(cmd, "screen")

        def score():
            col = "combined" if mode in _ESP_MODES else "shape"
            vals = []
            for _cmd, out_txt in plan:
                if not os.path.exists(out_txt):
                    continue
                by_name = _drv.parse_shaep(out_txt, col, query_names=[q_name]).get(q_name, {})
                vals.extend(by_name.values())
            if len(vals) != n_lib:
                raise ValueError(f"ShaEP returned {len(vals)} scores for {n_lib} conformers")
            return float(np.mean(vals))

        return Job(n_lib, run, score)

    def _clean_cells(self):
        """Purge stale per-cell scratch dirs (cells run sequentially in-process)."""
        for d in os.listdir(self.workdir):
            if d.startswith(("cell_", "screencell_")):
                shutil.rmtree(os.path.join(self.workdir, d), ignore_errors=True)


def _score_from_out(out_txt, q_names, want, mode):
    """Mean ShaEP score over the requested (qname, dname) pairs in the column ``mode`` selects."""
    if not os.path.exists(out_txt):
        return 0.0
    col = "combined" if mode in _ESP_MODES else "shape"
    table = _drv.parse_shaep(out_txt, col, query_names=q_names)
    vals = []
    for qn, dn in want:
        v = table.get(qn, {}).get(dn)
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else 0.0
