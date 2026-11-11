"""ESP-Sim adapter: ``hesther/espsim`` (Bolcato et al., JCIM 2022), shape + electrostatic-potential
similarity. CPU, in-process, serial, pure Python over RDKit/NumPy/SciPy.

One mode, ``vol_and_surf_esp``: both build an electrostatic potential from point charges over all
atoms including hydrogens and compare fields (``vol_esp`` weights heavy-atom volume overlap by
charge agreement instead). ESP-Sim's ESP score is signed and can go below zero. The alignment is
RDKit's ``rdMolAlign.GetCrippenO3A`` fit, which does not optimise the reported score, and it
dominates the timed loop. Only ``GetEspSim`` (Carbo metric, the package default) is scored, so the
cell compares with an fss ESP mode; FIG2_ESPSIM_SHAPE=1 adds ``GetShapeSim`` to the timed loop.
Charges are MMFF94, passed explicitly (the package default is Gasteiger) and computed in untimed
setup. 1:1 like the other CPU tools: one conformer per compound, store = ``library.sdf``.
"""
from __future__ import annotations

import os
import shutil

import numpy as np

from .base import Job, SubprocessAligner, count_sdf

#: The one fss mode this tool is bracketed against; see the module docstring for why not vol_esp.
_MODE = "vol_and_surf_esp"

# Optional diagnostic sum; the published comparator uses the ESP score alone.
SCORE_SHAPE = os.environ.get("FIG2_ESPSIM_SHAPE", "0") == "1"


def _espsim():
    try:
        import espsim
    except ImportError as exc:
        raise ImportError("espsim is not importable; install it into COMPARATORS_HOME/python and "
                          "put that directory on PYTHONPATH (benchmarks/comparators/install.sh espsim)"
                          ) from exc
    return espsim


class EspSimAligner(SubprocessAligner):
    """One (tool, backend); backend is always ``cpu``. Inherits the 1:1 build and the SDF store."""

    name = "espsim"
    supports_screen = True
    name_mols = False
    threads = 1

    def __init__(self, backend: str = "cpu", seed: int = 42):
        if backend != "cpu":
            raise ValueError(f"espsim backend must be cpu, got {backend!r}")
        self.backend = backend
        self.seed = seed
        self._es = _espsim()
        self._dir = self._workdir()

    # --- capability + identity -------------------------------------------------
    @property
    def supported_modes(self) -> set:
        return {_MODE}

    def device_label(self) -> str:
        import rdkit
        return f"CPU (espsim + rdkit {rdkit.__version__} CrippenO3A, {self.threads} thr)"

    def atom_basis(self, mode: str) -> str:
        return "all"                                  # the ESP integral spans hydrogens too

    @staticmethod
    def _check_mode(mode: str) -> None:
        if mode != _MODE:
            raise ValueError(f"espsim supports {_MODE!r} only, got {mode!r}")

    # --- per-molecule inputs, all computed in untimed setup --------------------
    def _prep(self, mol):
        from comparators.espsim_protocol import prepare
        return prepare(mol)

    def _extra(self, n_added_h: int) -> dict:
        return {"espsim_align": "rdMolAlign.GetCrippenO3A (a Crippen logP fit, not an optimiser of "
                                "either reported score)",
                "espsim_scores": ("GetShapeSim (grid ShapeTanimotoDist) + GetEspSim (Carbo), "
                                  "summed" if SCORE_SHAPE else
                                  "GetEspSim only (Carbo, the package default); GetShapeSim is "
                                  "outside the timed loop so the cell matches an fss ESP mode"),
                "espsim_timed_shape": SCORE_SHAPE,
                "espsim_charges": "MMFF94, passed explicitly (package default is Gasteiger)",
                "espsim_hydrogens_added": n_added_h}

    # --- PAIRWISE: in RAM ------------------------------------------------------
    def prepare_pairwise(self, mols: list, index_pairs: list, mode: str) -> Job:
        from comparators.espsim_protocol import align_and_score
        self._check_mode(mode)
        es = self._es
        prepped, added = {}, 0
        for k in {i for pair in index_pairs for i in pair}:
            before = mols[k].mol.GetNumAtoms()
            prepped[k] = self._prep(mols[k].mol)
            added += prepped[k][0].GetNumAtoms() != before
        vals = []

        def run():
            vals.clear()
            for i, j in index_pairs:
                ref, ref_cr, ref_q = prepped[i]
                fit, fit_cr, fit_q = prepped[j]
                prb, v = align_and_score(es, (fit, fit_cr, fit_q), (ref, ref_cr, ref_q))
                if SCORE_SHAPE:
                    v += float(es.GetShapeSim(prb, ref))
                vals.append(v)

        def score():
            return float(np.mean(vals)) if vals else 0.0

        return Job(len(index_pairs), run, score, extra=self._extra(added))

    # --- SCREEN: the query against every molecule of <store_path>/library.sdf ---
    def prepare_screen(self, store_path: str, query: object, mode: str, top_k: int = 0) -> Job:
        from rdkit import Chem
        from comparators.espsim_protocol import align_and_score
        self._check_mode(mode)
        es = self._es
        lib_sdf = os.path.join(store_path, "library.sdf")
        n_lib = count_sdf(lib_sdf)                    # screen throughput denominator
        library = [m for m in Chem.SDMolSupplier(lib_sdf, removeHs=False) if m is not None]
        if len(library) != n_lib:
            raise RuntimeError(f"{lib_sdf}: {n_lib} records, {len(library)} parsed")

        ref, ref_cr, ref_q = self._prep(query.mol)
        added = 0
        prepped = []
        for m in library:
            before = m.GetNumAtoms()
            mh, cr, q = self._prep(m)
            added += mh.GetNumAtoms() != before
            prepped.append((mh, cr, q))
        vals = []

        def run():
            vals.clear()
            for fit, fit_cr, fit_q in prepped:
                prb, v = align_and_score(es, (fit, fit_cr, fit_q), (ref, ref_cr, ref_q))
                if SCORE_SHAPE:
                    v += float(es.GetShapeSim(prb, ref))
                vals.append(v)

        def score():
            return float(np.mean(vals)) if vals else 0.0

        return Job(n_lib, run, score, extra=self._extra(added))

    def __del__(self):
        try:
            shutil.rmtree(self._dir, ignore_errors=True)
        except Exception:
            pass
