"""RDKit shape alignment adapter: ``rdkit.Chem.rdShapeAlign`` (RDKit >= 2024.09; the 2025.03+
``ShapeInputOptions`` form of ``PrepareConformer`` is handled too). CPU, in-process, serial.

``vol`` runs the shape-only alignment and scores the shape Tanimoto; ``vol_color`` runs the
shape+color alignment and scores shape + color Tanimoto (TanimotoCombo convention, 0..2), the score
that pairs with fss's ``vol_color``. The optimizer is the module's default (``opt_param=1.0``, 10
pre- and 30 post-iterations), disclosed rather than normalized. 1:1 like the other CPU tools:
``build`` (inherited) embeds one conformer per compound and the store is ``library.sdf``, which
``prepare_screen`` reads once (untimed); ``run()`` aligns a fresh copy of every library molecule so
the library stays unmoved between reps. Hydrogens do not enter the score (heavy-atom basis).
"""
from __future__ import annotations

import os
import shutil

import numpy as np

from .base import Job, SubprocessAligner, count_sdf


def _shape_align():
    try:
        from rdkit.Chem import rdShapeAlign
    except ImportError as exc:                       # RDKit < 2024.09
        import rdkit
        raise ImportError(f"rdkit.Chem.rdShapeAlign needs RDKit >= 2024.09; this environment has "
                          f"{rdkit.__version__}") from exc
    return rdShapeAlign


class RdkitShapeAligner(SubprocessAligner):
    """One (tool, backend); backend is always ``cpu``. Inherits the 1:1 build and the SDF store."""

    name = "rdkit"
    supports_screen = True
    name_mols = False

    def __init__(self, backend: str = "cpu", seed: int = 42):
        if backend != "cpu":
            raise ValueError(f"rdkit backend must be cpu, got {backend!r}")
        self.backend = backend
        self.seed = seed
        self._sa = _shape_align()
        self._dir = self._workdir()

    # --- capability + identity -------------------------------------------------
    @property
    def supported_modes(self) -> set:
        return {"vol", "vol_color"}

    def device_label(self) -> str:
        import rdkit
        return f"CPU (rdkit {rdkit.__version__} rdShapeAlign, {self.threads} thr)"

    def atom_basis(self, mode: str) -> str:
        return "heavy"

    @staticmethod
    def _use_colors(mode: str) -> bool:
        if mode not in ("vol", "vol_color"):
            raise ValueError(f"rdkit supports vol and vol_color, got {mode!r}")
        return mode == "vol_color"

    def _prepare(self, mol, colors: bool):
        """The query's ShapeInput. RDKit 2024.09 took ``useColors`` as a bool; 2025.03 and later
        take a ``ShapeInputOptions`` object (a bool there raises a converter TypeError)."""
        sa = self._sa
        opts_cls = getattr(sa, "ShapeInputOptions", None)
        if opts_cls is None:
            return sa.PrepareConformer(mol, -1, colors)
        opts = opts_cls()
        opts.useColors = colors
        return sa.PrepareConformer(mol, -1, opts)

    def _score_of(self, result, colors: bool) -> float:
        shape, color = result
        return float(shape + color) if colors else float(shape)

    # --- PAIRWISE: in RAM, one ShapeInput per distinct reference -----------------
    def prepare_pairwise(self, mols: list, index_pairs: list, mode: str) -> Job:
        from rdkit import Chem
        sa = self._sa
        colors = self._use_colors(mode)
        refs = {i: self._prepare(mols[i].mol, colors) for i in {i for i, _ in index_pairs}}
        fits = [(refs[i], mols[j].mol) for i, j in index_pairs]
        vals = []

        def run():
            vals.clear()
            for ref, fit in fits:
                vals.append(self._score_of(sa.AlignMol(ref, Chem.Mol(fit), -1, useColors=colors), colors))

        def score():
            return float(np.mean(vals)) if vals else 0.0

        return Job(len(index_pairs), run, score)

    # --- SCREEN: the query's ShapeInput vs every molecule of <store_path>/library.sdf ----------
    def prepare_screen(self, store_path: str, query: object, mode: str, top_k: int = 0) -> Job:
        from rdkit import Chem
        sa = self._sa
        colors = self._use_colors(mode)
        lib_sdf = os.path.join(store_path, "library.sdf")
        n_lib = count_sdf(lib_sdf)                   # screen throughput denominator
        library = [m for m in Chem.SDMolSupplier(lib_sdf, removeHs=False) if m is not None]
        if len(library) != n_lib:
            raise RuntimeError(f"{lib_sdf}: {n_lib} records, {len(library)} parsed")
        ref = self._prepare(query.mol, colors)
        vals = []

        def run():
            vals.clear()
            for m in library:
                vals.append(self._score_of(sa.AlignMol(ref, Chem.Mol(m), -1, useColors=colors), colors))

        def score():
            return float(np.mean(vals)) if vals else 0.0

        return Job(n_lib, run, score, extra={"rdshapealign_defaults": "opt_param=1.0, "
                                                "max_preiters=10, max_postiters=30"})

    def __del__(self):
        try:
            shutil.rmtree(self._dir, ignore_errors=True)
        except Exception:
            pass
