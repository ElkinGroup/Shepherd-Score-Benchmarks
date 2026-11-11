"""Aligner adapter interface. Featurization, indexing and batch assembly are untimed setup done in
build()/prepare_*; only Job.run() is timed. Architecture differences (atom basis, seeds, threads)
are disclosed per cell, not normalized."""
from __future__ import annotations

import abc
import os
import subprocess
import sys
import tempfile

# Put the directory holding the real `comparators` package (one with an __init__.py) on sys.path.
# A search rather than a fixed depth: the cluster run dir is an rsync of benchmarks/throughput, and the
# tool install prefix can also be named "comparators" without being a package.
_d = os.path.dirname(os.path.abspath(__file__))
while _d != os.path.dirname(_d):
    if os.path.isfile(os.path.join(_d, "comparators", "__init__.py")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _d = os.path.dirname(_d)
else:  # pragma: no cover
    raise ImportError(
        "the `comparators` package was not found above this file. The benchmark run dir "
        "must contain benchmarks/comparators/ alongside benchmarks/throughput/ -- rsync benchmarks/, not "
        "just benchmarks/throughput/. See the README.")

# The eight alignment channels fss exposes; a competitor advertises its subset via supported_modes.
MODES = ("vol", "vol_esp", "surf", "surf_esp", "vol_and_surf_esp", "pharm", "vol_color",
         "vol_lipo")

# Experimental SI modes are opt-in via FIG2_EXTRA_MODES (comma-separated).
_EXTRA = tuple(m.strip() for m in os.environ.get("FIG2_EXTRA_MODES", "").split(",") if m.strip())
if _EXTRA:
    MODES = MODES + tuple(m for m in _EXTRA if m not in MODES)


class Job:
    """A prepared, repeatedly-timeable unit of alignment work: ``n`` alignments per ``run()``,
    ``score()`` = mean overlap after a run (0/NaN flags a cell that did not align), ``extra`` =
    per-job metadata merged into the result row."""

    def __init__(self, n: int, run, score, extra=None):
        self.n = int(n)
        self._run = run
        self._score = score
        self.extra = dict(extra) if extra else {}

    def run(self):
        return self._run()

    def score(self) -> float:
        return float(self._score())


class MolHolder:
    """Picklable molecule holder returned by the black-box adapters' build(): the RDKit mol (3D
    conformer included), its Nx3 ``atom_pos`` (a hard contract: the driver reads
    ``.atom_pos.shape[0]`` over the whole pool), and an optional tool-side ``name``."""

    __slots__ = ("mol", "atom_pos", "name")

    def __init__(self, mol, atom_pos, name=None):
        self.mol = mol
        self.atom_pos = atom_pos
        self.name = name


def count_sdf(path) -> int:
    """Number of records in an SDF (``$$$$`` terminators)."""
    n = 0
    with open(path) as fh:
        for line in fh:
            if line.rstrip() == "$$$$":
                n += 1
    return n


class Aligner(abc.ABC):
    """Adapter the driver times. One instance == one (tool, backend)."""

    name: str = "?"
    backend: str = "?"
    supports_screen: bool = False

    @property
    @abc.abstractmethod
    def supported_modes(self) -> set:
        """The subset of :data:`MODES` this (tool, backend) can run."""

    @abc.abstractmethod
    def device_label(self) -> str:
        """Human-readable device, e.g. 'NVIDIA H100' or 'CPU (numba, 1 thr)'."""

    @abc.abstractmethod
    def build(self, smiles: list) -> list:
        """Featurize SMILES into the molecule objects prepare_* consume. Untimed; writes no files.

        Two return shapes: 1:1 tools (pharao, phesa, shaep) return one entry per SMILES with None
        for an embed failure; fss and roshambo2 return ``workloads.N_CONFORMERS`` contiguous
        entries per surviving compound (pool index i -> compound i // N_CONFORMERS) and drop any
        compound that cannot embed the full ensemble."""

    @abc.abstractmethod
    def prepare_pairwise(self, mols: list, index_pairs: list, mode: str) -> Job:
        """In-RAM batch for ``index_pairs`` -> a Job that aligns it in ``mode``."""

    # --- screen path (only if supports_screen) --------------------------------
    def build_store(self, mols: list, lib_indices: list, path: str, modes=None) -> None:
        raise NotImplementedError(f"{self.name} has no screen path")

    def prepare_screen(self, store_path: str, query: object, mode: str, top_k: int) -> Job:
        raise NotImplementedError(f"{self.name} has no screen path")

    # --- disclosure ------------------------------------------------------------
    def cache_key(self) -> str:
        """Extra state that changes what build() produces, folded into the pool-cache key."""
        return ""

    def atom_basis(self, mode: str) -> str:
        """Atom set this (tool, mode) aligns: 'heavy', 'all' (incl. H), or 'surface'."""
        return "heavy"

    # --- device timing ---------------------------------------------------------
    def sync(self) -> None:
        """Block until queued async (GPU) work is done. No-op for CPU tools."""
        return None

    def reset(self) -> None:
        """Clean device state between cells. No-op by default; GPU tools override."""
        return None


def _embed_one_1to1(args):
    """Embed one SMILES into one MMFF-optimised conformer; None on failure. Module-level so it pickles under spawn."""
    smi, seed = args
    try:
        from shepherd_score.conformer_generation import embed_conformer_from_smiles
        return embed_conformer_from_smiles(smi, MMFF_optimize=True, random_seed=seed)
    except Exception:
        return None


class SubprocessAligner(Aligner):
    """Base for CPU tools that align via an external binary / JVM: temp workdir, SDF writing, a
    return-code-checked ``_run`` and a library.sdf ``build_store``."""

    #: threads the tool is pinned to (default 1)
    threads: int = 1

    #: True when the tool's output parser keys molecules by SDF title (pharao does; ShaEP and
    #: PheSA do not, and naming their mols would change the titles they write).
    name_mols: bool = False

    def build(self, smiles: list) -> list:
        """One MMFF-optimised conformer per SMILES, embedded in parallel (spawn, not fork: torch may be imported)."""
        import numpy as np
        args = [(smi, self.seed) for smi in smiles]
        jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 1)
        if jobs <= 1 or len(smiles) < 64:
            rds = [_embed_one_1to1(a) for a in args]
        else:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(jobs) as pool:
                rds = pool.map(_embed_one_1to1, args, chunksize=8)

        out = []
        for i, rd in enumerate(rds):
            if rd is None:
                out.append(None)                    # keep indexing positional
                continue
            if self.name_mols:
                from rdkit import Chem
                rd = Chem.Mol(rd)
                rd.SetProp("_Name", f"mol{i}")
            out.append(MolHolder(rd, np.asarray(rd.GetConformer().GetPositions(), dtype=float)))
        return out

    def _workdir(self) -> str:
        return tempfile.mkdtemp(prefix=f"fig2_{self.name}_")

    def _run(self, cmd, *, env=None, timeout=None, input=None, cwd=None, preexec_fn=None) -> str:
        """Run ``cmd`` and return stdout; raises on a non-zero return code."""
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout,
                           input=input, cwd=cwd, preexec_fn=preexec_fn)
        if r.returncode != 0:
            raise RuntimeError(f"{os.path.basename(str(cmd[0]))} rc={r.returncode}: "
                               f"{(r.stderr or '')[-400:]}")
        return r.stdout

    def build_store(self, mols: list, lib_indices: list, path: str, modes=None) -> None:
        """No on-disk index: write the library mols to one ``<path>/library.sdf``. Untimed."""
        os.makedirs(path, exist_ok=True)
        self._write_sdf([mols[idx].mol for idx in lib_indices],
                        os.path.join(path, "library.sdf"),
                        names=[f"m{k}" for k in range(len(lib_indices))])

    def index_count(self, path: str) -> int:
        """Records in ``<path>/library.sdf``; the build-validity readout timing.measure_build reads."""
        import os
        sdf = os.path.join(path, "library.sdf")
        return count_sdf(sdf) if os.path.exists(sdf) else 0

    @staticmethod
    def _write_sdf(rdmols, path: str, names=None) -> int:
        """Write RDKit mols to one SDF at ``path`` with optional per-mol titles; returns the count."""
        from rdkit import Chem
        w = Chem.SDWriter(path)
        n = 0
        for i, m in enumerate(rdmols):
            if m is None:
                continue
            if names is not None:
                m = Chem.Mol(m)
                m.SetProp("_Name", str(names[i]))
            w.write(m)
            n += 1
        w.close()
        return n
