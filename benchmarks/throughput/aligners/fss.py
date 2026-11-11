"""fast_shepherd_score adapter: all modes, "triton" (GPU) or "numba" (CPU) backend, through the
public shepherd_score.container / shepherd_score.screen APIs at each method's default effort.
FSS_BENCH_CHARGES / FSS_BENCH_SURF_POINTS / FSS_BENCH_PHARM are read at import: one configuration per process."""
from __future__ import annotations

import os

import numpy as np

from .base import MODES, Aligner, Job

# shared xTB-charge helper (drivers.xtb_charges), so charge handling matches fig3
from comparators import drivers as _drv

_SCORE_ATTR = {  # result attr align_with_<mode> writes in place (no-H variants for vol*)
    "vol": "sim_aligned_vol_noH", "vol_esp": "sim_aligned_vol_esp_noH",
    "surf": "sim_aligned_surf", "surf_esp": "sim_aligned_surf_esp",
    "vol_and_surf_esp": "sim_aligned_vol_and_surf_esp", "pharm": "sim_aligned_pharm",
    "vol_color": "sim_aligned_vol_color",
    "vol_lipo": "sim_aligned_vol_lipo",
}

# experimental modes that take alpha
_SI_ALPHA_MODES = {"surf_tversky", "surf_esp_tversky", "vol_and_surf_esp_tversky"}
# atom set each mode aligns (disclosed, not normalized); vol_color is heavy: screen() refuses no_H=False
_ATOM_BASIS = {"vol": "heavy", "vol_esp": "heavy", "vol_color": "heavy",
               "surf": "surface", "surf_esp": "surface", "vol_and_surf_esp": "surface", "pharm": "heavy",
               "vol_lipo": "heavy"}

ALPHA = 0.81          # required by surf/surf_esp/vol_and_surf_esp (the library default)

# Charge model used when preparing new inputs. Stored-input benchmarks retain the
# charges in their frozen molecules; this switch does not replace those arrays.
CHARGE_MODEL = os.environ.get("FSS_BENCH_CHARGES", "xtb").lower()


def _charges_for(rd):
    """xTB partial charges (drivers.xtb_charges); None (fss's MMFF94 fallback) unless FSS_BENCH_CHARGES=xtb."""
    if CHARGE_MODEL != "xtb":
        return None
    return _drv.xtb_charges(rd)

# ESP lambdas are the library defaults (vol_esp 0.1, surf_esp 0.3, vol_and_surf_esp 0.001); lam is
# a width whose units differ by mode (surf_esp scales it internally), so the values are not interchangeable.
SURF_POINTS = int(os.environ.get("FSS_BENCH_SURF_POINTS", "200"))  # a store needs one fixed count
# Pharmacophore detection in Molecule.__init__: False = detect (pool default); FSS_BENCH_PHARM=0 skips it.
PHARM = None if os.environ.get("FSS_BENCH_PHARM", "1") == "0" else False


def embed_ensemble(smi, seed, n_conf):
    """SMILES -> (mol with exactly ``n_conf`` ETKDGv3 conformers, MMFF94-optimised, conformer ids); None if the full ensemble cannot be embedded."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    m = Chem.AddHs(m)
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    p.pruneRmsThresh = -1.0        # keep the full ensemble; RMS pruning would vary the count
    cids = AllChem.EmbedMultipleConfs(m, numConfs=n_conf, params=p)
    if len(cids) < n_conf:
        return None
    try:
        AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=200)
    except Exception:
        pass
    Chem.SanitizeMol(m)
    return m, list(cids)


def prepare_conformer(rd):
    """Prepare one existing conformer with the adapter's configured features."""
    from shepherd_score.container import Molecule
    ch = _charges_for(rd)
    if CHARGE_MODEL == "xtb" and ch is None:
        return None
    try:
        return Molecule(rd, num_surf_points=SURF_POINTS or None, partial_charges=ch,
                        charge_model=CHARGE_MODEL, pharm_multi_vector=PHARM)
    except Exception:
        return None


def _build_one(arg):
    """Prepare a complete ensemble, rejecting the compound if any conformer fails."""
    smi, seed, n_conf = arg
    from rdkit import Chem
    embedded = embed_ensemble(smi, seed, n_conf)
    if embedded is None:
        return None
    m, cids = embedded
    out = [prepare_conformer(Chem.Mol(m, confId=int(cid))) for cid in cids[:n_conf]]
    return None if any(mol is None for mol in out) else out


# Measured optimizer effort via the library's optional step recorder (shepherd_score.accel._stats);
# a library without it reports nothing.
_STATS_UNSET = "unset"
_STATS = _STATS_UNSET


def _stats_module():
    """``shepherd_score.accel._stats`` if the installed library has it, else None (cached)."""
    global _STATS
    if _STATS is _STATS_UNSET:
        try:
            from shepherd_score.accel import _stats as mod
            _STATS = mod
        except Exception:
            _STATS = None
    return _STATS


def _plain(v):
    """JSON-safe scalar (numpy/torch scalars are not json-serializable); unknown types become a bounded string."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    try:
        return v.item()                       # numpy / torch 0-d scalar
    except Exception:
        return str(v)[:200]


class FSSAligner(Aligner):
    name = "fss"
    supports_screen = True

    def __init__(self, backend: str = "triton", seed: int = 42, threads: int | None = None):
        if backend not in ("triton", "numba"):
            raise ValueError(f"fss backend must be triton/numba, got {backend!r}")
        self.backend = backend
        self.seed = seed
        self.threads = threads
        import torch
        self._torch = torch
        want_cuda = backend == "triton"
        if want_cuda and not torch.cuda.is_available():
            raise RuntimeError("backend='triton' needs a CUDA GPU but torch.cuda is unavailable")
        self.device = torch.device("cuda" if want_cuda else "cpu")
        if not want_cuda:
            # torch's per-step tail ops contend with numba's prange unless single-threaded
            torch.set_num_threads(1)
            if threads is not None:
                import numba
                numba.set_num_threads(int(threads))

    @property
    def supported_modes(self) -> set:
        return set(MODES)

    def device_label(self) -> str:
        if self.device.type == "cuda":
            return self._torch.cuda.get_device_name(0)
        thr = self.threads or os.environ.get("NUMBA_NUM_THREADS") or os.cpu_count()
        return f"CPU ({self.backend}, {thr} thr)"

    def cache_key(self) -> str:
        # the charge model and pharm flag change what build() produces
        return f"charges={CHARGE_MODEL};aromaticity=rdkit" + ("" if PHARM is False else ";pharm=off")

    def atom_basis(self, mode: str) -> str:
        return _ATOM_BASIS.get(mode, "heavy")

    # --- PREP: featurize SMILES -> Molecules (untimed) -------------------------
    def build(self, smiles: list) -> list:
        """Flat conformer-ensemble list: ``N_CONFORMERS`` Molecules per surviving compound, contiguous (pool index i -> compound i // N_CONFORMERS)."""
        from workloads import N_CONFORMERS
        args = [(smi, self.seed, N_CONFORMERS) for smi in smiles]
        jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 1)
        if jobs <= 1 or len(smiles) < 64:
            groups = [_build_one(a) for a in args]
        else:
            import multiprocessing as mp
            # spawn, not fork: this process has imported torch
            with mp.get_context("spawn").Pool(jobs) as pool:
                groups = pool.map(_build_one, args, chunksize=4)
        out = []
        for g in groups:
            if g:                                  # drop failed compounds; keep blocks exact
                out.extend(g)
        return out

    def _run_and_score(self, batch, pairs, mode):
        """align_with_<mode> dispatch behind the pairwise Job; only the mode's required kwarg + backend are set."""
        be = self.backend

        def run():
            if mode == "vol":
                batch.align_with_vol(no_H=True, backend=be)
            elif mode == "vol_esp":
                batch.align_with_vol_esp(no_H=True, backend=be)   # lam = library default (0.1)
            elif mode == "surf":
                batch.align_with_surf(alpha=ALPHA, backend=be)
            elif mode == "surf_esp":
                batch.align_with_surf_esp(alpha=ALPHA, backend=be)
            elif mode == "vol_and_surf_esp":
                batch.align_with_vol_and_surf_esp(alpha=ALPHA, backend=be)
            elif mode == "pharm":
                batch.align_with_pharm(backend=be)
            elif mode == "vol_color":
                batch.align_with_vol_color(backend=be)
            elif mode == "vol_lipo":
                batch.align_with_vol_lipo(backend=be)     # lipo_weight/alpha/lam = library defaults
            else:
                # experimental SI modes: generic dispatch through the library's align_with_<mode>
                kw = {"backend": be}
                if mode in _SI_ALPHA_MODES:
                    kw["alpha"] = ALPHA
                getattr(batch, f"align_with_{mode}")(**kw)

        def score():
            attr = _SCORE_ATTR.get(mode)
            if attr is None:                      # experimental mode: ask the library's registry
                from shepherd_score.accel._modes import MODE_ATTRS, canonical
                attr = MODE_ATTRS[canonical(mode)][1]
            return np.mean([float(getattr(p, attr)) for p in pairs])

        return run, score

    # --- PAIRWISE --------------------------------------------------------------
    def prepare_pairwise(self, mols: list, index_pairs: list, mode: str) -> Job:
        from shepherd_score.container import MoleculePair, MoleculePairBatch

        # fetch each distinct molecule once, in ascending index order (a sharded pool re-reads a
        # shard per random access)
        need = sorted({i for pr in index_pairs for i in pr})
        got = {i: mols[i] for i in need}
        # vol_fukui's field is a lazy property backed by xTB single points: materialise it here
        # (untimed) and drop molecules whose anion single point does not converge
        if mode == "vol_fukui":
            bad = set()
            for i, m in got.items():
                try:
                    _ = m.fukui
                except Exception:
                    bad.add(i)
            if bad:
                index_pairs = [(i, j) for i, j in index_pairs if i not in bad and j not in bad]
                print(f"vol_fukui: dropped {len(bad)} molecules (anion SP did not converge); "
                      f"{len(index_pairs)} pairs remain", flush=True)
                got = {i: m for i, m in got.items() if i not in bad}
        pairs = [MoleculePair(got[i], got[j], do_center=True, num_surf_points=SURF_POINTS,
                              device=self.device) for i, j in index_pairs]
        run, score = self._run_and_score(MoleculePairBatch(pairs), pairs, mode)
        return Job(len(index_pairs), run, score, extra={"timing": "kernel"})

    # --- SCREEN: on-disk ProfileStore + screen() --------------------------------
    def build_store(self, mols: list, lib_indices: list, path: str, modes=None) -> None:
        from shepherd_score.screen import ProfileStore
        # FSS_CANONICAL_STORE=1 rotates each molecule into its principal frame at build time so the
        # screen uses one constant seed set. This changes scores, so it is opt-in, and applies only
        # to the modes in FSS_CANONICAL_MODES (default "vol"; "all" for every mode).
        _cm = os.environ.get("FSS_CANONICAL_MODES", "vol")
        _cmodes = None if _cm.strip() == "all" else {m.strip() for m in _cm.split(",") if m.strip()}
        _canon = (os.environ.get("FSS_CANONICAL_STORE") == "1"
                  and (_cmodes is None
                       or any(m in _cmodes for m in (tuple(modes) if modes else MODES))))
        store = ProfileStore.create(path, num_surf_points=SURF_POINTS,
                                    modes=tuple(modes) if modes else MODES,
                                    pre_centered=True, overwrite=True, canonical=_canon)
        for idx in lib_indices:
            store.add(mols[idx])
        store.close()

    def region_a_boundary(self) -> dict:
        """Where this tool featurizes (fss: at pool time, so build_store is array extraction only); optional hook found via getattr."""
        return {
            "input_stage": "featurized",
            "includes_featurization": False,
            "pool_stage": "Molecule.__init__ -- surface mesh, charges and pharmacophores built at "
                          "pool time; that cost is panel D's per-molecule block",
            "build_stage": "array extraction from the prebuilt Molecule + ProfileStore shard write",
            "source": "screen.ProfileStore.add -> _profile_from_schema (extraction only)",
        }

    def index_count(self, path: str) -> int:
        """Molecules in the store, from its manifest (ProfileStore.open is a manifest read)."""
        from shepherd_score.screen import ProfileStore
        return len(ProfileStore.open(path))

    def prepare_screen(self, store_path: str, query: object, mode: str, top_k: int) -> Job:
        from shepherd_score.screen import ProfileStore, screen
        store = ProfileStore.open(store_path)          # manifest read; untimed
        n_actual = len(store)                          # one alignment per library molecule
        # screen() requires lam explicitly for vol_esp/surf_esp; pass the library defaults
        # (vol_esp 0.1 raw, surf_esp 0.3 scaled internally; not interchangeable)
        kw = {"surf": {"alpha": ALPHA},
              "surf_esp": {"alpha": ALPHA, "lam": 0.3},
              "vol_and_surf_esp": {"alpha": ALPHA},
              "vol_esp": {"lam": 0.1}}.get(mode, {})
        state = {}

        def run():
            state["hits"] = screen(query, store, mode=mode, backend=self.backend,
                                   top_k=top_k, **kw)

        def score():
            hits = state.get("hits") or []
            return np.mean([float(h.score) for h in hits]) if hits else 0.0

        return Job(n_actual, run, score, extra={"timing": "kernel"})

    # --- measured effort -------------------------------------------------------
    def effort_reset(self) -> None:
        """Arm + clear the library's step recorder for one timed region; no recorder -> no-op."""
        st = _stats_module()
        if st is None:
            return
        try:
            st.reset()
        except Exception:
            pass

    def effort_summary(self) -> dict:
        """Read the recorder and switch it off; returns the measured block, or {} when nothing was recorded."""
        st = _stats_module()
        if st is None:
            return {}
        try:
            s = st.summary() or {}
        except Exception:
            return {}
        finally:
            try:
                st.disable()          # never leave the counter running through untimed setup
            except Exception:
                pass
        if not isinstance(s, dict):
            return {}
        return {str(k): _plain(v) for k, v in s.items()}

    # --- device timing ---------------------------------------------------------
    def sync(self) -> None:
        if self.device.type == "cuda":
            self._torch.cuda.synchronize()

    def reset(self) -> None:
        """Free the accel caches (CUDA graphs, allocator, sub-batch footprint cache) so each cell starts clean."""
        # the footprint cache is keyed by device, so clear it before the CUDA guard: a CPU cell
        # can populate it too
        try:
            from shepherd_score.accel.batch import _pad
            _pad._PAIR_FOOTPRINT_BYTES.clear()
        except Exception:
            pass
        if self.device.type != "cuda":
            return
        try:
            # clear_caches() is the library's drop-every-accel-cache entry point; older installs fall back
            from shepherd_score.accel import clear_caches
            clear_caches()
        except Exception:
            try:
                from shepherd_score.accel.drivers._graphed import reset_graph_cache
                reset_graph_cache()
            except Exception:
                self._torch.cuda.empty_cache()
