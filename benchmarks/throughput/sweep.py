"""Run one benchmark cell, stamp provenance, and merge results into results/<panel>.json keyed by cell.

Environment checks and reusable workload execution.
"""
from __future__ import annotations

import gc
import datetime
import json
import os
from collections import OrderedDict
import shutil
import socket
import sys
import tempfile
import time

import workloads
from timing import measure, measure_build

_CELL_KEYS = ("tool", "backend", "mode", "workload", "size", "timing")


# --- provenance ------------------------------------------------------------
def library_identity() -> dict:
    """Path, commit, branch and dirty flag of the shepherd_score that was actually imported."""
    info = {"path": None, "commit": None, "branch": None, "dirty": None}
    try:
        import subprocess

        import shepherd_score
        root = os.path.dirname(os.path.dirname(os.path.abspath(shepherd_score.__file__)))
        info["path"] = root

        def _git(*a):
            return subprocess.run(("git", "-C", root) + a, capture_output=True, text=True,
                                  timeout=30).stdout.strip()

        info["commit"] = _git("rev-parse", "HEAD") or None
        info["branch"] = _git("rev-parse", "--abbrev-ref", "HEAD") or None
        info["dirty"] = bool(_git("status", "--porcelain"))
    except Exception:
        pass
    return info


def library_pin() -> dict:
    """Library identity plus the FIG2_FSS_COMMIT pin (None when unset)."""
    return {"fss_library": library_identity(),
            "fss_commit_pinned": os.environ.get("FIG2_FSS_COMMIT")}


class ExclusivityError(RuntimeError):
    """FIG2_EXCLUSIVE=1 was declared but the allocation is not a whole node."""


def assert_exclusivity_honest() -> dict:
    """Refuse to run when FIG2_EXCLUSIVE=1 but the job holds only part of the node; unset means undeclared."""
    declared = os.environ.get("FIG2_EXCLUSIVE")
    if declared != "1":
        return {"declared": declared, "checked": False}
    try:
        have = len(os.sched_getaffinity(0))
    except Exception:
        return {"declared": declared, "checked": False}
    total = os.cpu_count()
    info = {"declared": declared, "checked": True, "cpus_allocated": have, "cpus_on_node": total}
    if total and have < total:
        raise ExclusivityError(os.linesep.join([
            "NODE IS NOT EXCLUSIVE -- refusing to start.",
            "  FIG2_EXCLUSIVE=1 declares a dedicated node, but this allocation holds",
            "  {} of {} CPUs on {}.".format(have, total, os.environ.get("SLURMD_NODENAME", "?")),
            "  Every cell would be stamped node_exclusive=true while sharing the machine, and a",
            "  contended speed number is indistinguishable from a library regression.",
            "  Either submit with --exclusive, or pass FIG2_EXCLUSIVE=0 to measure and say so."]))
    return info


class StoreReuseError(RuntimeError):
    """``--store-mode reuse`` found no matching store; fatal for the whole run, not per cell."""


class LibraryPinError(RuntimeError):
    """The imported library is not at the FIG2_FSS_COMMIT commit."""


def assert_library_pinned() -> dict:
    """Refuse to run unless the imported library matches FIG2_FSS_COMMIT; unset means unpinned."""
    want = (os.environ.get("FIG2_FSS_COMMIT") or "").strip()
    ident = library_identity()
    if not want:
        return ident
    got = ident.get("commit")
    if not got:
        raise LibraryPinError(
            "FIG2_FSS_COMMIT={} is set but the library commit could not be read (path={}). "
            "Refusing to benchmark an unidentifiable library.".format(want, ident.get("path")))
    if not (got.startswith(want) or want.startswith(got)):
        raise LibraryPinError(os.linesep.join([
            "LIBRARY PIN MISMATCH -- refusing to start.",
            "  expected commit : {}  (FIG2_FSS_COMMIT)".format(want),
            "  imported commit : {}  (branch {})".format(got, ident.get("branch")),
            "  imported from   : {}".format(ident.get("path")),
            "Another process may have checked out a different branch in that repo. Either point",
            "FSS_REPO at the intended checkout, or update FIG2_FSS_COMMIT deliberately."]))
    if ident.get("dirty"):
        sys.stderr.write(
            "WARNING: the library at pinned commit {} has UNCOMMITTED changes, so the commit does "
            "not fully describe what is being measured.".format(got[:10]) + os.linesep)
    return ident


def describe_env(aligner) -> dict:
    """Node, GPU, environment and library provenance stamped on every cell."""
    info = {
        "tool": aligner.name, "backend": aligner.backend, "device": aligner.device_label(),
        "python": sys.version.split()[0], "hostname": socket.gethostname(),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    # FIG2_EXCLUSIVE: "1" dedicated node, "0" shared, unset = undeclared (None).
    _excl = os.environ.get("FIG2_EXCLUSIVE")
    info["node_exclusive"] = {"1": True, "0": False}.get(_excl)
    try:
        info["cpus_allocated"] = len(os.sched_getaffinity(0))
    except Exception:
        info["cpus_allocated"] = None
    info["cpus_on_node"] = os.cpu_count()

    # shepherd_score.__version__ is not recorded: it comes from pip's dist-info, not the checkout.
    for mod in ("triton", "rdkit"):
        try:
            info[f"{mod}_version"] = __import__(mod).__version__
        except Exception:
            info[f"{mod}_version"] = None

    # using_svml is stamped on every cell: CPU fss timings without SVML are understated.
    fp = {}
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from comparators import envcheck
        fp = envcheck.fingerprint()
    except Exception:
        pass
    info["using_svml"] = fp.get("using_svml")
    info["fss_cpu_soa"] = fp.get("fss_cpu_soa")

    info.update(library_pin())
    # None in fss_flags means the variable was unset (library default applied), not False.
    info["fss_flags"] = {k: os.environ.get(k) for k in
                         ("FSS_SCREEN_ARRAYS", "FSS_CANONICAL_STORE", "FSS_CANONICAL_MODES",
                          "FSS_POSES_PER_CTA", "FSS_SUBBATCH_CHUNK", "FSS_SEED_COORD_DEDUP")}
    # What the library resolved at import time, which can differ from the current environment.
    try:
        from shepherd_score.accel.batch import _arrays as _arr
        info["fss_arrays_enabled"] = bool(_arr.ENABLED)
    except Exception:
        info["fss_arrays_enabled"] = None
    # The library's own default for canonical stores (None in the signature means canonical).
    try:
        import inspect
        from shepherd_score.screen import ProfileStore
        _d = inspect.signature(ProfileStore.create).parameters["canonical"].default
        info["fss_canonical_default"] = True if _d is None else bool(_d)
    except Exception:
        info["fss_canonical_default"] = None
    for mod in ("torch", "numba", "numpy"):
        info[f"{mod}_version"] = fp.get(mod)
    return info


def evals_per_run(tool: str, mode: str):
    """``(evals, source)``: the configured seeds x fine-steps budget; ``steps_measured`` is what ran."""
    if tool == "fss":
        try:
            from shepherd_score.accel.batch.aligners import _seeds_for, _steps_for
            s, st = _seeds_for(mode), _steps_for(mode)
            return int(s) * int(st), f"MODE_SEEDS[{mode}]={s} x MODE_STEPS[{mode}]={st}"
        except Exception as e:
            return None, f"unread ({type(e).__name__})"
    if tool == "roshambo2":
        # ROSHAMBO2's shipped default; never overridden here.
        return 4 * 100, "start_mode=1 (4 orientations) x 100 steps"
    return None, "n/a"


# Below this fraction of the configured steps a cell is stamped effort_truncated.
EFFORT_MIN_FRAC = 0.8


def effort_check(row: dict, tool: str, mode: str, min_frac: float = EFFORT_MIN_FRAC):
    """Stamp ``effort_truncated`` from ``row['steps_measured']``; return a log line when truncated, else None."""
    st = row.get("steps_measured")
    if not isinstance(st, dict) or not st:
        return None
    cfg = st.get("steps_configured")
    mean = st.get("steps_mean")
    if not cfg or mean is None:
        return None
    cfg, mean = float(cfg), float(mean)
    frac = mean / cfg
    st["effort_frac"] = round(frac, 4)
    ev = row.get("evals_per_run")
    if ev:
        st["evals_measured"] = round(ev * frac)
    row["effort_truncated"] = bool(frac < min_frac)
    if not row["effort_truncated"]:
        return None
    return (f"EFFORT TRUNCATED {tool}/{mode}: ran {mean:.1f} of {cfg:g} configured steps "
            f"({frac * 100:.0f}%) over {st.get('calls')} fine-loop calls, "
            f"early_stop_frac={st.get('early_stop_frac')} -- this cell's throughput is NOT the "
            f"configured optimizer's and is not comparable")


# --- Region A: index construction ------------------------------------------
# Builds above FIG2_BUILD_STAGE_MAX conformers are skipped rather than measured unstaged.
BUILD_STAGE_MAX = int(os.environ.get("FIG2_BUILD_STAGE_MAX", 2_000_000))


def _dir_bytes(path: str) -> int:
    return sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(path) for f in fs)


def stage_library(pool, lib_indices):
    """Materialize ``lib_indices`` from ``pool`` into a resident list; returns ``(staged, seconds)``."""
    t0 = time.perf_counter()
    staged = [pool[i] for i in lib_indices]
    return staged, time.perf_counter() - t0


# Staged libraries are reused across modes. Entries hold references (they pin shards against the
# FIG2_POOL_RESIDENT LRU), so the total is capped at STAGE_CACHE_MAX molecules, oldest evicted first.
_STAGE_CACHE = OrderedDict()
STAGE_CACHE_MAX = 3_000_000


def stage_library_cached(pool, size, seed, *, n_conf=None):
    """``stage_library`` for a (size, seed) library, cached per pool; returns ``(staged, seconds, hit)``."""
    key = (id(pool), len(pool), int(size), int(seed), n_conf)
    hit = _STAGE_CACHE.get(key)
    if hit is not None:
        _STAGE_CACHE.move_to_end(key)
        return hit, 0.0, True
    staged, secs = stage_library(pool, workloads.screen_library(len(pool), size, seed,
                                                                n_conf=n_conf))
    _STAGE_CACHE[key] = staged
    while (sum(len(v) for v in _STAGE_CACHE.values()) > STAGE_CACHE_MAX
           and len(_STAGE_CACHE) > 1):
        _STAGE_CACHE.popitem(last=False)
    return staged, secs, False


def stage_cache_clear():
    """Drop every cached library."""
    _STAGE_CACHE.clear()


def _canonical_requested(mode: str) -> bool:
    """Whether this process builds a canonical-frame store for ``mode``; FSS_CANONICAL_STORE unset means non-canonical."""
    # For the library's own default, set FSS_CANONICAL_STORE=1 with FSS_CANONICAL_MODES=all.
    if os.environ.get("FSS_CANONICAL_STORE") != "1":
        return False
    modes = os.environ.get("FSS_CANONICAL_MODES")
    return True if not modes else (mode in [m.strip() for m in modes.split(",") if m.strip()])


# --- store format ------------------------------------------------------------
# Library symbols whose source decides what a persisted fss index contains; their source is hashed
# into the store identity. Update this list when the library renames one (a missing symbol is a
# hard error, so stale stores are never reused silently).
_STORE_FORMAT_SYMBOLS = {
    "fss": (("shepherd_score.screen", "MoleculeProfile"),
            ("shepherd_score.screen", "_schema_from_modes"),
            ("shepherd_score.screen", "_store_supports"),
            ("shepherd_score.screen", "_profile_from_schema"),
            ("shepherd_score.screen", "ProfileStore"),
            ("shepherd_score.screen", "_canonical_rot")),
}
_STORE_FORMAT_CACHE = {}


def store_format_fingerprint(aligner) -> str:
    """Hash of the adapter's ``build_store`` source plus, for fss, the library's store-layout symbols."""
    import hashlib
    import importlib
    import inspect

    key = (aligner.name, aligner.backend)
    if key in _STORE_FORMAT_CACHE:
        return _STORE_FORMAT_CACHE[key]
    h = hashlib.sha256()
    try:
        h.update(inspect.getsource(type(aligner).build_store).encode())
    except (OSError, TypeError, AttributeError) as exc:
        raise RuntimeError("cannot read %s.build_store source for the store-format fingerprint: %s"
                           % (type(aligner).__name__, exc))
    for modname, sym in _STORE_FORMAT_SYMBOLS.get(aligner.name, ()):
        mod = importlib.import_module(modname)
        obj = getattr(mod, sym, None)
        if obj is None:
            raise RuntimeError(
                "store-format symbol %s.%s no longer exists. It decides what a persisted "
                "index contains, so its disappearance must invalidate every store rather "
                "than quietly stop being hashed. Update _STORE_FORMAT_SYMBOLS in sweep.py "
                "to the new layout." % (modname, sym))
        h.update(("%s.%s" % (modname, sym)).encode())
        try:
            h.update(inspect.getsource(obj).encode())
        except (OSError, TypeError) as exc:
            raise RuntimeError(
                "cannot read source of store-format symbol %s.%s (%s). Its content decides what "
                "a persisted index holds, so it cannot be skipped; either keep it in readable "
                "Python or drop it from _STORE_FORMAT_SYMBOLS deliberately, which invalidates "
                "every existing store." % (modname, sym, exc))
    _STORE_FORMAT_CACHE[key] = h.hexdigest()[:16]
    return _STORE_FORMAT_CACHE[key]


def store_identity(aligner, *, mode, size, seed) -> dict:
    """The fields that determine an index's content; stamped on the build row."""
    return {"tool": aligner.name, "backend": aligner.backend, "mode": mode,
            "size": int(size), "seed": int(seed),
            "canonical": _canonical_requested(mode),
            "store_format": store_format_fingerprint(aligner)}


# --- persistent stores -----------------------------------------------------
# Indexes persist under FIG2_STORE_ROOT keyed by (tool, backend, mode, size, seed) plus the pool
# identity; an empty root disables persistence (temp dirs).
STORE_ROOT_ENV = "FIG2_STORE_ROOT"
_STORE_SIDECAR = "fig2_store.json"


def default_store_root() -> str:
    """Where persisted indexes live; empty string disables persistence."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.environ.get(STORE_ROOT_ENV, os.path.join(repo, "benchmark-storage", "throughput", "stores"))


def store_path(root: str, aligner, *, mode, size, seed) -> str:
    """Directory of one index, named by its identity fields."""
    return os.path.join(root, f"{aligner.name}_{aligner.backend}_{mode}_{int(size)}_{int(seed)}")


def _pool_identity(pool, aligner, pool_id=None) -> dict:
    """Pool identity a reuse must match; ``pool_id`` is ``pool.cache_identity(...)`` and None makes the store unverifiable."""
    from workloads import N_CONFORMERS
    return {"pool_id": pool_id, "n_pool": len(pool), "n_conformers": int(N_CONFORMERS),
            "cache_key": getattr(aligner, "cache_key", lambda: None)()}


def read_store_sidecar(path: str):
    try:
        with open(os.path.join(path, _STORE_SIDECAR)) as fh:
            return json.load(fh)
    except Exception:
        return None


def write_store_sidecar(path: str, *, aligner, mode, size, seed, pool, n_indexed,
                        build_seconds, pool_id=None, canonical=None):
    """Write the identity a reuse must match beside the store (atomically).

    ``canonical`` overrides the env-derived flag: ``_canonical_requested`` reads this process's
    FSS_CANONICAL_STORE, and a store assembled elsewhere (build_store_parallel's merge) can hold
    canonical data while this process's env says otherwise.
    """
    doc = dict(store_identity(aligner, mode=mode, size=size, seed=seed))
    if canonical is not None:
        doc["canonical"] = bool(canonical)
    doc.update(pool=_pool_identity(pool, aligner, pool_id), n_indexed=n_indexed,
               build_seconds=build_seconds, store_bytes=_dir_bytes(path),
               built_at=datetime.datetime.now().isoformat(timespec="seconds"))
    tmp = os.path.join(path, _STORE_SIDECAR + f".tmp.{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, os.path.join(path, _STORE_SIDECAR))
    return doc


def store_is_reusable(path: str, *, aligner, mode, size, seed, pool, pool_id=None):
    """``(ok, reason)``: every identity and pool field must match; a mismatch names the field."""
    doc = read_store_sidecar(path)
    if doc is None:
        return False, "no sidecar (not a harness-written store, or never finished)"
    want = store_identity(aligner, mode=mode, size=size, seed=seed)
    for k, v in want.items():
        if doc.get(k) != v:
            return False, f"{k}: store has {doc.get(k)!r}, this cell wants {v!r}"
    pid = _pool_identity(pool, aligner, pool_id)
    if pid["pool_id"] is None:
        return False, ("pool_id unavailable -- cannot verify which pool built this store "
                       "(pass pool_id=pool.cache_identity(...))")
    for k, v in pid.items():
        if doc.get("pool", {}).get(k) != v:
            return False, f"pool.{k}: store has {doc.get('pool', {}).get(k)!r}, pool is {v!r}"
    if not doc.get("n_indexed"):
        return False, "sidecar records n_indexed=0 (the build did not validate)"
    return True, "ok"

def _region_a(aligner) -> dict:
    """The adapter's declared featurization boundary (optional ``region_a_boundary`` hook)."""
    hook = getattr(aligner, "region_a_boundary", None)
    if hook is None:
        return {"declared": False, "note": "adapter declares no stage boundary; build_seconds is "
                                           "NOT cross-tool comparable without one"}
    return {**hook(), "declared": True,
            "pool_cost_measured_in": "results/P4.json (per-molecule preparation)",
            "do_not_blend": "env.pool_build_seconds is 0.0 on a cache hit, covers env.n_pool not "
                            "size, and is a multi-core wall time; it does not add to build_seconds"}


def run_build_cell(aligner, pool, *, mode, size, seed,
                   stage_max=BUILD_STAGE_MAX, store_root=None, store_mode="auto",
                   pool_id=None):
    """Time index construction (Region A); returns ``(row, store_dir)``, or ``(None, None)`` without a screen path."""
    if not getattr(aligner, "supports_screen", False):
        return None, None

    # A reused store emits no build row at all: an absent row or field means unmeasured.
    # --store-mode reuse hard-fails on a mismatch instead of rebuilding.
    keep_path = None
    if store_root:
        keep_path = store_path(store_root, aligner, mode=mode, size=size, seed=seed)
        if store_mode in ("reuse", "auto") and os.path.isdir(keep_path):
            ok, why = store_is_reusable(keep_path, aligner=aligner, mode=mode, size=size,
                                        seed=seed, pool=pool, pool_id=pool_id)
            if ok:
                return None, keep_path
            if store_mode == "reuse":
                raise StoreReuseError(
                    f"--store-mode reuse: {keep_path} is not reusable -- {why}")
            print(f"  [store] rebuilding {os.path.basename(keep_path)}: {why}", flush=True)
        elif store_mode == "reuse":
            raise StoreReuseError(f"--store-mode reuse: no store at {keep_path}")

    ident = store_identity(aligner, mode=mode, size=size, seed=seed)
    row = dict(ident, workload="build_index", timing="n/a", atom_basis=aligner.atom_basis(mode),
               region_a=_region_a(aligner))
    if size > stage_max:
        row["build_staged"] = False
        row["build_skipped"] = (f"size {size} > FIG2_BUILD_STAGE_MAX {stage_max}: staging would not "
                                f"fit in RAM and an unstaged build would measure pool shard I/O")
        # The build is not timed but the store is still built, unstaged (the pool streams in
        # ascending order, so the FIG2_POOL_RESIDENT LRU bounds RSS), into the persistent path;
        # no timing field is recorded, since absent means unmeasured.
        if keep_path and store_mode in ("build", "auto"):
            lib = workloads.screen_library(len(pool), size, seed)
            tmp = tempfile.mkdtemp(prefix=f"fig2_build_{aligner.name}_",
                                   dir=os.path.dirname(keep_path))
            try:
                aligner.build_store(pool, lib, tmp, modes=[mode])
                counter = getattr(aligner, "index_count", None)
                n_idx = counter(tmp) if counter is not None else 0
                if os.path.isdir(keep_path):
                    shutil.rmtree(keep_path, ignore_errors=True)
                os.makedirs(os.path.dirname(keep_path), exist_ok=True)
                shutil.move(tmp, keep_path)
            except BaseException:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
            write_store_sidecar(keep_path, aligner=aligner, mode=mode, size=size, seed=seed,
                                pool=pool, n_indexed=n_idx, build_seconds=None, pool_id=pool_id)
            row["store_persisted"] = keep_path
            row["build_n_indexed"] = n_idx
            return row, keep_path
        return row, None

    # The library is staged outside the timer so the build does not measure pool shard I/O.
    staged, gather_s, cached = stage_library_cached(pool, size, seed)
    idx = range(len(staged))
    counter = getattr(aligner, "index_count", None)

    # gc.freeze() keeps a gen-2 collection from walking the staged library mid-build.
    gc.freeze()

    def build_one():
        d = tempfile.mkdtemp(prefix=f"fig2_build_{aligner.name}_")
        aligner.build_store(staged, idx, d, modes=[mode])
        return d

    def probe(d):
        return {"n_indexed": counter(d) if counter is not None else 0, "bytes": _dir_bytes(d)}

    try:
        stats, keep = measure_build(build_one, aligner, n=size, probe=probe,
                                    discard=lambda d: shutil.rmtree(d, ignore_errors=True))
    finally:
        gc.unfreeze()
    row.update(stats)
    row.update(build_staged=True, lib_gather_seconds=gather_s,
               lib_gather_cached=cached,
               lib_gather_source=type(pool).__name__)

    # Move the surviving index into the persistent cache.
    if keep_path and keep and store_mode in ("build", "auto"):
        try:
            if os.path.isdir(keep_path):
                shutil.rmtree(keep_path, ignore_errors=True)
            os.makedirs(os.path.dirname(keep_path), exist_ok=True)
            shutil.move(keep, keep_path)
            write_store_sidecar(keep_path, aligner=aligner, mode=mode, size=size, seed=seed,
                                pool=pool, n_indexed=row.get("build_n_indexed"),
                                build_seconds=row.get("build_seconds"), pool_id=pool_id)
            row["store_persisted"] = keep_path
            keep = keep_path
        except Exception as e:
            print(f"  [store] could not persist {keep_path}: {type(e).__name__}: {e}", flush=True)
    return row, keep


# --- one cell --------------------------------------------------------------
def run_cell(aligner, pool, query, *, mode, workload, size, seed,
             top_k, store_dir=None) -> dict:
    """Prepare and time one cell; ``store_dir`` is an already-built index to screen against."""
    extra = {}
    owned = None
    if workload == "pairwise":
        pairs = workloads.pairwise_pairs(len(pool), size, seed)
        job = aligner.prepare_pairwise(pool, pairs, mode)
    elif workload == "screen":
        if store_dir is None:
            # store_build_staged=False marks an in-cell unstaged build; the certified build
            # number is the sibling build_index row.
            lib = workloads.screen_library(len(pool), size, seed)
            owned = store_dir = tempfile.mkdtemp(prefix=f"fig2_store_{aligner.name}_")
            t0 = time.perf_counter()
            aligner.build_store(pool, lib, store_dir, modes=[mode])
            extra["store_build_seconds"] = time.perf_counter() - t0
            extra["store_bytes"] = _dir_bytes(store_dir)
            extra["store_build_staged"] = False
        job = aligner.prepare_screen(store_dir, query, mode, top_k)
    else:
        raise ValueError(f"unknown workload {workload!r}")

    # gc.freeze() keeps a gen-2 collection from walking the prepared job mid-rep.
    gc.freeze()
    try:
        row = measure(job, aligner)
    finally:
        gc.unfreeze()
        if owned:
            shutil.rmtree(owned, ignore_errors=True)

    ev, evs = evals_per_run(aligner.name, mode)
    row.update({
        "tool": aligner.name, "backend": aligner.backend, "mode": mode,
        "workload": workload, "size": size,
        # The adapter may name its own timed region (job.extra["timing"]); default "kernel".
        "timing": job.extra.get("timing", "kernel"),
        "evals_per_run": ev, "evals_source": evs, "atom_basis": aligner.atom_basis(mode),
    })
    row.update(job.extra)
    row.update(extra)

    note = effort_check(row, aligner.name, mode)
    if note:
        print(f"  !! {note}  (workload={workload} N={size})", flush=True)
    return row


def advance_stop(prev, n, advance_seconds, cap):
    """Reason to stop a size series (previous cell >= advance_seconds, or next size projected over cap), else None."""
    if prev is None:
        return None
    if prev.get("compute_seconds", 0.0) >= advance_seconds:
        return f"prev {prev['compute_seconds']:.1f}s >= {advance_seconds:.0f}s advance"
    rate = prev.get("aligns_per_sec", 0.0)
    if rate > 0 and n / rate > cap:
        return f"projected {n / rate:.0f}s > {cap:.0f}s cap"
    return None


# --- incremental keyed results --------------------------------------------
def _cell_key(row) -> str:
    return "|".join(str(row.get(k)) for k in _CELL_KEYS)


def merge_partials(dest: str, srcs) -> int:
    """Fold per-leg part files into ``dest`` sequentially, deleting each; an unreadable part is kept and reported."""
    srcs = list(srcs)
    total, bad = 0, []
    for src in srcs:
        try:
            data = json.load(open(src))
        except Exception as exc:
            bad.append("%s: %s" % (src, exc))
            continue
        total = merge(dest, data.get("meta", {}), list(data.get("cells", {}).values()))
        os.remove(src)
    print("folded %d part file(s) -> %s (%d cells)" % (len(srcs) - len(bad), dest, total),
          flush=True)
    if bad:
        raise RuntimeError("merge_partials could not read %d part file(s), left in place: %s"
                           % (len(bad), "; ".join(bad)))
    return total


def merge(out_path: str, meta: dict, cells: list) -> int:
    """Merge ``cells`` into ``out_path`` keyed by cell; cells not in this run are kept. Returns the cell count."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    data = {"meta": {}, "cells": {}}
    if os.path.exists(out_path):
        try:
            data = json.load(open(out_path))
        except Exception:
            # An unreadable results file is moved aside, never treated as empty.
            aside = f"{out_path}.corrupt.{os.getpid()}"
            try:
                os.replace(out_path, aside)
                sys.stderr.write(f"WARNING: {out_path} was unreadable; moved to {aside} and "
                                 f"starting a fresh file.{os.linesep}")
            except OSError:
                pass
    data.setdefault("cells", {})
    data["meta"] = {**data.get("meta", {}), **meta}
    for row in cells:
        data["cells"][_cell_key(row)] = row
    # Write-then-rename so a killed job never leaves a partial file.
    tmp = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out_path)
    return len(data["cells"])
