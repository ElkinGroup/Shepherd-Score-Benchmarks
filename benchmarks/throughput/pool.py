"""Featurize a SMILES set once through the tool adapter, cache it beside the .smi, and reuse it.

Shared by input preparation and the manuscript benchmark suite.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import sys
import time
from bisect import bisect_right
from collections import OrderedDict


def read_smiles(path: str) -> list[str]:
    """SMILES from column 0; blank and '#' lines are skipped."""
    with open(path) as fh:
        return [ln.split()[0] for ln in fh
                if ln.strip() and not ln.lstrip().startswith("#")]


# Above FIG2_SHARD_ABOVE conformers the pool is built and read as shards of FIG2_SHARD_COMPOUNDS
# compounds each; ShardedPool is a read-only sequence, so callers index it like the list.
SHARD_ABOVE = int(os.environ.get("FIG2_SHARD_ABOVE", 2_000_000))   # conformers
SHARD_COMPOUNDS = int(os.environ.get("FIG2_SHARD_COMPOUNDS", 2_000))


class ShardedPool:
    """Molecules backed by shard pickles, with an LRU of FIG2_POOL_RESIDENT (default 3) resident shards."""

    def __init__(self, paths, counts, loader=pickle.load):
        max_resident = int(os.environ.get("FIG2_POOL_RESIDENT", 3))
        self._loader = loader
        self._paths = list(paths)
        self._starts = [0]
        for c in counts:
            self._starts.append(self._starts[-1] + int(c))
        self._max_resident = max(1, max_resident)
        self._resident = OrderedDict()

    def __len__(self):
        return self._starts[-1]

    def _shard(self, k):
        got = self._resident.get(k)
        if got is None:
            with open(self._paths[k], "rb") as fh:
                got = self._loader(fh)
            self._resident[k] = got
            while len(self._resident) > self._max_resident:
                self._resident.popitem(last=False)
        else:
            self._resident.move_to_end(k)
        return got

    def __getitem__(self, i):
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        k = bisect_right(self._starts, i) - 1
        return self._shard(k)[i - self._starts[k]]


def _shard_paths(cache: str, k: int) -> str:
    return f"{cache}.shard{k:04d}"


def _count_path(shard: str) -> str:
    return f"{shard}.n"


def _write_count(shard: str, n: int) -> None:
    """Record a shard's length in a sidecar so the manifest can count shards this task did not build."""
    tmp = f"{_count_path(shard)}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write(f"{int(n)}\n")
    os.replace(tmp, _count_path(shard))


SHARD_LOCK_STALE = 5400.0


def _claim(shard: str) -> bool:
    """Atomically claim a shard (O_EXCL lock) so concurrent jobs can work-steal; a lock older than SHARD_LOCK_STALE is stolen."""
    lock = f"{shard}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.uname().nodename}:{os.getpid()}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(lock) > SHARD_LOCK_STALE:
                os.remove(lock)
                return _claim(shard)
        except OSError:
            pass
        return False


def _release(shard: str) -> None:
    try:
        os.remove(f"{shard}.lock")
    except OSError:
        pass


def _scan_shards(cache: str):
    """Every shard on disk with a count sidecar, in k order; holes are valid (a shard boundary is a compound boundary)."""
    import glob as _glob
    paths, counts = [], []
    # The 4-digit pattern excludes the .n, .lock and .tmp files beside each shard.
    for sp in sorted(_glob.glob(cache + ".shard[0-9][0-9][0-9][0-9]")):
        cp = _count_path(sp)
        if not os.path.exists(cp):
            continue
        try:
            with open(cp) as fh:
                counts.append(int(fh.read().strip()))
        except Exception:
            continue
        paths.append(sp)
    return paths, counts


def _cache_path(aligner, molecules: str, seed: int, surf_points) -> str:
    """Pickle path keyed on (tool, seed, surf_points, file, mtime, cache_key, N_CONFORMERS); lives beside the .smi (or FIG2_POOL_CACHE), not TMPDIR."""
    try:
        mtime = int(os.path.getmtime(molecules))
    except OSError:
        mtime = 0
    extra = getattr(aligner, "cache_key", lambda: "")()   # e.g. fss's charge model
    from workloads import N_CONFORMERS
    key = (f"{aligner.name}|{seed}|{surf_points}|{os.path.basename(molecules)}|{mtime}|{extra}"
           f"|nconf={N_CONFORMERS}")
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    d = os.environ.get(
        "FIG2_POOL_CACHE",
        os.path.join(os.path.dirname(os.path.abspath(molecules)), ".pool_cache"),
    )
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"pool_{aligner.name}_{h}.pkl")


def _build_sharded(aligner, smiles, cache, n_conf):
    """Build the pool in SHARD_COMPOUNDS chunks, one pickle each, work-stealing across jobs; returns ``(ShardedPool, n_failed_compounds)``."""
    n_fail = 0
    manifest = f"{cache}.shards"
    n_shards = (len(smiles) + SHARD_COMPOUNDS - 1) // SHARD_COMPOUNDS
    for k in range(n_shards):
        lo = k * SHARD_COMPOUNDS
        chunk = smiles[lo:lo + SHARD_COMPOUNDS]
        path = _shard_paths(cache, k)
        # A completed shard (pickle + count sidecar) is reused; one without a sidecar is rebuilt.
        if os.path.exists(path):
            if os.path.exists(_count_path(path)):
                print(f"pool: shard {k} reused", flush=True)
                continue
            os.remove(path)
        if not _claim(path):
            continue                     # another job is building this one
        if os.path.exists(path):         # it finished while we were claiming
            _release(path)
            continue
        built = [m for m in aligner.build(chunk) if m is not None]
        n_fail += len(chunk) - (len(built) // n_conf if n_conf else len(built))
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "wb") as fh:
            pickle.dump(built, fh)
        os.replace(tmp, path)
        _write_count(path, len(built))
        _release(path)
        print(f"pool: shard {k} -> {len(built):,} molecules "
              f"({lo + len(chunk):,}/{len(smiles):,} compounds)", flush=True)
        del built
        # The manifest is rewritten after every shard so a partial build is loadable.
        paths, counts = _scan_shards(cache)
        _write_manifest(manifest, paths, counts)
    paths, counts = _scan_shards(cache)
    _write_manifest(manifest, paths, counts)
    return ShardedPool(paths, counts), n_fail


def _write_manifest(manifest, paths, counts):
    tmp = f"{manifest}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        # basenames, so the cache directory can be moved
        pickle.dump({"paths": [os.path.basename(x) for x in paths],
                     "counts": list(counts)}, fh)
    os.replace(tmp, manifest)


def cache_identity(aligner, molecules: str, seed: int, surf_points) -> str:
    """Basename of the pool pickle this configuration would load; recorded on persisted stores."""
    return os.path.basename(_cache_path(aligner, molecules, seed, surf_points))


def load_or_build(aligner, molecules: str, *, seed: int, surf_points) -> tuple[list, int, float]:
    """Return ``(pool, n_embed_failed, build_seconds)``; loads the cache if present, else builds via ``aligner.build`` (sharded above SHARD_ABOVE)."""
    cache = _cache_path(aligner, molecules, seed, surf_points)
    manifest = f"{cache}.shards"
    # FIG2_BUILD_ONLY=1 skips the cache-hit path so a build job completes a partial sharded pool
    # instead of loading it.
    build_only = os.environ.get("FIG2_BUILD_ONLY") == "1"
    if not build_only and os.path.exists(manifest):
        try:
            with open(manifest, "rb") as fh:
                m = pickle.load(fh)
            base = os.path.dirname(os.path.abspath(manifest))
            resolved = [q if os.path.isabs(q) else os.path.join(base, q) for q in m["paths"]]
            if all(os.path.exists(q) for q in resolved):
                pool = ShardedPool(resolved, m["counts"])
                print(f"pool: loaded {len(pool):,} cached across {len(m['paths'])} shards "
                      f"({os.path.basename(manifest)})", flush=True)
                return pool, 0, 0.0
        except Exception:
            pass
    if not build_only and os.path.exists(cache):
        try:
            with open(cache, "rb") as fh:
                pool = pickle.load(fh)
            if len(pool) >= 2:
                print(f"pool: loaded {len(pool)} cached ({os.path.basename(cache)})", flush=True)
                return pool, 0, 0.0
        except Exception:
            pass
    smiles = read_smiles(molecules)
    from workloads import N_CONFORMERS
    expected = len(smiles) * N_CONFORMERS
    print(f"pool: building {len(smiles):,} compounds x {N_CONFORMERS} conformers "
          f"= {expected:,} from {os.path.basename(molecules)} ...", flush=True)
    t0 = time.perf_counter()
    if expected > SHARD_ABOVE:
        pool, n_fail = _build_sharded(aligner, smiles, cache, N_CONFORMERS)
        build_seconds = time.perf_counter() - t0
        print(f"pool: built {len(pool):,} in {build_seconds:.1f}s (sharded"
              f"{f', {n_fail} compounds failed' if n_fail else ''})", flush=True)
        return pool, n_fail, build_seconds
    built = aligner.build(smiles)
    build_seconds = time.perf_counter() - t0
    pool = [m for m in built if m is not None]
    n_fail = len(built) - len(pool)
    if len(pool) < 2:
        sys.exit("need at least 2 successfully-built molecules")
    print(f"pool: built {len(pool)} in {build_seconds:.1f}s"
          f"{f' ({n_fail} embed-failed)' if n_fail else ''}", flush=True)
    try:
        # write-then-rename: concurrent builders never expose a partial pickle
        tmp = f"{cache}.tmp.{os.getpid()}"
        with open(tmp, "wb") as fh:
            pickle.dump(pool, fh)
        os.replace(tmp, cache)
    except Exception:
        pass
    return pool, n_fail, build_seconds
