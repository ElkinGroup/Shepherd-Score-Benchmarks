"""Compute vol_fukui's Fukui fields in parallel, then build the store from the cached arrays.

Why this exists: the vol_fukui SCREEN at N=100,000 needs 300,000 GFN2-xTB single points (three
per molecule, neutral/cation/anion). Serially that is unbounded -- job 23451373 spent six hours
on 0.7 cores and never reached build_store. build_store_parallel cannot help: shard_size is
100,000, so a 100k store is ONE shard and parts must divide the shard count.

The seam is on Molecule instead. ``fukui`` is a lazy property guarded by ``self._fukui`` and it
has a setter, so the expensive part can be computed anywhere and injected:

    @property
    def fukui(self):
        if self._fukui is None: self._fukui = self._generate_fukui()
        return self._fukui

phase=prep   one array task per slice; computes _generate_fukui() and writes the (n_atoms,)
             float32 arrays plus the indices that FAILED (a non-converging ion propagates --
             the benchmark's pairwise path drops such molecules, so they are recorded, never
             silently dropped into a short library).
phase=build  loads every slice, assigns mol.fukui = arr, and calls build_store, which is then
             pure array extraction with no xTB at all.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.getcwd())
import pool as poolmod                                                      # noqa: E402
import workloads                                                            # noqa: E402
from aligners import get_aligner                                            # noqa: E402

SEED = 42
MODE = "vol_fukui"
SURF = int(os.environ.get("FSS_BENCH_SURF_POINTS", "200"))


def load_pool(mols_file):
    a = get_aligner("fss", backend="numba")
    p, _, _ = poolmod.load_or_build(a, mols_file, seed=SEED, surf_points=SURF)
    return a, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["prep", "build"])
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--size", type=int, default=100000)
    ap.add_argument("--molecules", default="enamine_210k.smi")
    ap.add_argument("--work", required=True)
    ap.add_argument("--store-root", default="")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)

    a, p = load_pool(args.molecules)
    lib = workloads.screen_library(len(p), args.size, SEED)

    if args.phase == "prep":
        lo = (len(lib) * args.part) // args.parts
        hi = (len(lib) * (args.part + 1)) // args.parts
        mine = lib[lo:hi]
        t0 = time.perf_counter()
        vals, lens, ok_idx, failed = [], [], [], []
        for n, idx in enumerate(mine):
            try:
                f = np.asarray(p[idx].fukui, dtype=np.float32)
            except Exception:
                failed.append(idx)
                continue
            vals.append(f); lens.append(f.shape[0]); ok_idx.append(idx)
            if n and n % 200 == 0:
                print("  part %d: %d/%d  %.1f s" % (args.part, n, len(mine),
                                                    time.perf_counter() - t0), flush=True)
        out = os.path.join(args.work, "part_%05d.npz" % args.part)
        np.savez(out + ".tmp.npz",
                 idx=np.asarray(ok_idx, dtype=np.int64),
                 off=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64) if lens else np.zeros(1, np.int64),
                 vals=np.concatenate(vals) if vals else np.zeros(0, np.float32),
                 failed=np.asarray(failed, dtype=np.int64))
        os.replace(out + ".tmp.npz", out)
        print("part %d done: %d ok, %d failed, %.1f s" % (args.part, len(ok_idx), len(failed),
                                                          time.perf_counter() - t0), flush=True)
        return

    # ---- build -------------------------------------------------------------
    import glob
    parts = sorted(glob.glob(os.path.join(args.work, "part_*.npz")))
    if len(parts) != args.parts:
        sys.exit("expected %d part files, found %d -- refusing to build a short library"
                 % (args.parts, len(parts)))
    cache, failed = {}, []
    for f in parts:
        z = np.load(f)
        idx, off, vals = z["idx"], z["off"], z["vals"]
        for k, i in enumerate(idx):
            cache[int(i)] = vals[off[k]:off[k + 1]]
        failed.extend(int(x) for x in z["failed"])
    print("loaded %d fukui arrays, %d failures" % (len(cache), len(failed)), flush=True)

    keep = [i for i in lib if i in cache]
    if len(keep) != len(lib):
        print("WARNING: library shrinks %d -> %d (xTB failures)" % (len(lib), len(keep)), flush=True)
    import sweep as sweepmod
    root = args.store_root or sweepmod.default_store_root()
    dest = sweepmod.store_path(root, a, mode=MODE, size=len(keep), seed=SEED)
    tmp = dest + ".building"
    import shutil
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp, exist_ok=True)
    t0 = time.perf_counter()
    # ONE pass: mutate and consume the same object. Injecting in a separate loop loses every
    # assignment on a sharded pool, because pool[i] re-materialises from the shard on reload.
    from shepherd_score.screen import ProfileStore
    from aligners.fss import SURF_POINTS as _SP
    store = ProfileStore.create(tmp, num_surf_points=_SP, modes=(MODE,),
                                pre_centered=True, overwrite=True, canonical=False)
    for n, i in enumerate(keep):
        m = p[i]
        m.fukui = cache[i]
        store.add(m)
        if n and n % 10000 == 0:
            print("  added %d/%d  %.1f s" % (n, len(keep), time.perf_counter() - t0), flush=True)
    store.close()
    print("build_store: %.1f s (no xTB: every fukui was injected)" % (time.perf_counter() - t0), flush=True)
    if os.path.isdir(dest):
        shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)
    pid = poolmod.cache_identity(a, args.molecules, SEED, SURF)
    sweepmod.write_store_sidecar(dest, aligner=a, mode=MODE, size=len(keep), seed=SEED,
                                 pool=p, n_indexed=len(keep), build_seconds=None, pool_id=pid)
    print("store -> %s  (%d molecules)" % (dest, len(keep)), flush=True)


if __name__ == "__main__":
    main()
