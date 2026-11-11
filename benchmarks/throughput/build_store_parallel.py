#!/usr/bin/env python3
"""Build one screening store in parallel parts, then merge them into a single store.

Each part is built by the adapter's own ``build_store`` into its own directory (a store directory
is single-writer); the merge renumbers each part's ``ids`` by its global offset and writes the
sidecar, so the merged result screens as one ordinary store.

    python build_store_parallel.py --mode pharm --size 100000000 --molecules enamine_2100k.smi \
        --parts 8 --part 0 --work /path/to/parts                       # part K of N
    python build_store_parallel.py --mode pharm --size 100000000 --molecules enamine_2100k.smi \
        --parts 8 --merge --work /path/to/parts --store-root /path/to/stores
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SURF_POINTS = int(os.environ.get("FSS_BENCH_SURF_POINTS", "200"))


def library(pool_len: int, size: int, seed: int):
    import workloads
    return workloads.screen_library(pool_len, size, seed)


def build_part(args) -> None:
    import pool as poolmod
    from aligners import get_aligner

    aligner = get_aligner("fss", backend=args.backend)
    pool, _, _ = poolmod.load_or_build(aligner, args.molecules, seed=args.seed,
                                       surf_points=SURF_POINTS)
    lib = library(len(pool), args.size, args.seed)
    n = len(lib)
    lo = (n * args.part) // args.parts
    hi = (n * (args.part + 1)) // args.parts
    out = os.path.join(args.work, f"part{args.part:03d}")
    print(f"part {args.part}/{args.parts}: molecules [{lo:,}, {hi:,}) of {n:,} -> {out}", flush=True)

    t0 = time.perf_counter()
    aligner.build_store(pool, lib[lo:hi], out, modes=[args.mode])
    secs = time.perf_counter() - t0
    # Record the global offset so the merge can renumber ids without recomputing the library.
    with open(os.path.join(out, "part.json"), "w") as fh:
        json.dump({"part": args.part, "parts": args.parts, "lo": lo, "hi": hi,
                   "n": hi - lo, "build_seconds": secs}, fh, indent=1)
    print(f"part {args.part} done: {hi - lo:,} molecules in {secs:.1f}s", flush=True)


def merge(args) -> None:
    import numpy as np
    import pool as poolmod
    import sweep as sweepmod
    from aligners import get_aligner

    aligner = get_aligner("fss", backend=args.backend)
    dest = sweepmod.store_path(args.store_root or sweepmod.default_store_root(), aligner,
                               mode=args.mode, size=args.size,
                               seed=args.seed)
    parts = [os.path.join(args.work, f"part{k:03d}") for k in range(args.parts)]
    for p in parts:
        if not os.path.exists(os.path.join(p, "manifest.json")):
            sys.exit(f"missing part: {p}")

    os.makedirs(dest, exist_ok=True)
    base = None
    shards, n_total, sid = [], 0, 0
    for p in parts:
        man = json.load(open(os.path.join(p, "manifest.json")))
        meta = json.load(open(os.path.join(p, "part.json")))
        if base is None:
            base = {k: v for k, v in man.items() if k not in ("shards", "n_total")}
        else:                                   # every part must agree on how it was written
            for k, v in base.items():
                if man.get(k) != v:
                    sys.exit(f"part {p} disagrees on {k}: {man.get(k)!r} vs {v!r}")
        if meta["lo"] != n_total:
            sys.exit(f"part {p} starts at {meta['lo']} but {n_total} molecules precede it")
        for sh in man["shards"]:
            new = f"shard_{sid:05d}"
            for key in sh["keys"]:
                src = os.path.join(p, f"{sh['name']}__{key}.npy")
                dst = os.path.join(dest, f"{new}__{key}.npy")
                if key == "ids":                # global position, not position within the part
                    a = np.load(src)
                    np.save(dst, a + meta["lo"])
                    os.remove(src)
                else:
                    os.replace(src, dst)        # same filesystem: a rename, not a copy
            shards.append({"name": new, "n": sh["n"], "start": n_total, "keys": sh["keys"]})
            n_total += sh["n"]
            sid += 1
    man = dict(base, shards=shards, n_total=n_total)
    with open(os.path.join(dest, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)

    # Prove the merged store opens and holds what we think before advertising it for reuse.
    from shepherd_score.screen import ProfileStore
    got = len(ProfileStore.open(dest))
    if got != args.size:
        sys.exit(f"merged store holds {got:,} molecules, expected {args.size:,}")

    pool, _, _ = poolmod.load_or_build(aligner, args.molecules, seed=args.seed,
                                       surf_points=SURF_POINTS)
    pid = poolmod.cache_identity(aligner, args.molecules, args.seed, SURF_POINTS)
    build_s = sum(json.load(open(os.path.join(p, "part.json")))["build_seconds"] for p in parts)
    # The sidecar must describe the merged store, not this process: FSS_CANONICAL_STORE may differ
    # between the part builds and the merge, so the canonical flag is taken from the manifest
    # ProfileStore.create wrote.
    sweepmod.write_store_sidecar(dest, aligner=aligner, mode=args.mode, size=args.size,
                                 seed=args.seed, pool=pool, n_indexed=got,
                                 build_seconds=build_s, pool_id=pid,
                                 canonical=man.get("canonical"))
    ok, why = sweepmod.store_is_reusable(dest, aligner=aligner, mode=args.mode, size=args.size,
                                         seed=args.seed, pool=pool, pool_id=pid)
    print(f"merged {len(parts)} parts -> {dest}")
    print(f"  {got:,} molecules, {len(shards)} shards, {build_s / 3600:.2f} core-hours of build")
    print(f"  reusable by the harness: {ok} ({why})")
    if not ok:
        sys.exit(1)
    for p in parts:
        shutil.rmtree(p, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--molecules", required=True)
    ap.add_argument("--parts", type=int, required=True)
    ap.add_argument("--part", type=int, default=None, help="build this part (omit with --merge)")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--work", required=True, help="directory holding the part stores")
    ap.add_argument("--store-root", dest="store_root",
                    help="default: $FIG2_STORE_ROOT, else benchmark-storage/throughput/stores")
    ap.add_argument("--backend", default="triton")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if not os.path.isabs(a.molecules):
        a.molecules = os.path.join(HERE, a.molecules)
    os.makedirs(a.work, exist_ok=True)
    if a.merge:
        merge(a)
    elif a.part is not None:
        build_part(a)
    else:
        ap.error("pass --part K or --merge")
    return 0


if __name__ == "__main__":                       # spawn-safe: the pool build fans out
    raise SystemExit(main())
