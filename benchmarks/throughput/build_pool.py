#!/usr/bin/env python3
"""Build (or resume) one sharded conformer pool and exit, without running a panel.

``pool._build_sharded`` work-steals: each shard is claimed with an O_EXCL lock, built, written by
atomic rename with a count sidecar, and released, so any number of jobs can build one pool at once.

    python build_pool.py --molecules enamine_2100k.smi          # build or resume
    FIG2_SHARD_COMPOUNDS=2000 python build_pool.py -m ...       # shard granularity

Must run as a file, not ``python - <<EOF``: ``FSSAligner.build`` fans out over a spawn pool, which
re-imports ``__main__`` in every worker. FIG2_BUILD_ONLY=1 is set here so a task that starts once
the manifest exists finishes the partial pool instead of loading it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

#: Surface points per conformer; part of the pool cache key, so it must match the measurement runs.
SURF_POINTS = int(os.environ.get("FSS_BENCH_SURF_POINTS", "200"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-m", "--molecules", required=True,
                    help="the .smi file, relative to this directory or absolute")
    ap.add_argument("--tool", default="fss", help="aligner whose build() featurises (default: fss)")
    ap.add_argument("--backend", default="numba",
                    help="backend for the aligner; irrelevant to what build() produces")
    ap.add_argument("--seed", type=int, default=42, help="pool seed, part of the cache key")
    args = ap.parse_args()

    mols = args.molecules if os.path.isabs(args.molecules) else os.path.join(HERE, args.molecules)
    if not os.path.exists(mols):
        sys.exit(f"no such molecule file: {mols}")

    # Build, never load: a late task must finish the pool, not return the partial one.
    os.environ["FIG2_BUILD_ONLY"] = "1"

    import pool as poolmod
    from aligners import get_aligner

    aligner = get_aligner(args.tool, backend=args.backend)
    print(f"pool build: {os.path.basename(mols)} tool={args.tool} seed={args.seed} "
          f"surf_points={SURF_POINTS} shard={poolmod.SHARD_COMPOUNDS} compounds", flush=True)
    print(f"cache identity: {poolmod.cache_identity(aligner, mols, args.seed, SURF_POINTS)}",
          flush=True)

    t0 = time.perf_counter()
    built, n_fail, secs = poolmod.load_or_build(aligner, mols, seed=args.seed,
                                                surf_points=SURF_POINTS)
    print(f"pool now holds {len(built):,} conformers; this task spent {time.perf_counter() - t0:.1f}s"
          f"{f' ({n_fail} embed failures here)' if n_fail else ''}", flush=True)
    return 0


if __name__ == "__main__":                       # required: spawn re-imports this module
    raise SystemExit(main())
