"""One untimed warm-up followed by one timed execution."""
import math
import time


def measure(job, aligner):
    aligner.reset()
    aligner.sync()
    job.run()
    aligner.sync()
    reset = getattr(aligner, "effort_reset", None)
    if reset is not None:
        reset()
    aligner.sync()
    start = time.perf_counter()
    job.run()
    aligner.sync()
    seconds = time.perf_counter() - start
    score = float(job.score())
    if not math.isfinite(score) or seconds <= 0 or job.n <= 0:
        raise ValueError("invalid measured workload")
    row = dict(n=job.n, aligns_per_sec=job.n / seconds, compute_seconds=seconds,
               warmups=1, reps=1, rep_seconds=[seconds], overlap_score=score, valid=True)
    summary = getattr(aligner, "effort_summary", None)
    if summary is not None:
        row["steps_measured"] = summary()
    return row


def measure_build(build_one, aligner, *, n, probe=None, discard=None):
    aligner.reset()
    aligner.sync()
    warm = build_one()
    aligner.sync()
    if discard is not None:
        discard(warm)
    start = time.perf_counter()
    handle = build_one()
    aligner.sync()
    seconds = time.perf_counter() - start
    info = probe(handle) if probe else {}
    count = int(info.get("n_indexed", n))
    if count != n or seconds <= 0:
        raise ValueError(f"index contains {count} entries; expected {n}")
    size = int(info.get("bytes", 0))
    return dict(build_seconds=seconds, build_rep_seconds=[seconds], build_reps=1,
                warmups=1, build_valid=True, build_n_indexed=count, build_index_frac=1.0,
                store_bytes=size, store_bytes_per_mol=size / n,
                mols_indexed_per_sec=n / seconds, us_per_mol_indexed=seconds / n * 1e6), handle
