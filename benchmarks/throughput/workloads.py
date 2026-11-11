"""Index sets for the two workloads: screen (1 query vs N library conformers) and pairwise (all-vs-all over distinct compounds).

The pool is a conformer-ensemble library: pool index ``i`` belongs to compound ``i // N_CONFORMERS``, and sizes count conformers.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Shared with the accuracy benchmark (benchmarks/dudez/protocol.py); not overridable.
try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from dudez.protocol import N_CONFORMERS as _PROTOCOL_N_CONF
except Exception:                                   # standalone checkout without benchmarks/dudez
    _PROTOCOL_N_CONF = 50
N_CONFORMERS = int(_PROTOCOL_N_CONF)


def pairwise_pairs(pool_size: int, n: int, seed: int, n_conf: int = None) -> list[tuple[int, int]]:
    """Prefix of the frozen 250-query by 400-library compound selection."""
    from prepare_inputs import pair_indices
    k = n_conf or N_CONFORMERS
    if not 1 <= n <= 100000:
        raise ValueError('pairwise size must be between 1 and 100000')
    q, library = pair_indices(pool_size // k, seed)
    q = [i // 50 * k for i in q]
    library = [i // 50 * k for i in library]
    return [(i, j) for i in q for j in library][:n]


def screen_library(pool_size: int, n: int, seed: int, n_conf: int = None) -> list[int]:
    """N library conformer indices drawn as whole compound ensembles without replacement, truncated to exactly n."""
    k = n_conf or N_CONFORMERS
    if n > pool_size:
        raise ValueError(f"screen n={n} exceeds the pool of {pool_size} conformers")
    n_cmpd = pool_size // k
    need = -(-n // k)                                # ceil
    if need > n_cmpd:
        raise ValueError(f"screen n={n} needs {need} compounds; pool has {n_cmpd}")
    rng = np.random.default_rng(seed)
    # Ascending order so a sharded pool is read once; the set is still a seeded random sample.
    chosen = np.sort(rng.choice(n_cmpd, size=need, replace=False))
    lib = [int(c) * k + j for c in chosen for j in range(k)]
    return lib[:n]


QUERY_SAMPLE = 20_000


def _prefix_atom_counts(pool, sample: int = None):
    """Atom counts of the first ``QUERY_SAMPLE`` molecules (the SMILES file is shuffled, so a prefix is an unbiased sample)."""
    n = min(len(pool), sample or QUERY_SAMPLE)
    return np.array([pool[i].atom_pos.shape[0] for i in range(n)])


def median_heavy(pool, sample: int = None) -> int:
    """Median heavy-atom count of the pool (the atom-basis disclosure)."""
    return int(np.median(_prefix_atom_counts(pool, sample)))


def median_query_index(pool, sample: int = None) -> int:
    """Pool index of the median heavy-atom-count molecule, used as the single query."""
    counts = _prefix_atom_counts(pool, sample)
    return int(np.argsort(counts)[len(counts) // 2])
