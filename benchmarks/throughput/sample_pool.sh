#!/bin/bash
# Build the benchmark's molecule pool: N_KEEP distinct compounds drawn uniformly at random from an
# Enamine REAL Database release (cxsmiles.bz2): decompress, dedupe, shuffle, take the first N.
# Shuffle before head: REAL files are ordered by heavy-atom count, so any prefix of the unshuffled
# file is a size-biased subset.
#
#   SRC=/path/to/2026.01_Enamine_REAL_DB_13.6M.cxsmiles.bz2 bash sample_pool.sh [OUT]
#                                    # writes enamine_210k.smi next to this script by default
#
# No pipefail: head closes the pipe once it has N lines and SIGPIPEs the shuffle; the output is
# checked by counting lines instead.
set -e
: "${SRC:?Set SRC to the Enamine REAL CXSMILES file (.cxsmiles.bz2)}"
OUT=${1:-"$(dirname "$0")/enamine_210k.smi"}
# 10M conformers at 50 per compound needs 200,000 compounds; over-provision for embed failures.
N_KEEP=${N_KEEP:-210000}
SEED=${SEED:-20260813}

# Reproducible shuffle without `shuf --random-source`: tag each line with a seeded key, sort on it,
# drop the key. SMILES never contain spaces (the parser below takes the first whitespace token).
shuffle_seeded() {
  awk -v seed="$SEED" 'BEGIN{srand(seed)} {print rand(), $0}' \
    | sort -k1,1g -S 25% \
    | cut -d" " -f2-
}

# bzip2 decompression is the slow step; use a parallel decompressor when one is available.
if command -v lbzip2 >/dev/null;   then CAT="lbzip2 -dc"
elif command -v pbzip2 >/dev/null; then CAT="pbzip2 -dc"
else                                    CAT="bzcat"; fi

echo "source : $SRC"
echo "decomp : $CAT"
echo "keep   : $N_KEEP  (seed $SEED)"

# cxsmiles is tab-delimited with the structure in field 1 and a header row. Field 1 can carry
# CXSMILES extensions after a space, so take its first whitespace token; header words are dropped.
$CAT "$SRC" \
  | awk -F'\t' 'NR>1 && $1 != "" {n=split($1,a," "); t=a[1];
                if (tolower(t) != "smiles" && tolower(t) != "cxsmiles") print t}' \
  | sort -u \
  | shuffle_seeded \
  | head -n "$N_KEEP" > "$OUT"

n_out=$(wc -l < "$OUT")
echo "wrote $n_out distinct compounds to $OUT"
if [ "$n_out" -lt "$N_KEEP" ]; then
  echo "ERROR: wanted $N_KEEP compounds, got $n_out -- the source may have fewer unique SMILES" >&2
  exit 1
fi
echo
echo "Sanity-check the size range before benchmarking on it -- the previous source's ceiling was"
echo "invisible until the whole file was measured:"
echo "  python - <<'PY'"
echo "  from rdkit import Chem; import numpy as np"
echo "  h=[Chem.MolFromSmiles(l.split()[0]).GetNumHeavyAtoms() for l in open('$OUT')]"
echo "  h=np.array(h); print('heavy atoms', h.min(), h.max(), h.mean().round(1))"
echo "  PY"
