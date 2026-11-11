#!/bin/bash
# Fetch one DUDE-Z property_matched target and write actives.smi / decoys.smi (one SMILES per
# molecule). Exit 3 if the code is not a DUDE-Z target. Keep concurrency <= 4: docking.org throttles.
# Usage: dudez_fetch.sh TARGET [OUT_DIR]     (OUT_DIR defaults to $PWD/dudez_data/TARGET)
set -e
T="$1"
DD="${2:-$PWD/dudez_data/$T}"
base="https://dudez.docking.org/property_matched/$T"
mkdir -p "$DD"; cd "$DD"
[ -s actives.smi ] && [ -s decoys.smi ] && { echo "[$T] SMILES already present"; exit 0; }

get() { curl -sL --max-time "$1" --retry 2 --retry-delay 5 "$2" -o "$3"; }

echo "[$T] fetching ligands.tgz ..."
get 900 "$base/ligands.tgz" ligands.tgz
tar tzf ligands.tgz >/dev/null 2>&1 || { echo "[$T] not a DUDE-Z target"; rm -f ligands.tgz; exit 3; }
echo "[$T] fetching decoys.tgz ..."
get 1800 "$base/decoys.tgz" decoys.tgz

# one SMILES per unique molecule id (collapse the .1/.2/... protomer files; line 3 of each db2.gz is "M <SMILES>")
extract() {
  local dir="$1" out="$2"; : > "$out"; declare -A seen
  while IFS= read -r f; do
    local id; id=$(basename "$f"); id=${id%%.*}
    [ -n "${seen[$id]:-}" ] && continue; seen[$id]=1
    local smi; smi=$(zcat "$f" 2>/dev/null | sed -n '3s/^M[[:space:]]*//p' | tr -d '[:space:]')
    [ -n "$smi" ] && printf '%s %s\n' "$smi" "$id" >> "$out"
  done < <(find "$dir" -name '*.db2.gz')
}
rm -rf lig dec; mkdir lig dec
tar xzf ligands.tgz -C lig
tar xzf decoys.tgz  -C dec
extract lig actives.smi
extract dec decoys.smi
echo "[$T] actives=$(wc -l < actives.smi) decoys=$(wc -l < decoys.smi)"
rm -rf lig dec ligands.tgz decoys.tgz
