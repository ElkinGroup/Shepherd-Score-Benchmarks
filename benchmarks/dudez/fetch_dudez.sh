#!/bin/bash
# Download one DUDE-Z target: the property-matched DB2 archives, DUDE-Z's full SMILES lists and the
# crystal ligand. correct_smiles.py then writes actives.smi / decoys.smi from them.
# Exit 3 if the code is not a DUDE-Z target. Keep concurrency <= 4: docking.org throttles.
# Usage: fetch_dudez.sh TARGET [OUT_DIR]     (OUT_DIR defaults to $PWD/dudez_data/TARGET)
set -e
T="$1"
DD="${2:-$PWD/dudez_data/$T}"
site="https://dudez.docking.org"
mkdir -p "$DD"; cd "$DD"

# skip files already downloaded; write to .part so an interrupted download is not mistaken for one
get() { [ -s "$3" ] || { curl -fsSL --max-time "$1" --retry 2 --retry-delay 5 "$2" -o "$3.part" && mv "$3.part" "$3"; }; }

get 900 "$site/property_matched/$T/ligands.tgz" ligands.tgz
tar tzf ligands.tgz >/dev/null 2>&1 || { echo "[$T] not a DUDE-Z target"; rm -f ligands.tgz; exit 3; }
get 1800 "$site/property_matched/$T/decoys.tgz" decoys.tgz
get 300 "$site/DUDE-Z-benchmark-grids/$T/ligands.smi" grid_ligands.smi
get 300 "$site/DUDE-Z-benchmark-grids/$T/decoys.smi" grid_decoys.smi
get 300 "$site/DOCKING_GRIDS_AND_POSES/$T/xtal-lig.pdb" xtal-lig.pdb
echo "[$T] downloaded"
