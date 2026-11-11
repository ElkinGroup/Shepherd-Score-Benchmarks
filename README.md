# ShEPhERD-score benchmarks

Code to reproduce the ShEPhERD-score paper's benchmarks and figures:
virtual-screening enrichment on 43 DUDE-Z targets, and alignment throughput
(screening and pairwise, CPU and GPU) against six external tools. The archived
results behind the published figures are in `results/published.zip`.

The published measurements used an AMD EPYC 9474F CPU and NVIDIA L40S GPUs.
Benchmarks run on Linux; figures also run on Windows. Heavy steps belong in a
batch-scheduler allocation; the scripts do not submit jobs.

## Setup

```sh
conda env create -f benchmarks/comparators/envs/fss.yml && conda activate fss
git clone https://github.com/Nate-Somers/fast_shepherd_score ../fast_shepherd_score
export FSS_REPO="$(realpath ../fast_shepherd_score)" && python -m pip install -e "$FSS_REPO"
export PYTHONPATH="$FSS_REPO:$PWD/benchmarks:$PWD/benchmarks/throughput"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export FSS_CPU_SOA=1 FSS_FUSE_SURF_ESP=1 OMP_WAIT_POLICY=passive
python benchmarks/comparators/envcheck.py numba   # CPU timings are understated without SVML
bash benchmarks/comparators/install.sh all        # external tools, pinned in versions.lock
```

`install.sh` builds into `$COMPARATORS_HOME` (default `~/scratch/comparators`).
ROSHAMBO2 needs its own environment: `conda env create -f benchmarks/comparators/envs/roshambo2.yml`,
then, inside it, `ROSHAMBO2_SOURCE=<new dir> bash benchmarks/comparators/build_roshambo2.sh`.

## Enrichment inputs (DUDE-Z)

**Source.** [DUDE-Z](https://dudez.docking.org/), downloaded 2026-09-29 for the
43 targets in `benchmarks/dudez/published_benchmark.json`:
`property_matched/<target>/{ligands,decoys}.tgz` (DB2 files) and
`DUDE-Z-benchmark-grids/<target>/{ligands,decoys}.smi`. Query ligands are the
crystal poses `DOCKING_GRIDS_AND_POSES/<target>/xtal-lig.pdb` (retrieval date not
recorded).

**What was done.**
1. Actives and decoys are the compounds in the property-matched DB2 archives.
2. The DB2 SMILES field keeps only the last 76 characters of longer SMILES, so each
   compound's complete SMILES was taken from the benchmark-grid `.smi` files and
   accepted only if its formula (with hydrogens) and net charge matched the DB2
   atom records. DRD4 decoy DB2 file names do not match their ids; its truncated
   decoys were rebuilt from the DB2 atom records, without stereochemistry.
3. Query bond orders come from the RCSB Chemical Component Dictionary or
   `benchmarks/dudez/templates.json`; hydrogens are added without moving heavy atoms.
4. Each compound gets up to 50 ETKDGv3 conformers (seed 42, 0.5 Å RMS pruning,
   up to 500 MMFF94 iterations). Frozen features: 200 surface points, xTB ESP
   (total charge 0), pharmacophores, atomic fields and Fukui descriptors
   (seed 42 for the query, 43 + i for conformer i).
5. Compounds whose features failed persistently (16 decoys) are excluded from every
   method. Result: 2,376 actives, 134,437 decoys, 3,158,803 conformers.

`fetch_dudez.sh` downloads the property-matched archives but writes the truncated
DB2 SMILES; replace them as in step 2 before `export`. This repository does not
include that correction step. Then, per target:

```sh
bash benchmarks/dudez/fetch_dudez.sh T inputs/dudez/T     # writes actives.smi / decoys.smi ("SMILES id")
curl -fL https://dudez.docking.org/DOCKING_GRIDS_AND_POSES/T/xtal-lig.pdb -o inputs/dudez/T/xtal-lig.pdb
python benchmarks/dudez/prepare_queries.py --dudez-dir inputs/dudez                 # once, all targets
python benchmarks/dudez/benchmark.py export  --target T --dudez-dir inputs/dudez --sdf-dir inputs/sdf --jobs N
python benchmarks/dudez/benchmark.py prepare --target T --sdf-dir inputs/sdf --store inputs/enrichment --jobs N --exclude-failures
python benchmarks/dudez/benchmark.py verify  --target T --store inputs/enrichment --published
```

`verify --published` compares the store with the published input hashes in
`benchmarks/dudez/published_inputs.json`.

## Throughput inputs (Enamine REAL)

Source: [Enamine REAL](https://enamine.net/compound-collections/real-compounds/real-database)
release `2026.01_Enamine_REAL_DB_13.6M.cxsmiles.bz2` (retrieval date not recorded);
a different release gives a different benchmark. Pools are random compound samples
(seed 20260813) with exactly 50 conformers per compound.

```sh
export SRC=/path/to/2026.01_Enamine_REAL_DB_13.6M.cxsmiles.bz2 D=benchmark-storage/throughput
N_KEEP=210000  SEED=20260813 bash benchmarks/throughput/sample_pool.sh $D/enamine_210k.smi
N_KEEP=2100000 SEED=20260813 bash benchmarks/throughput/sample_pool.sh $D/enamine_2100k.smi
head -n 2000 $D/enamine_210k.smi > $D/enamine_2k.smi                       # preparation workload
export FIG2_POOL_CACHE=$D/pools FIG2_SHARD_ABOVE=0 FIG2_SHARD_COMPOUNDS=2000 FSS_BENCH_CHARGES=mmff \
       FSS_BENCH_SURF_POINTS=200 FSS_BENCH_PHARM=1 FSS_CANONICAL_STORE=1 FSS_SCREEN_ARRAYS=1
python benchmarks/throughput/build_pool.py --backend numba --seed 42 --molecules $D/enamine_210k.smi   # and 2100k
python benchmarks/throughput/export_selections.py --storage $D --main-pool MAIN.shards --extended-pool EXTENDED.shards
python benchmarks/throughput/prepare_inputs.py pairwise --pool MAIN.shards --out-dir $D/pairwise
python benchmarks/throughput/build_store_parallel.py --backend triton --mode vol --size 100000000 ...  # large indexes (GPU)
python benchmarks/throughput/fukui_prep.py --phase prep ...  then  --phase build ...                   # vol_fukui index
```

`MAIN.shards`/`EXTENDED.shards` are the pool files `build_pool.py` writes. Do not
rename or touch the `.smi` files afterwards: name and mtime key the pool cache.
Indexes go to `$FIG2_STORE_ROOT` (default `benchmark-storage/throughput/stores`).

## Run

```sh
mkdir -p outputs && cp benchmarks/suite.example.json outputs/config.json   # set its paths
python benchmarks/run_evaluations.py plan --config outputs/config.json --out outputs/run/plan.json
python benchmarks/run_evaluations.py run --plan outputs/run/plan.json --dry-run
python benchmarks/run_evaluations.py run --plan outputs/run/plan.json --group prep   # then cpu, gpu, cpu64, gpu4, roshambo
python benchmarks/run_evaluations.py assemble --plan outputs/run/plan.json
```

| Group | Resources | Run after |
|---|---|---|
| `prep`, `cpu` | 1 CPU core per task | – |
| `cpu64` | 64 cores | `cpu` |
| `gpu` | 1 GPU | `cpu` |
| `gpu4` | 4 GPUs | `gpu` |
| `roshambo` | 1 GPU, ROSHAMBO2 interpreter | `prep` |

`--task ID` runs one task (for job arrays). `engine` must be a clean
fast_shepherd_score checkout; the plan records its commit and refuses changed code
or inputs, so a change needs a new plan and output directory. Each timing is one
untimed warm-up plus one timed run; never share a core or GPU with another timing.

Results go to `outputs/run/`: `receipts/` (hashes), `measurements/`, `enrichment/`
and, after `assemble`, `tables/*.csv` (per-target and summary ROC-AUC/EF1%,
charge-model comparison, preparation, index construction, throughput, parallel scaling).

## Figures

Needs numpy, scipy, matplotlib, RDKit (all in the `fss` env) and Chrome for
rendering (set `CHROME` if it is not at the default path).

```sh
python figures/make_figures.py --restore   # unpack results/published.zip, then all figures and SI tables
python figures/make_figures.py --only enrichment   # or throughput
python figures/SI/ef1/make_figure.py       # SI EF1% figure
```

`figures/enrichment/fig3.svg` (Figure 2) and `figures/throughput/fig2_alt.svg`
(Figure 3) are hand-edited layouts: the scripts move only data marks and labels,
and stop if the target/mode set changes or a value leaves the fixed axes. Keep the
`<metadata id="throughput-data-layout">` element when editing `fig2_alt.svg`.
To preview other results without touching the layouts:

```sh
python figures/enrichment/make_figure.py --data-dir DIR --prefix review
python figures/throughput/make_figure.py --data-dir DIR --prefix review
```

`DIR` must hold the complete result files that `enrichment_data.py` and
`paper_results.py` read. The 3D artwork scripts (`figures/modes/render_3d.py`,
`figures/graphical_abstract/render_overlays.py`) also need `pip install pyvista`.
Paths and host names in the archived records are replaced by placeholders such
as `<repo>` and `node-1`.
