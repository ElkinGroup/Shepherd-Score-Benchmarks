"""Comparator drivers: the argv builder, output parser and run helper for every external tool.
Both figures build their tool command lines here; each owns its own batching and timing."""
from __future__ import annotations

import csv
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import paths

# fss mode -> the tools that cover it. vol_color has no phesa entry: PheSA's colour is directional,
# fss/ROSHAMBO2's is directionless.
MODES: dict[str, tuple[str, ...]] = {
    "roshambo2": ("vol", "vol_color"),
    "shaep":     ("vol", "vol_esp", "vol_and_surf_esp"),
    "align_it":  ("pharm",),
    "phesa":     ("vol", "pharm"),
}

# fss mode -> PheSA ppWeight
_PHESA_PPWEIGHT = {"vol": "0.0", "pharm": "1.0"}

# ShaEP needs "y" on stdin (it prompts before overwriting an existing output file), a memory cap and
# library chunking (its per-thread field grids are held in RAM).
SHAEP_TIMEOUT = int(os.environ.get("SHAEP_TIMEOUT", "600"))
SHAEP_MEM_CAP_GB = int(os.environ.get("SHAEP_MEM_CAP_GB", "12"))
SHAEP_STDIN = "y\ny\ny\ny\n"


# align-it: never pass --scoreOnly (disables alignment: scores the raw input coordinates), --rankBy
# (silently drops the HYBH/HYBL feature types on the pinned fork) or --best (truncates the output).
def align_it_argv(*, reference, dbase, scores) -> list[str]:
    return [str(paths.align_it_bin()), "--reference", str(reference),
            "--dbase", str(dbase), "--scores", str(scores)]


def _phesa_common(mode: str, scores, threads: int | None) -> list[str]:
    # ppWeight is always passed explicitly; PheSAScreen applies it via getPhesaSetting().setPpWeight()
    return ["--scores", str(scores), "--threads", str(threads or 1),
            "--ppweight", _PHESA_PPWEIGHT[mode]]


def phesa_screen_argv(*, mode, reference, dbase, scores, threads=None) -> list[str]:
    """PheSA screen: one reference vs the dbase. Output is 2 columns: '<dbName>\\t<sim>'."""
    return [paths.java_bin(), "-cp", paths.phesa_classpath(), "PheSAScreen",
            "--reference", str(reference), "--dbase", str(dbase),
            *_phesa_common(mode, scores, threads)]


def phesa_pairs_argv(*, mode, queries, dbase, pairs, scores, threads=None) -> list[str]:
    """PheSA pairwise: one JVM launch per cell. Output is 3 columns: '<qIndex>\\t<dIndex>\\t<sim>'."""
    return [paths.java_bin(), "-cp", paths.phesa_classpath(), "PheSAScreen",
            "--queries", str(queries), "--dbase", str(dbase), "--pairs", str(pairs),
            *_phesa_common(mode, scores, threads)]


def shaep_argv(*, queries, input_file, output_file, threads=1) -> list[str]:
    """ShaEP; ``queries`` is a list of query SDFs (len 1 = screen). --noconfs scores each record as
    its own molecule. One run yields both the shape and the ESP column (see parse_shaep). With SDF
    input ShaEP computes its own charges, so the enrichment ESP series is not charge-matched to fss."""
    qflags = []
    for q in queries:
        qflags += ["--query", str(q)]
    return [str(paths.shaep_bin()), "-j", str(threads), "--noconfs",
            *qflags, "--input-file", str(input_file), "--output-file", str(output_file)]


# ROSHAMBO2 runs at its own default start_mode (1 = 4 start orientations); override with ROSH_START_MODE.
ROSH_START_MODE = int(os.environ.get("ROSH_START_MODE",
                                     os.environ.get("FIG2_ROSH_START_MODE", "1")))


def roshambo2_kwargs(mode: str) -> dict:
    """compute() kwargs: ROSHAMBO2's own defaults plus optim_mode, which is required for the color
    column (without it tanimoto_combination comes back equal to tanimoto_shape)."""
    color = (mode == "vol_color")
    kw = dict(backend="cuda", start_mode=ROSH_START_MODE, n_gpus=1, write_scores=False,
              color_scores=color,
              optimizer_settings={"lr_q": 0.1, "lr_t": 0.1, "steps": 100})
    if color:
        kw.update(optim_mode="combination", combination_param=0.5)
    return kw


def roshambo2_column(mode: str) -> str:
    return "tanimoto_combination" if mode == "vol_color" else "tanimoto_shape"


# ---- parsers, keyed by molecule name ----------------------------------------------------------
def parse_align_it(path) -> dict:
    """align-it scores file: no header, tab-separated, 11 fields; field[2] = name, field[-3] = Tanimoto."""
    out = {}
    with open(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 11:
                try:
                    out[p[2]] = float(p[-3])
                except ValueError:
                    continue
    return out


def parse_phesa_screen(path) -> dict:
    """PheSAScreen screen output: '<dbName>\\t<similarity>' per library record."""
    out = {}
    with open(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                try:
                    out[p[0]] = float(p[1])
                except ValueError:
                    continue
    return out


def parse_shaep(path, column: str, query_names=None) -> dict:
    """ShaEP TSV by exact column name. Single query: shape_similarity / best_similarity (ShaEP's own
    0.5*(shape+ESP) blend) -> {name: score}. Multi query (pass ``query_names``): <query>_shape /
    <query>_ESP -> {query: {name: score}}."""
    with open(path, newline="") as fh:
        rdr = csv.reader(fh, delimiter="\t")
        header = next(rdr)
        rows = [r for r in rdr if r]
    idx = {h: i for i, h in enumerate(header)}
    name_i = idx.get("molecule", 0)

    def _pick(rows, si, ei):
        out = {}
        for p in rows:
            if len(p) <= max(si, ei if ei is not None else si):
                continue
            try:
                out[p[name_i]] = float(p[si]) if column == "shape" else float(p[ei])
            except (ValueError, TypeError):
                continue
        return out

    if query_names:
        per_q = {}
        for qn in query_names:
            s_h, e_h = f"{qn}_shape", f"{qn}_ESP"
            if s_h not in idx:
                continue
            if column == "shape":
                per_q[qn] = _pick(rows, idx[s_h], None)
            else:                                     # combined = 0.5*(shape+ESP), ShaEP's espweight
                d = {}
                for p in rows:
                    if len(p) > max(idx[s_h], idx[e_h]):
                        try:
                            d[p[name_i]] = 0.5 * (float(p[idx[s_h]]) + float(p[idx[e_h]]))
                        except ValueError:
                            continue
                per_q[qn] = d
        if not per_q:
            raise RuntimeError(f"ShaEP {path}: no per-query columns for {query_names[:3]}; "
                               f"header={header}")
        return per_q

    want = "shape_similarity" if column == "shape" else "best_similarity"
    if want not in idx:
        raise RuntimeError(f"ShaEP {path}: no {want!r} column; header={header}")
    ci = idx[want]
    out = {}
    for p in rows:
        if len(p) > ci:
            try:
                out[p[name_i]] = float(p[ci])
            except ValueError:
                continue
    return out


# ---- running ---------------------------------------------------------------------------------
@dataclass
class Spec:
    """A ready-to-exec subprocess; build it untimed, exec it with :func:`run`."""
    tool: str
    mode: str
    argv: list[str]
    env: dict
    out_path: Path
    stdin: str | None = None
    timeout: int | None = None
    mem_cap_gb: int | None = None


def supported(tool: str, mode: str) -> bool:
    return mode in MODES.get(tool, ())


def screen_spec(tool: str, mode: str, query_sdf, lib_sdf, out_path, threads: int | None = None) -> Spec:
    """Spec for one query vs a library (align_it, phesa)."""
    if not supported(tool, mode):
        raise ValueError(f"{tool} does not support mode {mode!r} (supports {MODES.get(tool, ())})")
    env, out_path = paths.tool_env(tool), Path(out_path)
    if tool == "align_it":
        return Spec(tool, mode, align_it_argv(reference=query_sdf, dbase=lib_sdf, scores=out_path),
                    env, out_path)
    if tool == "phesa":
        return Spec(tool, mode, phesa_screen_argv(mode=mode, reference=query_sdf, dbase=lib_sdf,
                                                  scores=out_path, threads=threads), env, out_path)
    # shaep Specs are built by the callers (chunked library, both columns from one run)
    raise ValueError(f"{tool!r} has no generic screen spec "
                     f"(roshambo2 is in-process; shaep is hand-driven)")


def parse_screen(sp: Spec) -> dict:
    """{db_molecule_name: score} from a screen Spec's output."""
    if sp.tool == "align_it":
        return parse_align_it(sp.out_path)
    if sp.tool == "phesa":
        return parse_phesa_screen(sp.out_path)
    raise ValueError(f"no screen parser for {sp.tool!r}")


def _preexec(mem_cap_gb):
    def _limit():
        n = mem_cap_gb * 1024 ** 3
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (n, n))
    return _limit


def run(sp: Spec) -> None:
    """Exec a Spec; a non-zero exit raises. Kept minimal because fig2 times this call."""
    kw = {}
    if sp.mem_cap_gb and os.name == "posix":
        kw["preexec_fn"] = _preexec(sp.mem_cap_gb)
    r = subprocess.run(sp.argv, env=sp.env, input=sp.stdin, timeout=sp.timeout,
                       capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{sp.tool} exited {r.returncode}\n"
                           f"  cmd: {' '.join(map(str, sp.argv))}\n  stderr: {r.stderr[-2000:]}")


def score(tool: str, mode: str, query_sdf, lib_sdf, expected: list[str], out_path,
          threads: int | None = None):
    """Score one query vs a library; returns (scores in ``expected`` order, missing names).
    Molecules the tool dropped are scored 0.0 and listed in ``missing``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if tool == "roshambo2":
        from roshambo2 import Roshambo2                     # only inside the roshambo2 env
        kwargs = dict(roshambo2_kwargs(mode), reduce_over_conformers=False,
                      keep_order=True, max_results=len(expected))
        res = Roshambo2(str(query_sdf), str(lib_sdf),
                        color=(mode == "vol_color"),
                        conformers_have_unique_names=True,
                        n_cpus_prepare=int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))).compute(**kwargs)
        col = roshambo2_column(mode)
        by_name = {}
        for _, df in res.items():
            for nm, v in zip(df["name"].values, df[col].values):
                by_name[str(nm)] = float(v)
    else:
        sp = screen_spec(tool, mode, query_sdf, lib_sdf, out_path, threads=threads)
        run(sp)
        by_name = parse_screen(sp)

    missing = [n for n in expected if n not in by_name]
    if tool == 'roshambo2' and (missing or set(by_name) != set(expected)):
        raise ValueError(f'ROSHAMBO2 returned a different conformer set: {len(missing)} missing')
    if missing:
        print(f"  WARNING {tool}/{mode} dropped {len(missing)}/{len(expected)} molecules "
              f"(scored 0.0): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return np.array([by_name.get(n, 0.0) for n in expected], dtype=float), missing


def _split_sdf(sdf_path, chunk_size, out_dir):
    """Split a multi-record SDF into chunk files under out_dir/_chunks (used by benchmarks/throughput/aligners/shaep.py)."""
    out_dir = Path(out_dir) / "_chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts, buf, n, idx = [], [], 0, 0
    with open(sdf_path) as fh:
        for line in fh:
            buf.append(line)
            if line.rstrip() == "$$$$":
                n += 1
                if n >= chunk_size:
                    p = out_dir / f"chunk_{idx}.sdf"
                    p.write_text("".join(buf)); parts.append(p)
                    buf, n, idx = [], 0, idx + 1
    if buf:
        p = out_dir / f"chunk_{idx}.sdf"
        p.write_text("".join(buf)); parts.append(p)
    return parts


def xtb_charges(rd):
    """gfn2-xTB partial charges for one RDKit conformer, or None if xTB fails (the caller drops it)."""
    from shepherd_score.conformer_generation import charges_from_single_point_conformer_with_xtb
    try:
        return charges_from_single_point_conformer_with_xtb(rd)
    except Exception as e:
        smi = ""
        try:
            from rdkit import Chem
            smi = Chem.MolToSmiles(Chem.RemoveHs(rd))
        except Exception:
            pass
        print(f"  WARNING xTB failed on a molecule ({smi or '?'}): {str(e)[:80]} -- DROPPED")
        return None
