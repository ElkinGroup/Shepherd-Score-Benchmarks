#!/usr/bin/env python3
"""Write each DUDE-Z target's crystal-ligand query to <target>/xtal-lig.sdf, taking bond orders from
the PDB Chemical Component Dictionary entry (xtal-lig.pdb has none).

A template is accepted only when its heavy-atom formula matches the crystal ligand: for some targets
(DRD4, MT1) the coordinate records carry a placeholder code (DRG, RES) whose dictionary entry is an
unrelated molecule. When the dictionary entry does not match, the template is taken from
templates.json beside this file (keyed by target), then from the target's actives.smi when exactly
one active carries the crystal formula; a target that still does not match is refused. queries.json
records, per target, the template's origin and whether the query carries the crystal pose.

Run: python prepare_queries.py --dudez-dir DIR
"""

import argparse
import json
import os
import urllib.request
from collections import Counter

CCD_URL = "https://files.rcsb.org/ligands/download/{code}_ideal.sdf"

#: Codes DOCK and DUDE-Z write for "the ligand", which the dictionary also assigns to real molecules.
PLACEHOLDER_CODES = {"DRG", "RES", "LIG", "UNK", "INH"}


def residue_code(pdb_path):
    """The ligand's PDB chemical-component code (columns 18-20 of the coordinate records)."""
    for line in open(pdb_path):
        if line.startswith(("ATOM", "HETATM")):
            return line[17:20].strip()
    raise ValueError(f"no coordinate records in {pdb_path}")


def fetch_template(code, cache_dir):
    """The Chemical Component Dictionary entry for ``code`` as an RDKit Mol (cached on disk)."""
    from rdkit import Chem
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{code}_ideal.sdf")
    if not os.path.exists(path):
        with urllib.request.urlopen(CCD_URL.format(code=code), timeout=60) as r:
            open(path, "wb").write(r.read())
    return Chem.MolFromMolBlock(open(path).read(), sanitize=True)


def heavy_formula(mol):
    """Element counts over heavy atoms: the identity a template has to match."""
    return Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() != "H")


def pdb_heavy_formula(pdb_path):
    """Element counts from the coordinate records, read without any chemistry perception."""
    c = Counter()
    for line in open(pdb_path):
        if line.startswith(("ATOM", "HETATM")):
            el = (line[76:78].strip() or line[12:16].strip()[0]).capitalize()
            if el != "H":
                c[el] += 1
    return c


def overrides():
    """templates.json beside this file: {"TARGET": "SMILES"}, for ligands the dictionary cannot give."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates.json")
    if not os.path.exists(path):
        return {}
    return {k: v for k, v in json.load(open(path)).items() if not k.startswith("_")}


def candidate_templates(pdb_path, code, cache_dir, target):
    """(templates, crystal formula): each candidate already matches the crystal formula, best first."""
    from rdkit import Chem
    want = pdb_heavy_formula(pdb_path)
    out = []
    if code not in PLACEHOLDER_CODES:
        try:
            t = fetch_template(code, cache_dir)
        except Exception:
            t = None
        if t is not None and heavy_formula(t) == want:
            out.append((t, "dictionary"))
    smi = overrides().get(target)
    if smi:
        t = Chem.MolFromSmiles(smi)
        if t is None or heavy_formula(t) != want:
            raise ValueError(f"{target}: the templates.json SMILES is not the crystal ligand "
                             f"(crystal formula {dict(want)})")
        out.append((t, "override"))
    actives = os.path.join(os.path.dirname(pdb_path), "actives.smi")
    if os.path.exists(actives):
        hits = []
        for line in open(actives):
            if line.strip():
                t = Chem.MolFromSmiles(line.split()[0])
                if t is not None and heavy_formula(t) == want:
                    hits.append(t)
        if len(hits) == 1:            # a unique formula match identifies the ligand; several prove nothing
            out.append((hits[0], "actives"))
    return out, want


def build_query(pdb_path, cache_dir, target):
    """(query Mol, pose, template origin); pose is "crystal" when the coordinates are the crystal's."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    code = residue_code(pdb_path)
    cands, want = candidate_templates(pdb_path, code, cache_dir, target)
    if not cands:
        raise ValueError(f"{target}: no template matches the crystal ligand (code {code}, formula "
                         f"{dict(want)}). Add its SMILES to benchmarks/dudez/templates.json.")
    pdb = Chem.MolFromPDBFile(pdb_path, removeHs=True, sanitize=False)
    for template, origin in cands:
        try:
            mol = AllChem.AssignBondOrdersFromTemplate(Chem.RemoveHs(template), pdb)
            Chem.SanitizeMol(mol)
            return mol, "crystal", origin
        except Exception:
            continue
    template, origin = cands[0]       # the right molecule, whose connectivity would not map onto the pose
    Chem.SanitizeMol(template)
    return template, "dictionary", origin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dudez-dir", required=True, help="directory of per-target DUDE-Z folders")
    ap.add_argument("--targets", nargs="*", default=None, help="only these targets (default: all)")
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    cache = os.path.join(args.dudez_dir, "_ccd_cache")
    path = os.path.join(args.dudez_dir, "queries.json")
    report = json.load(open(path)) if os.path.exists(path) else {}
    for target in sorted(os.listdir(args.dudez_dir)):
        if args.targets and target not in args.targets:
            continue
        tdir = os.path.join(args.dudez_dir, target)
        pdb = os.path.join(tdir, "xtal-lig.pdb")
        if not os.path.isfile(pdb):
            continue
        mol, pose, origin = build_query(pdb, cache, target)
        mol.SetProp("_Name", f"{target}_query")
        w = Chem.SDWriter(os.path.join(tdir, "xtal-lig.sdf"))
        w.write(mol)
        w.close()
        report[target] = {"code": residue_code(pdb), "source": pose, "template": origin,
                          "smiles": Chem.MolToSmiles(Chem.RemoveHs(mol))}
        print(f"  {target:8} {report[target]['code']:>4}  {pose:10} template={origin}", flush=True)

    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    n_xtal = sum(1 for v in report.values() if v["source"] == "crystal")
    n_dict = sum(1 for v in report.values() if v.get("template", "dictionary") == "dictionary")
    print(f"\nwrote {len(report)} queries: {n_xtal} crystal pose, {len(report) - n_xtal} idealized; "
          f"templates: {n_dict} dictionary, {len(report) - n_dict} actives or override")


if __name__ == "__main__":
    main()
