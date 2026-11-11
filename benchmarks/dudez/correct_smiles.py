#!/usr/bin/env python3
"""Write complete SMILES for DUDE-Z targets from the DB2 archives and DUDE-Z's own SMILES lists.

Line 3 of every DUDE-Z db2.gz file holds the SMILES in a 76-character field that keeps the END
of longer SMILES, so reading only that field truncates large molecules. Per target and set, the
first DB2 file of each identifier (archive order) gives one compound, whose SMILES is:

  1. the field, if it is shorter than 76 characters or parses to the DB2 atom records' elements;
  2. else a DUDE-Z-benchmark-grids SMILES for that identifier whose formula (hydrogens included)
     and net charge equal the DB2 atom records, preferring one that ends with the field;
  3. else the molecule rebuilt from the DB2 atom and bond records (no stereochemistry);
  4. else none: the compound is left out.

Writes DIR/<T>/actives.smi and decoys.smi ("SMILES id", for benchmark.py export) and
corrections.tsv (the source of every SMILES).

usage: correct_smiles.py DIR TARGET [TARGET ...]
DIR/<T>/ holds ligands.tgz, decoys.tgz, grid_ligands.smi and grid_decoys.smi (fetch_dudez.sh).
"""
import collections
import gzip
import re
import sys
import tarfile
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

RDLogger.DisableLog('rdApp.*')
FIELD = 76
SETS = (('ligands', 'active', 'actives.smi'), ('decoys', 'decoy', 'decoys.smi'))


def elements(formula):
    counts = collections.Counter()
    for element, n in re.findall(r'([A-Z][a-z]?)(\d*)', re.sub(r'[+-]\d*$', '', formula)):
        counts[element] += int(n or 1)
    return counts


def describe(smiles):
    """(element counts with hydrogens, net charge) of a SMILES, or None if it does not parse."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    heavy = Chem.RemoveHs(mol)
    return elements(CalcMolFormula(Chem.AddHs(heavy))), Chem.GetFormalCharge(heavy)


def read_db2(stream):
    """M lines, atoms (sybyl type, charge) and bonds up to the first coordinate record."""
    mlines, atoms, bonds = [], [], []
    for raw in stream:
        line = raw.decode('ascii', 'replace').rstrip('\n')
        if line[:1] == 'M':
            mlines.append(line)
        elif line[:1] == 'A':
            f = line.split()
            atoms.append((int(f[1]), f[2], f[3], float(f[6])))
        elif line[:1] == 'B':
            f = line.split()
            bonds.append((int(f[2]), int(f[3]), f[4]))
        elif line[:1] in ('X', 'R', 'C', 'S', 'D'):
            break
    return mlines, atoms, bonds


def element(sybyl):
    e = sybyl.split('.')[0]
    return e if e in ('Cl', 'Br') else e[:1].upper() + e[1:].lower()


def rebuild(name, atoms, bonds):
    """Stereo-free SMILES of the molecule written by the DB2 atom and bond records, or ''."""
    block = ['@<TRIPOS>MOLECULE', name, f'{len(atoms)} {len(bonds)} 0 0 0', 'SMALL',
             'USER_CHARGES', '', '@<TRIPOS>ATOM']
    for i, (num, aname, sybyl, charge) in enumerate(atoms, 1):
        if num != i:
            return ''
        sybyl = sybyl.replace('.o2', '.O2').replace('.o', '.O')   # DB2 writes S.o2; Mol2 wants S.O2
        block.append(f'{i} {aname} 0.0000 0.0000 0.0000 {sybyl} 1 LIG {charge:.4f}')
    block.append('@<TRIPOS>BOND')
    block += [f'{i} {a} {b} {order}' for i, (a, b, order) in enumerate(bonds, 1)]
    try:
        mol = Chem.MolFromMol2Block('\n'.join(block) + '\n', sanitize=True, removeHs=False,
                                    cleanupSubstructures=True)
    except Exception:                                          # noqa: BLE001
        return ''
    if mol is None:
        return ''
    heavy = Chem.RemoveHs(mol)
    Chem.RemoveStereochemistry(heavy)
    return Chem.MolToSmiles(heavy)


def read_smi(path):
    listed = collections.defaultdict(list)
    for line in open(path):
        parts = line.split()
        if parts:
            listed[parts[1] if len(parts) > 1 else ''].append(parts[0])
    return listed


def correct(folder):
    rows = []
    for archive, label, out_name in SETS:
        listed = read_smi(folder / f'grid_{archive}.smi')
        seen, lines = set(), []
        with tarfile.open(folder / f'{archive}.tgz') as tar:
            for member in tar:
                if not (member.isfile() and member.name.endswith('.db2.gz')):
                    continue
                ident = member.name.split('/')[-1].split('.')[0]
                if ident in seen:
                    continue
                seen.add(ident)
                with gzip.open(tar.extractfile(member)) as stream:
                    mlines, atoms, bonds = read_db2(stream)
                line3 = mlines[2] if len(mlines) > 2 else ''
                field_smiles = ''.join(line3[1:].split()) if line3.startswith('M') else ''
                if not field_smiles:
                    continue                                   # no SMILES: never part of the set
                line2 = mlines[1].split() if len(mlines) > 1 else []
                charge = int(round(float(line2[1]))) if len(line2) > 1 else None   # None never matches
                db2 = (collections.Counter(element(a[2]) for a in atoms), charge)

                def matches(smiles):
                    d = describe(smiles)
                    return d is not None and d == db2

                parsed = describe(field_smiles)
                if len(line3[2:2 + FIELD].rstrip()) < FIELD or (parsed and parsed[0] == db2[0]):
                    smiles, source = field_smiles, 'db2_smiles_field'
                else:
                    listed_ok = [s for s in listed.get(ident, []) if matches(s)]
                    ending = [s for s in listed_ok if s.endswith(field_smiles)]
                    rebuilt = rebuild(ident, atoms, bonds)
                    if ending:
                        smiles, source = ending[0], 'dudez_smiles_list:suffix+formula+charge'
                    elif listed_ok:
                        smiles, source = listed_ok[0], 'dudez_smiles_list:formula+charge'
                    elif rebuilt and matches(rebuilt):
                        smiles, source = rebuilt, 'db2_atom_bond_records'
                    else:
                        smiles, source = '', 'unresolved'
                rows.append((label, ident, source, smiles))
                if smiles:
                    lines.append(f'{smiles} {ident}\n')
        (folder / out_name).write_text(''.join(lines))
    with open(folder / 'corrections.tsv', 'w', newline='\n') as f:
        f.write('label\tid\tsource\tsmiles\n')
        f.writelines('\t'.join(r) + '\n' for r in rows)
    return collections.Counter(r[2] for r in rows)


def main():
    root = Path(sys.argv[1])
    for target in sys.argv[2:]:
        print(target, dict(correct(root / target)), flush=True)


if __name__ == '__main__':
    main()
