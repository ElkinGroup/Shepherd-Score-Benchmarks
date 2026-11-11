"""Freeze rectangular pairwise inputs without timing alignments."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pair_indices(n_compounds, seed=42):
    if n_compounds < 650:
        raise ValueError('pairwise preparation needs at least 650 compounds')
    selected = np.random.default_rng(seed).permutation(n_compounds)[:650]
    return [int(i) * 50 for i in sorted(selected[:250])], [int(i) * 50 for i in sorted(selected[250:])]


def pairwise(args):
    from rdkit import Chem
    from pool import ShardedPool
    info = pickle.loads(args.pool.read_bytes())
    pool = ShardedPool([str(args.pool.parent / p) for p in info['paths']], info['counts'])
    if len(pool) % 50:
        raise ValueError('pool must contain complete 50-conformer compound blocks')
    query, library = pair_indices(len(pool) // 50)
    manifest = {'seed': 42, 'n_query': 250, 'n_library': 400, 'n_pairs': 100000,
                'disjoint_compounds': True, 'pool_manifest_sha256': digest(args.pool),
                'conformers_per_compound': 50, 'selections': {}}
    for label, indices in [('query', query), ('library', library)]:
        molecules = [pool[i] for i in indices]
        for molecule in molecules:
            # Surface arrays come from the saved pool; never resample them here.
            for field in ('surf_pos', 'surf_esp', 'pharm_types', 'pharm_ancs', 'pharm_vecs',
                          'partial_charges', 'fukui'):
                values = getattr(molecule, field)
                if values is None or not np.isfinite(values).all():
                    raise ValueError(f'{label}: missing or nonfinite {field}')
        path = args.out_dir / f'{label}.pkl'
        path.write_bytes(pickle.dumps(molecules, protocol=5))
        rdkit_path = args.out_dir / f'{label}-rdkit.pkl'
        rdkit_path.write_bytes(pickle.dumps([Chem.Mol(m.mol) for m in molecules], protocol=5))
        sdf = args.out_dir / f'{label}.sdf'
        with Chem.SDWriter(str(sdf)) as writer:
            for i, molecule in enumerate(molecules):
                rd = Chem.Mol(molecule.mol)
                rd.SetProp('_Name', f'{label}_{indices[i]}')
                writer.write(rd)
        manifest['selections'][label] = {
            'file': path.name, 'sha256': digest(path),
            'rdkit_file': rdkit_path.name, 'rdkit_sha256': digest(rdkit_path),
            'sdf_file': sdf.name, 'sdf_sha256': digest(sdf),
            'conformer_indices': indices, 'compound_indices': [i // 50 for i in indices],
            'n': len(indices)}
    (args.out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    pairs = sub.add_parser('pairwise')
    pairs.add_argument('--pool', type=Path, required=True, help='main pool .pkl.shards manifest')
    pairs.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    pairwise(args)
    print('Prepared', args.out_dir)


if __name__ == '__main__':
    main()
