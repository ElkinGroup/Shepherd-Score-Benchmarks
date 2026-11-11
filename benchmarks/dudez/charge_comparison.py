"""MMFF charge overlays on frozen coordinates and surface samples."""
import json
import pickle
from pathlib import Path
import numpy as np
from suite import digest, write

MODES = ('vol_esp', 'surf_esp', 'vol_and_surf_esp', 'vol_esp_tversky',
         'surf_esp_tversky', 'vol_and_surf_esp_tversky')


def replace_charges(arrays, charges):
    from shepherd_score.generate_point_cloud import get_electrostatics_given_point_charges
    charges = np.asarray(charges, dtype=np.float32)
    centers = arrays['centers_w_H']
    if charges.shape != (len(centers),) or not np.isfinite(charges).all():
        raise ValueError('invalid MMFF charges')
    esp = get_electrostatics_given_point_charges(charges, centers, arrays['surf_pos'])
    if not np.isfinite(esp).all():
        raise ValueError('nonfinite MMFF surface ESP')
    return dict(partial_charges=charges, surf_esp=esp)


def prepare(store, target, folder):
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem
    from dudez import benchmark as B
    data, meta, manifest = B.load(store, target)
    if manifest.get('charge_model') != 'xtb':
        raise ValueError('charge comparison requires the frozen xTB baseline')
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    payload = folder / f'{target}.pkl'; side = folder / f'{target}.json'
    if payload.exists() or side.exists():
        raise FileExistsError('incomplete charge task; inspect existing overlay before retrying')
    cache = {}; formal_counts = {}
    def convert(rd, arrays):
        if rd is None:
            raise ValueError('invalid saved SDF molecule')
        centers = rd.GetConformer().GetPositions()
        if centers.shape != arrays['centers_w_H'].shape or not np.allclose(
                centers, arrays['centers_w_H'], rtol=0, atol=1e-4):
            raise ValueError('SDF atom order/coordinates differ from frozen features')
        # Include ordered atoms and bonds: atom-indexed charges must never be permuted.
        key = (Chem.MolToSmiles(rd, canonical=False),
               tuple((a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsAromatic()) for a in rd.GetAtoms()),
               tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType())) for b in rd.GetBonds()))
        if key not in cache:
            props = AllChem.MMFFGetMoleculeProperties(Chem.Mol(rd), mmffVariant='MMFF94')
            if props is None:
                raise ValueError('MMFF94 cannot parameterize a benchmark molecule; no exclusions permitted')
            cache[key] = np.array([props.GetMMFFPartialCharge(i) for i in range(rd.GetNumAtoms())], dtype=np.float32)
        charge = int(Chem.GetFormalCharge(rd))
        formal_counts[str(charge)] = formal_counts.get(str(charge), 0) + 1
        return replace_charges(arrays, cache[key])
    root = Path(store)/target
    queries = list(Chem.SDMolSupplier(str(root/f'{target}_query.sdf'), removeHs=False))
    if len(queries) != 1:
        raise ValueError('expected one frozen query')
    query = convert(queries[0], data['query'])
    library = []
    for i, rd in enumerate(Chem.SDMolSupplier(str(root/f'{target}.sdf'), removeHs=False)):
        if i >= len(data['library']) or rd is None or rd.GetProp('_Name') != meta['names'][i]:
            raise ValueError('SDF conformer identities/order differ')
        library.append(convert(rd, data['library'][i]))
    if len(library) != len(data['library']):
        raise ValueError('SDF conformer count differs')
    tmp = payload.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        pickle.dump(dict(query=query, library=library), stream, protocol=5)
    tmp.replace(payload)
    write(side, dict(benchmark_id=manifest['benchmark_id'], sha256=digest(payload),
                     charge_model='MMFF94', rdkit_version=rdBase.rdkitVersion, n_conformers=len(library),
                     xtb_total_charge=manifest.get('esp_total_charge', 'not recorded'),
                     mmff_total_charge='molecular formal charge',
                     query_and_conformer_formal_charge_counts=formal_counts,
                     changed_fields=['partial_charges', 'surf_esp'],
                     coordinate_validation_atol_angstrom=1e-4))
    return [payload, side]


def apply(data, folder, target, benchmark_id):
    folder = Path(folder); payload = folder/f'{target}.pkl'
    meta = json.loads((folder/f'{target}.json').read_text())
    if meta['benchmark_id'] != benchmark_id or digest(payload) != meta['sha256']:
        raise ValueError('MMFF overlay does not match frozen benchmark')
    with payload.open('rb') as stream:
        overlay = pickle.load(stream)
    if len(overlay['library']) != len(data['library']):
        raise ValueError('MMFF overlay length mismatch')
    def merge(base, patch):
        if set(patch) != {'partial_charges', 'surf_esp'}:
            raise ValueError('MMFF overlay changes non-charge features')
        return dict(base, **patch)
    return dict(query=merge(data['query'], overlay['query']),
                library=[merge(a,b) for a,b in zip(data['library'],overlay['library'])]), meta
