"""Prepare once, then score main-text and SI modes from the same frozen benchmark."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dudez import protocol as P

MAIN_MODES = ('vol', 'vol_esp', 'surf', 'surf_esp', 'vol_and_surf_esp',
              'pharm', 'vol_color', 'vol_lipo')
SI_MODES = ('vol_pharm', 'vol_atomtype', 'vol_mr', 'surf_tversky',
            'surf_esp_tversky', 'vol_and_surf_esp_tversky', 'vol_color_tversky',
            'vol_lipo_tversky', 'pharm_tversky', 'vol_fukui')
MODES = MAIN_MODES + SI_MODES


def _keep_molecule(rd):
    return rd


def export(target, dudez_dir, sdf_dir, jobs=1):
    """Generate a new input set explicitly; published reruns should use the saved store."""
    from rdkit import Chem
    source = Path(dudez_dir) / target
    destination = Path(sdf_dir)
    if any(Path(p).exists() for p in P.shared_library_paths(destination, target)):
        raise FileExistsError(f'{target}: export exists; use a new directory')
    actives = P.read_smiles(source / 'actives.smi')
    decoys = P.read_smiles(source / 'decoys.smi')
    for i, smi in enumerate(actives + decoys):
        if Chem.MolFromSmiles(smi) is None:
            raise ValueError(f'{target}: invalid input SMILES at compound {i}')
    mols, owner, kept = P.build_library(actives + decoys, jobs, _keep_molecule)
    if len(kept) != len(actives) + len(decoys):
        raise ValueError('conformer generation failed for a compound; no export written')
    query = P.load_query(source)
    destination.mkdir(parents=True, exist_ok=True)
    lib_path, query_path, meta_path = P.shared_library_paths(destination, target)
    names, seen = [], {}
    with Chem.SDWriter(lib_path) as writer:
        for rd, comp in zip(mols, owner):
            k = seen.get(int(comp), 0)
            seen[int(comp)] = k + 1
            tag = f'act_{comp}' if comp < len(actives) else f'dec_{comp - len(actives)}'
            name = f'{tag}_c{k}'
            rd.SetProp('_Name', name)
            writer.write(rd)
            names.append(name)
    with Chem.SDWriter(query_path) as writer:
        writer.write(query)
    Path(meta_path).write_text(json.dumps({'names': names, 'owner': owner.tolist(),
                                         'n_actives': len(actives), 'n_compounds': len(kept)}))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def environment():
    import rdkit
    import shepherd_score
    import torch
    root = Path(shepherd_score.__file__).resolve().parents[1]
    files = sorted((root / 'shepherd_score').rglob('*.py'))
    code = identity({str(p.relative_to(root)): digest(p) for p in files})
    return {'fss_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
            'fss_source_sha256': code, 'rdkit': rdkit.__version__, 'numpy': np.__version__,
            'torch': torch.__version__, 'cuda': torch.version.cuda}


def xtb_charges(rd, charge, uhf=0):
    """Retry a failed SCC calculation once with a larger iteration limit."""
    from rdkit import Chem
    from shepherd_score.conformer_generation import charges_from_single_point_conformer_with_xtb
    try:
        return np.asarray(charges_from_single_point_conformer_with_xtb(
            rd, charge=charge, uhf=uhf, timeout=120), dtype=np.float64)
    except subprocess.CalledProcessError:
        with tempfile.TemporaryDirectory(prefix='dudez_xtb_') as directory:
            path = Path(directory)
            Chem.MolToXYZFile(rd, str(path / 'input.xyz'))
            command = ['xtb', 'input.xyz', '--scc', '--parallel', '1', '--chrg', str(charge),
                       '--iterations', '1000']
            if uhf:
                command += ['--uhf', str(uhf)]
            result = subprocess.run(command, cwd=path, capture_output=True, text=True, timeout=120)
            if result.returncode:
                raise RuntimeError(f'xTB charge={charge}, uhf={uhf} failed after retry: '
                                   + (result.stdout + result.stderr)[-1500:])
            return np.loadtxt(path / 'charges', ndmin=1)


def prepare_molecule(rd, seed):
    from rdkit import Chem
    import open3d as o3d
    from shepherd_score.container import Molecule
    rd = Chem.Mol(rd)
    Chem.SanitizeMol(rd)
    np.random.seed(seed)
    o3d.utility.random.seed(seed)
    # Preserve the existing ESP benchmark's neutral-system charge convention.
    charges = xtb_charges(rd, charge=0)
    formal = int(Chem.GetFormalCharge(rd))
    reference = charges if formal == 0 else xtb_charges(rd, charge=formal)
    anion = xtb_charges(rd, charge=formal - 1, uhf=1)
    cation = xtb_charges(rd, charge=formal + 1, uhf=1)
    fukui = ((reference - anion) - (cation - reference)).astype(np.float32)
    return Molecule(rd, num_surf_points=P.SURF_POINTS, partial_charges=charges,
                    pharm_multi_vector=False, fukui=fukui)


def profile_arrays(m):
    """Plain arrays: scoring never invokes RDKit, xTB, or surface sampling."""
    return dict(atom_pos=m.atom_pos, surf_pos=m.surf_pos, surf_esp=m.surf_esp,
                partial_charges=m.partial_charges, radii=m.radii,
                nonH_atoms_idx=m._nonH_atoms_idx,
                pharm_types=m.pharm_types, pharm_ancs=m.pharm_ancs, pharm_vecs=m.pharm_vecs,
                centers_w_H=m.mol.GetConformer().GetPositions(),
                lipo_pos=m.get_lipo_positions(), lipophilicity=m.get_lipophilicity(no_H=True),
                mr_pos=m.get_mr_positions(), molar_refractivity=m.get_molar_refractivity(no_H=True),
                atomtype_pos=m.get_atomtype_positions(), atomic_numbers=m.get_atomic_numbers(no_H=True),
                fukui_pos=m.get_fukui_positions(), fukui=m.get_fukui(no_H=True))


def normalize_query(rd):
    """Add missing hydrogens without moving the crystal atoms; use the saved SDF precision."""
    from rdkit import Chem
    rd = Chem.AddHs(rd, addCoords=True)
    return Chem.MolFromMolBlock(Chem.MolToMolBlock(rd), removeHs=False)


def _prepare_one(arg):
    i, rd = arg
    try:
        m = prepare_molecule(rd, P.SEED + i + 1)
        arrays = profile_arrays(m)
        if any(v is None or not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('missing or nonfinite feature array')
        return i, arrays, None
    except Exception as exc:
        return i, None, f'{type(exc).__name__}: {exc}'


def prepare(store, target, sdf_dir, jobs=1, cache_dir=None, exclude_failures=False):
    """Import existing conformers; never regenerate a library while scoring."""
    from rdkit import Chem
    store = Path(store)
    destination = store / target
    if destination.exists():
        raise FileExistsError(f'{destination} exists; use a new store version to change inputs')
    mols, query_rd, meta = P.read_shared_library(sdf_dir, target)
    if any(Chem.AddHs(m).GetNumAtoms() != m.GetNumAtoms() for m in mols):
        raise ValueError('library SDF must contain explicit hydrogens before preparation')
    query_rd = normalize_query(query_rd)
    profiles = [None] * len(mols)
    reused = 0
    source_hashes = {Path(p).name: digest(p) for p in P.shared_library_paths(sdf_dir, target)}
    # Optional one-time migration of an existing, trusted local cache. Validate each record.
    cache_provenance = None
    if cache_dir:
        candidates = []
        for side in Path(cache_dir).glob(target + '_*.json'):
            doc = json.loads(side.read_text())
            if doc.get('sdf_sha256') == source_hashes[target + '.sdf'][:16] and doc.get('want_fukui'):
                candidates.append((side, doc))
        if len(candidates) != 1:
            raise ValueError(f'{target}: expected one matching cache, found {len(candidates)}')
        side, doc = candidates[0]
        payload = side.with_suffix('.pkl')
        cache_provenance = {'metadata': doc, 'sha256': digest(payload)}
        with payload.open('rb') as f:
            cached, fukui_ok = pickle.load(f)
        if len(cached) != len(mols) or len(fukui_ok) != len(mols):
            raise ValueError('cache length differs from SDF')
        for i, (rd, m) in enumerate(zip(mols, cached)):
            if m is None or not fukui_ok[i]:
                continue
            # RDKit's default pickle stores conformer coordinates as float32.
            if (Chem.MolToSmiles(rd) != Chem.MolToSmiles(m.mol)
                    or [a.GetAtomicNum() for a in rd.GetAtoms()] != [a.GetAtomicNum() for a in m.mol.GetAtoms()]
                    or not np.array_equal(rd.GetConformer().GetPositions().astype(np.float32),
                                          m.mol.GetConformer().GetPositions().astype(np.float32))):
                raise ValueError(f'{target}: cached molecule {i} differs from exported molecule')
            arrays = profile_arrays(m)
            if any(v is None or not np.isfinite(v).all() for v in arrays.values()):
                continue
            profiles[i] = arrays
            reused += 1
        del cached
    missing = [(i, rd) for i, rd in enumerate(mols) if profiles[i] is None]
    print(f'{target}: reusing {reused}; preparing {len(missing)} conformers and one query', flush=True)
    errors = {}
    if jobs > 1 and missing:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(jobs) as pool:
            for i, arrays, error in pool.imap_unordered(_prepare_one, missing):
                profiles[i] = arrays
                if error:
                    errors[str(i)] = error
    else:
        for item in missing:
            i, arrays, error = _prepare_one(item)
            profiles[i] = arrays
            if error:
                errors[str(i)] = error
    _, query, error = _prepare_one((-1, query_rd))
    if error:
        raise ValueError(f'{target}: query preparation failed: {error}')
    if errors and not exclude_failures:
        store.mkdir(parents=True, exist_ok=True)
        (store / (target + '.failures.json')).write_text(json.dumps(errors, indent=2))
        raise ValueError(f'{target}: {len(errors)} conformers failed; no benchmark published')
    keep = [i for i, item in enumerate(profiles) if item is not None]
    if set(meta['owner'][i] for i in keep) != set(range(meta['n_compounds'])):
        raise ValueError('preparation would remove an entire compound; no benchmark published')
    exclusions = [{'index': int(i), 'name': meta['names'][int(i)], 'error': e} for i, e in errors.items()]
    original_count = len(profiles)
    if errors:
        profiles = [profiles[i] for i in keep]
        meta = dict(meta, names=[meta['names'][i] for i in keep], owner=[meta['owner'][i] for i in keep])
    store.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{target}-', dir=store))
    try:
        for path in P.shared_library_paths(sdf_dir, target):
            shutil.copyfile(path, staging / Path(path).name)
        copied_hashes = {p.name: digest(p) for p in staging.iterdir()}
        if any(copied_hashes[k] != v for k, v in source_hashes.items()):
            raise ValueError('source export changed during preparation')
        with Chem.SDWriter(str(staging / (target + '_query.sdf'))) as writer:
            writer.write(query_rd)
        if errors:
            # Retain the exact SDF bytes of every included conformer.
            source = staging / (target + '.sdf')
            filtered = source.with_suffix('.sdf.tmp')
            excluded = {int(i) for i in errors}
            with source.open('rb') as src, filtered.open('wb') as dst:
                i = 0
                for line in src:
                    if i not in excluded:
                        dst.write(line)
                    if line.strip() == b'$$$$':
                        i += 1
            os.replace(filtered, source)
            (staging / (target + '_meta.json')).write_text(json.dumps(meta))
        with (staging / 'features.pkl').open('wb') as f:
            pickle.dump({'query': query, 'library': profiles}, f, protocol=5)
        files = {p.name: digest(p) for p in staging.iterdir()}
        manifest = {'schema': 1, 'target': target, 'files': files, 'modes': list(MODES),
                    'n_conformers': len(profiles), 'n_compounds': meta['n_compounds'],
                    'n_actives': meta['n_actives'], 'environment': environment(),
                    'preparation_sha256': digest(__file__), 'cache_import': cache_provenance,
                    'charge_model': 'xtb', 'esp_total_charge': 0, 'fukui_reference_charge': 'formal',
                    'surf_points': P.SURF_POINTS, 'seed': P.SEED,
                    'query_hydrogens': 'RDKit AddHs(addCoords=True); coordinates saved at SDF precision',
                    'source_files': source_hashes, 'source_n_conformers': original_count,
                    'exclusions': exclusions, 'failure_policy': 'shared-exclusion' if exclude_failures else 'error'}
        manifest['benchmark_id'] = identity(manifest)
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(f'{target}: sealed {destination} ({manifest["benchmark_id"]})', flush=True)


def load(store, target):
    root = Path(store) / target
    manifest = verify(store, target)
    meta = P.read_meta(root, target)
    with (root / 'features.pkl').open('rb') as f:
        data = pickle.load(f)
    if len(data['library']) != len(meta['names']):
        raise ValueError('features and metadata lengths differ')
    return data, meta, manifest


def verify(store, target):
    root = Path(store) / target
    manifest = json.loads((root / 'manifest.json').read_text())
    expected = manifest.pop('benchmark_id')
    if identity(manifest) != expected:
        raise ValueError('benchmark manifest identity mismatch')
    manifest['benchmark_id'] = expected
    for name, sha in manifest['files'].items():
        if Path(name).name != name or digest(root / name) != sha:
            raise ValueError(f'benchmark file changed: {name}')
    return manifest


def parameters(mode):
    from shepherd_score.accel._modes import MODE_SEEDS, MODE_STEPS
    from shepherd_score.container import MoleculePairBatch
    method = getattr(MoleculePairBatch, 'align_with_' + mode)
    sig = inspect.signature(method)
    kw = {k: v.default for k, v in sig.parameters.items()
          if k != 'self' and v.default is not inspect.Parameter.empty}
    kw.update(backend='triton', num_repeats=MODE_SEEDS[mode], max_num_steps=MODE_STEPS[mode])
    if 'alpha' in sig.parameters:
        kw['alpha'] = P.ALPHA
    return kw


def align(query, library, mode, kwargs):
    import torch
    from shepherd_score.screen import MoleculeProfile
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    if not torch.cuda.is_available():
        raise RuntimeError('this benchmark protocol requires a CUDA device')
    def profile(arrays):
        return MoleculeProfile(**{k: np.array(v, copy=True) for k, v in arrays.items()})
    q = profile(query)
    q.center_to(q.atom_pos.mean(0))
    pairs = []
    for m in library:
        fit = profile(m)
        fit.center_to(fit.atom_pos.mean(0))
        pairs.append(MoleculePair(q, fit, do_center=False, device=torch.device('cuda')))
    batch = MoleculePairBatch(pairs)
    getattr(batch, 'align_with_' + mode)(**kwargs)
    attr = 'sim_aligned_vol_noH' if mode == 'vol' else 'sim_aligned_vol_esp_noH' if mode == 'vol_esp' else 'sim_aligned_' + mode
    scores = np.array([float(getattr(p, attr)) for p in pairs])
    if not np.isfinite(scores).all():
        raise ValueError(f'{mode}: nonfinite conformer score')
    return scores


def score(store, target, out_dir, modes, charge_overlay=None):
    import torch
    data, meta, manifest = load(store, target)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    charge_meta = None
    if charge_overlay is not None:
        from dudez.charge_comparison import apply
        data, charge_meta = apply(data, charge_overlay, target, manifest['benchmark_id'])
    run = {'benchmark_id': manifest['benchmark_id'], 'environment': environment(),
           'runner_sha256': digest(__file__), 'protocol_sha256': digest(P.__file__),
           'device': torch.cuda.get_device_name()}
    if charge_meta is not None:
        run.update(charge_model='MMFF94', charge_overlay=charge_meta)
    for mode in modes:
        kwargs = parameters(mode)
        path = out_dir / f'{target}.{mode}.json'
        score_path = out_dir / f'{target}.{mode}.npy'
        provenance = dict(run, parameters=kwargs, mode=mode)
        if path.exists():
            prior = json.loads(path.read_text())
            if prior['provenance'] != provenance or digest(score_path) != prior['scores_sha256']:
                raise ValueError(f'{path}: existing result has different inputs or settings')
            print(f'{target} {mode}: verified existing result', flush=True)
            continue
        scores = align(data['query'], data['library'], mode, kwargs)
        doc = dict(P.evaluate_from_meta(meta, scores), provenance=provenance,
                   target=target, n_conformers=len(scores), n_actives=meta['n_actives'],
                   n_decoys=meta['n_compounds'] - meta['n_actives'])
        tmp = score_path.with_suffix('.npy.tmp')
        with tmp.open('wb') as f:
            np.save(f, scores)
        os.replace(tmp, score_path)
        doc['scores_sha256'] = digest(score_path)
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(doc, indent=2))
        os.replace(tmp, path)
        print(f'{target} {mode}: {doc["metrics"]}', flush=True)
    # Main and SI mode summaries are built from the same per-mode result files.
    for prefix, group in [('enrichment', MAIN_MODES), ('si_enrichment', SI_MODES)]:
        paths = [out_dir / f'{target}.{mode}.json' for mode in group]
        if not all(p.exists() for p in paths):
            continue
        docs = [json.loads(p.read_text()) for p in paths]
        if any({k:d['provenance'][k] for k in run} != run for d in docs):
            raise ValueError('cannot combine results from different benchmark runs')
        for mode, doc in zip(group, docs):
            if digest(out_dir / f'{target}.{mode}.npy') != doc['scores_sha256']:
                raise ValueError(f'{target}/{mode}: saved scores changed')
        result = {'target': target, 'n_actives': meta['n_actives'],
                  'n_decoys': meta['n_compounds'] - meta['n_actives'],
                  'n_conformers': len(meta['names']), 'charge_model': 'xtb',
                  'benchmark_id': manifest['benchmark_id'],
                  'scoring_id': identity(run),
                  'results': {mode:{'metrics':d['metrics'], 'n_scored':d['n_conformers']}
                              for mode, d in zip(group, docs)}}
        path = out_dir / f'{prefix}_{target}.json'
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(result, indent=2))
        os.replace(tmp, path)


def validate_publication_inputs(main_dir, si_dir):
    main = {p.stem.removeprefix('enrichment_'): json.loads(p.read_text())
            for p in Path(main_dir).glob('enrichment_*.json')}
    si = {p.stem.removeprefix('si_enrichment_'): json.loads(p.read_text())
          for p in Path(si_dir).glob('si_enrichment_*.json')}
    if not main or main.keys() != si.keys():
        raise ValueError('main-text and SI target sets differ or are empty')
    for target, a in main.items():
        b = si[target]
        for key in ('benchmark_id', 'scoring_id', 'n_conformers', 'n_actives', 'n_decoys'):
            if a.get(key) is None or a[key] != b.get(key):
                raise ValueError(f'{target}: main-text/SI mismatch in {key}')
        if set(a['results']) != set(MAIN_MODES) or set(b['results']) != set(SI_MODES):
            raise ValueError(f'{target}: incomplete mode set')
        if any(r.get('n_scored') != a['n_conformers'] for d in (a, b) for r in d['results'].values()):
            raise ValueError(f'{target}: modes scored different conformer sets')
    return len(main)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['export', 'prepare', 'score', 'verify'])
    ap.add_argument('--store', type=Path)
    ap.add_argument('--published', action='store_true',
                    help='verify also requires file hashes to match the published inputs')
    ap.add_argument('--target', required=True)
    ap.add_argument('--sdf-dir', type=Path)
    ap.add_argument('--dudez-dir', type=Path)
    ap.add_argument('--cache-dir', type=Path, help='one-time import of a trusted local preparation cache')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--exclude-failures', action='store_true',
                    help='exclude persistently failing conformers from every mode and record their IDs')
    ap.add_argument('--out-dir', type=Path)
    ap.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    a = ap.parse_args()
    if a.published and a.action != 'verify':
        ap.error('--published applies only to verify')
    if a.action != 'export' and a.store is None:
        ap.error('--store is required')
    if a.action == 'export':
        if a.sdf_dir is None or a.dudez_dir is None or a.jobs < 1:
            ap.error('export requires --sdf-dir, --dudez-dir and positive --jobs')
        export(a.target, a.dudez_dir, a.sdf_dir, a.jobs)
    elif a.action == 'prepare':
        if a.sdf_dir is None or a.jobs < 1:
            ap.error('prepare requires --sdf-dir and positive --jobs')
        prepare(a.store, a.target, a.sdf_dir, a.jobs, a.cache_dir, a.exclude_failures)
    elif a.action == 'score':
        if a.out_dir is None:
            ap.error('score requires --out-dir')
        score(a.store, a.target, a.out_dir, a.modes)
    else:
        manifest = verify(a.store, a.target)
        print(manifest['benchmark_id'])
        if a.published:
            reference = json.loads(Path(__file__).with_name('published_inputs.json').read_text())[a.target]
            if manifest['files'] != reference['files']:
                changed = sorted(k for k in set(manifest['files']) | set(reference['files'])
                                 if manifest['files'].get(k) != reference['files'].get(k))
                raise SystemExit('Different from published inputs: ' + ', '.join(changed))
            print('All input file hashes match the published benchmark.')


if __name__ == '__main__':
    main()
