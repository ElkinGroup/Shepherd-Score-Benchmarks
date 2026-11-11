"""Stored-input workloads used by the single manuscript measurement plan."""
import json
import os
from pathlib import Path
import pickle
import tempfile
import time
import numpy as np
import hashlib

from suite import digest, identity, write


def checked_pickle(path, expected):
    if digest(path) != expected: raise ValueError(f'changed input: {path}')
    with Path(path).open('rb') as stream: return pickle.load(stream)


def rectangular_pairs(manifest, size):
    q = manifest['selections']['query']; lib = manifest['selections']['library']
    if set(q['compound_indices']) & set(lib['compound_indices']):
        raise ValueError('pairwise compound sets overlap')
    if (q['n'],lib['n'],manifest['n_pairs']) != (250,400,100000):
        raise ValueError('pairwise selection must be 250 x 400')
    if size not in (1000,100000): raise ValueError('unsupported pairwise size')
    # CPU uses the first 1,000 of the SAME ordered GPU pair list.
    return [(i,250+j) for i in range(250) for j in range(400)][:size]


def pairwise(storage, spec, aligner):
    from aligners.base import MolHolder
    folder=Path(storage)/'pairwise'; man=json.loads((folder/'manifest.json').read_text())
    arrays=[]
    for label in ('query','library'):
        item=man['selections'][label]
        prefix='' if spec['tool']=='fss' else 'rdkit_'
        values=checked_pickle(folder/item[prefix+'file'],item[prefix+'sha256'])
        if len(values)!=item['n']: raise ValueError('pairwise selection length mismatch')
        if spec['tool']!='fss':
            values=[MolHolder(m,m.GetConformer().GetPositions()) for m in values]
        arrays.extend(values)
    pairs=rectangular_pairs(man,spec['size'])
    job=aligner.prepare_pairwise(arrays,pairs,spec['mode'])
    if job.n!=len(pairs) or job.extra.get('cross_overlays',job.n)!=len(pairs):
        raise ValueError('adapter changed the requested pair set')
    return job,dict(manifest_sha256=digest(folder/'manifest.json'),
                    pair_indices_sha256=identity(pairs),query_count=len({q for q,_ in pairs}),
                    library_count=len({d for _,d in pairs}),selection=man)




class StoredFields:
    """Read serialized molecule fields without importing the FSS execution backend."""
    def __setstate__(self,state):
        if isinstance(state,tuple):
            for part in state:
                if part:self.__dict__.update(part)
        else:self.__dict__.update(state)


class RDKitUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module.startswith('shepherd_score') and name in ('Molecule','Surface','Pharmacophore'):
            return StoredFields
        return super().find_class(module,name)


def rdkit_load(stream):return RDKitUnpickler(stream).load()


class RDKitView:
    def __init__(self,pool):self.pool=pool
    def __getitem__(self,index):
        from aligners.base import MolHolder
        molecule=self.pool[index]
        return MolHolder(molecule.mol,molecule.mol.GetConformer().GetPositions())


class FukuiPool:
    """Cache missing fields once, outside timing; never drop a failing conformer."""
    def __init__(self,pool,folder):
        self.pool=pool;self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
    def materialize(self,molecule,key):
        path=self.folder/(str(key)+'.npz')
        if path.exists():
            with np.load(path) as saved:
                values=saved['values']
                if hashlib.sha256(values.tobytes()).hexdigest()!=str(saved['sha256']):
                    raise ValueError('Fukui cache changed')
            molecule.fukui=values
        else:
            lock=path.with_suffix('.lock')
            with lock.open('x'):
                try:
                    values=np.asarray(molecule.fukui)
                    if not np.isfinite(values).all():raise ValueError('nonfinite Fukui field')
                    tmp=path.with_suffix('.tmp')
                    with tmp.open('wb') as stream:
                        np.savez(stream,values=values,sha256=hashlib.sha256(values.tobytes()).hexdigest())
                    tmp.replace(path)
                finally:lock.unlink()
        return molecule
    def __getitem__(self,index):return self.materialize(self.pool[index],int(index))


def screen_inputs(storage, spec, cache, aligner, *, reuse=None):
    from pool import ShardedPool
    import workloads
    from aligners.base import MolHolder
    storage=Path(storage); selection=storage/'selections'
    label='extended' if spec['size']>10497450 else 'main'
    item=json.loads((selection/'manifest.json').read_text())[label]
    prefix='query' if spec['tool']=='fss' else 'query_rdkit'
    # Library size changes the pool, not the query used for the scaling sweep.
    query_item=json.loads((selection/'manifest.json').read_text())['main']
    query=checked_pickle(selection/query_item[prefix+'_file'],query_item[prefix+'_sha256'])
    if spec['tool']!='fss':query=MolHolder(query,query.GetConformer().GetPositions())
    pool_path=Path(item['pool_manifest'])
    pool_info=checked_pickle(pool_path,item['pool_manifest_sha256'])
    count=sum(pool_info['counts'])
    n=spec['size']
    if n>count:raise ValueError('requested size exceeds frozen pool')
    chosen=np.sort(np.random.default_rng(42).choice(count//50,size=(n+49)//50,replace=False))
    indices=(chosen[:,None]*50+np.arange(50)).reshape(-1)[:n]
    selection_hash=hashlib.sha256(indices.astype('<i8',copy=False).tobytes()).hexdigest()
    record=dict(pool_manifest_sha256=item['pool_manifest_sha256'],
                query_sha256=item['query_sha256'],selection_sha256=selection_hash,
                n=spec['size'],tool=spec['tool'],mode=spec['mode'],backend=spec['backend'],
                canonical=True,engine_commit=os.environ['FIG2_FSS_COMMIT'])
    store=Path(cache)/identity(record)[:24]
    if reuse:
        previous=dict(record,engine_commit=reuse['engine_commit'])
        candidate=Path(reuse['path'])/identity(previous)[:24]
        previous_sidecar=candidate/'suite-inputs.json'
        if previous_sidecar.is_file():
            if json.loads(previous_sidecar.read_text())!=previous:
                raise ValueError('reused index input identity mismatch')
            if aligner.index_count(str(candidate))!=spec['size']:
                raise ValueError('reused index has the wrong conformer count')
            store,record=candidate,previous
    sidecar=store/'suite-inputs.json'
    if sidecar.exists():
        if json.loads(sidecar.read_text())!=record:raise ValueError('index identity mismatch')
    else:
        if store.exists():raise ValueError(f'incomplete index: {store}; inspect before retrying')
        store.mkdir(parents=True)
        pool=ShardedPool([str(pool_path.parent/p) for p in pool_info['paths']],pool_info['counts'],
                         loader=pickle.load if spec['tool']=='fss' else rdkit_load)
        if spec['tool']=='fss':
            if spec['mode']=='vol_fukui':
                pool=FukuiPool(pool,Path(cache).parent/'features'/item['pool_manifest_sha256'])
                query=pool.materialize(query,'query-'+item['query_sha256'])
            molecules=pool; selected=indices
        else:
            # All adapters consume the same frozen coordinates, not new conformers.
            molecules=RDKitView(pool)
            selected=indices
        aligner.build_store(molecules,selected,str(store),modes=[spec['mode']])
        if aligner.index_count(str(store))!=spec['size']:raise ValueError('incomplete screening index')
        write(sidecar,record)
    if spec['mode']=='vol_fukui' and spec['tool']=='fss':
        query=FukuiPool(None,Path(cache).parent/'features'/item['pool_manifest_sha256']).materialize(query,'query-'+item['query_sha256'])
    return query,store,dict(record, query_sha256=query_item['query_sha256'],
                           index_identity=store.name)


def measure(spec, cfg, root):
    from aligners import get_aligner
    from aligners.base import Job
    from timing import measure as timed
    import sweep
    if spec['tool']=='fss':sweep.assert_library_pinned()
    if spec['backend'] in ('cpu','numba') and spec['workload']!='parallel':
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    aligner=get_aligner(spec['tool'],backend=spec['backend']);aligner.threads=1
    if spec['backend']=='triton':
        import torch
        torch.set_num_threads(1)
    if spec['workload']=='pairwise':
        job,inputs=pairwise(cfg['storage'],spec,aligner)
    else:
        query,store,inputs=screen_inputs(cfg['storage'],spec,root/'indexes',aligner,
                                       reuse=cfg.get('reuse_indexes'))
        if spec['workload']=='parallel':
            from shepherd_score.screen import ProfileStore
            from shepherd_score.accel.screen_parallel import screen_parallel,screen_parallel_close
            st=ProfileStore.open(str(store))
            library=[p for sh in range(st.num_shards) for p in st.read_profiles(sh)]
            if len(library)!=spec['size']:raise ValueError('parallel library count mismatch')
            if len(os.sched_getaffinity(0))<spec['workers']:raise ValueError('not enough allocated CPU cores')
            holder={}
            def call():
                scores=screen_parallel(query,library,spec['mode'],n_workers=spec['workers'])
                if len(scores)!=len(library):raise ValueError('parallel score count mismatch')
                holder['scores']=scores
            job=Job(len(library),call,lambda:float(np.mean(holder['scores'])))
        elif spec.get('devices',1)>1:
            import torch
            from shepherd_score.screen import ProfileStore,screen
            if torch.cuda.device_count()<spec['devices']:raise ValueError('not enough allocated GPUs')
            st=ProfileStore.open(str(store))
            holder={}
            def call():
                hits=screen(query,st,mode=spec['mode'],backend='triton',top_k=1000,
                            ndev=spec['devices'],worker_threads=1)
                if len(hits)!=min(1000,len(st)):raise ValueError('multi-GPU hit count mismatch')
                holder['scores']=[h.score for h in hits]
            job=Job(len(st),call,lambda:float(np.mean(holder['scores'])))
        else:
            job=aligner.prepare_screen(str(store),query,spec['mode'],top_k=1000)
    if job.n!=spec['size']:raise ValueError('incorrect workload size')
    try:
        row=timed(job,aligner)
    finally:
        if spec['workload']=='parallel':screen_parallel_close()
    row.update(spec,inputs=inputs,env=sweep.describe_env(aligner),timing_boundary='alignment call; setup excluded')
    if spec['tool']=='fss' and spec['backend']=='triton':
        row['host_compute_threads_per_gpu']=1
        profile=cfg.get('triton_configs')
        row['triton_configs_sha256']=digest(profile) if profile else None
        if spec.get('devices',1)==1:
            from shepherd_score.accel.kernels.tuning import export_configurations
            folder=root/'tuning';folder.mkdir(exist_ok=True)
            path=folder/(identity(spec)[:20]+'.json')
            export_configurations(path)
            row['triton_configs_observed']=str(path.relative_to(root))
    row.update(job.extra)
    return row
