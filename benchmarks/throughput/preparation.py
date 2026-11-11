"""Shared 100-compound preparation, timed once and reused by every representation."""
import json
import os
from pathlib import Path
import time
import pickle
import tempfile
from suite import digest, write

CONFIGS = {'vol': ('mmff',0,'0'), 'pharm': ('mmff',0,'1'), 'surf': ('mmff',200,'1'),
           'vol_esp': ('xtb',0,'1'), 'surf_esp': ('xtb',200,'1')}


def cpu_seconds():
    import resource
    return sum(r.ru_utime+r.ru_stime for r in
               (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)))


def timed(fn):
    fn()  # Separate objects, full workload, untimed.
    cpu=cpu_seconds();start=time.perf_counter()
    result=fn()
    elapsed=time.perf_counter()-start;used=cpu_seconds()-cpu
    return result,dict(seconds=elapsed,reps=1,warmups=1,rep_seconds=[elapsed],
                       cpu_seconds=used,effective_cores=used/elapsed,
                       affinity_cores=len(os.sched_getaffinity(0)))


def run(spec,cfg,root):
    from rdkit import Chem
    import numpy as np
    folder=Path(root)/'preparation';folder.mkdir(parents=True,exist_ok=True)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    if spec['stage']=='embed':
        from rdkit.Chem import AllChem
        smiles=[s.split()[0] for s in Path(cfg['prep_smiles']).read_text().splitlines()
                if s.strip() and not s.lstrip().startswith('#')][:100]
        if len(smiles)!=100:raise ValueError('preparation needs 100 compounds')
        def embed():
            mols=[]
            for i,smi in enumerate(smiles):
                rd=Chem.AddHs(Chem.MolFromSmiles(smi))
                p=AllChem.ETKDGv3();p.randomSeed=42;p.pruneRmsThresh=-1.;p.numThreads=1
                ids=list(AllChem.EmbedMultipleConfs(rd,numConfs=50,params=p))
                if len(ids)!=50:raise ValueError(f'compound {i}: fewer than 50 conformers')
                AllChem.MMFFOptimizeMoleculeConfs(rd,maxIters=500,numThreads=1)
                Chem.SanitizeMol(rd)
                mols.extend(Chem.Mol(rd,confId=int(cid)) for cid in ids)
            return mols
        mols,row=timed(embed)
        sdf=folder/'shared.sdf'
        with Chem.SDWriter(str(sdf)) as writer:
            for i,m in enumerate(mols):m.SetProp('_Name',f'c{i}');writer.write(m)
        man=dict(sdf=sdf.name,sdf_sha256=digest(sdf),n_conformers=len(mols),n_compounds=100,
                 seed=42,n_conformers_each=50,mmff_max_iters=500,prune_rms_thresh=-1.,
                 source_smiles_sha256=digest(cfg['prep_smiles']))
        write(folder/'shared.manifest.json',man)
        row.update(stage='embed',n=5000,input_sha256=man['source_smiles_sha256'],
                   output_sdf_sha256=man['sdf_sha256'],timing_boundary='ETKDGv3 + MMFF94 + aromaticity restoration')
    elif spec['stage']=='index':
        config=spec['config'];path=folder/f'{config}.pkl'
        meta=json.loads((folder/f'{config}.manifest.json').read_text())
        if digest(path)!=meta['sha256']:raise ValueError('prepared feature cache changed')
        mols=pickle.loads(path.read_bytes())
        charge,surface,pharm=CONFIGS[config]
        os.environ.update(FSS_BENCH_CHARGES=charge,FSS_BENCH_SURF_POINTS=str(surface),FSS_BENCH_PHARM=pharm)
        from aligners.fss import FSSAligner
        aligner=FSSAligner('numba')
        from timing import measure_build
        import shutil
        def build():
            out=tempfile.mkdtemp(prefix='index-',dir=folder)
            aligner.build_store(mols,list(range(len(mols))),out,modes=[spec['mode']])
            return out
        def probe(out):
            return dict(n_indexed=aligner.index_count(out),bytes=sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file()))
        stats,handle=measure_build(build,aligner,n=len(mols),probe=probe,discard=shutil.rmtree)
        shutil.rmtree(handle)
        row=dict(stats,stage='index',mode=spec['mode'],config=config,n=len(mols),reps=1,
                 seconds=stats['build_seconds'],input_sha256=meta['sha256'],
                 timing_boundary='profile extraction and store write; prepared features reused')
    else:
        man=json.loads((folder/'shared.manifest.json').read_text());sdf=folder/man['sdf']
        if digest(sdf)!=man['sdf_sha256']:raise ValueError('prepared SDF changed')
        mols=list(Chem.SDMolSupplier(str(sdf),removeHs=False))
        if len(mols)!=5000 or any(m is None for m in mols):raise ValueError('invalid preparation input')
        config=spec['config'];calls=[0]
        if config in CONFIGS:
            charge,surface,pharm=CONFIGS[config]
            os.environ.update(FSS_BENCH_CHARGES=charge,FSS_BENCH_SURF_POINTS=str(surface),FSS_BENCH_PHARM=pharm)
            from aligners.fss import prepare_conformer
            from comparators import drivers
            orig=drivers.xtb_charges
            def counted(*a,**kw):calls[0]+=1;return orig(*a,**kw)
            drivers.xtb_charges=counted
            def convert():
                calls[0]=0;out=[]
                for i,m in enumerate(mols):
                    np.random.seed(42+i)
                    if surface:
                        import open3d as o3d
                        o3d.utility.random.seed(42+i)
                    prepared=prepare_conformer(Chem.Mol(m))
                    if prepared is None:raise ValueError(f'{config}: preparation failed at {i}')
                    out.append(prepared)
                expected=5000 if charge=='xtb' else 0
                if calls[0]!=expected:raise ValueError('unexpected xTB call count')
                return out
        elif config.startswith('roshambo2'):
            from roshambo2.prepare import prepare_from_rdkitmols
            def convert():
                return prepare_from_rdkitmols([Chem.Mol(m) for m in mols],
                                             color=config.endswith('color'),quiet=True,n_cpus=1)
        elif config=='sdf':
            def convert():
                out=folder/'handoff.sdf'
                with Chem.SDWriter(str(out)) as writer:
                    for m in mols:writer.write(m)
                return out
        else:raise ValueError(config)
        prepared,row=timed(convert)
        if config in CONFIGS:
            cache=folder/f'{config}.pkl';cache.write_bytes(pickle.dumps(prepared,protocol=5))
            write(folder/f'{config}.manifest.json',dict(sha256=digest(cache),input_sdf_sha256=man['sdf_sha256']))
        row.update(stage='convert',config=config,n=5000,input_sha256=man['sdf_sha256'],
                   xtb_calls=calls[0],timing_boundary='conversion of the shared 5000 conformers; embedding excluded')
    from comparators.envcheck import fingerprint
    row['environment']=fingerprint()
    return row
