"""Export small immutable workload selections from existing prepared pools."""
import argparse,hashlib,io,json,os,pickle,sys
from pathlib import Path
import numpy as np

repo=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--storage',type=Path,default=repo/'benchmark-storage/throughput')
parser.add_argument('--main-pool', type=Path, help='existing main pool .pkl.shards manifest')
parser.add_argument('--extended-pool', type=Path, help='existing expanded pool .pkl.shards manifest')
args=parser.parse_args()
if (args.main_pool is None) != (args.extended_pool is None):
    parser.error('supply both --main-pool and --extended-pool')
os.environ['FIG2_POOL_RESIDENT']='1'
sys.path[:0]=[str(args.storage/'engine'),str(repo/'benchmarks/throughput')]
import pool as P
import workloads as W
cache=args.storage/'pools'
out=args.storage/'selections'
out.mkdir(parents=True,exist_ok=True)
report={}

def save(path,data):
    if path.exists():
        if path.read_bytes()!=data:raise ValueError(f'Existing input differs: {path}')
    else:
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_bytes(data)
        temporary.replace(path)

def adapters(report):
    changed=False
    for label,item in report.items():
        if 'query_rdkit_file' not in item:
            molecule=pickle.loads((out/item['query_file']).read_bytes())
            path=out/f'{label}-query-rdkit.pkl'
            save(path,pickle.dumps(molecule.mol,protocol=4))
            item.update(query_rdkit_file=path.name,query_rdkit_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            changed=True
    return changed

manifest=out/'manifest.json'
if manifest.exists():
    report=json.loads(manifest.read_text())
    for item in report.values():
        expected={item['query_file']:item['query_sha256'],**item.get('pairwise_files',{})}
        for prefix in ['query_rdkit','pairwise_rdkit']:
            if prefix+'_file' in item:expected[item[prefix+'_file']]=item[prefix+'_sha256']
        for name,digest in expected.items():
            if hashlib.sha256((out/name).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Changed selection: {name}')
    for label, supplied in [('main', args.main_pool), ('extended', args.extended_pool)]:
        if supplied and hashlib.sha256(supplied.read_bytes()).hexdigest() != report[label]['pool_manifest_sha256']:
            raise ValueError('Existing selections belong to a different pool; use a new storage directory')
    if adapters(report):
        temporary=manifest.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report,indent=2))
        temporary.replace(manifest)
    print('Existing selections and comparator exports verified.')
    sys.exit(0)

pools = [('main', args.main_pool or cache/'pool_fss_0a6ba79ece.pkl.shards'),
         ('extended', args.extended_pool or cache/'pool_fss_1014f342b0.pkl.shards')]
for label,path in pools:
    path=path.resolve()
    d=pickle.loads(path.read_bytes())
    paths=[str(path.parent/p) for p in d['paths']]
    pool=P.ShardedPool(paths,d['counts'])
    if len(pool) % 50 or not len(pool):
        raise ValueError("pool must contain complete 50-conformer compound blocks")
    qi=W.median_query_index(pool)
    q=pool[qi]
    qpath=out/f'{label}-query.pkl'
    save(qpath,pickle.dumps(q,protocol=4))
    item={'pool_manifest':str(path),'pool_manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'n_conformers':len(pool),'query_index':qi,'query_file':qpath.name,
          'query_sha256':hashlib.sha256(qpath.read_bytes()).hexdigest()}
    report[label]=item
    print(label,len(pool),qi,flush=True)
adapters(report)
save(out/'manifest.json',json.dumps(report,indent=2).encode())
print('Saved fixed queries without embedding or alignment.',flush=True)
