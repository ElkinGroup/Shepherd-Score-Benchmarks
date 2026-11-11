"""Execute and validate one planned measurement."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from suite import ROOT, MODES, EXTERNAL_MODES, digest, write, verify_plan, completed


def execute(plan_path,task_id,required_kind=None):
    plan_path=Path(plan_path).resolve();root=plan_path.parent
    doc=json.loads(plan_path.read_text());verify_plan(doc)
    task=next(t for t in doc['tasks'] if t['id']==task_id);spec=task['spec'];cfg=doc['config']
    os.environ.pop('FSS_TRITON_CONFIGS', None)
    if cfg.get('triton_configs'):
        os.environ['FSS_TRITON_CONFIGS']=cfg['triton_configs']
    if required_kind and spec['kind']!=required_kind:raise ValueError('wrong task kind')
    receipt=root/'receipts'/f'{task_id}.json'
    if completed(receipt,doc,task):return
    sys.path[:0]=[cfg['engine'],str(ROOT/'benchmarks'),str(ROOT/'benchmarks/throughput')]
    from suite_provenance import snapshot
    artifacts=[snapshot(root,cfg['engine'])]
    if spec['kind']=='charge_inputs':
        from dudez.charge_comparison import prepare
        artifacts += prepare(cfg['store'],spec['target'],root/'charge_inputs')
    elif spec['kind'] in ('enrichment','charge_reference','charge_enrichment'):
        from dudez import benchmark as B
        folder=root/'enrichment';target=spec['target'];tool=spec.get('tool','fss')
        modes=MODES
        charge_kind=spec['kind']
        if charge_kind!='enrichment':
            from dudez.charge_comparison import MODES as CHARGE_MODES
            modes=CHARGE_MODES if charge_kind=='charge_enrichment' else tuple(m for m in CHARGE_MODES if m not in MODES)
            folder=root/('enrichment_mmff' if charge_kind=='charge_enrichment' else 'charge_reference')
        manifest=B.verify(cfg['store'],target)
        if tool=='fss':
            B.score(cfg['store'],target,folder,modes,
                    charge_overlay=root/'charge_inputs' if charge_kind=='charge_enrichment' else None)
            files=[folder/f'{target}.{m}.json' for m in modes]
            views=[folder/f'{prefix}_{target}.json' for prefix in ('enrichment','si_enrichment')] if charge_kind=='enrichment' else []
        else:
            prefixes=['phesa','phesapharm'] if tool=='phesa' else [tool]
            files=[folder/f'_ext_{prefix}_{target}.json' for prefix in prefixes];views=[]
            if not all(p.exists() for p in files):
                subprocess.run([sys.executable,str(ROOT/'benchmarks/enrichment/run_benchmark.py'),
                                '--what',tool,'--target',target,'--store',cfg['store'],
                                '--out-dir',str(folder),'--phesa-threads','1'],check=True)
        seen=set()
        for path in files:
            d=json.loads(path.read_text())
            bench=d.get('benchmark_id',d.get('provenance',{}).get('benchmark_id'))
            if bench!=manifest['benchmark_id']:raise ValueError('enrichment input mismatch')
            results=d.get('results',{d.get('provenance',{}).get('mode'):d})
            for mode,result in results.items():
                if mode in seen:raise ValueError('duplicate enrichment output')
                seen.add(mode)
                if not {'auc','ef1','hit_rate'} <= set(result.get('metrics',{})):
                    raise ValueError('missing enrichment metrics')
                import math
                if not all(math.isfinite(float(v['mean'])) for v in result['metrics'].values()):
                    raise ValueError('nonfinite enrichment metric')
                scores=folder/f'{target}.{mode}.npy'
                if digest(scores)!=result['scores_sha256']:raise ValueError('score vector changed')
                artifacts.append(scores)
            artifacts.append(path)
        if seen!=set(modes if tool=='fss' else EXTERNAL_MODES[tool]):
            raise ValueError('incomplete enrichment output')
        artifacts+=views
    else:
        if spec['kind']=='throughput':
            from throughput.suite_measure import measure
            row=measure(spec,cfg,root)
            if row.get('triton_configs_observed'):
                artifacts.append(root/row['triton_configs_observed'])
            if cfg.get('triton_configs') and spec['backend']=='triton':
                # Archive the supplied file even for configurations run in child processes.
                import shutil
                frozen=root/'tuning'/'frozen.json';frozen.parent.mkdir(exist_ok=True)
                if not frozen.exists():
                    temp=frozen.with_suffix('.'+task_id+'.tmp')
                    shutil.copyfile(cfg['triton_configs'],temp);temp.replace(frozen)
                if digest(frozen)!=digest(cfg['triton_configs']):
                    raise ValueError('frozen Triton configurations changed')
                artifacts.append(frozen)
            if row['reps']!=1 or row['warmups']!=1 or row['n']!=spec['size'] or not row['valid']:
                raise ValueError('invalid throughput result')
        else:
            from throughput.preparation import run
            row=run(spec,cfg,root)
            if row['n']!=5000 or row['reps']!=1 or row['warmups']!=1:
                raise ValueError('invalid preparation result')
            if spec['stage']=='embed':
                artifacts += [root/'preparation/shared.sdf',root/'preparation/shared.manifest.json']
            elif spec['stage']=='convert' and not spec['config'].startswith('roshambo') and spec['config']!='sdf':
                artifacts += [root/'preparation'/f"{spec['config']}.pkl",root/'preparation'/f"{spec['config']}.manifest.json"]
        row.update(plan_id=doc['plan_id'],task_id=task_id,engine_commit=doc['engine_commit'])
        path=root/'measurements'/f'{task_id}.json';write(path,row);artifacts.append(path)
    write(receipt,dict(plan_id=doc['plan_id'],task=task,
                       artifacts={str(p.relative_to(root)):digest(p) for p in artifacts}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--task',required=True)
    a=p.parse_args();execute(a.plan,a.task)
