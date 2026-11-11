"""Assemble data for the six SI tables and parallel-processing figure, without editing TeX."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel
from suite import MODES, MAIN, EXTERNAL_MODES, completed, write

MATCH = {'roshambo2_shape':'vol','shaep_shape':'vol','rdkit_shape':'vol','phesa_shape':'vol',
         'roshambo2_combo':'vol_color','rdkit_color':'vol_color','shaep_esp':'vol_and_surf_esp',
         'espsim_esp':'vol_and_surf_esp','pharao':'pharm','phesa_pharm':'pharm'}


def csv_write(path,rows):
    rows=list(rows)
    if not rows:raise ValueError(f'no rows for {path}')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def stats(values):
    x=np.asarray(values,dtype=float)
    if not len(x) or not np.isfinite(x).all():raise ValueError('missing or nonfinite metric')
    return float(x.mean()),float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None


def paired(a,b):
    a=np.asarray(a);b=np.asarray(b);delta=a-b
    mean,sem=stats(delta)
    p=1.0 if np.all(delta==0) else float(ttest_rel(a,b).pvalue)
    return mean,sem,p if np.isfinite(p) else None


def charge_comparison(root, targets, sources=None):
    from dudez.charge_comparison import MODES as CHARGE_MODES
    where=lambda target,key:(sources or {}).get((target,key),root)
    rows=[]; per_target=[]
    for mode in CHARGE_MODES:
        pairs=[]
        for target in targets:
            baseline=(where(target,'fss')/'enrichment' if mode in MODES else
                      where(target,'charge_reference')/'charge_reference')/f'{target}.{mode}.json'
            a=json.loads(baseline.read_text())
            b=json.loads((where(target,'charge_enrichment')/'enrichment_mmff'/f'{target}.{mode}.json').read_text())
            for key in ('n_conformers','n_actives','n_decoys'):
                if a[key]!=b[key]:raise ValueError('charge comparison populations differ')
            if a['provenance']['benchmark_id']!=b['provenance']['benchmark_id']:
                raise ValueError('charge comparison inputs differ')
            if b['provenance'].get('charge_model')!='MMFF94':raise ValueError('missing MMFF provenance')
            if a['provenance']['parameters']!=b['provenance']['parameters']:
                raise ValueError('charge comparison alignment settings differ')
            if set(a['metrics'])!=set(b['metrics']):raise ValueError('charge metrics differ')
            pairs.append((a,b))
            per_target.append(dict(target=target,mode=mode,xtb=a['metrics'],mmff=b['metrics'],
                                   charge_overlay=b['provenance']['charge_overlay']))
        for metric in pairs[0][0]['metrics']:
            a=[p[0]['metrics'][metric]['mean'] for p in pairs]
            b=[p[1]['metrics'][metric]['mean'] for p in pairs]
            am,ase=stats(a);bm,bse=stats(b);delta,dse,p=paired(b,a)
            rows.append(dict(mode=mode,metric=metric,n_targets=len(targets),xtb_mean=am,xtb_sem=ase,
                             mmff_mean=bm,mmff_sem=bse,delta_mmff_minus_xtb=delta,delta_sem=dse,
                             p_unadjusted=p,test='two-sided paired t-test across targets'))
    # Holm correction across every mode/metric comparison in this table.
    valid=sorted((r for r in rows if r['p_unadjusted'] is not None),key=lambda r:r['p_unadjusted'])
    previous=0.
    for i,row in enumerate(valid):
        previous=max(previous,min(1.,(len(valid)-i)*row['p_unadjusted']))
        row.update(p_holm=previous,significant_holm_0_05=previous<.05)
    return rows,per_target


TOOL_OF={mode:tool for tool,modes in EXTERNAL_MODES.items() for mode in modes}


def enrichment(folder,targets,sources=None):
    """``sources``: (target, tool) -> run root holding that result; default ``folder``'s run."""
    where=lambda target,tool:(sources[(target,tool)]/'enrichment' if sources and (target,tool) in sources else folder)
    data={};counts={}
    for target in targets:
        rows={}
        for mode in MODES:
            d=json.loads((where(target,'fss')/f'{target}.{mode}.json').read_text())
            rows[mode]={k:float(v['mean']) for k,v in d['metrics'].items()}
            population={k:d[k] for k in ('n_actives','n_decoys','n_conformers')}
            if target in counts and counts[target]!=population:raise ValueError('target populations differ across modes')
            counts[target]=population
        for base in sorted({where(target,tool) for tool in EXTERNAL_MODES}):
          for path in sorted(base.glob(f'_ext_*_{target}.json')):
            d=json.loads(path.read_text())
            for mode,block in d['results'].items():
                if where(target,TOOL_OF.get(mode))!=base:continue   # that tool's result lives elsewhere
                if mode in rows:raise ValueError(f'duplicate result {target}/{mode}')
                rows[mode]={k:float(v['mean']) for k,v in block['metrics'].items()}
        data[target]=rows
    methods=list(MODES)+sorted(set(data[targets[0]])-set(MODES))
    expected=set(MODES)|{m for modes in EXTERNAL_MODES.values() for m in modes}
    if set(methods)!=expected:raise ValueError('missing comparator outputs')
    for t in targets:
        if set(data[t])!=set(methods):raise ValueError(f'{t}: incomplete method set')
    metric_sets=[set(r) for d in data.values() for r in d.values()]
    if any(keys!=metric_sets[0] for keys in metric_sets):raise ValueError('metric coverage differs across methods')
    metrics=sorted(metric_sets[0])
    if not {'auc','ef1','hit_rate'}<=set(metrics):raise ValueError('missing enrichment metrics')
    summary=[];tests=[]
    for method in methods:
        row=dict(method=method,n_targets=len(targets))
        for metric in metrics:
            values=[data[t][method][metric] for t in targets]
            row[metric+'_mean'],row[metric+'_sem']=stats(values)
        base=method.removesuffix('_tversky') if method.endswith('_tversky') else None
        if base:
            values=[data[t][base]['auc'] for t in targets]
            row['tanimoto_mode']=base
            row['tanimoto_auc'],row['tanimoto_auc_sem']=stats(values)
            row['delta_auc'],row['delta_auc_sem'],row['delta_auc_p']=paired(
                [data[t][method]['auc'] for t in targets],values)
        match=MATCH.get(method) or base
        if match:
            for metric in metrics:
                delta,sem,p=paired([data[t][method][metric] for t in targets],
                                   [data[t][match][metric] for t in targets])
                tests.append(dict(method=method,reference=match,metric=metric,n_targets=len(targets),
                                  delta=delta,delta_sem=sem,p=p,test='two-sided paired t; unadjusted'))
        summary.append(row)
    return summary,data,counts,methods,tests


def assemble(plan_path):
    plan_path=Path(plan_path).resolve();root=plan_path.parent;doc=json.loads(plan_path.read_text())
    missing=[t['id'] for t in doc['tasks'] if not completed(root/'receipts'/f"{t['id']}.json",doc,t)]
    if missing:raise ValueError(f'{len(missing)} unfinished tasks: {missing[:10]}')
    out=root/'tables';out.mkdir(exist_ok=True)
    sources={}
    for task in doc['tasks']:
        spec=task['spec']
        if spec['kind']=='enrichment':sources[(spec['target'],spec['tool'])]=root
        elif spec['kind'] in ('charge_reference','charge_enrichment'):sources[(spec['target'],spec['kind'])]=root
    summary,data,counts,methods,tests=enrichment(root/'enrichment',doc['targets'],sources)
    csv_write(out/'table1_enrichment.csv',summary)
    csv_write(out/'enrichment_paired_tests.csv',tests)
    for name,group in [('table2_fss_ef1',MODES),('table3_comparator_ef1',[m for m in methods if m not in MODES])]:
        csv_write(out/(name+'.csv'),[dict(target=t,**{m:data[t][m]['ef1'] for m in group}) for t in doc['targets']])
    write(out/'enrichment.json',dict(per_target=data,counts=counts,summary=summary,paired_tests=tests,
                                   normalized_early_enrichment='hit_rate = EF1% / target maximum EF1%',
                                   sem='sample standard deviation across targets / sqrt(n_targets)'))
    charge_rows,charge_targets=charge_comparison(root,doc['targets'],sources)
    csv_write(out/'enrichment_charge_comparison.csv',charge_rows)
    write(out/'enrichment_charge_comparison.json',dict(summary=charge_rows,per_target=charge_targets,
          delta='MMFF94 minus stored xTB',correction='Holm across all mode/metric tests in this table',
          note='Frozen coordinates, surface points and non-charge features are unchanged. '
               'MMFF uses formal molecular charge; inspect recorded xTB charge convention. '
               'vol_fukui requires xTB finite-difference calculations and has no MMFF replacement.'))
    rows=[];prep=[];sizes=[];destinations={}
    for task in doc['tasks']:
        for use in task['uses']:destinations.setdefault(use,[]).append(task['id'])
        spec=task['spec']
        if spec['kind'] in ('enrichment','charge_inputs','charge_reference','charge_enrichment'):continue
        row=json.loads((root/'measurements'/f"{task['id']}.json").read_text())
        if spec['kind']=='preparation':prep.append(row);continue
        rows.append(row)
        sizes.append(dict(task_id=task['id'],tool=spec['tool'],mode=spec['mode'],
                          workload=spec['workload'],backend=spec['backend'],n=spec['size'],
                          workers=spec.get('workers',1),devices=spec.get('devices',1),
                          n_queries=row['inputs'].get('query_count',1),
                          n_library=row['inputs'].get('library_count',spec['size']),
                          destinations=';'.join(task['uses']),
                          input_identity=row['inputs'].get('selection_sha256',row['inputs'].get('pair_indices_sha256'))))
    for target,c in counts.items():sizes.append(dict(workload='enrichment',target=target,**c))
    sizes.append(dict(workload='preparation',n_compounds=100,n_conformers=5000))
    csv_write(out/'table4_benchmark_sizes.csv',sizes)
    embed=next(r for r in prep if r['stage']=='embed')
    totals=[dict(configuration='shared_embedding',n=5000,shared_seconds=embed['seconds'],
                 conversion_seconds=0,total_seconds=embed['seconds'],task_id=embed['task_id'])]
    for row in prep:
        if row['stage']!='convert':continue
        if row['input_sha256']!=embed['output_sdf_sha256']:raise ValueError('preparation inputs differ')
        total=embed['seconds']+row['seconds']
        totals.append(dict(configuration=row['config'],n=5000,shared_seconds=embed['seconds'],
                           conversion_seconds=row['seconds'],total_seconds=total,
                           ms_per_conformer=total/5,conformers_per_minute=300000/total,
                           task_id=row['task_id'],shared_task_id=embed['task_id']))
    csv_write(out/'table5_preparation.csv',totals)
    csv_write(out/'table5_index_construction.csv',[dict(mode=r['mode'],configuration=r['config'],n=r['n'],seconds=r['seconds'],store_bytes=r['store_bytes'],task_id=r['task_id']) for r in prep if r['stage']=='index'])
    # One row per displayed tool/mode; ShaEP aliases retain their one measurement id.
    selected=[]
    for row in rows:
        task=next(t for t in doc['tasks'] if t['id']==row['task_id'])
        if 'table6' not in task['uses']:continue
        modes=['vol','vol_and_surf_esp'] if row['tool']=='shaep' else [row['mode']]
        for mode in modes:
            selected.append(dict(tool=row['tool'],mode=mode,backend=row['backend'],
                                 workload=row['workload'],n=row['n'],seconds=row['compute_seconds'],
                                 alignments_per_second=row['aligns_per_sec'],task_id=row['task_id']))
    csv_write(out/'table6_throughput.csv',selected)
    wide={}
    for r in selected:
        d=wide.setdefault((r['tool'],r['mode']),dict(tool=r['tool'],mode=r['mode']))
        key=f"{r['backend']}_{r['workload']}_{r['n']}"
        d[key]=r['alignments_per_second'];d[key+'_task_id']=r['task_id']
    csv_write(out/'table6_throughput_wide.csv',wide.values())
    parallel=[]
    for row in rows:
        task=next(t for t in doc['tasks'] if t['id']==row['task_id'])
        if not set(task['uses'])&{'figure4','figure4-reference'}:continue
        family='cpu_reference' if 'figure4-reference' in task['uses'] else 'cpu' if row['backend']=='numba' else 'gpu'
        parallel.append(dict(family=family,count=row.get('workers',row.get('devices',1)),
                             n=row['n'],seconds=row['compute_seconds'],rate=row['aligns_per_sec'],task_id=row['task_id']))
    for r in parallel:
        base=next(x for x in parallel if x['family']==r['family'] and x['count']==1)
        r['speedup']=r['rate']/base['rate']
    csv_write(out/'figure4_parallel.csv',parallel)
    write(out/'measurements.json',dict(plan_id=doc['plan_id'],throughput=rows,preparation=prep,
                                     destinations=destinations))
    from suite_provenance import tables
    receipts=[dict(json.loads((root/'receipts'/f"{t['id']}.json").read_text()),_root=str(root))
              for t in doc['tasks']]
    tables(root,doc,receipts,out,csv_write)
    write(out/'sources.json',dict(plan_id=doc['plan_id'],engine_commit=doc['engine_commit'],
                                files={p.name:__import__('suite').digest(p) for p in out.iterdir() if p.name!='sources.json'}))
    return out
