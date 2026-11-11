"""One measurement plan covering every enrichment, preparation and throughput experiment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MAIN = ('vol', 'vol_esp', 'surf', 'surf_esp', 'vol_and_surf_esp', 'pharm', 'vol_color', 'vol_lipo')
ADDITIONAL = ('vol_pharm', 'vol_atomtype', 'vol_mr', 'surf_tversky', 'surf_esp_tversky',
              'vol_and_surf_esp_tversky', 'vol_color_tversky', 'vol_lipo_tversky',
              'pharm_tversky', 'vol_fukui')
MODES = MAIN + ADDITIONAL
EXTERNAL = ('roshambo2', 'shaep', 'pharao', 'phesa', 'rdkit', 'espsim')
EXTERNAL_MODES = {'roshambo2': ('roshambo2_shape','roshambo2_combo'),
                  'shaep': ('shaep_shape','shaep_esp'), 'pharao': ('pharao',),
                  'phesa': ('phesa_shape','phesa_pharm'), 'rdkit': ('rdkit_shape','rdkit_color'),
                  'espsim': ('espsim_shape','espsim_esp','espsim_combo')}
CPU_TOOLS = {'rdkit': ('vol', 'vol_color'), 'phesa': ('vol', 'pharm'),
             'shaep': ('vol',), 'pharao': ('pharm',), 'espsim': ('vol_and_surf_esp',)}
PREP = ('vol', 'pharm', 'surf', 'vol_esp', 'surf_esp', 'roshambo2_shape', 'roshambo2_color', 'sdf')
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def source_hash():
    return identity({p.relative_to(ROOT).as_posix(): digest(p)
                     for p in sorted((ROOT / 'benchmarks').rglob('*.py'))
                     if '__pycache__' not in p.parts})


def engine_commit(engine):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(engine), *args], text=True).strip()
    if git('status', '--porcelain'):
        raise ValueError('the FSS checkout must be clean')
    return git('rev-parse', 'HEAD')


def tasks(targets):
    result = {}
    def add(group, kind, uses, **kw):
        # Figure/table destinations never participate in measurement identity.
        spec = dict(kind=kind, **kw)
        key = identity(spec)[:20]
        if key not in result:
            result[key] = dict(id=key, group=group, spec=spec, uses=[])
        result[key]['uses'] = sorted(set(result[key]['uses']) | set(uses))
        return key
    def speed(tool, backend, mode, workload, n, uses, **kw):
        group = 'roshambo' if tool == 'roshambo2' else ('gpu' if backend == 'triton' else 'cpu')
        return add(group, 'throughput', uses, tool=tool, backend=backend, mode=mode,
                   workload=workload, size=n, **kw)
    for target in targets:
        for tool in ('fss',) + EXTERNAL:
            add('roshambo' if tool == 'roshambo2' else ('gpu' if tool == 'fss' else 'cpu'),
                'enrichment', ['main-enrichment', 'table1', 'table2' if tool == 'fss' else 'table3'],
                target=target, tool=tool)
        charge_input = add('cpu', 'charge_inputs', ['charge-comparison'], target=target)
        extra = add('gpu', 'charge_reference', ['charge-comparison'], target=target)
        mmff = add('gpu', 'charge_enrichment', ['charge-comparison'], target=target)
        result[mmff]['requires'] = [charge_input, extra]
    for mode in MODES:
        for backend, workload, n in [('numba', 'screen', 1000), ('numba', 'pairwise', 1000),
                                     ('triton', 'screen', 100000), ('triton', 'pairwise', 100000)]:
            uses = ['table6']
            if mode in MAIN and (backend, workload, n) in [('numba','screen',1000),('triton','pairwise',100000)]:
                uses += ['main-throughput-C']
            speed('fss', backend, mode, workload, n, uses)
    for mode in MAIN:
        cap = 1000 if mode in ('surf', 'surf_esp', 'vol_and_surf_esp') else 100000
        for n in (10, 100, 1000, 10000, 100000):
            if n <= cap: speed('fss', 'numba', mode, 'screen', n, ['main-throughput-B'])
        for n in (10, 100, 1000, 10000, 100000, 1000000, 10000000):
            speed('fss', 'triton', mode, 'screen', n,
                  ['main-throughput-B'] + (['main-throughput-C','table6'] if n == 10000000 else []))
    for mode in ('vol', 'vol_esp', 'vol_color', 'vol_lipo', 'pharm'):
        speed('fss', 'triton', mode, 'screen', 100000000, ['main-throughput-B'])
    for tool, modes in CPU_TOOLS.items():
        for mode in modes:
            speed(tool, 'cpu', mode, 'screen', 1000, ['main-throughput-C', 'table6'])
    # ShaEP's two plotted columns come from the same executable call.
    # Its one measurement is referenced twice by the report, never run twice.
    for mode in ('vol','vol_color'):
        for workload, n in [('screen',100000),('screen',10000000),('pairwise',100000)]:
            speed('roshambo2', 'cuda', mode, workload, n, ['table6'] + ([] if (workload,n)==('screen',100000) else ['main-throughput-C']))
    embed = add('prep', 'preparation', ['main-throughput-A','table5'], stage='embed')
    conversion = {}
    for config in PREP:
        key = add('roshambo' if config.startswith('roshambo') else 'prep', 'preparation',
                  ['main-throughput-A','table5'], stage='convert', config=config)
        result[key]['requires'] = [embed]
        conversion[config] = key
    for mode in ('vol','pharm','surf','vol_esp','surf_esp','vol_lipo'):
        config = 'pharm' if mode == 'vol_lipo' else mode
        key = add('prep','preparation',['table5-index'],stage='index',config=config,mode=mode)
        result[key]['requires'] = [conversion[config]]
    for workers in (1,8,16,32,64):
        add('cpu64', 'throughput', ['figure4'], tool='fss', backend='numba', mode='vol',
            workload='parallel', size=10000, workers=workers)
    # The in-process CPU reference and one-GPU baseline are already sweep cells.
    speed('fss','numba','vol','screen',10000,['figure4-reference'])
    speed('fss','triton','vol','screen',100000000,['figure4'])
    for devices in (2,4):
        add('gpu4','throughput',['figure4'],tool='fss',backend='triton',mode='vol',
            workload='screen',size=100000000,devices=devices)
    cpu_ref=speed('fss','numba','vol','screen',10000,['figure4-reference'])
    gpu_ref=speed('fss','triton','vol','screen',100000000,['figure4'])
    for task in result.values():
        if task['group']=='cpu64':task['requires']=[cpu_ref]
        if task['group']=='gpu4':task['requires']=[gpu_ref]
    return list(result.values())


def plan(config, out):
    cfg = json.loads(Path(config).read_text())
    for key in ('engine','store','storage','prep_smiles','roshambo_python'):
        cfg[key] = str(Path(os.path.expandvars(os.path.expanduser(cfg[key]))).resolve())
    if cfg.get('triton_configs'):
        cfg['triton_configs'] = str(Path(cfg['triton_configs']).expanduser().resolve())
    if cfg.get('reuse_indexes'):
        previous=cfg['reuse_indexes']
        previous['path']=str(Path(os.path.expandvars(os.path.expanduser(previous['path']))).resolve())
        if len(previous.get('engine_commit',''))!=40:
            raise ValueError('reuse_indexes requires the full index-building engine commit')
    cfg['source_records'] = [str(Path(os.path.expandvars(os.path.expanduser(p))).resolve())
                             for p in cfg.get('source_records', [])]
    targets = sorted(p.parent.name for p in Path(cfg['store']).glob('*/manifest.json'))
    if len(targets) != 43:
        raise ValueError(f'expected 43 frozen target stores, found {len(targets)}')
    manifests = {t:digest(Path(cfg['store'])/t/'manifest.json') for t in targets}
    inputs = {name:digest(Path(cfg['storage'])/name) for name in
              ('selections/manifest.json','pairwise/manifest.json')}
    inputs['prep_smiles'] = digest(cfg['prep_smiles'])
    if cfg.get('triton_configs'):
        inputs['triton_configs'] = digest(cfg['triton_configs'])
    from suite_provenance import source_records
    body = dict(schema=2, source_records=source_records(cfg), config=cfg, engine_commit=engine_commit(cfg['engine']),
                runner_sha256=source_hash(), targets=targets, enrichment_manifests=manifests,
                input_manifests=inputs, timing=dict(warmups=1,timed_runs=1), tasks=tasks(targets))
    body['plan_id'] = identity(body)
    out = Path(out)
    if out.exists() and json.loads(out.read_text()) != body:
        raise ValueError('existing plan differs; use a new run directory')
    write(out, body)
    return body


def verify_plan(doc):
    if engine_commit(doc['config']['engine']) != doc['engine_commit']:
        raise ValueError('engine changed after planning')
    if source_hash() != doc['runner_sha256']:
        raise ValueError('benchmark code changed after planning')
    cfg = doc['config']
    for record in doc['source_records']:
        if digest(record['path']) != record['sha256']:
            raise ValueError('source record changed: '+record['path'])
    for target, expected in doc['enrichment_manifests'].items():
        if digest(Path(cfg['store'])/target/'manifest.json') != expected:
            raise ValueError(f'{target}: input manifest changed')
    for name, expected in doc['input_manifests'].items():
        path = cfg[name] if name in ('prep_smiles', 'triton_configs') else Path(cfg['storage'])/name
        if digest(path) != expected: raise ValueError(f'{name}: input changed')


def completed(path, doc, task):
    if not path.exists(): return False
    d = json.loads(path.read_text())
    if d.get('plan_id') != doc['plan_id'] or d.get('task') != task or not d.get('artifacts'):
        raise ValueError(f'{path}: incompatible or incomplete result')
    for name, sha in d['artifacts'].items():
        if digest(path.parent.parent/name) != sha:
            raise ValueError(f'{path}: result artifact changed: {name}')
    return True


def run(plan_path, groups=None, task_id=None, dry_run=False):
    plan_path = Path(plan_path).resolve()
    doc = json.loads(plan_path.read_text()); verify_plan(doc)
    root = plan_path.parent; cfg = doc['config']
    if task_id and task_id not in {t['id'] for t in doc['tasks']}:raise ValueError('unknown task ID')
    env = os.environ.copy()
    env.update({k:'1' for k in THREADS})
    env.update(FSS_REPO=cfg['engine'], FIG2_FSS_COMMIT=doc['engine_commit'], FSS_CPU_SOA='1',
               FSS_BENCH_CHARGES='mmff',FSS_CANONICAL_STORE='1',FSS_CANONICAL_MODES='all',
               FIG2_EXTRA_MODES=','.join(MODES))
    # A caller's shell must not silently change the planned launch configurations.
    env.pop('FSS_TRITON_CONFIGS', None)
    if cfg.get('triton_configs'):
        env['FSS_TRITON_CONFIGS'] = cfg['triton_configs']
    env['PYTHONPATH'] = os.pathsep.join([cfg['engine'], str(ROOT/'benchmarks'),
                                       str(ROOT/'benchmarks/throughput'), env.get('PYTHONPATH','')])
    for task in doc['tasks']:
        if groups and task['group'] not in groups: continue
        if task_id and task['id'] != task_id: continue
        receipt = root/'receipts'/f"{task['id']}.json"
        if completed(receipt,doc,task):
            print('verified',task['id'],flush=True);continue
        for dep in task.get('requires',[]):
            parent=next(t for t in doc['tasks'] if t['id']==dep)
            if not dry_run and not completed(root/'receipts'/f'{dep}.json',doc,parent):
                raise ValueError(f"run prerequisite task {dep} first")
        interpreter = cfg['roshambo_python'] if task['group']=='roshambo' else sys.executable
        cmd=[interpreter,str(ROOT/'benchmarks/suite_worker.py'),'--plan',str(plan_path),'--task',task['id']]
        print(subprocess.list2cmdline(cmd),flush=True)
        if dry_run: continue
        lock=root/'locks'/task['id'];lock.parent.mkdir(parents=True,exist_ok=True)
        lock.mkdir()  # Atomic exclusion, including concurrent Slurm allocations.
        try:
            if not completed(receipt,doc,task):
                subprocess.run(cmd,env=env,cwd=ROOT,check=True)
                if not completed(receipt,doc,task): raise RuntimeError('worker returned without valid result')
        finally:
            lock.rmdir()
