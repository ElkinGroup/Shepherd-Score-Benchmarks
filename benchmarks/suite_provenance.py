"""Observed environments and source records; retrieval dates are never inferred."""
import configparser
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from suite import ROOT, digest, identity, write


def command(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return dict(argv=argv, exit_code=p.returncode, output=(p.stdout+p.stderr).strip()[:12000])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(argv=argv, error=str(exc))


def snapshot(root, engine):
    from comparators import paths
    packages = []
    for dist in list(metadata.distributions()) + list(metadata.distributions(path=[str(paths.HOME/'python')])):
        packages.append(dict(name=dist.metadata.get('Name'), version=dist.version,
                             project_urls=dist.metadata.get_all('Project-URL') or [],
                             home_page=dist.metadata.get('Home-page'),
                             license=dist.metadata.get('License-Expression') or dist.metadata.get('License'),
                             direct_url=json.loads(dist.read_text('direct_url.json') or 'null')))
    lock = configparser.ConfigParser(); lock.read(paths.LOCK_PATH)
    artifacts = {'shaep': paths.shaep_bin(), 'align_it': paths.align_it_bin(),
                 'openchemlib': paths.phesa_jar(),
                 'phesa_screen': paths.phesa_classdir()/'PheSAScreen.class'}
    for name in ('xtb', 'obabel', 'java'):
        found = paths.java_bin() if name == 'java' else shutil.which(name)
        if found: artifacts[name] = Path(found)
    observed = []
    for name, path in artifacts.items():
        row = dict(name=name, path=str(path), available=path.is_file(),
                   declared=dict(lock[name]) if name in lock else {})
        if path.is_file():
            row['sha256'] = digest(path)
            if row['declared'].get('sha256'):
                row['matches_declared_sha256'] = row['sha256']==row['declared']['sha256']
        observed.append(row)
    conda = []
    conda_paths=set((Path(sys.prefix)/'conda-meta').glob('*.json'))
    conda_paths.update((paths.OPENBABEL_LIB.parent/'conda-meta').glob('openbabel-*.json'))
    for path in sorted(conda_paths):
        d = json.loads(path.read_text())
        conda.append(dict(prefix=str(path.parent.parent),**{k:d.get(k) for k in ('name','version','build','channel','url','sha256','md5')}))
    import importlib.util
    modules=[]
    for name in ('roshambo2','espsim'):
        spec=importlib.util.find_spec(name)
        module_path=Path(spec.origin).parent if spec and spec.origin else paths.HOME/'python'/name
        if not module_path.is_dir():continue
        files={p.relative_to(module_path).as_posix():digest(p) for p in sorted(module_path.rglob('*'))
               if p.is_file() and p.suffix in ('.py','.so','.pyd')}
        modules.append(dict(name=name,path=str(module_path),source_sha256=identity(files),
                            files=files,git_commit=command(['git','-C',str(module_path),'rev-parse','HEAD'])))
    driver=shutil.which('nvidia-smi')
    gpu_driver=command([driver,'--query-gpu=name,driver_version','--format=csv,noheader']) if driver else None
    body = dict(python=sys.version,modules=modules,gpu_driver=gpu_driver, executable=sys.executable, platform=platform.platform(),
                packages=sorted(packages,key=lambda d:(d['name'] or '',d['version'])),
                conda_packages=conda, artifacts=observed,
                declared_software={s:dict(lock[s]) for s in lock.sections()},
                lock_sha256=digest(paths.LOCK_PATH),
                engine=command(['git','-C',str(engine),'rev-parse','HEAD']),
                engine_source=command(['git','-C',str(engine),'remote','get-url','origin']))
    key = identity(body); path = Path(root)/'provenance'/f'{key}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    lock=path.with_suffix('.lock')
    if path.exists():return path
    import time
    deadline=time.monotonic()+60
    while True:
        try:
            lock.mkdir();break
        except FileExistsError:
            if path.exists():return path
            if time.monotonic()>deadline:raise RuntimeError('provenance snapshot lock is stale or busy')
            time.sleep(.1)
    try:
        if path.exists():return path
        probes = []
        for name,args in [('xtb',['--version']),('obabel',['-V']),('java',['-version'])]:
            executable = paths.java_bin() if name == 'java' else shutil.which(name)
            if executable:probes.append(command([str(executable),*args]))
        for name in ('shaep','align_it'):
            executable=artifacts[name]
            if executable.is_file():probes.append(command([str(executable),'--version']))
        write(path, dict(body, observed_at_utc=datetime.now(timezone.utc).isoformat(),
                         version_probes=probes,
                         note='Environment inventory, not a claim that every installed dependency was invoked. '
                              'Lock entries are declared sources/versions; artifact hashes identify installed files. '
                              'Observation time is not a download/access date.'))
    finally:
        lock.rmdir()
    return path


def source_records(cfg):
    """Keep external acquisition manifests alongside exact input identities."""
    records = []
    paths = [('enrichment_store',p) for p in sorted(Path(cfg['store']).glob('*/manifest.json'))]
    paths += [('throughput_inputs',Path(cfg['storage'])/p) for p in
              ('selections/manifest.json','pairwise/manifest.json')]
    paths += [('acquisition_record',Path(p)) for p in cfg.get('source_records',[])]
    for role,path in paths:
        records.append(dict(role=role,path=str(path),sha256=digest(path),
                            record=json.loads(path.read_text())))
    records.append(dict(role='preparation_smiles',path=cfg['prep_smiles'],
                        sha256=digest(cfg['prep_smiles']),
                        acquisition='See supplied acquisition records; unrecorded fields remain unknown.'))
    return records


def tables(root, doc, receipts, out, csv_write):
    records = {}
    for receipt in receipts:
        base = Path(receipt.get('_root', root))
        for name in receipt['artifacts']:
            if name.startswith('provenance/'):
                path = base/name
                records[name if base == Path(root) else str(path)] = json.loads(path.read_text())
    software = []
    for source,d in records.items():
        common = dict(environment=source, observed_at_utc=d['observed_at_utc'])
        software.append(dict(common, name='Python', version=d['python'],path=d['executable']))
        software.append(dict(common,name='FSS',commit=d['engine'].get('output'),source=d['engine_source'].get('output')))
        for probe in d['version_probes']:
            software.append(dict(common,kind='observed version command',name=probe['argv'][0],
                                 version_report=probe.get('output'),probe_error=probe.get('error'),exit_code=probe.get('exit_code')))
        if d.get('gpu_driver'):
            software.append(dict(common,name='NVIDIA GPU/driver',version_report=d['gpu_driver'].get('output'),
                                 probe_error=d['gpu_driver'].get('error'),exit_code=d['gpu_driver'].get('exit_code')))
        for module in d.get('modules',[]):
            software.append(dict(common,kind='observed module files',name=module['name'],path=module['path'],
                                 source_sha256=module['source_sha256'],
                                 commit=module['git_commit'].get('output') if module['git_commit'].get('exit_code')==0 else None))
        for p in d['packages']:
            software.append(dict(common,name=p['name'],version=p['version'],kind='Python distribution',license=p.get('license'),
                                 sources=json.dumps(dict(home=p['home_page'],urls=p['project_urls'],direct=p['direct_url']))))
        for p in d['conda_packages']:
            software.append(dict(common,kind='Conda package',**p))
        for a in d['artifacts']:
            software.append(dict(common,kind='executable or JAR',name=a['name'],path=a['path'],
                                 available=a['available'],sha256=a.get('sha256'),matches_declared_sha256=a.get('matches_declared_sha256'),
                                 declared_version=a['declared'].get('version'),
                                 declared_source=a['declared'].get('url',a['declared'].get('repo')),
                                 declared_commit=a['declared'].get('commit')))
        for name,p in d['declared_software'].items():
            software.append(dict(common,kind='declared software manifest',name=name,
                                 declared_version=p.get('version'),declared_commit=p.get('commit'),
                                 declared_source=p.get('url',p.get('repo',p.get('source'))),
                                 citation=p.get('citation'),license=p.get('license')))
    if software:csv_write(out/'software_provenance.csv',software)
    else:raise ValueError('no observed software provenance')
    inputs = doc['source_records']
    # Flatten nested acquisition records without assigning an invented source/date.
    rows = []
    keys = {'url','urls','source','repo','doi','citation','license','version','commit','sha256',
            'last_modified','etag','server_date','downloaded_at','download_time','downloaded_at_utc','retrieved_at','accessed_at','download_date'}
    def walk(value,location,base):
        if isinstance(value,dict):
            found={k:(json.dumps(v) if isinstance(v,(dict,list)) else v) for k,v in value.items()
                   if k in keys or 'download' in k or 'retriev' in k or 'access' in k}
            if found:rows.append(dict(base,record_location=location,**found))
            for k,v in value.items():walk(v,location+'/'+str(k),base)
        elif isinstance(value,list):
            for i,v in enumerate(value):walk(v,location+'/'+str(i),base)
    for item in inputs:
        base=dict(role=item['role'],manifest_path=item['path'],manifest_sha256=item['sha256'])
        rows.append(dict(base,record_location='/',note='Acquisition dates absent from source records are unknown.'))
        walk(item.get('record',{}),'',base)
    for source in doc.get('config',{}).get('dataset_sources',[]):
        rows.append(dict(role='declared dataset source',**source))
    csv_write(out/'data_provenance.csv',rows)
    write(out/'provenance.json',dict(software=records,data=inputs,
          declared_dataset_sources=doc.get('config',{}).get('dataset_sources',[]),engine_commit=doc['engine_commit']))
