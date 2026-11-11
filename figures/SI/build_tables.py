"""Render the assembled campaign as SI tables without changing manuscript prose."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'benchmarks/publication_data'
OUT = Path(__file__).resolve().parent / 'tables'
OMITTED_ENRICHMENT = {'espsim_shape', 'espsim_combo'}
MANIFEST = r'$^{\dagger}$'      # version taken from the release manifest, not probed
PINNED = r'--$^{\ddagger}$'     # documented source; retrieved files pinned by hash instead
# Significance markers live in a fixed-width box after the number so they cannot shift the digits
# of a right-aligned column. Column headings carry the same box, which keeps them over the digits.
MARKPAD = r'\makebox[1.6em][l]{%s}'
# The provenance record names the package by its internal name; the manuscript calls it \sss{}.
DISPLAY_NAME = {'FSS': r'\sss{}'}


def rows(name):
    with (DATA / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def esc(s):
    return str(s).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('#', r'\#')


def num(v, digits=3):
    return '--' if v in ('', None) else f'{float(v):.{digits}f}'


def table(name, headers, records, spec=None):
    OUT.mkdir(exist_ok=True)
    text = '\\begin{tabular}{' + (spec or 'l'+'r'*(len(headers)-1)) + '}\n\\toprule\n'
    text += ' & '.join(headers) + r' \\' + '\n\\midrule\n'
    text += '\n'.join(r if isinstance(r, str) else ' & '.join(map(str, r)) + r' \\' for r in records)
    text += '\n\\bottomrule\n\\end{tabular}\n'
    (OUT / (name+'.tex')).write_text(text, encoding='utf-8')


def main():
    summary = [r for r in rows('table1_enrichment.csv')
               if r['method'] not in OMITTED_ENRICHMENT]
    tests={(r['method'],r['metric']):r for r in rows('enrichment_paired_tests.csv')}
    def meansem(r, key, digits=3):
        test=tests.get((r['method'],key))
        mark=''
        if test:
            p=float(test['p'])
            mark='$^{'+('**' if p<.005 else '*' if p<.05 else r'\mathrm{n.s.}')+'}$'
        return '$'+num(r[key+'_mean'],digits)+r'\pm '+num(r[key+'_sem'],digits)+'$'+MARKPAD%mark
    records=[]
    for i,r in enumerate(summary):
        if i in (0,18):
            if i: records.append(r'\midrule')
            records.append(r'\multicolumn{7}{l}{\textbf{'+(r'\sss{}' if i==0 else 'External implementations')+r'}} \\')
        reference=tests.get((r['method'],'auc'),{}).get('reference','--')
        records.append([esc(r['method']),esc(reference),meansem(r,'auc'),meansem(r,'ef1',2),meansem(r,'hit_rate'),
                        num(r['tanimoto_auc']),
                        f"${float(r['delta_auc']):+.3f}$" if r['delta_auc'] else '--'])
    table('enrichment', ['Method','Reference']+[h+MARKPAD%'' for h in ('ROC-AUC','EF1\\%','Hit rate')]
          +['Tanimoto ROC-AUC','$\\Delta$'],records,'llrrrrr')
    for source, name in [('table2_fss_ef1.csv','fss_ef1'),('table3_comparator_ef1.csv','comparator_ef1')]:
        data=rows(source);keys=[k for k in data[0] if k not in OMITTED_ENRICHMENT]
        table(name,['Target']+[r'\makebox[2.6em][l]{\rotatebox[origin=l]{65}{'+esc(k)+'}}' for k in keys[1:]],
              [[r['target']]+[num(r[k],1) for k in keys[1:]] for r in data],
              'l'+'p{2.6em}'*(len(keys)-1))
    sizes=rows('table4_benchmark_sizes.csv'); groups={}
    for r in sizes:
        if r['workload'] in ('enrichment','preparation'):continue
        key=(r['tool'],r['backend'],r['workload'],r['workers'],r['devices'])
        groups.setdefault(key,set()).add(int(r['n']))
    size_rows=[]
    for (tool,backend,workload,workers,devices),ns in sorted(groups.items()):
        size_rows.append([esc(tool),esc(backend),esc(workload),workers,devices if backend in ('triton','cuda') else '0',', '.join(f'{n:,}' for n in sorted(ns))])
    table('sizes',['Tool','Backend','Workload','Workers','GPUs','$N$'],size_rows,'lllllp{6cm}')
    table('preparation',['Configuration','Shared (s)','Conversion (s)','Total (s)'],
          [[esc(r['configuration'])]+[num(r[k]) for k in ['shared_seconds','conversion_seconds','total_seconds']]
           for r in rows('table5_preparation.csv')])
    table('index',['Mode','Conformers','Construction (s)','Storage (MB)'],
          [[esc(r['mode']),r['n'],num(r['seconds']),num(int(r['store_bytes'])/1e6,2)] for r in rows('table5_index_construction.csv')])
    data=rows('table6_throughput_wide.csv')
    def rate(r, alternatives):
        value=next((r.get(k) for k in alternatives if r.get(k)),None)
        return '--' if value is None else f'{float(value):,.1f}'
    columns=[['numba_screen_1000','cpu_screen_1000'],['numba_pairwise_1000'],['triton_screen_100000','cuda_screen_100000'],
             ['triton_pairwise_100000','cuda_pairwise_100000']]
    names={'fss':r'\sss{}','rdkit':'RDKit','roshambo2':'ROSHAMBO2','phesa':'PheSA','shaep':'ShaEP','pharao':'Pharao','espsim':'ESP-Sim'}
    headers=[r'\shortstack{'+label+r'\\$N='+n+r'$}' for label,n in
             [('CPU screen','10^3'),('CPU pairs','10^3'),('GPU screen','10^5'),('GPU pairs','10^5')]]
    table('throughput',['Tool / mode']+headers,
          [[names[r['tool']]+' / '+esc(r['mode'])]+[rate(r,c) for c in columns] for r in data])
    charge=rows('enrichment_charge_comparison.csv')
    metrics=('auc','ef1','hit_rate')
    grouped={}
    for r in charge:
        grouped.setdefault(r['mode'],{})[r['metric']]=r
    records=[]
    for mode, values in grouped.items():
        if records: records.append(r'\midrule')
        for j,quantity in enumerate(('xTB','MMFF94',r'$\Delta$',r'Paired $p$')):
            cells=[]
            for metric in metrics:
                r=values[metric]
                if j<2:
                    prefix=('xtb','mmff')[j]
                    cells.append('$'+num(r[prefix+'_mean'])+r'\pm '+num(r[prefix+'_sem'])+'$')
                elif j==2:
                    cells.append('$'+f"{float(r['delta_mmff_minus_xtb']):+.3f}"+r'\pm '+num(r['delta_sem'])+'$')
                else:
                    cells.append(f"{float(r['p_unadjusted']):.3g}")
            records.append([esc(mode) if j==0 else '',quantity]+cells)
    table('charge',['Mode','Quantity',r'\multicolumn{3}{c}{Enrichment metric} \\'+
                   '\n'+r'\cmidrule(lr){3-5} & & ROC-AUC & EF1\% & Hit rate'],records,'llrrr')
    sw=rows('software_provenance.csv')
    wanted=['FSS','Python','rdkit','roshambo2','shaep','align_it','phesa','espsim','xtb','torch','triton','numba','numpy','scipy','open3d','openchemlib']
    def env_of(r):
        prefix=r['prefix'] or r['path'] or ''
        return prefix.split('/envs/')[-1].split('/')[0] if '/envs/' in prefix else ''
    def homepage(r):
        """Project URL recorded for a pip distribution that carries no acquisition URL."""
        try:meta=json.loads(r['sources'] or '{}')
        except ValueError:return ''
        return meta.get('home') or next((u.split(', ')[-1] for u in (meta.get('urls') or [])
                                         if u.startswith(('Homepage','Repository','Source'))),'')
    def canonical(v):
        return '.'.join(str(int(p)) if p.isdigit() else p for p in v.split('.'))
    records=[]
    for name in wanted:
        rs=[r for r in sw if r['name'].lower()==name.lower()]
        if not rs:continue
        probed={}
        for r in rs:
            if not r['version']:continue
            shown=r['version'].split(' | ')[0]
            entry=probed.setdefault(canonical(shown),[shown,set()])
            entry[0]=max(entry[0],shown,key=len)   # keep the spelling the source URL uses
            entry[1].add(env_of(r))
        for shown,envs in probed.values():envs.discard('')
        from_manifest=False
        if probed:
            items=sorted(probed.items())
            version=', '.join(shown+(' ('+'/'.join(sorted(envs))+')' if envs and len(items)>1 else '')
                              for _key,(shown,envs) in items)
        else:
            version=', '.join(sorted({r['declared_version'] for r in rs if r['declared_version']}))
            from_manifest=bool(version)
        commits=sorted({r['commit'][:12] for r in rs if r['commit']})
        declared_commits=sorted({r['declared_commit'][:12] for r in rs if r['declared_commit']})
        if commits:
            version=(version+'; ' if version else '')+', '.join(commits)
        elif declared_commits:
            version=(version+'; ' if version else '')+', '.join(declared_commits);from_manifest=True
        # Only a prebuilt download can be hash-checked; a declared git source is compiled locally,
        # so its source hash and the installed executable are not comparable.
        if (any(r['declared_source'] and not r['declared_source'].endswith('.git') for r in rs)
                and any(r['matches_declared_sha256']=='False' for r in rs)):
            version+=' (installed binary does not match the declared download)'
        sources=sorted({r['source'] or r['declared_source'] or r['url'] for r in rs
                        if r['source'] or r['declared_source'] or r['url']})
        src=sources[0] if sources else next((h for h in map(homepage,rs) if h),'')
        records.append([DISPLAY_NAME.get(name,esc(name)),esc(version)+(MANIFEST if from_manifest else ''),
                        r'\url{'+src+'}' if src else '--'])
    table('software',['Software','Version / commit','Source'],records,'lp{4.5cm}p{5cm}')
    dr=rows('data_provenance.csv');sources={}
    for r in dr:
        u=r['url']
        if not u:continue
        if 'property_matched/' in u:u=u.split('property_matched/')[0]+'property_matched/'
        elif 'DUDE-Z-benchmark-grids/' in u:u=u.split('DUDE-Z-benchmark-grids/')[0]+'DUDE-Z-benchmark-grids/'
        sources.setdefault(u,set()).add(r['downloaded_utc'][:10] or PINNED)
    table('datasets',['Source','Download date (UTC)'],
          [[r'\url{'+u+'}',', '.join(sorted(ds))] for u,ds in sorted(sources.items())],'p{11cm}l')
    print('Rendered SI tables from',DATA)


if __name__=='__main__':main()
