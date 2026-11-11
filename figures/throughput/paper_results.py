"""Figure series selected from the assembled, verified publication campaign."""
import json
from pathlib import Path
DATA = Path(__file__).resolve().parents[2] / 'benchmarks/publication_data'
PAPER = ('vol', 'vol_esp', 'surf', 'surf_esp', 'vol_and_surf_esp', 'pharm', 'vol_color', 'vol_lipo')
ADDITIONAL = ('vol_pharm', 'vol_atomtype', 'vol_mr', 'surf_tversky', 'surf_esp_tversky',
              'vol_and_surf_esp_tversky', 'vol_color_tversky', 'vol_lipo_tversky',
              'pharm_tversky', 'vol_fukui')
ALL = PAPER + ADDITIONAL


def document():
    return json.loads((DATA / 'measurements.json').read_text())

def select(use=None, **filters):
    doc = document()
    ids = set(doc['destinations'][use]) if use else None
    return [dict(r, source='measurements.json', cell_key=r['task_id'])
            for r in doc['throughput']
            if (ids is None or r['task_id'] in ids)
            and all(r.get(k) == v for k, v in filters.items())]

def gpu_ladder():
    return select('main-throughput-B', tool='fss', backend='triton')

def cpu_ladder():
    return [r for r in select('main-throughput-B', tool='fss', backend='numba')
            if r['mode'] not in ('surf', 'surf_esp', 'vol_and_surf_esp') or r['size'] <= 1000]

def gpu_screen():
    return [r for r in gpu_ladder() if r['size'] == 10_000_000]

def gpu_100m():
    return [r for r in gpu_ladder() if r['size'] == 100_000_000]

def cpu_comparison():
    rows = select('main-throughput-C', workload='screen', size=1000)
    return rows + [dict(r, mode='vol_and_surf_esp') for r in rows if r['tool']=='shaep']

def read_external_cpu():
    return [r for r in cpu_comparison() if r['tool']!='fss']

def screen_comparison():
    return select('main-throughput-C', workload='screen', size=10_000_000)

def pair_comparison():
    return select('main-throughput-C', workload='pairwise', size=100_000)

def pairwise_campaign(tool='fss'):
    return select('table6', tool=tool, workload='pairwise', size=100_000)

def pairwise():
    return pairwise_campaign()

def si_cpu():
    return select('table6', tool='fss', workload='screen', size=1000)

def si_cpu_pairwise():
    return select('table6', tool='fss', workload='pairwise', size=1000)

def si_screen():
    return select('table6', tool='fss', workload='screen', size=100_000)

def as_cells(rows):
    return {r['task_id']: r for r in rows}
