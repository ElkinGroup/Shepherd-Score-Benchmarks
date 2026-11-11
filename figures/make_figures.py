"""Regenerate the publication plots from verified archived results."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from publication import restore
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--only',choices=['all','enrichment','throughput'],default='all')
p.add_argument('--restore',action='store_true',help='restore archived inputs before plotting')
a=p.parse_args()
if a.restore:
    restore()
commands=[['figures/SI/build_tables.py']]
if a.only in ('all','enrichment'):
    commands += [['figures/SI/aggregate.py'],
                 ['figures/enrichment/make_figure.py']]
if a.only in ('all','throughput'):
    commands += [['figures/throughput/make_figure.py'],
                 ['figures/SI/multiproc/make_figure.py']]
for args in commands:
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
