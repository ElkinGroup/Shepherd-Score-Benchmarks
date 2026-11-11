"""Restore the small, checksummed records used by the published plots."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def restore():
    with zipfile.ZipFile(ROOT/'results/published.zip') as archive:
        manifest = json.loads(archive.read('manifest.json'))
        for name, expected in manifest.items():
            # Archive member names retain their original paths and checksums.
            destination = name
            for old, new in (
                ('paper/fig2_speed/', 'benchmarks/throughput/'),
                ('paper/fig3_enrichment/data/', 'benchmarks/enrichment/data/'),
                ('paper/SI/accuracy/data/', 'benchmarks/accuracy/data/'),
            ):
                if name.startswith(old):
                    destination = new + name[len(old):]
                    break
            target = (ROOT/destination).resolve()
            if not target.is_relative_to(ROOT):
                raise ValueError(f'Unsafe archive path: {name}')
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f'Corrupt publication record: {name}')
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError(f'Publication record changed: {name}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
    data = ROOT/'benchmarks/publication_data'
    sources = json.loads((data/'sources.json').read_text())
    for name, expected in sources['files'].items():
        if hashlib.sha256((data/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Assembled table changed: {name}')
    plan = json.loads((data/'plan.json').read_text())
    if plan['plan_id'] != sources['plan_id']:
        raise ValueError('Publication plan differs from assembled results')
    for task in plan['tasks']:
        receipt = json.loads((data/'receipts'/f"{task['id']}.json").read_text())
        if receipt['plan_id'] != plan['plan_id']:
            raise ValueError('Receipt belongs to a different plan')
    return len(manifest)


if __name__ == '__main__':
    print('Verified', restore(), 'publication records')
