#!/usr/bin/env python3
"""Plan, run and tabulate one benchmark suite."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    plan=sub.add_parser('plan');plan.add_argument('--config',type=Path,required=True)
    plan.add_argument('--out',type=Path,required=True)
    run=sub.add_parser('run');run.add_argument('--plan',type=Path,required=True)
    run.add_argument('--group',nargs='+',choices=['cpu','gpu','cpu64','gpu4','prep','roshambo'])
    run.add_argument('--task');run.add_argument('--dry-run',action='store_true')
    assemble=sub.add_parser('assemble');assemble.add_argument('--plan',type=Path,required=True)
    a=p.parse_args()
    if a.action=='plan':
        from suite import plan
        d=plan(a.config,a.out);print(f"{len(d['tasks'])} unique tasks; plan {d['plan_id']}")
    elif a.action=='run':
        from suite import run
        run(a.plan,a.group,a.task,a.dry_run)
    else:
        from suite_tables import assemble
        print(assemble(a.plan))


if __name__=='__main__':main()
