"""Execute one throughput task from a frozen manuscript benchmark plan."""
import argparse
from suite import run

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True)
    p.add_argument('--task',required=True)
    a=p.parse_args()
    run(a.plan,task_id=a.task)
