"""Restore an exact archived historical checkpoint to a fresh path."""
import argparse
import json
import os
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.nice(max(0, 19 - os.nice(0)))
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from slm_lab.exact_checkpoint_gzip_archive_v1 import restore
    print(json.dumps(restore(args.archive, args.output), indent=2))
