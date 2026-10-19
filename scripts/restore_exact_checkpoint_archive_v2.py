"""Restore either exact archival format to a fresh historical checkpoint path."""
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
    metadata = json.loads((args.archive / 'manifest.json').read_text())
    if metadata.get('format') == 'own-opaque-checkpoint-exact-zstd-archive-v2':
        from slm_lab.exact_checkpoint_zstd_archive_v2 import restore
    elif metadata.get('format') == 'own-opaque-checkpoint-exact-gzip-archive-v1':
        from slm_lab.exact_checkpoint_gzip_archive_v1 import restore
    else:
        raise ValueError('Unknown exact checkpoint archive format')
    print(json.dumps(restore(args.archive, args.output), indent=2))
