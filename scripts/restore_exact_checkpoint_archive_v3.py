"""Restore opaque V1/V2 or compressed-anchor XOR V3 to a fresh original path."""
import argparse
import json
import os
from pathlib import Path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scratch',type=Path,default=Path('/tmp'))
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.nice(max(0,19-os.nice(0)))
    metadata=json.loads((args.archive/'manifest.json').read_text())
    if metadata.get('format')=='own-opaque-checkpoint-tensor-aligned-XOR-zstd-exact-v3':
        from slm_lab.exact_checkpoint_xor_zstd_archive_v3 import restore
        result=restore(args.archive,args.output,args.scratch)
    elif metadata.get('format')=='own-opaque-checkpoint-exact-zstd-archive-v2':
        from slm_lab.exact_checkpoint_zstd_archive_v2 import restore
        result=restore(args.archive,args.output)
    elif metadata.get('format')=='own-opaque-checkpoint-exact-gzip-archive-v1':
        from slm_lab.exact_checkpoint_gzip_archive_v1 import restore
        result=restore(args.archive,args.output)
    else:raise ValueError('Unknown exact checkpoint archive format')
    print(json.dumps(result,indent=2))
