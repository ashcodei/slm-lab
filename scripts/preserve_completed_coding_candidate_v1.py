"""Physically preserve a completed research candidate on the persistent volume.

Only a completed epoch with actual cold/query receipts is eligible. Complete
file copies are checksum-verified above the existing48GiB free-space floor.
This is research evidence/model delivery, never quality-based promotion.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/tmp/slm-completed-public-coding-standalone-CPU-v1')
DEST=ROOT/'checkpoints/public-coding-research-v1'
GIB=1024**3


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def preserve(profile):
    source=SOURCE/profile;dest=DEST/profile
    if dest.exists():raise ValueError('Preserve every original published candidate; use a new destination for changes')
    receipt=json.loads((source/'complete.json').read_text());query=json.loads((source/'fresh-query.json').read_text())
    readback=json.loads((source/'complete-training-readback.json').read_text())
    if (receipt.get('complete') is not True or receipt.get('completed_updates')!=230
            or receipt.get('actual_registered_parameters')!=999706442
            or receipt.get('original_complete_epoch_reconciled') is not True
            or receipt.get('all_actual_values_and_full4096text448photo16wholeword_and_cached_actions_CPU_exact') is not True
            or receipt.get('candidate_promoted') is not False or receipt.get('goal_achieved') is not False
            or query.get('complete') is not True or query.get('blocked_ancestor_or_training_read_attempts')!=[]
            or query.get('network_attempts')!=[] or query.get('completed_epoch_updates')!=230
            or query['actual_fresh_query']['checkpoint_sha256']!=receipt['checkpoint_sha256']
            or readback.get('complete') is not True or readback.get('completed_updates')!=230):
        raise ValueError('Complete real source epoch, all native cold outputs and fresh standalone query are required')
    files={'model.pt':receipt['checkpoint_sha256'],'tokenizer.json':receipt['tokenizer_sha256']}
    for name in ['complete.json','fresh-query.json','complete-training-readback.json']:files[name]=sha(source/name)
    for name,expected in files.items():
        path=source/name
        if path.is_symlink() or not path.is_file() or sha(path)!=expected:
            raise ValueError('Every exact original complete package/evidence byte is required')
    total=sum((source/name).stat().st_size for name in files)
    if shutil.disk_usage(ROOT).free<48*GIB+total+16*1024**2:
        raise ValueError('The complete persistent copy must fit above the original48GiB writer floor')
    dest.mkdir(parents=True)
    try:
        for name,expected in files.items():
            path=dest/name;pending=path.with_suffix(path.suffix+'.pending')
            with (source/name).open('rb') as original,pending.open('xb') as copy:
                while True:
                    chunk=original.read(8*1024**2)
                    if not chunk:break
                    if shutil.disk_usage(ROOT).free<48*GIB+len(chunk):
                        raise ValueError('Stop only this unadmitted copy before crossing the persistent reserve')
                    copy.write(chunk)
                copy.flush();os.fsync(copy.fileno())
            if sha(pending)!=expected or sha(source/name)!=expected:
                raise ValueError('The entire physical persistent copy and unchanged original must match')
            os.replace(pending,path)
        fd=os.open(dest,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        artifact=dict(complete=True,at=datetime.now(timezone.utc).isoformat(),profile=profile,
            checkpoint=str(dest/'model.pt'),checkpoint_sha256=files['model.pt'],tokenizer_sha256=files['tokenizer.json'],
            actual_registered_parameters=999706442,completed_original_epoch_updates=230,
            total_preserved_bytes=total,files_sha256=files,physical_persistent_files_independently_rehashed=True,
            no_ancestor_or_scratch_file_needed_for_inference=True,source_inputs_untouched=True,
            query_module='slm_lab.own_coding_policy_standalone_v1',source_query_implementation_sha256=sha(ROOT/'src/slm_lab/own_coding_policy_standalone_v1.py'),
            matched_quality_and_actual_CUDA_BF16_cold_still_required=True,retained_as_research_candidate=True,
            candidate_promoted=False,goal_achieved=False)
        path=dest/'PRESERVED.json'
        with path.open('x') as stream:
            stream.write(json.dumps(artifact,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
        print(json.dumps(artifact,indent=2))
    except BaseException as error:
        (dest/'failure.json').write_text(json.dumps(dict(error=repr(error),original_inputs_retained=True,
            partial_own_copy_retained_for_inspection=True,candidate_promoted=False))+'\n')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',required=True,choices=['plain_tokens','eos_weight4'])
    preserve(parser.parse_args().profile)
