"""Retain the complete pinned public CodeI/O raw function file, data only.

Publisher LFS bytes/hash and README Git blob are verified. Original code,
metadata and all cached IO remain untouched. Nothing is executed or admitted
to training; source roles, overlap and confined execution still need review.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/tmp/slm-CodeIO-raw-candidate-v1')
REPO='hkust-nlp/CodeIO-PyEdu-Reasoning-Raw'
REVISION='413a78caa01e634318ef8aa0e84f5ff242f25807'
NAME='0_368500_filtered_v2_ds25.sced.jsonl'
SIZE=988821246
SHA256='e6f0b1d1db181764a5d6ca6d2dcf8b641b63c632468672d1abfc24837700cca4'
GIB=1024**3


def write(name,value):
    path=OUT/name;pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(pending,path)


def main():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fresh CUDA-hidden source acquisition required')
    os.nice(10);os.sched_setaffinity(0,{22})
    if shutil.disk_usage(OUT.parent).free<SIZE+42*GIB:raise ValueError('The complete source file and original writer reserves must fit')
    OUT.mkdir()
    write('plan.json',dict(repo=REPO,revision=REVISION,filename=NAME,publisher_bytes=SIZE,publisher_sha256=SHA256,
        at=datetime.now(timezone.utc).isoformat(),code_execution_permitted=False,GPU_usage_permitted=False,
        source_only_no_training_admission=True,original_code_and_every_cached_IO_retained=True,
        data_is_public_synthetic_not_human_feedback=True,model_updates=0,goal_achieved=False))
    try:
        url='https://huggingface.co/api/datasets/'+REPO+'/revision/'+REVISION+'?blobs=true'
        with urllib.request.urlopen(url,timeout=30) as stream:raw=stream.read()
        metadata=json.loads(raw)
        if metadata['sha']!=REVISION or metadata.get('cardData',{}).get('license')!='odc-by':
            raise ValueError('The pinned publisher revision and declared dataset license must remain exact')
        entries={v['rfilename']:v for v in metadata['siblings']}
        if entries[NAME]['lfs']['sha256']!=SHA256 or entries[NAME]['lfs']['size']!=SIZE:
            raise ValueError('The complete original publisher LFS checksum and byte count are required')
        (OUT/'publisher-metadata.json').write_bytes(raw)
        base='https://huggingface.co/datasets/'+REPO+'/resolve/'+REVISION+'/'
        with urllib.request.urlopen(base+'README.md',timeout=30) as stream:readme=stream.read()
        blob=hashlib.sha1(b'blob '+str(len(readme)).encode()+b'\0'+readme).hexdigest()
        if blob!=entries['README.md']['blobId']:raise ValueError('The original publisher README Git blob differs')
        (OUT/'publisher-README.md').write_bytes(readme)
        path=OUT/NAME;digest=hashlib.sha256();total=0
        with urllib.request.urlopen(base+NAME+'?download=true',timeout=60) as remote,path.open('xb') as local:
            if remote.status!=200 or int(remote.headers.get('Content-Length','0'))!=SIZE:
                raise ValueError('A complete original full-file response is required, never a cached range prefix')
            while True:
                chunk=remote.read(8*1024**2)
                if not chunk:break
                total+=len(chunk)
                if total>SIZE or shutil.disk_usage(OUT).free<40*GIB:raise ValueError('Complete original file size/reserve exceeded')
                local.write(chunk);digest.update(chunk)
                if total%(64*1024**2)==0:write('progress.json',dict(phase='download',actual_bytes=total,expected_bytes=SIZE))
            local.flush();os.fsync(local.fileno())
        if total!=SIZE or digest.hexdigest()!=SHA256:raise ValueError('Every complete source byte must match the pinned publisher SHA')
        categories=Counter();sources=Counter();repositories=set();rows=0;with_ios=0;ios=0;missing=set()
        with path.open() as stream:
            for line in stream:
                row=json.loads(line);rows+=1
                required={'problem_description','io_requirements','refcode','funcname','ios','source','category','meta'}
                missing.update(required-set(row))
                if required-set(row):continue
                categories[row['category']]+=1;sources[row['source']]+=1
                if isinstance(row['meta'],dict) and isinstance(row['meta'].get('repo_name'),str):repositories.add(row['meta']['repo_name'])
                if isinstance(row['ios'],list):
                    ios+=len(row['ios']);with_ios+=bool(row['ios'])
                if rows%10000==0:write('progress.json',dict(phase='source_inventory',original_rows=rows))
        with path.open('rb') as stream:observed=hashlib.file_digest(stream,'sha256').hexdigest()
        if observed!=SHA256 or missing:raise ValueError('The unchanged full original source and complete publisher field inventory are required')
        write('complete.json',dict(complete=True,repo=REPO,revision=REVISION,original_source_file=str(path),
            original_source_bytes=total,original_source_sha256=observed,publisher_LFS_hash_matches=True,
            original_rows=rows,rows_with_cached_IO=with_ios,cached_IO_pairs=ios,distinct_origin_repositories=len(repositories),
            categories=dict(categories),sources=dict(sources),publisher_dataset_license='odc-by',
            original_repository_path_attribution_preserved=True,original_code_and_IO_bytes_untouched=True,
            roles_overlap_source_license_and_confined_execution_review_pending=True,
            publisher_notes_some_descriptions_may_be_incomplete=True,
            dataset_loader_or_source_code_executed=False,external_model_weights_or_live_teacher_API_used=False,
            data_admitted_to_training=False,human_feedback=False,RL_reward_oracle_admitted=False,
            model_quality_improvement_proved=False,new_model_updates=0,goal_achieved=False))
    except BaseException as error:
        write('failure.json',dict(error=repr(error),partial_source_retained=True,no_automatic_retry=True,data_admitted_to_training=False))
        raise


if __name__=='__main__':main()
