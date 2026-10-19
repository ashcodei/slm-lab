"""Exact inactive private-capacity archive while keeping a40GiBwriter floor.

The original protected experiment remains failed and terminal. Only its one
unbound two-update capacity snapshot may be retired, after a complete physical
restore and independent full-file hash. Its original initial native stays raw.
"""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

from archive_inactive_code_capacity_CPU_v1 import metadata_bindings,no_readers_or_anchors
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.local_source_closure_v2 import python_sources

ROOT=Path(__file__).resolve().parents[1]
OWNER=dict(pid=3078946,ticks='37395463')
MAXIMUM=17*1024**3//2
RESERVE=40*1024**3


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)!=19
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Only a CUDA-hidden nice19 single-thread inactive-artifact archiver is eligible')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0))
    package=Path('/opt/conda/lib/python3.11/site-packages/zstandard')
    codec_files=[package/'__init__.py',package/'backend_c.cpython-311-x86_64-linux-gnu.so']
    original_codec=read(ROOT/'reports/inactive-code-capacity-after1-exact-zstd-v1/plan.json')['source_files_sha256']
    if any(original_codec.get(str(p))!=sha(p) for p in codec_files):raise ValueError('Every exact physically tested installed codec byte must remain')
    if importlib.util.find_spec('zstandard') is None:
        spec=importlib.util.spec_from_file_location('zstandard',package/'__init__.py',submodule_search_locations=[str(package)])
        module=importlib.util.module_from_spec(spec);sys.modules['zstandard']=module;spec.loader.exec_module(module)
    from slm_lab.exact_checkpoint_raw_anchor_v4 import archive,atomic,state
    source=ROOT/'checkpoints/protected-joint-v1/capacity-resume-step2.pt';anchor=source.parent/'initial/model.pt'
    destination=source.with_name(source.name+'.exact-raw-XOR-v1')
    output=ROOT/'reports/inactive-protected-capacity-step2-exact-raw-XOR-v1'
    old_plan=ROOT/'reports/protected-joint-v1/plan.json';old_failure=old_plan.with_name('failure.json')
    prior=ROOT/'reports/inactive-code-final-resume-exact-raw-XOR-v1/complete.json';proof=read(prior)
    tests=ROOT/'reports/exact-checkpoint-raw-anchor-v4-tests.log'
    if (output.exists() or destination.exists() or not terminal(OWNER)
            or {k:read(old_plan)['owner'][k] for k in OWNER}!=OWNER or not old_failure.is_file()
            or proof.get('complete') is not True or not terminal(proof['owner'])
            or proof.get('actual_archive_manifest',{}).get('physical_restore_checked') is not True
            or source.is_symlink() or anchor.is_symlink() or source.stat().st_nlink!=1
            or source.stat().st_size!=11804390971 or anchor.stat().st_size!=3937741157
            or '4 passed' not in tests.read_text() or shutil.disk_usage(ROOT).free<RESERVE+MAXIMUM+1024**3
            or shutil.disk_usage('/tmp').free<RESERVE+source.stat().st_size):
        raise ValueError('Exact inactive failed-source bytes, prior full private restoration and both full physical budgets must fit')
    for path,digest in proof['source_files_sha256'].items():
        if path.endswith('.py') and sha(path)!=digest:raise ValueError('Every actual previously verified full private archive/restore source must remain exact')
    no_readers_or_anchors(source);mentions=metadata_bindings(source)
    paths=[old_plan,old_failure,prior,tests,*codec_files,
        *python_sources(ROOT,[Path(__file__),ROOT/'scripts/restore_exact_checkpoint_raw_anchor_v4.py'])]
    files={str(p):sha(p) for p in paths};signatures={p:state(p) for p in files}
    original_state,anchor_state=state(source),state(anchor)
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    output.mkdir();atomic(output/'owner-plan.json',dict(owner=owner,at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=files,source=str(source),retained_raw_anchor=str(anchor),retained_raw_anchor_must_not_be_retired=True,
        original_failed_experiment_owner=OWNER,original_failure_preserved_and_not_relabelled_success=True,
        actual_immutable_source_bindings=0,actual_live_readers=0,inspected_metadata=mentions,
        maximum_payload_bytes=MAXIMUM,minimum_original_writer_free_bytes=RESERVE,
        full_physical_restore_and_independent_hash_required_before_retirement=True,
        only_this_attempt_unadmitted_duplicate_may_be_discarded_after_exact_original_and_anchor_rehash=True,
        active_workers_and_shared_caches_unchanged=True,GPU_contexts=0,new_neural_updates=0,goal_achieved=False))
    atomic(output/'progress.json',dict(phase='hashing whole inactive capacity state and retained raw initial',original_retired=False))
    source_sha,anchor_sha=sha(source),sha(anchor)
    atomic(output/'plan.json',dict(owner=owner,source_files_sha256=files,source=str(source),original_sha256=source_sha,
        original_state=original_state,retained_raw_anchor=str(anchor),retained_raw_anchor_sha256=anchor_sha,
        retained_raw_anchor_state=anchor_state,archive=str(destination),maximum_payload_bytes=MAXIMUM,
        minimum_free_bytes=RESERVE,minimum_reclaimed_bytes=2*1024**3,goal_achieved=False))
    started=time.monotonic();last=[started]
    def guard():
        if (os.nice(0)!=19 or os.sched_getscheduler(0)!=os.SCHED_IDLE or not terminal(OWNER)
                or state(source)!=original_state or state(anchor)!=anchor_state or time.monotonic()-started>3*3600):
            raise ValueError('The exact inactive source/retained anchor and bounded idle archive context must remain')
        if any(state(p)!=expected for p,expected in signatures.items()):raise ValueError('An exact prior proof or archival runtime byte changed')
        if time.monotonic()-last[0]>=30:
            no_readers_or_anchors(source);metadata_bindings(source);last[0]=time.monotonic()
    def progress(row):
        atomic(output/'progress.json',dict(phase='encoding whole inactive protected capacity',**row,
            original_retired=False,all_original_bytes_retained=True))
    try:
        guard()
        with tempfile.TemporaryDirectory(prefix='slm-inactive-protected-capacity-',dir='/tmp') as scratch:
            result=archive(source,anchor,destination,Path(scratch),expected_sha256=source_sha,expected_anchor_sha256=anchor_sha,
                maximum_payload_bytes=MAXIMUM,minimum_free_bytes=RESERVE,minimum_reclaimed_bytes=2*1024**3,
                guard=guard,progress=progress,retire_original=True)
        if source.exists() or not result['original_retired'] or not result['physical_restore_checked']:
            raise ValueError('Only actual full physical restoration and independent whole-file hashes may retire the raw duplicate')
        atomic(output/'complete.json',dict(complete=True,owner=owner,at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files,actual_archive_manifest=result,actual_archive_manifest_sha256=sha(destination/'manifest.json'),
            actual_reclaimed_allocated_bytes=result['reclaimed_allocated_bytes_if_retired'],retained_raw_anchor=str(anchor),
            explicit_restore_command='.venv/bin/python scripts/restore_exact_checkpoint_raw_anchor_v4.py --archive '+str(destination)+' --output '+str(source),
            every_original_tensor_optimizer_RNG_and_metadata_byte_retained=True,original_failed_experiment_not_relabelled_success=True,
            actual_future_training_funding_still_requires_fresh_full_free_check=True,model_quality_gain=False,goal_achieved=False))
        print(json.dumps(dict(complete=True,actual_reclaimed_bytes=result['reclaimed_allocated_bytes_if_retired'])),flush=True)
    except BaseException as error:
        payload=destination/'payload.zst';discarded=None
        if (source.is_file() and state(source)==original_state and sha(source)==source_sha
                and state(anchor)==anchor_state and sha(anchor)==anchor_sha and payload.is_file()):
            discarded=dict(path=str(payload),bytes=payload.stat().st_size,sha256=sha(payload),
                only_this_new_unadmitted_duplicate_discarded=True,complete_original_and_anchor_still_exact=True);payload.unlink()
        atomic(output/'failure.json',dict(owner=owner,error=repr(error),traceback=traceback.format_exc(),original_retained=source.exists(),
            discarded_unadmitted_duplicate=discarded,all_original_and_partial_metadata_preserved=True,automatic_retry=False,goal_achieved=False));raise


if __name__=='__main__':main()
