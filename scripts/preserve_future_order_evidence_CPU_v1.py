"""Preserve exact closed research evidence, without changing scratch inputs.

The bundle is not a serving model: it contains an interim four-rank checkpoint
and a two-source mechanism study. Original absolute paths remain provenance
values, with explicit aliases to the verified persistent byte copies.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from slm_lab.committed_snapshot_copy_v1 import copy_committed
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.local_source_closure_v2 import python_sources

ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=Path('/tmp/slm-broad-code-step512-snapshot-CPU-v1')
PROBE=Path('/tmp/slm-compact-future-order-full-CPU-v1')
DESTINATION=ROOT/'checkpoints/research-evidence-future-order-v1'
REPORT=ROOT/'reports/durable-future-order-evidence-v1'
FORMAT='own-durable-exact-research-evidence-copy-v1'
RESERVE=40*1024**3
MAXIMUM=2*1024**3


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())


def verify(folder):
    folder=Path(folder).resolve();manifest=read(folder/'manifest.json')
    if (manifest.get('format')!=FORMAT or manifest.get('complete') is not True
            or manifest.get('not_a_serving_or_production_checkpoint') is not True):
        raise ValueError('An explicitly research-only complete exact-byte bundle is required')
    seen=set();total=0
    for row in manifest['artifacts']:
        relative=Path(row['relative_path']);target=folder/relative
        if (relative.is_absolute() or '..' in relative.parts or relative in seen or target.is_symlink()
                or not target.resolve().is_relative_to(folder) or not target.is_file()
                or target.stat().st_size!=row['bytes'] or sha(target)!=row['sha256']):
            raise ValueError('Every unique persistent copy must retain its exact complete admitted bytes')
        seen.add(relative);total+=row['bytes']
    if total!=manifest['actual_copied_bytes'] or not 0<total<=MAXIMUM:
        raise ValueError('The entire declared bounded evidence population must reconcile')
    return dict(verified=True,artifacts=len(seen),actual_bytes=total,not_a_serving_checkpoint=True,
                model_quality_gain_proved=False,goal_achieved=False)


def publish():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)!=19 or 'torch' in sys.modules
            or any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))):
        raise ValueError('A CUDA-hidden nice19 single-thread non-neural exact publisher is required')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0))
    review_path=ROOT/'reports/completed-future-order-probe-and-fast-archive-v1-review/review.json'
    comparison_path=ROOT/'reports/compact-future-order-full-comparison-v1/complete.json'
    review=read(review_path);comparison=read(comparison_path)
    if review['source_files_sha256'].get(str(comparison_path))!=sha(comparison_path):
        raise ValueError('The original independently closed paired-study comparison must remain exact')
    expected=dict(comparison['source_files_sha256']);producer=read(PROBE/'complete.json');snapshot=read(SNAPSHOT/'complete.json')
    who=producer['owner'];capture=snapshot['owner']
    if (DESTINATION.exists() or REPORT.exists() or not producer['complete'] or producer['total_actual_CPU_Adam_updates']!=4
            or not comparison['experimental_mechanism_passes'] or not comparison['efficacy_unproven']
            or not terminal(dict(pid=who['pid'],ticks=str(who.get('birth',who.get('ticks')))))
            or not terminal(dict(pid=capture['pid'],ticks=str(capture.get('birth',capture.get('ticks')))))
            or snapshot['completed']!=512 or not snapshot['source_checkpoint_is_interim_not_completed_epoch']
            or DESTINATION.parent.resolve()!=DESTINATION.parent or DESTINATION.parent.stat().st_dev!=ROOT.stat().st_dev
            or shutil.disk_usage(ROOT).free<RESERVE+MAXIMUM):
        raise ValueError('Fresh funded destinations and exact terminal closed producers are required; active consumers may keep reading')
    inputs={str(p):sha(p) for p in python_sources(ROOT,[Path(__file__)])}
    for p in (review_path,comparison_path,PROBE/'complete.json',SNAPSHOT/'complete.json'):
        digest=sha(p)
        if str(p) in expected and expected[str(p)]!=digest:raise ValueError('A sealed original evidence receipt differs')
        inputs[str(p)]=digest
    artifacts=[]
    def admit(source,relative,digest):
        source=Path(source);artifacts.append(dict(original_path=str(source),relative_path=relative,sha256=digest,bytes=source.stat().st_size))
    for path in sorted(SNAPSHOT.iterdir()):
        digest=(expected.get(str(path)) if path.name=='complete.json' else snapshot['source_files_sha256'].get(str(path)))
        if not path.is_file() or digest is None:raise ValueError('Every original immutable snapshot artifact requires its external committed SHA')
        admit(path,'snapshot/'+path.name,digest)
    probe_expected={name:expected[str(PROBE/name)] for name in ('owner-plan.json','actual-full-size-admission.json','complete.json')}
    for arm,row in producer['arms'].items():
        checkpoint=Path(row['complete_control_checkpoint']);probe_expected[checkpoint.name]=row['complete_control_checkpoint_sha256']
        for record in row['records']:
            raw=(json.dumps(record,indent=2,allow_nan=False)+'\n').encode()
            probe_expected[f'{arm}-step-{record["step"]}.json']=hashlib.sha256(raw).hexdigest()
    if set(probe_expected)!={p.name for p in PROBE.iterdir() if p.is_file()}:
        raise ValueError('Every original closed-study file must have an authoritative whole-byte commitment')
    for name,digest in sorted(probe_expected.items()):admit(PROBE/name,'probe/'+name,digest)
    for path in (comparison_path,ROOT/'reports/compact-future-order-full-comparison-v1/compare.py',
                 ROOT/'reports/compact-future-order-selected-readback-v1/read_arm.py',
                 ROOT/'reports/compact-future-order-selected-readback-v1/CE_only-readback.json',
                 ROOT/'reports/compact-future-order-selected-readback-v1/CE_plus_future_order-readback.json'):
        digest=sha(path) if path==comparison_path else expected[str(path)]
        admit(path,'evidence/'+path.name,digest)
    placement=ROOT/'reports/owned-CPU-affinity-placement-v1/complete.json'
    placement_review=read(ROOT/'reports/first-matched-CPU-delivery-and-placement-v1-review/review.json')
    admit(placement,'evidence/CPU-placement-complete.json',placement_review['source_files_sha256'][str(placement)])
    total=sum(r['bytes'] for r in artifacts)
    if total>MAXIMUM:raise ValueError('The full unshortened byte population must fit the declared2GiBcopy budget')
    def checked():
        if shutil.disk_usage(ROOT).free<RESERVE+1024**2 or any(sha(p)!=h for p,h in inputs.items()):
            raise ValueError('Every publisher/receipt byte and the unchanged40GiBwriter floor must remain')
    checked();DESTINATION.mkdir();REPORT.mkdir()
    write(REPORT/'owner-plan.json',dict(at=datetime.now(timezone.utc).isoformat(),source_files_sha256=inputs,
        actual_planned_bytes=total,artifacts=artifacts,persistent_destination=str(DESTINATION),minimum_writer_free_bytes=RESERVE,
        original_scratch_files_not_modified_or_removed=True,no_Torch_models_GPU_contexts_or_training=True,goal_achieved=False))
    try:
        copied=[]
        for row in artifacts:
            target=DESTINATION/row['relative_path'];target.parent.mkdir(parents=True,exist_ok=True)
            proof=copy_committed(row['original_path'],target,expected_sha256=row['sha256'],maximum_bytes=256*1024**2,checked=checked)
            if proof['captured_bytes']!=row['bytes']:raise ValueError('Every complete original artifact must retain its exact boundary')
            copied.append(proof)
        checked();write(DESTINATION/'manifest.json',dict(format=FORMAT,complete=True,at=datetime.now(timezone.utc).isoformat(),
            artifacts=artifacts,actual_copied_bytes=total,source_files_sha256=inputs,
            publication_git_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            original_absolute_paths_are_provenance_with_explicit_persistent_copy_aliases=True,
            original_scratch_files_not_changed=True,source_datasets_and_frozen_native_remain_required=True,
            interim_snapshot_updates=512,original_epoch_updates=3721,probe_train_sources_per_arm=2,probe_updates_per_arm=2,
            not_a_serving_or_production_checkpoint=True,no_model_quality_gain_claim=True,goal_achieved=False))
        result=verify(DESTINATION);bound=dict(inputs)
        bound.update({str(DESTINATION/r['relative_path']):r['sha256'] for r in artifacts});bound[str(DESTINATION/'manifest.json')]=sha(DESTINATION/'manifest.json')
        write(REPORT/'complete.json',dict(complete=True,source_files_sha256=bound,manifest=str(DESTINATION/'manifest.json'),
            verification=result,actual_artifact_copies=copied,original_scratch_inputs_and_live_workers_unchanged=True,
            standalone_model_or_quality_not_proved=True,goal_achieved=False));print(json.dumps(result),flush=True)
    except BaseException as error:
        write(REPORT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),original_inputs_unchanged=True,
            own_partial_copies_retained=True,no_automatic_retry=True,goal_achieved=False));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',type=Path,help='Independently hash an existing exact research bundle; no model loads or path restoration.')
    args=parser.parse_args()
    if args.verify:print(json.dumps(verify(args.verify)),flush=True)
    else:publish()


if __name__=='__main__':main()
