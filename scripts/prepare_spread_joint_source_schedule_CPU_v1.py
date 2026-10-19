"""Bind the fully read-back original streams to a paced joint recipe only."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/complete-joint-source-schedule-CPU-v1'
READER=ROOT/'reports/complete-joint-source-schedule-readback-CPU-v1'
OUT=ROOT/'reports/spread-joint-source-schedule-CPU-v1'
RESERVE=40*1024**3


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if len(raw)>16*1024**2 or shutil.disk_usage(ROOT).free-len(raw)<RESERVE:raise ValueError('Complete paced metadata must fit above40GiB')
    pending=Path(path).with_name(Path(path).name+'.pending')
    with pending.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or shutil.disk_usage(ROOT).free<RESERVE+32*1024**2):
        raise ValueError('Fresh CUDA-hidden one-thread nice19 complete metadata preparation required')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0))
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.complete_joint_source_schedule_v1 import STREAMS,digest,require_original_scope
    from slm_lab.spread_joint_source_schedule_v1 import spread
    closed=read(READER/'complete.json');reader_owner=read(READER/'plan.json')['owner']
    if (not terminal(reader_owner) or (READER/'failure.json').exists() or not closed['complete']
            or closed['source_joint_sha256']!=sha(SOURCE/'joint.json') or closed['actual_complete_joint_steps']!=5362):
        raise ValueError('The complete exact original source recipe and its terminal independent reader are required')
    if '3 passed' not in (ROOT/'reports/spread-joint-source-schedule-v1-tests.log').read_text():raise ValueError('The independent spacing oracle must pass')
    files=dict(closed['source_files_sha256'])
    entrypoints=[Path(__file__),ROOT/'tests/test_spread_joint_source_schedule_v1.py']
    for path in [*python_sources(ROOT,entrypoints),READER/'complete.json',READER/'plan.json',ROOT/'reports/spread-joint-source-schedule-v1-tests.log']:
        if str(path) in files and files[str(path)]!=sha(path):raise ValueError('Immutable source closures disagree')
        files[str(path)]=sha(path)
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    OUT.mkdir();write(OUT/'plan.json',dict(owner=owner,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        original_complete_source_reader=reader_owner,models_constructed=0,GPU_contexts=0,goal_achieved=False))
    try:
        for path,expected in files.items():
            if sha(path)!=expected:raise ValueError('A complete original source binding changed')
        original=read(SOURCE/'joint.json');require_original_scope(original)
        streams={name:read(SOURCE/(name+'.json')) for name in STREAMS};paced=spread(streams)
        if paced['actual_joint_updates']!=5362 or paced['original_stream_budgets']!=original['original_stream_budgets']:
            raise ValueError('Pacing cannot change any original source population, target budget or complete horizon')
        # Independently recover each referenced step without using the
        # producer's position formula. Every row, hash and order is checked.
        for name in STREAMS:
            selected=[(row['step'],part) for row in paced['entries'] for part in row['components'] if part['stream']==name]
            expected=streams[name]['entries']
            if ([part['source_step'] for _,part in selected]!=list(range(1,len(expected)+1))
                    or any(part['original_entry_sha256']!=digest(row) for (_,part),row in zip(selected,expected))
                    or selected[0][0]!=1 or selected[-1][0]!=5362):raise ValueError('Every full original component must remain ordered once, including both endpoints')
        if (paced['actual_spaced_gaps']['code']['longest_consecutive_steps_without_task']!=1
                or paced['actual_spaced_gaps']['visual']['longest_consecutive_steps_without_task']!=1
                or paced['actual_spaced_gaps']['human']['longest_consecutive_steps_without_task']!=0):
            raise ValueError('The full5362step schedule must retain the declared complete task-spacing bound')
        paced['component_files']=original['component_files'];paced['original_prefix_join_file_sha256']=sha(SOURCE/'joint.json')
        write(OUT/'joint.json',paced)
        if 'torch' in sys.modules:raise ValueError('A pacing-only metadata process must not import the neural runtime')
        for path,expected in files.items():
            if sha(path)!=expected:raise ValueError('Every original complete source must remain exact')
        write(OUT/'complete.json',dict(complete=True,owner=owner,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            actual_joint_updates=5362,original_stream_budgets=paced['original_stream_budgets'],actual_task_combinations=paced['task_combinations'],
            original_prefix_gaps=paced['original_prefix_gaps'],actual_spaced_gaps=paced['actual_spaced_gaps'],joint_sha256=sha(OUT/'joint.json'),
            every_complete_original_stream_step_independently_referenced_in_order_once=True,
            production_training_admitted=False,models_constructed=0,neural_forwards=0,optimizer_updates=0,
            model_retention_or_quality_gain_unmeasured=True,goal_achieved=False))
    except BaseException as exc:
        write(OUT/'failure.json',dict(owner=owner,error=repr(exc),automatic_retry=False,goal_achieved=False));raise


if __name__=='__main__':main()
