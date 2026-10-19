"""Resource accounting for report-only inference while checkpoint writers are absent."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
MINIMUM_FREE=16*1024**3
REPORT_BUDGET=400*1024**2
ENV='SLM_EXCLUSIVE_INFERENCE_LEASE'


def identity(pid):
    try:
        proc=Path('/proc')/str(pid);fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),ticks=fields[19],state=fields[0],
            argv=[a.decode() for a in (proc/'cmdline').read_bytes().split(b'\0') if a])
    except FileNotFoundError:return None


def checkpoint_writers(root=ROOT):
    root=Path(root);writers=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:info=identity(proc.name)
        except (PermissionError,ProcessLookupError):continue
        if info is None or info['state'] in {'Z','X'}:continue
        args=info['argv']
        if 'slm_lab.training' in args:
            writers.append(info);continue
        scripts=[Path(a) if Path(a).is_absolute() else root/a for a in args if a.endswith('.py')]
        scripts=[p for p in scripts if p.parent==root/'scripts' and p.name.startswith('train_')]
        if not scripts:continue
        if all(p.name=='train_human_feedback_pilot.py' for p in scripts):
            # This retained controller cannot begin until its exact predecessor
            # completes. The lease also binds that stopped predecessor's bytes.
            try:
                previous=Path(args[args.index('--preceding-status')+1]);previous=previous if previous.is_absolute() else root/previous
                expected=args[args.index('--preceding-completed-state')+1]
                destination=Path(args[args.index('--reports')+1]);destination=destination if destination.is_absolute() else root/destination
                waiting=json.loads((destination/'status.json').read_text())['status'].startswith('waiting')
                incomplete=json.loads(previous.read_text())['status']!=expected
                if waiting and incomplete:continue
            except (OSError,ValueError,KeyError):pass
        writers.append(info)
    return writers


def gpu_memory_for_pid(pid):
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
        '--format=csv,noheader,nounits'],text=True)
    return {r[1].strip():int(r[2].strip()) for r in csv.reader(rows.splitlines()) if int(r[0].strip())==pid}


def validate_active_lease(root=ROOT,*,commit_bytes=0,for_start=False):
    root=Path(root)
    if type(commit_bytes) is not int or commit_bytes<0:raise ValueError('Nonnegative exact commit size required')
    path=Path(os.environ.get(ENV,''))
    expected=root/'reports/code-objective-exclusive-assessment-v1/lease.json'
    if path!=expected:raise ValueError('Exact workspace exclusive-inference lease required')
    lease=json.loads(path.read_text());parent=identity(lease['parent_pid'])
    source=root/'scripts/run_exclusive_code_objective_assessments_v1.py'
    if (lease.get('format')!='slm-exclusive-inference-lease-v1' or parent is None
            or parent['ticks']!=str(lease['parent_start_ticks']) or parent['state'] in {'Z','X'}
            or str(source) not in parent['argv']
            or hashlib.sha256(source.read_bytes()).hexdigest()!=lease['parent_source_sha256']):
        raise ValueError('Exact live exclusive-inference owner required; no stale/PID-reused lease')
    for name,digest in lease['stopped_predecessor_status_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Original predecessor changed; inspect writer state before continuing')
    writers=checkpoint_writers(root)
    if writers:raise ValueError('Checkpoint writer/controller became live during exclusive inference')
    allocations=gpu_memory_for_pid(lease['parent_pid'])
    if any(allocations.get(uuid,0)<=128 for uuid in lease['reserved_GPU_UUIDs']):
        raise ValueError('Owner must retain actual GPU allocations blocking the idle-GPU resumer')
    minimum=MINIMUM_FREE+(REPORT_BUDGET if for_start else 0)
    if shutil.disk_usage(root).free-commit_bytes<minimum:
        raise ValueError('Exclusive inference report allocation would consume its16GiB remaining reserve')
    return lease


def inference_reservation(root=ROOT):
    lease=validate_active_lease(root,for_start=True)
    return dict(credited_bytes=0,minimum_reserve_bytes=MINIMUM_FREE,owner_pid=lease['parent_pid'],
        policy='Report-only exclusive inference: verified terminal checkpoint writers; live GPU owner blocks checked curriculum resumption.16GiB remaining free and400MiB all-report commitment; no occupied-file credit. Original25GiB curriculum and26GiB shared-inference drivers are unchanged.')


def retire_idle_waiter(receipt,status_path,not_started_paths):
    """Stop only an exact own idle actor, after freezing and rechecking its state."""
    current=identity(receipt['pid'])
    if current is None or current['state'] in {'Z','X'}:return dict(retired=False,already_terminal=True)
    if current['ticks']!=str(receipt['start_ticks']) or current['argv']!=receipt['argv']:
        raise ValueError('Never signal a reused PID or a different process')
    def checked_signal(value):
        observed=identity(receipt['pid'])
        if observed is None or observed['state'] in {'Z','X'}:return False
        if observed['ticks']!=str(receipt['start_ticks']) or observed['argv']!=receipt['argv']:
            raise ValueError('Signal target identity changed; never signal the replacement')
        os.kill(receipt['pid'],value)
        return True
    stopped=False
    try:
        checked_signal(signal.SIGSTOP);stopped=True
        deadline=time.monotonic()+3
        while True:
            observed=identity(receipt['pid'])
            if observed is None or observed['state'] in {'Z','X'}:return dict(retired=False,already_terminal=True)
            if observed['state'] in {'T','t'}:break
            if time.monotonic()>deadline:raise ValueError('Could not verify stopped idle actor; inspect this handle')
            time.sleep(.02)
        status=json.loads(Path(status_path).read_text())
        if not status['status'].startswith('waiting') or any(Path(p).exists() for p in not_started_paths):
            raise ValueError('Existing assessment has started; leave it running instead of replacing it')
        checked_signal(signal.SIGTERM);checked_signal(signal.SIGCONT);stopped=False
        deadline=time.monotonic()+5
        while True:
            observed=identity(receipt['pid'])
            if observed is None or observed['state'] in {'Z','X'}:break
            if time.monotonic()>deadline:raise ValueError('Idle actor termination not verified; do not launch duplicate work')
            time.sleep(.02)
        return dict(retired=True,pid=receipt['pid'],start_ticks=str(receipt['start_ticks']),
            previous_status=status,no_generations_started=True,signal='SIGTERM after SIGSTOP; PID/start-tick/argv verified before every signal')
    finally:
        if stopped:
            try:checked_signal(signal.SIGCONT)
            except ProcessLookupError:pass
