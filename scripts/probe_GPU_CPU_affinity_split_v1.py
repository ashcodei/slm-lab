"""Bounded original/split/restored CPU-evaluation affinity experiment.

Only exact own CPU evaluation processes and their descendants change affinity.
Their training is already complete. GPU workers, numerical settings, model
files and source bytes are untouched. A separate watchdog restores masks on
owner death, timeout or ordinary completion. This is not a quality experiment.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'reports/GPU-CPU-affinity-split-v1'
GPU_OWNERS=[dict(pid=p,birth=str(b)) for p,b in [(293164,60395445),(293165,60395446),(293166,60395446),(293167,60395446)]]
CPU_OWNERS=[dict(pid=18545,birth='50128787',name='full-function-repair-comparison-CPU-v1',script='train_full_function_repair_comparison_CPU_v1.py'),
    dict(pid=101564,birth='54147411',name='public-edge-preference-comparison-CPU-v2',script='train_public_edge_preferences_CPU_v2.py')]


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if fields[0] in ('Z','X'):return None
        return dict(pid=int(pid),birth=fields[19])
    except FileNotFoundError:return None


def same(actor):
    return identity(actor['pid'])=={k:actor[k] for k in ['pid','birth']}


def write(path,value):
    path=Path(path);pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('w') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(pending,path)


def descendants(actor):
    if not same(actor):return []
    seen={actor['pid']:dict(pid=actor['pid'],birth=actor['birth'])};pending=[actor['pid']]
    while pending:
        parent=pending.pop()
        for task in Path(f'/proc/{parent}/task').glob('*'):
            try:children=(task/'children').read_text().split()
            except FileNotFoundError:continue
            for child in children:
                actual=identity(int(child))
                if actual is not None and actual['pid'] not in seen:
                    seen[actual['pid']]=actual;pending.append(actual['pid'])
    return list(seen.values())


def threads(actor):
    result=[]
    for process in descendants(actor):
        if not same(process):continue
        for task in Path(f'/proc/{process["pid"]}/task').glob('*'):
            actual=identity(int(task.name))
            if actual is None:continue
            try:status=Path(f'/proc/{actual["pid"]}/status').read_text()
            except FileNotFoundError:continue
            uid=int(next(line for line in status.splitlines() if line.startswith('Uid:')).split()[1])
            if uid!=os.getuid():raise ValueError('Only this user exact own evaluation threads may change affinity')
            try:affinity=sorted(os.sched_getaffinity(actual['pid']))
            except ProcessLookupError:continue
            result.append(dict(**actual,process=process,affinity=affinity))
    return result


def apply_split(directory,plan):
    directory=Path(directory)
    with (directory/'restore.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (directory/'restored.json').exists():raise ValueError('A restored trial cannot be reapplied')
        path=directory/'affected.json';registered=json.loads(path.read_text()) if path.exists() else {}
        active=[]
        for actor in plan['targets']:
            if not same(actor):raise ValueError('An exact original CPU evaluation ended; restore before interpreting the trial')
            for task in threads(actor):
                key=str(task['pid'])+':'+task['birth'];current=task['affinity']
                if key not in registered:
                    if current not in (plan['original_cpus'],plan['restricted_cpus']):
                        raise ValueError('Unexpected private affinity; never broaden or replace it')
                    registered[key]=dict(pid=task['pid'],birth=task['birth'],original_affinity=plan['original_cpus'])
                active.append(task)
        # Publish restoration information before the first scheduler mutation.
        write(path,registered)
        for task in active:
            if same(task):
                try:os.sched_setaffinity(task['pid'],set(plan['restricted_cpus']))
                except ProcessLookupError:continue
        return len(active)


def restore(directory,plan,reason):
    directory=Path(directory)
    with (directory/'restore.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (directory/'restored.json').exists():return
        path=directory/'affected.json';registered=json.loads(path.read_text()) if path.exists() else {}
        # Include newly spawned descendants, which inherited the temporary
        # main-thread mask but may not have reached the next observation tick.
        for actor in plan['targets']:
            for task in threads(actor):
                key=str(task['pid'])+':'+task['birth']
                if key not in registered and task['affinity']==plan['restricted_cpus']:
                    registered[key]=dict(pid=task['pid'],birth=task['birth'],original_affinity=plan['original_cpus'])
        changed=[]
        for task in registered.values():
            if not same(task):continue
            try:
                os.sched_setaffinity(task['pid'],set(task['original_affinity']))
                assert sorted(os.sched_getaffinity(task['pid']))==task['original_affinity']
            except ProcessLookupError:continue
            changed.append(task)
        write(directory/'restored.json',dict(at=datetime.now(timezone.utc).isoformat(),reason=reason,
            restored_live_threads=changed,exact_original_affinities_restored=True,processes_signalled=0,model_updates=0))


def watchdog(directory):
    directory=Path(directory);plan=json.loads((directory/'owner-plan.json').read_text())
    while not (directory/'restored.json').exists():
        if not same(plan['owner']) or time.time()>plan['expires_at_unix'] or (directory/'restore-requested').exists():
            restore(directory,plan,'Owner ended, requested restoration or bounded lifetime expired');return
        time.sleep(.2)


def sample(plan):
    ranks=[]
    for actor in GPU_OWNERS:
        if not same(actor):raise ValueError('Original GPU worker changed; inspect the real handoff without restarting it')
        path=Path(f'/proc/{actor["pid"]}')
        runtime,wait,slices=map(int,(path/'schedstat').read_text().split())
        ranks.append(dict(**actor,runtime_ns=runtime,eligible_wait_ns=wait,timeslices=slices,
            affinity=sorted(os.sched_getaffinity(actor['pid'])),nice=os.getpriority(os.PRIO_PROCESS,actor['pid']),
            scheduler=os.sched_getscheduler(actor['pid'])))
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
    gpu=[dict(index=int(line.split(',')[0]),utilization_percent=float(line.split(',')[1]),memory_MiB=float(line.split(',')[2])) for line in raw.splitlines()]
    if len(gpu)!=4:raise ValueError('All four original NVML observations required')
    return dict(monotonic=time.monotonic(),GPU_workers=ranks,GPUs=gpu,
        CPU_evaluations=[dict(**{k:a[k] for k in ['pid','birth']},current_progress=json.loads((ROOT/'reports'/a['name']/'progress.json').read_text())) for a in CPU_OWNERS])


def main(directory):
    directory=Path(directory).resolve()
    original=sorted(os.sched_getaffinity(0));restricted=[c for c in original if c not in (5,20)]
    if (directory.exists() or original!=[5,7,9,11,13,15,20,22] or len(restricted)!=6
            or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'torch' in sys.modules
            or os.nice(0)!=0 or os.sched_getscheduler(0)!=os.SCHED_OTHER):
        raise ValueError('Fresh normal CUDA-hidden own trial within the actual original eight-core allocation required')
    files={str(Path(__file__).resolve()):sha(__file__)}
    for actor in CPU_OWNERS:
        if not same(actor) or actor['script'] not in Path(f'/proc/{actor["pid"]}/cmdline').read_bytes().decode():
            raise ValueError('Only the exact already-trained own CPU evaluation owners may participate')
        report=ROOT/'reports'/actor['name'];done=report/'all-arms-training-complete.json'
        if json.loads(done.read_text()).get('complete') is not True or json.loads((report/'progress.json').read_text())['phase']!='evaluation':
            raise ValueError('Both original CPU training studies must already be complete and evaluating')
        files[str(done)]=sha(done)
        for path in (ROOT/'checkpoints'/actor['name']).glob('*-actor.pt'):files[str(path)]=sha(path)
        if any(task['affinity']!=original for task in threads(actor)):raise ValueError('Every original participant must start with its recorded full mask')
    directory.mkdir()
    plan=dict(owner=identity(os.getpid()),at=datetime.now(timezone.utc).isoformat(),targets=CPU_OWNERS,
        original_cpus=original,restricted_cpus=restricted,GPU_support_cores_not_taken_by_CPU_evaluators=[5,20],
        expires_at_unix=time.time()+600,source_files_sha256=files,phase_seconds=90,
        training_completed_before_intervention=True,CPU_intraop_threads_not_changed=True,CUDA_contexts_or_leases=0,
        GPU_workers_not_reconfigured=True,no_source_model_optimizer_or_checkpoint_mutation=True,goal_achieved=False)
    write(directory/'owner-plan.json',plan)
    def interrupted(*args):raise KeyboardInterrupt('Bounded affinity trial interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    watcher=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--watchdog',str(directory)],
        cwd=ROOT,stdout=(directory/'watchdog.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True)
    phases=[]
    metrics=ROOT/'reports/broad-anchored-code-v7/anchor_CE-metrics.jsonl'
    def step():
        with metrics.open('rb') as stream:
            stream.seek(0,2);stream.seek(max(0,stream.tell()-2*1024**2));raw=stream.read()
        return json.loads(next(line for line in reversed(raw.split(b'\n')[:-1]) if line))['step']
    try:
        for name in ['original_before','six_core_evaluations','restored_after']:
            if name=='six_core_evaluations':apply_split(directory,plan)
            if name=='restored_after':restore(directory,plan,'Bounded split phase complete; original masks restored before final observation')
            start=time.monotonic();first_step=step();samples=[]
            with (directory/(name+'.jsonl')).open('x') as stream:
                while time.monotonic()-start<90:
                    if name=='six_core_evaluations':apply_split(directory,plan)
                    row=sample(plan);samples.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                    time.sleep(1)
            last_step=step();seconds=time.monotonic()-start
            ranks=[]
            for rank in range(4):
                a,b=samples[0]['GPU_workers'][rank],samples[-1]['GPU_workers'][rank]
                runtime=b['runtime_ns']-a['runtime_ns'];wait=b['eligible_wait_ns']-a['eligible_wait_ns']
                ranks.append(dict(pid=a['pid'],CPU_runtime_seconds=runtime/1e9,eligible_wait_seconds=wait/1e9,
                    eligible_wait_fraction_of_runtime_plus_wait=wait/(runtime+wait) if runtime+wait else 0.))
            phases.append(dict(phase=name,seconds=seconds,starting_complete_step=first_step,ending_complete_step=last_step,
                observed_completed_updates=last_step-first_step,mean_GPU_utilization_percent=[sum(r['GPUs'][i]['utilization_percent'] for r in samples)/len(samples) for i in range(4)],
                actual_GPU_main_thread_scheduling=ranks,samples=len(samples)))
            write(directory/'progress.json',dict(completed_phases=phases,original_masks_restored=(directory/'restored.json').exists()))
        assert all(sha(path)==wanted for path,wanted in files.items())
        write(directory/'complete.json',dict(complete=True,phases=phases,all_original_affinities_restored=True,
            actual_model_and_source_file_hashes_unchanged=True,unmatched_sequential_source_workloads_and_short_windows_not_a_controlled_kernel_benchmark=True,
            numerical_forward_or_gradient_equivalence_not_proved_by_OS_observation=True,model_quality_gain=False,goal_achieved=False))
    except BaseException:
        write(directory/'failure.json',dict(traceback=traceback.format_exc(),partial_outputs_retained=True,goal_achieved=False));raise
    finally:
        restore(directory,plan,'Trial ended; restore every exact live affected thread')
        watcher.wait(timeout=10)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path);parser.add_argument('--watchdog',type=Path);args=parser.parse_args()
    if args.watchdog:watchdog(args.watchdog)
    elif args.run:main(args.run)
    else:parser.error('Use explicit bounded --run OUTPUT or restoration --watchdog OUTPUT')
