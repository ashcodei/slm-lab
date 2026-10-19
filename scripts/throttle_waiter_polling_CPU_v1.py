"""Temporary bounded polling duty cycle, preserving every original controller.

Only the nine exact unstarted childless CUDA-hidden parents from the completed
five-window experiment are suspended. Every suspension lasts 45 seconds and has an independent
55-second rescue; 15-second running windows retain their original checks. Stop at
20 minutes or a predecessor/stage transition. No neural worker receives a signal.
"""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import time
import traceback

from slm_lab.bounded_waiter_pause_v1 import atomic,pause,state
from slm_lab.exact_waiting_process_retirement_v1 import observe,terminal
import profile_waiter_polling_CPU_v1 as profile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/waiter-polling-duty-cycle-CPU-v1'
MAXIMUM_SECONDS=1200
PAUSE_SECONDS=45
RUN_SECONDS=15


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists():
        raise ValueError('Fresh CUDA-hidden low-priority temporary polling coordinator required')
    lock=(ROOT/'reports/waiter-polling-CPU.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=ROOT/'reports/waiter-polling-throughput-CPU-v1'
    done=profile.read(prior/'complete.json');plan=profile.read(prior/'plan.json')
    if (not done.get('complete') or not done.get('all_nine_waiting_parents_resumed')
            or [v['window'] for v in done['actual_windows']]!=['baseline','paused1','restored1','paused2','restored2']):
        raise ValueError('Complete repeated measured pause/restoration experiment required')
    for value in done['actual_windows']:
        paused=value['window'].startswith('paused')
        rate=value['actual_average_CPU_cores']['waiter_pollers']
        if paused and abs(rate)>.05 or not paused and rate<.3:
            raise ValueError('A repeatable substantial polling cost and observed negligible paused CPU required')
        for row in plan['actual_idle_targets']:
            expected_states={'T','t'} if paused else {'R','S'}
            if any(snapshot['processes'][str(row['pid'])]['state'] not in expected_states for snapshot in [value['before'],value['after']]):
                raise ValueError('Every target must match the actual intended measurement state')
    tests=ROOT/'reports/bounded-waiter-pause-protected-exit-v1-tests.log'
    if '1 passed' not in tests.read_text():raise ValueError('Actual protected-predecessor exit rescue must pass')
    files=dict(done['source_files_sha256'])
    for path in [Path(__file__),prior/'complete.json',prior/'plan.json',tests,ROOT/'tests/test_bounded_waiter_pause_protected_exit_v1.py']:
        files[str(path)]=profile.sha(path)
    targets=plan['actual_idle_targets'];protected=plan['protected_neural_and_GPU_study_owners']
    if len(targets)!=9 or {v['pid'] for v in targets}&{v['pid'] for v in protected}:
        raise ValueError('Exact disjoint idle-controller and protected neural scopes required')
    def eligibility():
        if any(profile.sha(name)!=expected for name,expected in files.items()):
            raise ValueError('An original source/plan or tested scheduling implementation changed')
        if any(terminal(v) for v in protected):return 'protected_predecessor_transition'
        if profile.journal()['updates']>=profile.read(profile.MATH)['planned_steps']-128:return 'public_training_near_handoff'
        for row in targets:
            if terminal(row):return 'original_waiting_controller_transition'
            if (Path(row['report_folder'])/'plan.json').exists():return 'original_stage_started'
            observe(row,row['command'])
            if state(row) not in {'R','S'}:return 'controller_not_available_for_owned_bounded_pause'
        return None
    reason=eligibility()
    if reason:raise ValueError('Original processes already changed: '+reason)
    OUT.mkdir();owner=profile.identity(os.getpid());at=datetime.now(timezone.utc).isoformat()
    atomic(OUT/'plan.json',dict(at=at,owner=owner,source_files_sha256=files,actual_idle_targets=targets,
        protected_neural_and_GPU_study_owners=protected,maximum_seconds=MAXIMUM_SECONDS,
        pause_seconds=PAUSE_SECONDS,original_check_running_seconds=RUN_SECONDS,independent_rescue_seconds=55,
        original_training_validation_and_benchmark_budgets_unchanged=True,
        no_original_neural_worker_signals_or_runtime_source_or_queue_edits=True,
        no_new_neural_optimizer_updates=True,no_direct_model_quality_or_decoding_speed_claim=True,goal_achieved=False))
    started=time.monotonic();deadline=started+MAXIMUM_SECONDS;cycles=[];reason='maximum_bounded_lifetime'
    def interrupted(signum,frame):raise KeyboardInterrupt('Owned polling coordinator interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        while time.monotonic()<deadline:
            reason=eligibility()
            if reason:break
            number=len(cycles)+1;folder=OUT/('cycle-'+str(number))
            atomic(OUT/'progress.json',dict(owner=owner,phase='bounded_idle_polling_suspension',cycle=number,
                original_main=profile.journal(),maximum_lifetime_seconds=MAXIMUM_SECONDS,new_neural_updates=0,goal_achieved=False))
            try:
                with pause(targets,folder,maximum_seconds=55,release_on_exit=protected):
                    time.sleep(min(PAUSE_SECONDS,max(0,deadline-time.monotonic())))
            except ValueError:
                if any(terminal(v) for v in protected):reason='protected_predecessor_transition';break
                raise
            receipt=profile.read(folder/'rescue-complete.json')
            if not receipt.get('complete') or any(state(v) in {'T','t'} for v in targets):
                raise ValueError('Every owned bounded suspension must be observed fully released')
            cycle=dict(cycle=number,rescue_complete=receipt,all_owned_suspensions_released=True,
                       actual_main=profile.journal(),elapsed_seconds=time.monotonic()-started)
            cycles.append(cycle);atomic(OUT/('cycle-'+str(number)+'-complete.json'),cycle)
            reason=eligibility()
            if reason:break
            atomic(OUT/'progress.json',dict(owner=owner,phase='original_polling_running',complete_cycles=len(cycles),
                actual_main=profile.journal(),new_neural_updates=0,goal_achieved=False))
            time.sleep(min(RUN_SECONDS,max(0,deadline-time.monotonic())))
        atomic(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),owner=owner,
            source_files_sha256=files,complete_cycles=cycles,finish_reason=reason or 'maximum_bounded_lifetime',
            all_owned_suspensions_released=True,no_original_neural_worker_signals=True,
            no_original_runtime_source_or_queue_edited=True,actual_new_neural_updates=0,goal_achieved=False))
        atomic(OUT/'progress.json',dict(owner=owner,phase='complete',reason=reason,all_owned_suspensions_released=True,goal_achieved=False))
    except BaseException as error:
        atomic(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,
            active_bounded_context_restores_its_waiters=True,actual_new_neural_updates=0,goal_achieved=False))
        raise


if __name__=='__main__':main()
