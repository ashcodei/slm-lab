"""Observe restored polling, launch its bounded continuation, verify a rescue.

Only an already-tested controller may pause the nine exact idle parents. This
observer never signals a neural worker or changes its priority/affinity/source.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/polling-continuation-observation-CPU-v1'
PREVIOUS=ROOT/'reports/waiter-polling-duty-cycle-CPU-v2'
NEXT=ROOT/'reports/waiter-polling-duty-cycle-CPU-v3'


def read(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    pending=path.with_suffix(path.suffix+'.pending');pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists() or NEXT.exists():raise ValueError('Fresh low-priority CPU observer required')
    prior=read(PREVIOUS/'plan.json');done=read(PREVIOUS/'complete.json')
    if (not done.get('complete') or not done.get('all_owned_suspensions_released')
            or (Path('/proc')/str(prior['owner']['pid'])).exists()):raise ValueError('The exact old coordinator must have completed and fully released its waiters')
    targets=prior['actual_idle_targets'];protected=prior['protected_neural_and_GPU_study_owners'];owners=targets+protected
    if len(targets)!=9:raise ValueError('Only the original nine tested idle parents may be controlled')
    hz=os.sysconf('SC_CLK_TCK')
    def snapshot():
        processes={}
        for owner in owners:
            values=(Path('/proc')/str(owner['pid'])/'stat').read_text().split(') ',1)[1].split()
            if values[19]!=owner['ticks'] or values[0] in {'Z','X'}:raise ValueError('Every exact original process birth must remain live')
            processes[str(owner['pid'])]=dict(state=values[0],CPU_ticks=int(values[11])+int(values[12]))
        return dict(monotonic=time.monotonic(),processes=processes)
    def window(name,states):
        before=snapshot()
        if any(before['processes'][str(row['pid'])]['state'] not in states for row in targets):raise ValueError('Every target must start in its declared observation state')
        time.sleep(30);after=snapshot();elapsed=after['monotonic']-before['monotonic']
        if any(after['processes'][str(row['pid'])]['state'] not in states for row in targets):raise ValueError('Every target must finish in its declared observation state')
        groups=dict(waiter_pollers=[row['pid'] for row in targets],CPU_training=[protected[1]['pid']],
            CPU_assessment=[protected[2]['pid']],GPU_workers=[row['pid'] for row in protected[4:]])
        result=dict(window=name,before=before,after=after,wall_seconds=elapsed,actual_average_CPU_cores={
            key:sum(after['processes'][str(pid)]['CPU_ticks']-before['processes'][str(pid)]['CPU_ticks'] for pid in values)/hz/elapsed
            for key,values in groups.items()})
        atomic(OUT/(name+'.json'),result)
        return result
    source_files={str(path):sha(path) for path in [Path(__file__),ROOT/'scripts/throttle_waiter_polling_CPU_v3.py',PREVIOUS/'plan.json',PREVIOUS/'complete.json']}
    OUT.mkdir();atomic(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),source_files_sha256=source_files,
        original_idle_targets=targets,protected_original_workers=protected,window_seconds=30,
        no_neural_worker_signals_or_source_priority_affinity_changes=True,goal_achieved=False))
    baseline=window('original-polling-restored',{'R','S'})
    log=ROOT/'reports/waiter-polling-duty-cycle-CPU-v3.log'
    with log.open('xb') as stream:
        process=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/throttle_waiter_polling_CPU_v3.py')],
            cwd=ROOT,env=dict(os.environ),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    # Observe actual new ownership/held stops, never infer it from intent.
    deadline=time.monotonic()+20
    while True:
        if process.poll() is not None:raise ValueError('The new bounded coordinator ended before initial admission')
        if (NEXT/'plan.json').exists() and (NEXT/'progress.json').exists():
            current=read(NEXT/'progress.json')
            if current['phase']=='bounded_idle_polling_suspension' and all(snapshot()['processes'][str(row['pid'])]['state'] in {'T','t'} for row in targets):break
        if time.monotonic()>deadline:raise ValueError('Initial owned suspension was not observed; inspect exact existing owner without restarting')
        time.sleep(.1)
    paused=window('bounded-polling-suspension',{'T','t'})
    deadline=time.monotonic()+30
    while not (NEXT/'cycle-1-complete.json').exists():
        if process.poll() is not None or time.monotonic()>deadline:raise ValueError('Full first-cycle release not observed; inspect the exact owner/rescue receipt')
        time.sleep(.1)
    cycle=read(NEXT/'cycle-1-complete.json');plan=read(NEXT/'plan.json')
    if (not cycle.get('all_owned_suspensions_released') or not cycle['rescue_complete'].get('complete')
            or plan['maximum_seconds']!=14400 or plan['pause_seconds']!=45
            or plan['original_check_running_seconds']!=15 or plan['independent_rescue_seconds']!=55):
        raise ValueError('Only the unchanged tested pause/rescue mechanics and an actual full release are admitted')
    after=snapshot()
    if any(after['processes'][str(row['pid'])]['state'] not in {'R','S'} for row in targets):raise ValueError('Every owned first-cycle suspension must actually be released')
    for path,digest in source_files.items():
        if sha(path)!=digest:raise ValueError('An original or measured resource source changed')
    atomic(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=source_files,
        previous_hour_controller_terminal_and_all_waiters_released=True,new_actual_bounded_owner=plan['owner'],
        maximum_adopted_lifetime_seconds=14400,pause_seconds=45,original_running_seconds=15,independent_rescue_seconds=55,
        actual_normal_polling=baseline,actual_paused_polling=paused,actual_first_cycle_release=cycle,actual_resumed_snapshot=after,
        no_neural_worker_signals_or_priority_affinity_runtime_source_or_queue_changes=True,
        original_all_source_and_benchmark_budgets_unchanged=True,no_new_neural_jobs_or_optimizer_updates=True,
        measured_CPU_allocation_not_direct_model_throughput_or_quality=True,goal_achieved=False))
    print(json.dumps(dict(observation=str(OUT),new_actual_bounded_owner=plan['owner'],normal_CPU=baseline['actual_average_CPU_cores'],paused_CPU=paused['actual_average_CPU_cores'],first_cycle_all_released=True)),flush=True)


if __name__=='__main__':main()
