"""Five bounded CPU-allocation windows with original neural workers untouched."""
from datetime import datetime,timezone
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from slm_lab.bounded_waiter_pause_v1 import atomic,pause
from slm_lab.exact_waiting_process_retirement_v1 import observe,terminal

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/waiter-polling-throughput-CPU-v1'
ADMISSION=ROOT/'reports/function-repair-GPU-lineage-v1/admission.json'
RUNTIME=ROOT/'reports/public-pretraining-recovery-v1/plan.json'
MATH=ROOT/'reports/public-pretraining-v1/plan.json'
TESTS=ROOT/'reports/bounded-waiter-pause-v1-tests.log'


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):value.update(block)
    return value.hexdigest()


def identity(pid):
    fields=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()
    if fields[0] in {'Z','X'}:raise ValueError('Exact process must remain live')
    return dict(pid=pid,ticks=fields[19])


def report_folder(script):
    values=[]
    for node in ast.parse(Path(script).read_text()).body:
        if not isinstance(node,ast.Assign) or not any(isinstance(k,ast.Name) and k.id in {'OUT','REPORTS'} for k in node.targets):continue
        value=node.value
        if (isinstance(value,ast.BinOp) and isinstance(value.op,ast.Div) and isinstance(value.left,ast.Name)
                and value.left.id=='ROOT' and isinstance(value.right,ast.Constant) and isinstance(value.right.value,str)):
            values.append(ROOT/value.right.value)
    if len(values)!=1:raise ValueError('One explicit original production report path required')
    return values[0]


def journal():
    raw=(ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes()
    rows=list(map(json.loads,raw[:raw.rfind(b'\n')+1].splitlines()))
    if [v['step'] for v in rows]!=list(range(1,len(rows)+1)):raise ValueError('Actual contiguous complete neural prefix required')
    return dict(updates=len(rows),targets=sum(int(v['response_EOS_targets']) for v in rows))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or os.nice(0)<19:
        raise ValueError('Fresh CUDA-hidden low-priority bounded CPU profile required')
    if '3 passed' not in TESTS.read_text():raise ValueError('Actual disposable deadline/error/killed-owner rescue tests required')
    lock=(ROOT/'reports/waiter-polling-CPU.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    admitted=read(ADMISSION);runtime=read(RUNTIME)
    targets=[];skipped=[];paths=[Path(__file__),TESTS,ADMISSION,RUNTIME,MATH,
        ROOT/'src/slm_lab/bounded_waiter_pause_v1.py',ROOT/'src/slm_lab/exact_waiting_process_retirement_v1.py',
        ROOT/'tests/test_bounded_waiter_pause_v1.py',ROOT/'reports/bounded-waiter-pause-v1-initial-tests.log']
    for item in admitted['new_controllers']:
        script=ROOT/'scripts'/item['script'];folder=report_folder(script);queue=read(ROOT/item['queue'])
        if queue['owner']!=item['owner']:raise ValueError('Every original queue must retain its actual process birth')
        if (folder/'plan.json').exists():
            skipped.append(dict(stage=item['stage'],reason='already-published reader plan; leave it running'));continue
        if queue['source_files_sha256'].get(str(script))!=sha(script):raise ValueError('Original source-bound controller must remain unchanged')
        targets.append(dict(**observe(item['owner'],str(script)),stage=item['stage'],report_folder=str(folder)))
        paths.append(script)
    if len(targets)!=9 or len(skipped)!=1:raise ValueError('Exactly nine unstarted controllers and the existing CPU strata reader required')
    protected=[runtime['owner'],read(ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json')['owner'],
               read(ROOT/'reports/public-pretraining-interim-CPU-v2/evaluation-plan.json')['owner'],admitted['GPU_study_owner']]
    compute=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()]
    if len(compute)!=4 or {v['pid'] for v in targets}&set(compute):raise ValueError('Exactly the original four GPU workers, none among idle controls, required')
    protected += [identity(pid) for pid in compute]
    if any(terminal(v) for v in protected) or journal()['updates']>=read(MATH)['planned_steps']-128:
        raise ValueError('Every original neural process must be live and far from its handoff')
    files={str(p):sha(p) for p in paths}
    def checked():
        if any(sha(name)!=expected for name,expected in files.items()):raise ValueError('Original controller/model plan or bounded profiler source changed')
        if any(terminal(v) for v in protected):raise ValueError('A protected predecessor ended; finish with every waiter resumed')
        for row in targets:
            observe(row,row['command'])
            if (Path(row['report_folder'])/'plan.json').exists():raise ValueError('An original controller has started its stage; leave it running')
    checked();OUT.mkdir()
    atomic(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=identity(os.getpid()),source_files_sha256=files,
        actual_idle_targets=targets,existing_reader_left_running=skipped,protected_neural_and_GPU_study_owners=protected,
        fixed_windows=['baseline','paused1','restored1','paused2','restored2'],seconds_per_window=20,
        watchdog_seconds_per_pause=35,all_signals_through_exact_held_pidfds=True,
        no_neural_worker_or_main_or_active_CPU_assessment_signalled=True,
        no_original_runtime_source_or_queue_edited=True,CPU_activity_is_not_model_quality_or_direct_decoding_speed=True,goal_achieved=False))
    groups=dict(waiter_pollers=[v['pid'] for v in targets],CPU_training=[protected[1]['pid']],
                CPU_assessment=[protected[2]['pid']],GPU_workers=compute)
    everyone={v['pid']:v for v in [*targets,*protected]};hz=os.sysconf('SC_CLK_TCK')
    def snapshot():
        processes={}
        for pid,value in everyone.items():
            fields=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()
            if fields[19]!=value['ticks'] or fields[0] in {'Z','X'}:raise ValueError('Observed process birth ended during profile')
            processes[str(pid)]=dict(user=int(fields[11])/hz,system=int(fields[12])/hz,state=fields[0])
        return dict(monotonic=time.monotonic(),processes=processes,main=journal(),
                    actual_GPU_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True))
    def window(name):
        checked();before=snapshot();time.sleep(20);after=snapshot();wall=after['monotonic']-before['monotonic']
        cores={group:sum(after['processes'][str(p)]['user']+after['processes'][str(p)]['system']-
                        before['processes'][str(p)]['user']-before['processes'][str(p)]['system'] for p in pids)/wall
               for group,pids in groups.items()}
        value=dict(window=name,actual_wall_seconds=wall,actual_average_CPU_cores=cores,
                   complete_main_updates=after['main']['updates']-before['main']['updates'],
                   complete_main_response_EOS_targets=after['main']['targets']-before['main']['targets'],before=before,after=after)
        atomic(OUT/(name+'.json'),value);atomic(OUT/'progress.json',dict(last_complete_window=name,actual_average_CPU_cores=cores,goal_achieved=False))
        print(json.dumps(dict(window=name,CPU_cores=cores)),flush=True);return value
    results=[]
    try:
        results.append(window('baseline'))
        for number in (1,2):
            checked()
            with pause(targets,OUT/('pause-'+str(number)),maximum_seconds=35,release_on_exit=protected):
                results.append(window('paused'+str(number)))
            if any((Path('/proc')/str(v['pid'])/'stat').read_text().split(') ',1)[1].split()[0] in {'T','t'} for v in targets):
                raise ValueError('Every exact idle waiter must be observed running after bounded resume')
            results.append(window('restored'+str(number)))
        checked()
        atomic(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            actual_windows=results,all_nine_waiting_parents_resumed=True,original_all_source_and_benchmark_scopes_preserved=True,
            new_neural_optimizer_updates_by_profiler=0,no_model_quality_or_direct_decoding_speed_claim=True,goal_achieved=False))
    except BaseException as error:
        atomic(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,
            original_neural_workers_not_signalled=True,goal_achieved=False));raise


if __name__=='__main__':main()
