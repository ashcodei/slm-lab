"""Admit optional GPU research by retiring only ten verified unstarted waiters.

The actual live pretraining ranks and independent CPU experiment are observed
and left running. All old sources, queues and evidence remain on disk. Fresh
controllers retain their original numerical functions and full populations.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import observe,retire,terminal
from slm_lab.function_repair_GPU_contract_v1 import QUEUE as GPU_QUEUE,REPORT as GPU_REPORT,ADMISSION
from slm_lab.local_source_closure_v2 import python_sources
import own_training_gpu_lease_v2 as owner

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/function-repair-GPU-lineage-v1'
MAP=ROOT/'reports/function-repair-GPU-lineage-v1-file-map.json'
OLD=ROOT/'reports/instance-lineage-recovery-v2/complete.json'


def read(path):return json.loads(Path(path).read_text())


def scope(queue):
    return {k:v for k,v in queue.items() if k not in {'at','status','source_files_sha256','expires_at_unix'}
        and 'owner' not in k and 'predecessor' not in k}


def unstarted(row,birth):
    observed=observe(birth,row['previous_script'])
    source=(ROOT/'scripts'/row['previous_script']).read_text()
    # Module-level output path constants are independent of the queue labels;
    # CPU finalizers use distinct report and model names.
    paths=re.findall(r"(?:REPORTS|OUT|OUTPUT)\s*=\s*ROOT\s*/\s*['\"]([^'\"]+)['\"]",source)
    if not paths:raise ValueError('Original controller output paths must be explicit')
    for name in paths:
        path=ROOT/name
        if not path.exists():continue
        if row['stage']=='AI2D-fixed-exposure-strata-v5' and name=='reports/AI2D-fixed-exposure-strata-recovery-v1':
            if sorted(p.name for p in path.iterdir())!=['plan.json'] or read(path/'plan.json')!=read(ROOT/row['previous_queue']):
                raise ValueError('Only the exact zero-neural-update strata reader plan may already exist')
        else:raise ValueError('A scientific output exists; this is no longer an unstarted waiting controller: '+name)
    return observed


def launch(script,queue,log):
    path=ROOT/queue
    if path.exists():raise ValueError('A fresh actual process queue is required')
    with (OUT/log).open('xb') as stream:
        child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts'/script)],cwd=ROOT,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ))
    birth=owner.resources.identity(child.pid)
    if not birth:raise ValueError('New controller ended before actual birth registration')
    while not path.exists():
        if child.poll() is not None:raise ValueError('Controller failed before publishing its real queue: '+script)
        time.sleep(2)
    queue_value=read(path)
    if (queue_value['owner']['pid']!=birth['pid'] or queue_value['owner']['ticks']!=birth['ticks'] or terminal(queue_value['owner'])):
        raise ValueError('Actual new process birth and published queue must match')
    FileBindings(queue_value['source_files_sha256']).check()
    return queue_value


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or (ROOT/GPU_QUEUE).exists() or (ROOT/GPU_REPORT).exists():
        raise ValueError('Fresh CUDA-hidden handoff and GPU queue required; never overwrite a prior attempt')
    mapping=read(MAP);old=read(OLD)['actual_ten_new_controllers'];routes=mapping['stages']
    test=ROOT/'reports/function-repair-GPU-lineage-v1-tests.log'
    if '5 passed' not in test.read_text():
        raise ValueError('Exact ten scientific scopes and human GPU restoration gate must pass')
    if len(routes)!=10 or len(old)!=10:raise ValueError('All ten original stages required')
    files=dict(mapping['source_files_sha256'])
    for path in [MAP,OLD,test,ROOT/'reports/function-repair-GPU-controller-v1-tests.log',
            *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_function_repair_GPU_lineage_v1.py',
                ROOT/'scripts/train_full_function_repair_comparison_GPU_v1.py'])]:
        files[str(path)]=file_hash(path)
    guard=FileBindings(files);snapshots=[];observations=[]
    pretrain=read(ROOT/'reports/public-pretraining-recovery-v1-queue.json')['owner']
    cpu=read(ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json')['owner']
    if terminal(pretrain) or terminal(cpu):raise ValueError('Live original GPU and CPU studies required for this operational handoff')
    gpu_before=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    for row,previous in zip(routes,old):
        if row['stage']!=previous['stage'] or row['previous_script']!=previous['recovery_script']:
            raise ValueError('Exact original stage order and tested recovery source required')
        queue=read(ROOT/row['previous_queue']);birth=queue['owner']
        if birth['pid']!=previous['actual_owner']['pid'] or birth['ticks']!=previous['actual_owner']['ticks']:
            raise ValueError('Only the exact ten current original queue owners may be replaced')
        observations.append(unstarted(row,birth));snapshots.append(queue)
    OUT.mkdir();launched=[];gpu=None
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner.resources.identity(os.getpid()),
        source_files_sha256=files,original_pretraining_owner=pretrain,original_CPU_comparison_owner=cpu,
        original_exact_waiting_observations=observations,original_queue_scopes=[scope(v) for v in snapshots],
        no_learned_output_present=True,actual_neural_updates=0,goal_achieved=False))
    try:
        gpu=launch('train_full_function_repair_comparison_GPU_v1.py',GPU_QUEUE,'GPU-controller-launcher.log')
        if (ROOT/ADMISSION).exists() or (ROOT/GPU_REPORT/'plan.json').exists():
            raise ValueError('The new GPU controller must remain behind closed admission with no claimed GPU')
        # Refresh every observation immediately before the pidfd operation.
        for row,q in zip(routes,snapshots):unstarted(row,q['owner'])
        guard.check(full=True);retire(observations)
        atomic_json(OUT/'retired.json',dict(complete=True,exact_original_waiting_owners=observations,
            all_terminal=all(terminal(row) for row in observations),original_queues_and_sources_preserved=True,
            signals_to_neural_workers=0,goal_achieved=False))
        for row,previous in zip(routes,snapshots):
            current=launch(row['recovery_script'],row['queue'],row['stage']+'-launcher.log')
            if scope(current)!=scope(previous):raise ValueError('Original full numerical source and benchmark queue scope changed: '+row['stage'])
            item=dict(stage=row['stage'],queue=row['queue'],owner=current['owner'],script=row['recovery_script'],
                original_queue_scope=scope(previous),actual_new_queue_scope=scope(current))
            launched.append(item)
            atomic_json(OUT/'progress.json',dict(actual_new_controllers=len(launched),new_controllers=launched,admission_open=False,goal_achieved=False))
        if any(terminal(row['owner']) for row in launched) or terminal(gpu['owner']):
            raise ValueError('All actual new gated owners must remain live before admission')
        if terminal(pretrain) or terminal(cpu):raise ValueError('Original in-progress neural owners changed during operational routing')
        gpu_after=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        if sorted(gpu_before.splitlines())!=sorted(gpu_after.splitlines()):
            raise ValueError('Actual original GPU compute owners must remain unchanged during this CPU-only handoff')
        guard.check(full=True)
        # Every new controller and the GPU queue have independently read back
        # their immutable sources. Never bind the periodically rewritten human
        # queue inode; its actual birth and mathematical scope are retained.
        admission=dict(complete=True,at=datetime.now(timezone.utc).isoformat(),GPU_study_owner=gpu['owner'],
            old_ten_waiting_owners_retired=True,all_original_neural_functions_and_benchmark_populations_preserved=True,
            new_controllers=launched,source_files_sha256=files,original_pretraining_owner=pretrain,
            original_CPU_comparison_owner=cpu,actual_GPU_compute_PIDs_unchanged=True,
            all_original_sources_queues_reports_and_checkpoints_preserved=True,signals_to_neural_workers=0,
            canonical_human_RL_initializer_unchanged=True,candidate_promoted=False,goal_achieved=False)
        atomic_json(ROOT/ADMISSION,admission)
        atomic_json(OUT/'complete.json',dict(**admission,admission_sha256=file_hash(ROOT/ADMISSION)))
        print(json.dumps(dict(admission=str(ROOT/ADMISSION),actual_new_controllers=10,neural_workers_signalled=0)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),new_controllers=launched,
            GPU_study_owner=None if gpu is None else gpu['owner'],admission_open=(ROOT/ADMISSION).exists(),
            original_waiting_owners_terminal=[terminal(row) for row in observations],automatic_retry=False,goal_achieved=False));raise


if __name__=='__main__':main()
