"""Queue complete matched math assessment after the existing GPU owner ends."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import assess_joint_java_comparison_v1 as resources
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/apps-full-language-GPU-math-v1'
QUEUE=ROOT/'reports/apps-full-language-GPU-math-v1-queue.json'
PREDECESSOR=ROOT/'reports/apps-full-language-v1'
WAITER=dict(pid=2754743,ticks='27602681')
WAIT_STATE='waiting for finalized original paired real-photo questions'
BASE=ROOT/'checkpoints/dialogue-retention-v1/systems-variety-code/model.pt'
CANDIDATE=ROOT/'checkpoints/apps-full-language-v1/model.pt'
DATA=ROOT/'data/math-reasoning-v2/test.jsonl'
MAXIMUM_SECONDS=4*60*60
resources.REPORTS=REPORTS


def idle_original():
    actor=resources.same(WAITER)
    children=Path('/proc')/str(WAITER['pid'])/'task'/str(WAITER['pid'])/'children'
    if (not actor or actor['state']!='S' or children.read_text().strip()
            or json.loads((ROOT/'reports/dialogue-retention-v1/status.json').read_text())['status']!=WAIT_STATE):
        raise ValueError('Never preempt original active training; exact childless photo-data waiter required')


def owned(plan):
    waiter=resources.same(plan['waiter'])
    if not resources.same(plan['owner']) or not waiter or waiter['state'] not in {'T','t'}:
        raise ValueError('Exact live math owner and its held original waiter required')


def worker(shard):
    plan=json.loads((REPORTS/'plan.json').read_text());me=resources.identity(os.getpid())
    while True:
        owned(plan)
        actors=json.loads((REPORTS/'children.json').read_text())
        if any(a['pid']==me['pid'] and a['ticks']==me['ticks'] and a['subject']=='math-shard-'+str(shard) for a in actors):break
        time.sleep(.1)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(shard):raise ValueError('Exact physical math GPU assignment required')
    guard=FileBindings({**plan['source_files_sha256'],str(REPORTS/'plan.json'):file_hash(REPORTS/'plan.json')})
    guard.check(full=True)
    import torch
    from slm_lab.full_math_shards_v1 import evaluate_shard
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.4,0)
    def checked():owned(plan);guard.check()
    for phase,checkpoint in [('baseline',BASE),('candidate',CANDIDATE)]:
        checked()
        evaluate_shard(checkpoint,DATA,REPORTS/f'{phase}-{shard}.json',shard=shard,shards=4,
            device='cuda:0',max_new_tokens=512,expected_count=1319,checked=checked)
        guard.check(full=True)


def parent(files):
    if REPORTS.exists():raise ValueError('Preserve earlier math generations; no implicit restart')
    idle_original()
    completed=json.loads((PREDECESSOR/'training-complete.json').read_text())
    if completed['steps']!=3203 or not completed['whole_APPS_source_pass'] or completed['checkpoint_sha256']!=file_hash(CANDIDATE):
        raise ValueError('Actual completed full3203update own candidate required')
    import torch
    state=torch.load(CANDIDATE,map_location='cpu',weights_only=True,mmap=True)
    predecessor_plan=json.loads((PREDECESSOR/'plan.json').read_text())
    if (state.get('steps')!=3203 or state.get('planned_steps')!=3203
            or state.get('run_plan_sha256')!=file_hash(PREDECESSOR/'plan.json')
            or state.get('source_checkpoint_sha256')!=predecessor_plan['base_checkpoint_sha256']
            or state.get('tokenizer_sha256')!=predecessor_plan['tokenizer_sha256']
            or state.get('training_kind')!='complete_APPS_human_solution_language_v1'
            or state.get('truncated_sequences')!=0
            or sum(t.numel() for n,t in state['model'].items() if n!='lm_head.weight')!=706012672):
        raise ValueError('Actual complete own706M candidate metadata/source/tensor inventory required')
    del state
    if resources.same(json.loads((PREDECESSOR/'plan.json').read_text())['owner']):
        raise ValueError('Original four-GPU predecessor must actually end before math acquisition')
    old_actors=json.loads((PREDECESSOR/'children.json').read_text())
    if any(resources.same(a) for a in old_actors):raise ValueError('Every registered predecessor GPU actor must actually end')
    for p in [CANDIDATE,CANDIDATE.parent/'tokenizer.json',PREDECESSOR/'training-complete.json',PREDECESSOR/'restored.json']:
        files[str(p)]=file_hash(p)
    guard=FileBindings(files)
    from slm_lab.full_math_shards_v1 import load_source
    load_source(DATA,expected_count=1319)
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        idle_original()
        memory=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
            capture_output=True,text=True,check=True).stdout.splitlines()
        if len(memory)!=4 or any(int(x.strip())>128 for x in memory):raise ValueError('All four math GPUs must be unused')
        guard.check(full=True);REPORTS.mkdir()
        plan=dict(created_at=datetime.now(timezone.utc).isoformat(),owner=resources.identity(os.getpid()),waiter=WAITER,
            source_files_sha256=files,baseline_checkpoint_sha256=file_hash(BASE),candidate_checkpoint_sha256=file_hash(CANDIDATE),
            official_questions=1319,shards=4,phases=['baseline','candidate'],generation_budget=512,
            arithmetic='CUDA BF16 autocast; distinct from original CPU FP32 runs',native_prompt_and_scoring_unchanged=True,
            maximum_seconds=MAXIMUM_SECONDS,model_updates=0,candidate_promoted=False,goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[])
        watcher=subprocess.Popen(['.venv/bin/python','-u',str(Path(__file__).resolve()),'--watchdog'],cwd=ROOT,
            stdout=(REPORTS/'watchdog.log').open('xb'),stderr=subprocess.STDOUT,start_new_session=True)
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned math controller interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        children=[]
        try:
            idle_original();os.kill(WAITER['pid'],signal.SIGSTOP);until=time.monotonic()+5
            while (resources.same(WAITER) or {}).get('state') not in {'T','t'}:
                if time.monotonic()>until:raise ValueError('Original idle waiter did not enter the owned hold')
                time.sleep(.05)
            started=time.monotonic()
            for shard in range(4):
                with (REPORTS/f'shard-{shard}.log').open('xb') as stream:
                    child=subprocess.Popen(['.venv/bin/python','-u',str(Path(__file__).resolve()),'--worker',str(shard)],
                        cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(shard),'OMP_NUM_THREADS':'1'})
                children.append(child)
                actors=json.loads((REPORTS/'children.json').read_text())
                actor=resources.identity(child.pid)
                if not actor:raise ValueError('Math worker ended before exact registration')
                actors.append(dict(**actor,subject='math-shard-'+str(shard)));atomic_json(REPORTS/'children.json',actors)
            while any(p.poll() is None for p in children):
                owned(plan)
                if time.monotonic()-started>MAXIMUM_SECONDS:raise TimeoutError('Owned whole math window expired; preserve partial generations')
                if any(p.poll() not in [None,0] for p in children):raise ValueError('A math shard failed; preserve partial results without retry')
                atomic_json(REPORTS/'status.json',dict(observed_at=datetime.now(timezone.utc).isoformat(),
                    status='complete native math assessment active',workers=[dict(pid=p.pid,exit_code=p.poll()) for p in children],
                    model_updates=0,goal_achieved=False));time.sleep(2)
            if any(p.returncode!=0 for p in children):raise ValueError('Every complete native math shard must succeed')
            from slm_lab.full_math_shards_v1 import reconcile
            full={}
            for phase,checkpoint in [('baseline',BASE),('candidate',CANDIDATE)]:
                full[phase]=reconcile([REPORTS/f'{phase}-{i}.json' for i in range(4)],DATA,
                    checkpoint.parent/'tokenizer.json',REPORTS/f'{phase}-complete.json',expected_count=1319)
            before={r['id']:r['correct'] for r in full['baseline']['results']}
            after={r['id']:r['correct'] for r in full['candidate']['results']}
            guard.check(full=True)
            atomic_json(REPORTS/'comparison.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
                baseline_correct=full['baseline']['correct'],candidate_correct=full['candidate']['correct'],questions=1319,
                gained_correct_ids=sorted(i for i in before if after[i] and not before[i]),
                lost_correct_ids=sorted(i for i in before if before[i] and not after[i]),
                complete_native_original_questions=True,precision='CUDA BF16',generation_budget=512,
                baseline_report_sha256=file_hash(REPORTS/'baseline-complete.json'),
                candidate_report_sha256=file_hash(REPORTS/'candidate-complete.json'),
                math_answers_not_reasoning_proof=True,candidate_promoted=False,goal_achieved=False))
            atomic_json(REPORTS/'status.json',dict(status='full matched math assessment complete; inspect actual results',goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS/'failure.json',dict(error=str(error),automatic_retry=False,goal_achieved=False));raise
        finally:
            resources.restore(plan,'Owned math assessment ended; exact original idle waiter restored')
            watcher.wait(timeout=20)


def queue():
    if QUEUE.exists() or REPORTS.exists():raise ValueError('Preserve queued or prior math assessment; no implicit duplicate')
    predecessor=json.loads((PREDECESSOR/'plan.json').read_text())['owner']
    from slm_lab.evaluation_provenance import capture
    from slm_lab.full_math_shards_v1 import load_source
    load_source(DATA,expected_count=1319)
    files={str(ROOT/name):sha for name,sha in capture()['files_sha256'].items()}
    paths=[Path(__file__).resolve(),ROOT/'src/slm_lab/full_math_shards_v1.py',ROOT/'scripts/assess_joint_java_comparison_v1.py',
        DATA,DATA.parent/'manifest.json',BASE,BASE.parent/'tokenizer.json',PREDECESSOR/'plan.json',
        ROOT/'tests/test_full_math_shards_v1.py',ROOT/'reports/full-math-shards-v1-tests.log']
    for p in paths:files[str(p)]=file_hash(p)
    if '2 passed' not in (ROOT/'reports/full-math-shards-v1-tests.log').read_text():
        raise ValueError('Actual native CPU protocol reproduction and active-original-work rejection checks required')
    if file_hash(BASE)!='fc8e248eb8304cfd3895ed103f5cf172641e6151017b9aa0d83e19a443e444f0':
        raise ValueError('Exact completed own systems baseline required')
    guard=FileBindings(files)
    atomic_json(QUEUE,dict(queued_at=datetime.now(timezone.utc).isoformat(),queue_owner=resources.identity(os.getpid()),
        exact_predecessor=predecessor,source_files_sha256=files,all_original_questions=1319,checkpoints=2,
        device='four idle L40s after original predecessor exits',generation_budget=512,precision='BF16',
        pending_not_started=True,automatic_retry=False,model_updates=0,goal_achieved=False))
    while resources.same(predecessor):
        guard.check();time.sleep(5)
    while True:
        guard.check()
        if not resources.same(WAITER):raise ValueError('Original idle waiter ended; inspect a new math resource window')
        try:idle_original();break
        except ValueError:time.sleep(5)
    parent(files)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=int,choices=[0,1,2,3]);parser.add_argument('--watchdog',action='store_true')
    args=parser.parse_args()
    if args.watchdog:resources.watchdog()
    elif args.worker is not None:worker(args.worker)
    else:queue()
