"""Queue all original175 public problems for the own conditional-code prototype."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import train_point_grounding_joint_v1 as owner
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/own-code-delta-public-v1'
QUEUE=ROOT/'reports/own-code-delta-public-v1-queue.json'
DATA=ROOT/'data/public-code-assessment-v1'
COMPONENTS=ROOT/'checkpoints/own-code-delta-v1/model.pt'
BASE=ROOT/'checkpoints/extended-curriculum-v1/joint/model.pt'
PREPARATION=ROOT/'reports/own-code-delta-v1'
BEFORE=ROOT/'reports/apps-full-language-v1/candidate/public-Python.json'
owner.REPORTS=REPORTS;owner.resources.REPORTS=REPORTS


def seal_shards(original,output):
    manifest=json.loads((original.parent/'manifest.json').read_text())
    if file_hash(original)!=manifest['questions_sha256']:
        raise ValueError('Exact original eligible public problem bytes required')
    lines=original.read_bytes().splitlines(keepends=True);rows=[json.loads(line) for line in lines]
    if len(rows)!=len({r['id'] for r in rows}):
        raise ValueError('Every original eligible problem must be unique')
    for shard in range(4):
        folder=output/('shard-'+str(shard));folder.mkdir(parents=True)
        (folder/'questions.jsonl').write_bytes(b''.join(lines[shard::4]))
        atomic_json(folder/'manifest.json',dict(manifest,questions_sha256=file_hash(folder/'questions.jsonl'),
            full_supported_problems=len(rows),actual_shard_problems=len(rows[shard::4]),shard=shard,shards=4,
            original_complete_questions_sha256=file_hash(original),
            limitations=[*manifest['limitations'],'This disjoint shard needs all four original parts for a complete score.']))
    return rows


def worker(shard):
    plan=owner.registered('own-code-delta-public-shard-'+str(shard))
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(shard):
        raise ValueError('Exact owned isolated physical public-evaluation GPU required')
    guard=FileBindings({**plan['source_files_sha256'],str(REPORTS/'plan.json'):file_hash(REPORTS/'plan.json')})
    guard.check(full=True)
    import torch
    from tokenizers import Tokenizer
    from slm_lab import public_code_benchmark as evaluation
    from slm_lab.own_code_delta_inference_v1 import load
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.7,0)
    if torch.cuda.get_device_name(0)!='NVIDIA L40':
        raise ValueError('Same original L40 inference device required')
    tokenizer=Tokenizer.from_file(str(BASE.parent/'tokenizer.json'))
    original_loader=evaluation.load_checkpoint
    def own_loader(path,device,training_state=False):
        if Path(path)!=COMPONENTS or training_state:
            raise ValueError('Only the exact own inference components may use this explicit loader')
        model,metadata=load(COMPONENTS,BASE,BASE.parent/'tokenizer.json',PREPARATION/'plan.json',
            expected_checkpoint_sha256=plan['components_sha256'],expected_base_sha256=plan['base_sha256'],
            expected_target_sha256=plan['own_code_target_sha256'],expected_plan_sha256=plan['preparation_plan_sha256'],device=device)
        rows=list(map(json.loads,(DATA/'questions.jsonl').read_text().splitlines()))
        def prefix(row):
            text=f"User: Write a Python function with signature {row['signature']} {row['specification']}\nUse built-in Python types; only public math imports are available. Return only complete code.\nAssistant:"
            return [1,*tokenizer.encode(text).ids]
        longest=max(rows,key=lambda r:len(prefix(r)));ids=torch.tensor([prefix(longest)],device=device)
        if ids.shape[1]+512>model.config.max_text_length:
            raise ValueError('Whole original longest public prompt must fit original4096context')
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            first=model.code.prefill(ids);decoded=model.code.decode(first['logits'][:,-1].argmax(-1,keepdim=True),first['cache'])
        if not bool(torch.isfinite(decoded['logits']).all()):
            raise ValueError('Actual full985M longest-prompt cached inference must be finite')
        atomic_json(REPORTS/f'capacity-{shard}.json',dict(actual_GPU=torch.cuda.get_device_name(0),
            actual_model_counts=model.counts(),actual_longest_public_prompt_tokens=ids.shape[1],
            one_cached_decode_completed=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            actual_full_code_source_not_loaded=True,model_updates=0,goal_achieved=False))
        return model.code,metadata
    evaluation.load_checkpoint=own_loader
    try:
        evaluation.evaluate(COMPONENTS,REPORTS/'inputs'/f'shard-{shard}'/'questions.jsonl',
            REPORTS/f'candidate-{shard}.json',device='cuda:0',max_new_tokens=512)
    finally:
        evaluation.load_checkpoint=original_loader
    owner.owned(plan);guard.check(full=True)


def combine(plan):
    from slm_lab.code_quality import execute,instruction_code
    from slm_lab.apps_native_metrics_v1 import summarize
    rows=list(map(json.loads,(DATA/'questions.jsonl').read_text().splitlines()));source={r['id']:r for r in rows}
    before=json.loads(BEFORE.read_text());baseline=summarize(before,'public-Python.json')
    if (before['data_sha256']!=file_hash(DATA/'questions.jsonl')
            or before['checkpoint_sha256']!=plan['own_code_target_sha256']
            or before['inference_device_type']!='cuda' or before['inference_precision']!='bf16'):
        raise ValueError('Exact actual original full-code source policy and native comparison protocol required')
    native=[];reports=[]
    for shard in range(4):
        path=REPORTS/f'candidate-{shard}.json';report=json.loads(path.read_text());reports.append(path)
        expected=rows[shard::4]
        if ([r['id'] for r in report['results']]!=[r['id'] for r in expected]
                or report['checkpoint_sha256']!=plan['components_sha256'] or report['generation_budget']!=512
                or report['inference_device_type']!='cuda' or report['inference_precision']!='bf16'
                or report['evaluation_provenance']!=before['evaluation_provenance']
                or report['tokenizer_sha256']!=before['tokenizer_sha256']
                or report['data_sha256']!=file_hash(REPORTS/'inputs'/f'shard-{shard}'/'questions.jsonl')):
            raise ValueError('Exact original subset, native runtime, precision, prompts and budget required')
        summarize(report,'public-Python.json');native.extend(report['results'])
    mapped={r['id']:r for r in native}
    if len(mapped)!=175 or len(native)!=175 or mapped.keys()!=source.keys():
        raise ValueError('All original175public problems exactly once required')
    passing=[]
    for identity,row in mapped.items():
        if instruction_code(row['raw_response'])!=row['code'] or row['task_id']!=source[identity]['task_id']:
            raise ValueError('Every actual delivered code and original task identity must reconstruct')
        if row['execution'].get('functional_passed'):
            cases=source[identity]['cases']
            actual=execute(dict(language='Python',code=row['code'],function=source[identity]['function'],cases=cases,stress=[]))
            if not actual.get('functional_passed') or actual['passed_cases']!=len(cases):
                raise ValueError('Every newly claimed pass must reproduce against all original published cases')
            passing.append(dict(id=identity,reexecution=actual))
    old={r['id']:r['execution'].get('functional_passed',False) for r in before['results']}
    after={i:r['execution'].get('functional_passed',False) for i,r in mapped.items()}
    if old.keys()!=after.keys() or before['generation_budget']!=512:
        raise ValueError('Exact complete original matched source baseline required')
    atomic_json(REPORTS/'comparison.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
        baseline_own_full_code=baseline,compressed_own_code=dict(tasks=175,
            syntax=sum(r['execution'].get('syntax_valid',False) for r in native),functional=sum(after.values())),
        gained_ids=sorted(i for i in after if after[i] and not old[i]),lost_ids=sorted(i for i in old if old[i] and not after[i]),
        every_new_claimed_pass_reexecuted_against_all_original_cases=passing,
        original_full_questions_sha256=file_hash(DATA/'questions.jsonl'),baseline_report_sha256=file_hash(BEFORE),
        actual_new_loader_source_sha256=file_hash(ROOT/'src/slm_lab/own_code_delta_inference_v1.py'),
        actual_new_architecture_source_sha256=file_hash(ROOT/'src/slm_lab/own_code_delta_v1.py'),
        all_original_native_generation_and_execution_runtime_files_and_precision_match=True,
        model_architecture_and_checkpoint_are_the_experimental_change=True,
        reports_sha256={str(p):file_hash(p) for p in reports},
        native_report_keeps_raw_response_not_generation_token_ids=True,
        not_full_standardized_MBPP_or_certified_independent_test=True,
        finite_case_passes_not_algorithmic_efficiency_proof=True,model_updates=0,candidate_promoted=False,goal_achieved=False))


def parent(files,predecessors):
    if REPORTS.exists():
        raise ValueError('Preserve earlier own-delta benchmark outputs')
    if any(owner.resources.same(a) for a in predecessors):
        raise ValueError('All exact original math/grounding owners must end before new GPU acquisition')
    for folder in [ROOT/'reports/apps-full-language-GPU-math-v1',ROOT/'reports/point-grounding-joint-v1']:
        if (folder/'children.json').exists() and any(owner.resources.same(a) for a in json.loads((folder/'children.json').read_text())):
            raise ValueError('All registered preceding GPU workers must actually end')
    guard=FileBindings(files)
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);owner.idle_original()
        memory=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.splitlines()
        if len(memory)!=4 or any(int(v.strip())>128 for v in memory):
            raise ValueError('All four public-diagnostic GPUs must actually be unused')
        guard.check(full=True);REPORTS.mkdir();rows=seal_shards(DATA/'questions.jsonl',REPORTS/'inputs')
        if len(rows)!=175:
            raise ValueError('Complete original175problem subset required')
        for shard in range(4):
            for name in ['questions.jsonl','manifest.json']:
                path=REPORTS/'inputs'/f'shard-{shard}'/name;files[str(path)]=file_hash(path)
        guard=FileBindings(files)
        plan=dict(owner=owner.resources.identity(os.getpid()),waiter=owner.WAITER,
            source_files_sha256=files,predecessors=predecessors,expires_at_unix=time.time()+2*60*60,
            components_sha256=file_hash(COMPONENTS),base_sha256=file_hash(BASE),
            own_code_target_sha256=json.loads((PREPARATION/'plan.json').read_text())['own_code_target_sha256'],
            preparation_plan_sha256=file_hash(PREPARATION/'plan.json'),max_new_tokens=512,precision='CUDA BF16',
            automatic_retry=False,model_updates=0,goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[])
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned code-delta diagnostic interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        watcher=subprocess.Popen(['.venv/bin/python','-u',str(Path(__file__).resolve()),'--watchdog'],cwd=ROOT,
            stdout=(REPORTS/'watchdog.log').open('xb'),stderr=subprocess.STDOUT,start_new_session=True)
        try:
            owner.idle_original();os.kill(owner.WAITER['pid'],signal.SIGSTOP);deadline=time.monotonic()+5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in ['T','t']:
                if time.monotonic()>deadline:raise ValueError('Original idle waiter did not enter held state')
                time.sleep(.05)
            if (Path('/proc')/str(owner.WAITER['pid'])/'task'/str(owner.WAITER['pid'])/'children').read_text().strip():
                raise ValueError('Original waiter began work before hold; do not start new GPU workers')
            children=[owner.launch(plan,['-u',str(Path(__file__).resolve()),'--worker',str(i)],'public-'+str(i),str(i)) for i in range(4)]
            owner.wait_children(plan,children);combine(plan);guard.check(full=True)
        except BaseException as problem:
            atomic_json(REPORTS/'failure.json',dict(error=str(problem),automatic_retry=False,goal_achieved=False));raise
        finally:
            owner.restore_owned(plan,'Owned own-code-delta public diagnostic ended');watcher.wait(timeout=20)


def queue():
    if QUEUE.exists() or REPORTS.exists():raise ValueError('Preserve earlier queued own-delta comparison')
    preparation=json.loads((PREPARATION/'plan.json').read_text());files=dict(preparation['source_files_sha256'])
    from slm_lab.evaluation_provenance import capture
    files.update({str(ROOT/n):s for n,s in capture()['files_sha256'].items()})
    names=['scripts/assess_own_code_delta_public_v1.py','scripts/train_point_grounding_joint_v1.py',
        'scripts/assess_joint_java_comparison_v1.py','src/slm_lab/own_code_delta_inference_v1.py',
        'src/slm_lab/apps_native_metrics_v1.py','tests/test_own_code_delta_public_v1.py',
        'reports/own-code-delta-public-v1-tests.log','reports/own-code-delta-inference-v1-tests.log',
        'reports/own-code-delta-v1/complete.json','reports/own-code-delta-v1/CPU-calibration-complete.json',
        'data/public-code-assessment-v1/questions.jsonl','data/public-code-assessment-v1/manifest.json',
        'checkpoints/own-code-delta-v1/model.pt','checkpoints/own-code-delta-v1/tokenizer.json']
    for name in names:files[str(ROOT/name)]=file_hash(ROOT/name)
    for path in [BEFORE,PREPARATION/'plan.json']:files[str(path)]=file_hash(path)
    if '2 passed' not in (ROOT/'reports/own-code-delta-public-v1-tests.log').read_text():
        raise ValueError('Actual whole-source shard coverage and active-owner rejection checks required')
    predecessors=[]
    for name in ['apps-full-language-GPU-math-v1','point-grounding-joint-v1']:
        path=ROOT/'reports'/(name+'-queue.json');files[str(path)]=file_hash(path)
        predecessors.append(json.loads(path.read_text())['queue_owner'])
    guard=FileBindings(files)
    atomic_json(QUEUE,dict(queued_at=datetime.now(timezone.utc).isoformat(),owner=owner.resources.identity(os.getpid()),
        exact_predecessors=predecessors,source_files_sha256=files,original_public_problems=175,
        pending_not_started=True,model_updates=0,goal_achieved=False))
    while any(owner.resources.same(a) for a in predecessors):guard.check();time.sleep(5)
    while True:
        guard.check()
        if not owner.resources.same(owner.WAITER):raise ValueError('Original data waiter ended; inspect a new resource window')
        try:owner.idle_original();break
        except ValueError:time.sleep(5)
    parent(files,predecessors)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',type=int,choices=[0,1,2,3]);parser.add_argument('--watchdog',action='store_true');args=parser.parse_args()
    if args.watchdog:owner.watchdog()
    elif args.worker is not None:worker(args.worker)
    else:queue()
