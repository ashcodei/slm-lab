"""Record actual source-bound warmup queue and tested complete native reader."""
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from slm_lab.function_repair_GPU_contract_v1 import ADMISSION,QUEUE as GPU_QUEUE,handoff_admitted

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/public-reasoning-warmup-readiness-v1-review'
QUEUE=ROOT/'reports/public-reasoning-warmup-CPU-v1-queue.json'


def read(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    pending.replace(path)


def live(owner):
    fields=(Path('/proc')/str(owner['pid'])/'stat').read_text().split(') ',1)[1].split()
    if fields[19]!=owner['ticks'] or fields[0] in {'Z','X'}:raise ValueError('Every actual original and new process birth must remain live')
    return dict(owner=owner,state=fields[0])


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists():raise ValueError('Fresh low-priority CPU queue review required')
    queue=read(QUEUE);new_owner=live(queue['owner']);pid=queue['owner']['pid']
    command=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    environ=(Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
    if ('train_public_reasoning_warmup_CPU_v1.py' not in command or b'CUDA_VISIBLE_DEVICES=' not in environ
            or any((ROOT/name).exists() for name in ['reports/public-reasoning-warmup-CPU-v1/plan.json',
                'reports/public-reasoning-warmup-CPU-v1-parent-failure.json','checkpoints/public-reasoning-warmup-CPU-v1'])):
        raise ValueError('The exact new CUDA-hidden controller must be live and unstarted, without a model or failure')
    if (queue['planned_updates']!=483 or queue['response_EOS_targets']!=57240 or queue['reserved_examples']!=289
            or queue['exact_two_thread_predecessor']!=dict(pid=28339,ticks='51862171')):
        raise ValueError('Complete original source/evaluation scope and exact CPU allocation predecessor required')
    tests=[('public-reasoning-stage-v1-tests.log','6 passed'),('public-reasoning-reserved-evaluation-v1-tests.log','2 passed'),
           ('public-reasoning-warmup-gate-v1-tests.log','2 passed')]
    if any(expected not in (ROOT/'reports'/name).read_text() for name,expected in tests):raise ValueError('All actual stage/native/gate controls must pass')
    audit=read(ROOT/'reports/completion-audit.json');previous=Path(audit['latest_full_scope_review']);review=copy.deepcopy(read(previous))
    gpu=read(ROOT/GPU_QUEUE);handoff=read(ROOT/ADMISSION)
    if not handoff_admitted(ROOT,gpu['owner']):raise ValueError('All ten original full-scope successors must remain admitted')
    maps=[queue['source_files_sha256'],gpu['source_files_sha256'],
        read(ROOT/'reports/public-pretraining-recovery-v1/plan.json')['source_files_sha256'],
        read(ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json')['source_files_sha256'],
        read(ROOT/'reports/public-pretraining-interim-CPU-v2/evaluation-plan.json')['source_files_sha256'],
        read(ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue.json')['source_files_sha256']]
    maps += [read(ROOT/item['queue'])['source_files_sha256'] for item in handoff['new_controllers']]
    for name in ['reports/completion-audit.json','README.md','RESEARCH.md']:
        if any(str(ROOT/name) in mapping for mapping in maps):raise ValueError('Never edit a live source-bound index/document')
    files=dict(queue['source_files_sha256'])
    for path in [Path(__file__),QUEUE,previous,*[ROOT/'reports'/name for name,_ in tests]]:files[str(path)]=sha(path)
    if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('Every tested new queue/source/data byte must stay exact')
    raw=(ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes();prefix=raw[:raw.rfind(b'\n')+1]
    rows=list(map(json.loads,prefix.splitlines()))
    if [row['step'] for row in rows]!=list(range(1,len(rows)+1)):raise ValueError('Exact contiguous actual main update prefix required')
    current=dict(owner=dict(pid=1241,ticks='49351203'),complete_actual_updates=len(rows),planned_updates=4447,
        response_EOS_targets=sum(int(row['response_EOS_targets']) for row in rows),complete_prefix_sha256=hashlib.sha256(prefix).hexdigest())
    protected=[live(owner) for owner in [current['owner'],dict(pid=18545,ticks='50128787'),dict(pid=25240,ticks='51635952'),queue['exact_two_thread_predecessor'],gpu['owner']]]
    if any(row['state'] in {'T','t'} for row in protected+[new_owner]):raise ValueError('All active original neural/research work and new waiter must remain running')
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if {int(value) for value in compute.split()}!={2287,2288,2289,2290}:raise ValueError('The four exact original main GPU workers must remain the only compute contexts')
    at=datetime.now(timezone.utc).isoformat();OUT.mkdir()
    proof=dict(complete=True,at=at,source_files_sha256=files,
        previous_goal_turn_classification='progress: actual nine source/native/neural controls for whole-answer warmup and fresh verifier-group gradients',
        current_goal_turn_classification='progress: tested source-bound Adam/RNG stage, complete289case native evaluation, real predecessor gates and actual live full-size warmup queue',
        actual_new_waiting_owner=new_owner,actual_controller_command=command,actual_wait_status=read(ROOT/'reports/public-reasoning-warmup-CPU-v1-queue-progress.json'),
        actual_stage_recovery_controls_passed=6,actual_reserved_native_controls_passed=2,actual_live_process_gate_controls_passed=2,
        actual_complete_reserved_transport_test_population=289,reserved_transport_test_used_explicit_empty_output_small_model=True,
        actual_full_size_capacity_or_generation_throughput_proved=False,actual_new_production_updates=0,
        complete_warmup_budget=dict(source_examples=2568,epochs=3,updates=483,presentations=7704,response_EOS_targets=57240),
        baseline_and_final_native_scope=dict(reserved=289,reserved_action_budgets=[64,256],source_functions=15,source_budget=1024,
            local_code=48,code_budget=512,ordinary_Python=24,knowledge=56,math=16,OCR=20,objects=15,photos=32,image_side=448),
        no_GPU_lease_or_context=True,reuses_only_exact_released_two_thread_CPU_lock=True,
        new_RL_stage_checkpoint_controls_are_mechanics_only_and_no_CPU_RL_production_is_started=True,
        actual_original_main=current,actual_original_live_processes=protected,actual_original_GPU_compute_PIDs=sorted(map(int,compute.split())),
        actual_GPU_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True),
        original_CPU_comparison=read(ROOT/'reports/full-function-repair-comparison-CPU-v1/progress.json'),
        original_interim_assessment=read(ROOT/'reports/public-pretraining-interim-CPU-v2/progress.json'),
        all_original17families_human_RL_and_nine_endpoint_benchmark_scopes_unchanged=True,
        all_original_ten_downstream_owners_live=True,candidate_promoted=False,goal_achieved=False)
    atomic(OUT/'complete.json',proof)
    lines=['# Complete-source reasoning warmup queued','',
        'The new warmup is an actual live CUDA-hidden CPU controller, not a running model. It waits for the exact existing two-thread '
        'preference-comparison owner to finish all original training and native assessments, verifies complete public pretraining, then acquires the same released CPU lock. '
        'A stale complete receipt never substitutes for an actually terminal process birth. It preserves the original writer reserve and requires a32GiB working set plus64GiB other memory headroom.', '',
        'All **2,568** admitted train sources appear in each of **three complete epochs**: **483 updates**, **7,704 presentations** and **57,240 response/EOS targets**. '
        'The initializer must be the unchanged complete own4,447-update public model. Its complete984,399,690dense/head values plus the original actor/unused critic must total999,706,442unique inference parameters. '
        'Before production, the queued job must actually pass full-size16-source capacity, exact next-update actor/Adam/metrics/private-RNG recovery, '
        'source-mask/RNG/options rejection, all561selected Adam leaves and longest-source cold BF16 equality. These checks have not yet run.', '',
        'A checkpointable stage now binds whole-source schedules, native masks, exact optimizer groups, precision/options, source hashes and own-reference identities. '
        'Six actual small own-neural controls pass for FP32/BF16 warmup and FP32anchor_CE/anchor_RL. They reject corrupted source masks, cursors, options, Adam state, '
        'RNG, sampled rewards/actions and behavior logs before neural mutation. Next-update weights, optimizer state, metrics and private RNG reproduce exactly; cold outputs match. '
        'A reduced source schedule or small neural graph cannot enter production mode. A failed partial source update clears gradients and cannot be committed or exported.', '',
        'The queued warmup requires matched baseline and final evaluation on **all289reserved reasoning cases**, **15original source-validation functions at1024tokens**, '
        '**48local code cases at512tokens**, **24ordinaryPython executions**, and all original knowledge/math/OCR/object/photo/efficiency checks. '
        'The new reserved evaluator preserves every whole original prompt and native64/256budget, records actual generated tensors unchanged, and never places a reference answer into the model input. '
        'Two controls pass, including all289actual native deliveries from a deliberately initialized small model that returns an empty answer. '
        'That transport control proves empty answers do not receive credit; it is not production-model accuracy or full-size generation capacity. '
        'Two real disposable-process tests prove predecessor/receipt gates.', '',
        '**The new warmup has0production updates and no production plan/model yet.** The actual parent is confirmed live and waiting. '
        'The RL checkpoint mechanics are available for later integration, but this queue launches only supervised warmup; it does not launch a long CPU RL job. '
        'The289reserved examples are withheld only from this new source curriculum. Historical exposure and arbitrary semantic equivalence remain uncertified. '
        'No candidate is automatically adopted.', '',
        f'The original main continues at **{len(rows):,}/4,447** updates and **{current["response_EOS_targets"]:,}** actual targets. '
        'The four original GPU compute PIDs remain2287–2290; no new GPU context or lease exists. All original17source families, public human preference/DPO, '
        'execution RL, machine preferences and complete nine-endpoint final assessments remain intact. '
        '**Latest complete local functional correctness remains0/48; no new model-quality gain or frontier parity is established. All ten final requirements remain incomplete.**','']
    (OUT/'metrics.md').write_text('\n'.join(lines))
    review['requirements']['serious_full_model_training']['evidence']+=(
        ' An actual live CUDA-hidden nice19CPU parent now queues the complete2568source/3epoch/483update/57240target reasoning warmup '
        'after exact priorCPUcomparison completion and verified release of its two-thread lock. '
        'Six actual small own-neural source-bound stage recovery/rejection controls, two complete reserved/native controls and two real-process gates pass. '
        'Full-size16source capacity, all561Adam leaves, exact cold/next-update recovery and all production updates remain pending. '
        'The new queue must assess baseline and final289reserved/15source/48code/24ordinaryPython/broad cases. '
        'It creates no GPUcontext/lease and no production RL job; canonical sources and all final benchmarks remain unchanged.')
    if len(review['requirements'])!=10 or any(value['status']!='incomplete_for_requested_final_model' for value in review['requirements'].values()):
        raise ValueError('Queue readiness proves none of the requested final model capabilities')
    review.update(at=at,previous_full_scope_review=str(previous),previous_goal_turn_classification=proof['previous_goal_turn_classification'],
        current_goal_turn_classification=proof['current_goal_turn_classification'],latest_actual_reasoning_warmup_queue=proof,
        source_files_sha256={**files,str(OUT/'complete.json'):sha(OUT/'complete.json')},goal_achieved=False)
    atomic(OUT/'review.json',review)
    if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('Every frozen queue/source must remain exact before index update')
    audit.update(at=at,requirements=review['requirements'],latest_full_scope_review=str(OUT/'review.json'),latest_actual_main_runtime=current,
        latest_actual_reasoning_warmup_queue=str(OUT/'complete.json'),goal_achieved=False)
    atomic(ROOT/'reports/completion-audit.json',audit)
    paragraph=(f'Complete-source reasoning warmup queued, **{at}**: an actual live CUDA-hidden parent waits for the exact existing two-thread predecessor to finish. '
        '**10new source-bound stage/native/process-gate controls pass**. The future full-size warmup keeps all2,568sources/3epochs/483updates/57,240targets, '
        'then baseline/final289reserved/15source/48code/24ordinaryPython and full local broad checks. '
        '**Full-size capacity and production updates remain pending; the reserved transport fixture is not a model-quality result.** '
        f'The original main continues at **{len(rows)}/4,447** with the same four GPUworkers and unchanged source/human/RL/nine-endpoint scopes. '
        '**Latest complete functional correctness remains0/48; all ten final requirements remain incomplete.** '
        '[Actual queue and controls](reports/public-reasoning-warmup-readiness-v1-review/metrics.md), '
        '[full review](reports/public-reasoning-warmup-readiness-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md','RESEARCH.md']:
        path=ROOT/name;first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_waiter_owner=queue['owner'],new_production_updates=0,main_actual_updates=len(rows),goal_achieved=False)),flush=True)


if __name__=='__main__':main()
