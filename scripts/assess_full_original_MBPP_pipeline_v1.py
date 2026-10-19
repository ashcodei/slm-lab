"""Assess all600original reserved problems at eight fixed own-model endpoints.

Runs after the complete new-code continuation and its original assessments.
Uses all four idle original L40s, every original test/challenge case and fixed
native512greedy generation. Publisher test500 remains the primary denominator.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from assess_full_held_QA_pipeline_v1 import descriptors as original_descriptors
import train_whole_human_SFT_v1 as legacy
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/full-original-MBPP-pipeline-v1'
QUEUE=ROOT/'reports/full-original-MBPP-pipeline-v1-queue.json'
PREVIOUS=ROOT/'reports/focused-code-continuation-v1'
PREVIOUS_QUEUE=ROOT/'reports/focused-code-continuation-v1-queue.json'
ENDPOINTS=('phase_initial','phase_trained','human_chosen_CE','human_DPO','code_anchor_CE','code_anchor_execution','focused_initial','focused_trained')
owner=legacy.owner


def configure():
    owner.REPORTS=REPORTS;owner.resources.REPORTS=REPORTS;legacy.REPORTS=REPORTS


def descriptors(root):
    root=Path(root).resolve();result=original_descriptors(root)
    for arm in ['initial','trained']:
        name='focused_'+arm
        result[name]=dict(name=name,kind='foundation',arm=arm,
            checkpoint=str(root/'checkpoints/focused-code-continuation-v1'/arm/'model.pt'),
            plan=str(root/'reports/focused-code-continuation-v1/plan.json'))
    if tuple(result)!=ENDPOINTS:raise ValueError('All eight fixed endpoints required without score selection')
    return result


def previous_ready(identity):
    if owner.resources.same(identity):return False
    children=PREVIOUS/'children.json'
    if children.exists() and any(owner.resources.same(r) for r in json.loads(children.read_text())):return False
    if ((PREVIOUS/'failure.json').exists() or (ROOT/'reports/focused-code-continuation-v1-parent-failure.json').exists()
            or any(not (PREVIOUS/name).exists() for name in ['plan.json','complete.json','restored.json','training-complete.json'])):
        raise ValueError('Complete exact continuation, native assessments and all-worker restoration required')
    plan=json.loads((PREVIOUS/'plan.json').read_text());done=json.loads((PREVIOUS/'complete.json').read_text())
    training=json.loads((PREVIOUS/'training-complete.json').read_text())
    if (done.get('all58488new_code_and_all_admitted_execution_examples_once') is not True
            or done.get('actual_original_QA_question_endpoint_assessments')!=33922
            or done.get('actual_original_AI2D_question_endpoint_condition_assessments')!=18528
            or done.get('standalone_model_and_tokenizer_with_no_older_checkpoint_dependency') is not True
            or training.get('actual_updates')!=plan['planned_steps']
            or training.get('all58488new_code_and_all_admitted_public_execution_targets_once') is not True):
        raise ValueError('Every original source update and full text/visual assessment must complete before new code judging')
    for rank in range(4):
        receipt=json.loads((PREVIOUS/f'standalone-cold-reload-{rank}.json').read_text())
        if (receipt.get('physical_GPU')!=rank or receipt.get('actual_completed_updates')!=plan['planned_steps']
                or receipt.get('every_exported_parameter_bit_exact') is not True):
            raise ValueError('Every actual completed native standalone cold-load receipt required')
    return True


def worker(endpoint,gpu):
    plan=legacy.registered('MBPP-'+endpoint+'-'+str(gpu),gpu)
    import torch
    from slm_lab.MBPP_original_population_v2 import FullOriginalMBPP
    from slm_lab.MBPP_all_native_evaluation_v1 import evaluate_shard
    from slm_lab.dense_foundation_inference_v1 import DenseNativeView
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85,0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name(0)!='NVIDIA L40':raise ValueError('Only the owned original isolated physical L40 is admitted')
    spec=plan['endpoints'][endpoint];guard=FileBindings(plan['source_files_sha256'])
    if spec['kind']=='foundation':from slm_lab.fast_joint_inference_v2 import load
    elif spec['kind']=='human':from slm_lab.phase_human_policy_v1 import load
    elif spec['kind']=='code':from slm_lab.broad_anchored_policy_v1 import load
    else:raise ValueError('Exact own native policy loader required')
    model,metadata=load(spec['checkpoint'],spec['plan'],spec['arm'],expected_checkpoint_sha256=spec['checkpoint_sha256'],
        expected_plan_sha256=spec['plan_sha256'],device='cuda:0')
    native=DenseNativeView(model).eval() if spec['kind']=='foundation' else model.policy.eval()
    count=sum(p.numel() for p in native.parameters())
    if count!=(984399690 if spec['kind']=='foundation' else 999706442):raise ValueError('Every loaded own student value must match its fixed under1B endpoint')
    population=FullOriginalMBPP(ROOT)
    if metadata['tokenizer_sha256']!=population.tokenizer_sha256 or population.identity_sha256!=plan['original_population_sha256']:
        raise ValueError('Exact original600cases and own tokenizer required')
    folder=REPORTS/endpoint;folder.mkdir(exist_ok=True);journal=folder/f'MBPP-shard-{gpu}-actual-deliveries.jsonl'
    with journal.open('x') as stream:
        completed=0
        def record(row):
            nonlocal completed
            stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n');stream.flush();completed+=1
            if completed%10==0:atomic_json(folder/f'MBPP-shard-{gpu}-progress.json',dict(actual_original_problems=completed,
                original_selected_problems=150,last_original_task_id=row['original_task_id'],model_updates=0,goal_achieved=False))
        with inference_cache():report=evaluate_shard(native,population,folder/f'MBPP-shard-{gpu}.json',rank=gpu,world=4,
            checkpoint_sha256=spec['checkpoint_sha256'],checked=lambda:owner.owned(plan),record=record)
        os.fsync(stream.fileno())
    guard.check(full=True);owner.owned(plan)
    atomic_json(folder/f'MBPP-shard-{gpu}-complete.json',dict(endpoint=endpoint,physical_GPU=gpu,actual_original_problems=len(report['results']),
        actual_student_values=count,actual_delivery_journal_sha256=file_hash(journal),model_updates=0,goal_achieved=False))


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or REPORTS.exists() or QUEUE.exists():raise ValueError('Fresh CPU-hidden all-original MBPP evaluation queue required')
    from slm_lab.MBPP_original_population_v2 import FullOriginalMBPP
    from slm_lab.MBPP_all_native_evaluation_v1 import reconcile
    from slm_lab.local_source_closure_v2 import python_sources
    population=FullOriginalMBPP(ROOT);previous=json.loads(PREVIOUS_QUEUE.read_text());me=owner.resources.identity(os.getpid())
    tests={'mbpp-original-function-cases-v2':4,'MBPP-all-native-evaluation-v1-final':2,'full-original-MBPP-pipeline-v1':2}
    files=dict(population.bindings)
    for name,count in tests.items():
        path=ROOT/'reports'/(name+'-tests.log')
        if str(count)+' passed' not in path.read_text():raise ValueError('Actual original-case, raw-neural-delivery and resource controls required')
        files[str(path)]=file_hash(path)
    seeds=[Path(__file__),ROOT/'tests/test_full_original_MBPP_pipeline_v1.py',ROOT/'tests/test_MBPP_all_native_evaluation_v1.py']
    files.update({str(p):file_hash(p) for p in [PREVIOUS_QUEUE,*python_sources(ROOT,seeds)]})
    guard=FileBindings(files);expires=time.time()+14*86400
    atomic_json(QUEUE,dict(at=time.time(),owner=me,exact_predecessor=previous['owner'],endpoints=descriptors(ROOT),
        source_files_sha256=files,status='waiting for exact complete new-code continuation and all original assessments/restoration',
        all_eight_endpoints_fixed_before_new_model_MBPP_outputs=True,original_problems_each_endpoint=600,
        primary_publisher_test_problems=500,validation_and_prompting_scored_separately=True,
        original_population_sha256=population.identity_sha256,actual_neural_deliveries=0,
        original_reference_qualifications_never_filter_primary_questions=True,
        not_official_few_shot_protocol=True,expires_at_unix=expires,automatic_retry=False,model_updates=0,goal_achieved=False))
    while not previous_ready(previous['owner']):
        if time.time()>expires:raise ValueError('Queue observation lifetime exceeded; original jobs left intact')
        guard.check();time.sleep(5)
    specs=descriptors(ROOT)
    for spec in specs.values():
        spec['checkpoint_sha256']=file_hash(spec['checkpoint']);spec['plan_sha256']=file_hash(spec['plan'])
        files[spec['checkpoint']]=spec['checkpoint_sha256'];files[spec['plan']]=spec['plan_sha256']
    for path in [PREVIOUS/'complete.json',PREVIOUS/'restored.json']:files[str(path)]=file_hash(path)
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);owner.idle_original()
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(used)!=4 or any(int(x)>128 for x in used):raise ValueError('All four original GPUs must be idle before acquisition')
        if shutil.disk_usage(ROOT).free<27*1024**3:raise ValueError('Preserve complete report storage and original persistent reserve')
        guard=FileBindings(files);guard.check(full=True);REPORTS.mkdir()
        plan=dict(owner=me,waiter=owner.WAITER,endpoints=specs,source_files_sha256=files,expires_at_unix=time.time()+72*3600,
            original_population_sha256=population.identity_sha256,original_problems_each_endpoint=600,
            primary_publisher_test_problems=500,all_four_L40s_each_endpoint_in_sequence=True,
            no_original_case_removed_or_source_rewritten=True,automatic_retry=False,model_updates=0,candidate_promoted=False,goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[]);driver=str(Path(__file__).resolve())
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned original MBPP evaluation interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);watcher=None
        try:
            owner.idle_original();os.kill(owner.WAITER['pid'],signal.SIGSTOP);deadline=time.monotonic()+5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in ['T','t']:
                if time.monotonic()>deadline:raise ValueError('Exact original idle waiter hold failed')
                time.sleep(.05)
            watcher=subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,stdout=(REPORTS/'watchdog.log').open('x'),
                stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            summaries={}
            for endpoint in ENDPOINTS:
                children=[owner.launch(plan,['-u',driver,'--worker',endpoint,'--gpu',str(gpu)],'MBPP-'+endpoint+'-'+str(gpu),str(gpu)) for gpu in range(4)]
                owner.wait_children(plan,children)
                report=reconcile(population,[REPORTS/endpoint/f'MBPP-shard-{gpu}.json' for gpu in range(4)],REPORTS/endpoint/'full-original-MBPP.json')
                summaries[endpoint]=report['summary']
            atomic_json(REPORTS/'all-eight-endpoint-summaries.json',summaries);guard.check(full=True)
            atomic_json(REPORTS/'complete.json',dict(all_eight_fixed_endpoints_assessed=True,original_problems_each_endpoint=600,
                primary_publisher_test_problems=500,total_actual_original_problem_endpoint_deliveries=4800,
                all_original_cases_and_generated_actions_and_raw_values_retained=True,
                every_raw_value_verdict_rederived_without_source_eligibility_exclusions=True,
                finite_original_cases_not_general_correctness_or_asymptotic_efficiency_proof=True,
                not_official_few_shot_protocol=True,model_updates=0,candidate_promoted=False,goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS/'failure.json',dict(error=repr(error),automatic_retry=False,model_updates=0,goal_achieved=False));raise
        finally:
            owner.restore_owned(plan,'All-original MBPP evaluation ended; original work restored')
            if watcher is not None:watcher.wait(timeout=20)


if __name__=='__main__':
    configure();parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watchdog',action='store_true')
    parser.add_argument('--worker',choices=ENDPOINTS);parser.add_argument('--gpu',type=int,choices=range(4));args=parser.parse_args()
    if args.watchdog:owner.watchdog()
    elif args.worker:worker(args.worker,args.gpu)
    else:parent()
