"""Assess all600original reserved problems at eight fixed own-model endpoints.

Runs after all code-first training and the complete original text/visual assessments.
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

from slm_lab.code_first_endpoints_v3 import ENDPOINTS,descriptors,load_native
import train_whole_human_SFT_v1 as legacy
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/full-original-MBPP-pipeline-v4'
QUEUE=ROOT/'reports/full-original-MBPP-pipeline-v4-queue.json'
PREVIOUS=ROOT/'reports/full-AI2D-pipeline-v4'
PREVIOUS_QUEUE=ROOT/'reports/full-AI2D-pipeline-v4-queue.json'
import own_training_gpu_lease_v2 as owner
legacy.owner=owner


def configure():
    owner.REPORTS=REPORTS;owner.resources.REPORTS=REPORTS;legacy.REPORTS=REPORTS


def previous_ready(identity):
    if owner.resources.same(identity):return False
    children=PREVIOUS/'children.json'
    if children.exists() and any(owner.resources.same(r) for r in json.loads(children.read_text())):return False
    if ((PREVIOUS/'failure.json').exists()
            or any(not (PREVIOUS/name).exists() for name in ['plan.json','complete.json','restored.json'])):
        raise ValueError('Complete exact original visual predecessor and all-worker restoration required')
    plan=json.loads((PREVIOUS/'plan.json').read_text());done=json.loads((PREVIOUS/'complete.json').read_text())
    if (done.get('all_eight_predeclared_endpoints_assessed') is not True
            or done.get('original_questions_each_endpoint')!=3088
            or done.get('total_question_endpoint_assessments')!=24704
            or done.get('total_question_endpoint_condition_assessments')!=74112
            or set(plan.get('endpoints',{}))!=set(ENDPOINTS)
            or done.get('all_actual_native_pixel_identities_and_delivered_tokens_and_likelihoods_retained') is not True):
        raise ValueError('All eight full original visual assessments must finish before new code judging')
    for endpoint in ENDPOINTS:
        if not (PREVIOUS/endpoint/'full-AI2D.json').is_file():raise ValueError('Every complete original visual endpoint report required')
        for rank in range(4):
            receipt=json.loads((PREVIOUS/endpoint/f'AI2D-shard-{rank}-complete.json').read_text())
            if (receipt.get('physical_GPU')!=rank or receipt.get('endpoint')!=endpoint
                    or receipt.get('actual_original_questions')!=772 or receipt.get('actual_question_condition_assessments')!=2316
                    or receipt.get('all_loaded_values_unchanged') is not True or receipt.get('model_updates')!=0):
                raise ValueError('All32 complete original visual shard receipts required')
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
    guard.check(full=True);native,metadata=load_native(spec,device='cuda:0')
    count=sum(p.numel() for p in native.parameters())
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
    if '24 passed' not in (ROOT/'reports/terminal-waiter-pipeline-drivers-v1-tests.log').read_text() or '3 passed' not in (ROOT/'reports/terminal-waiter-gpu-lease-v2-tests.log').read_text():raise ValueError('Actual revised pipeline and terminal-waiter kernel-lease controls required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or REPORTS.exists() or QUEUE.exists():raise ValueError('Fresh CPU-hidden all-original MBPP evaluation queue required')
    from slm_lab.MBPP_original_population_v2 import FullOriginalMBPP
    from slm_lab.MBPP_all_native_evaluation_v1 import reconcile
    from slm_lab.local_source_closure_v2 import python_sources
    population=FullOriginalMBPP(ROOT);previous=json.loads(PREVIOUS_QUEUE.read_text());me=owner.resources.identity(os.getpid())
    tests={'mbpp-original-function-cases-v2':4,'MBPP-all-native-evaluation-v1-final':2,'state-code-first-full-assessment-drivers-v1':11}
    files=dict(population.bindings)
    for name,count in tests.items():
        path=ROOT/'reports'/(name+'-tests.log')
        if str(count)+' passed' not in path.read_text():raise ValueError('Actual original-case, raw-neural-delivery and resource controls required')
        files[str(path)]=file_hash(path)
    seeds=[Path(__file__),ROOT/'tests/test_full_original_MBPP_pipeline_v4.py',ROOT/'tests/test_MBPP_all_native_evaluation_v1.py']
    files.update({str(p):file_hash(p) for p in [PREVIOUS_QUEUE,*python_sources(ROOT,seeds)]})
    guard=FileBindings(files);expires=time.time()+14*86400
    atomic_json(QUEUE,dict(at=time.time(),owner=me,exact_predecessor=previous['owner'],endpoints=descriptors(ROOT),
        source_files_sha256=files,status='waiting for complete code-first text/visual assessments and restoration',
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
    files.update(owner.lease_sources())
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);owner.idle_original()
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(used)!=4 or any(int(x)>128 for x in used):raise ValueError('All four original GPUs must be idle before acquisition')
        if shutil.disk_usage(ROOT).free<27*1024**3:raise ValueError('Preserve complete report storage and original persistent reserve')
        guard=FileBindings(files);guard.check(full=True);REPORTS.mkdir()
        plan=dict(owner=me,waiter=owner.WAITER,terminal_waiter_receipt_sha256=owner.terminal_receipt_sha256(),endpoints=specs,source_files_sha256=files,expires_at_unix=time.time()+72*3600,
            original_population_sha256=population.identity_sha256,original_problems_each_endpoint=600,
            primary_publisher_test_problems=500,all_four_L40s_each_endpoint_in_sequence=True,
            no_original_case_removed_or_source_rewritten=True,automatic_retry=False,model_updates=0,candidate_promoted=False,goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[]);driver=str(Path(__file__).resolve())
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned original MBPP evaluation interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);watcher=None
        try:
            owner.hold_original(plan)
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
