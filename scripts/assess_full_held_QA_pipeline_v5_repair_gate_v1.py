"""Queue full native QA assessment of nine fixed completed own-model endpoints.

Waits for complete code-v4 source epochs, original assessments and restoration.
All four original GPUs are then used for evaluation only. No endpoint is
selected as a winner and no training, automatic retry or promotion occurs.
"""
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

import train_whole_human_SFT_v1 as legacy
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.public_pretraining_lineage_tests_v1 import bind_tests
from slm_lab.diverse_images import atomic_json
from slm_lab.public_pretraining_endpoints_v1 import ENDPOINTS,COMPARISONS,descriptors,load_native

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/full-held-QA-pipeline-v5'
QUEUE=ROOT/'reports/full-held-QA-pipeline-v5-repair-gate-v1-queue.json'
PREVIOUS=ROOT/'reports/broad-anchored-code-v7'
PREVIOUS_QUEUE=ROOT/'reports/broad-anchored-code-v7-repair-gate-v1-queue.json'
PREPARATION=ROOT/'reports/full-held-QA-evaluation-v1-preparation/complete.json'
FINALIZER=ROOT/'reports/public-pretraining-final-standalone-CPU-v1'
FINALIZER_QUEUE=ROOT/'reports/public-pretraining-final-standalone-CPU-v1-repair-gate-v1-queue.json'
import own_training_gpu_lease_v2 as owner
legacy.owner=owner


def configure():
    owner.REPORTS=REPORTS;owner.resources.REPORTS=REPORTS;legacy.REPORTS=REPORTS


def previous_ready(identity):
    if owner.resources.same(identity):return False
    children=PREVIOUS/'children.json'
    if children.exists() and any(owner.resources.same(p) for p in json.loads(children.read_text())):return False
    if (PREVIOUS/'failure.json').exists() or any(not (PREVIOUS/name).exists() for name in ['plan.json','complete.json','restored.json']):
        raise ValueError('Complete exact code predecessor, native assessments and restoration required')
    plan=json.loads((PREVIOUS/'plan.json').read_text());done=json.loads((PREVIOUS/'complete.json').read_text())
    if (not done.get('all_3721_original_coding_sources_once_each_arm') or
            not done.get('matched_native_code_broad_human_preference_and_held_APPS_assessments_complete')
            or plan['original_coding_sources']!=3721 or plan['planned_steps']!=3721):
        raise ValueError('Complete fixed original coding source population and all predecessor assessments required')
    for arm in ['anchor_CE','anchor_execution']:
        trained=json.loads((PREVIOUS/(arm+'-training-complete.json')).read_text())
        receipt=json.loads((PREVIOUS/('train-'+arm+'-resume-distributed-state.json')).read_text())
        if (trained.get('actual_updates')!=3721 or trained.get('all_3721_original_coding_sources_once') is not True
                or receipt.get('completed')!=3721 or {p['rank'] for p in receipt['all_four_private_states']}!=set(range(4))):
            raise ValueError('Both exact complete four-worker code epoch receipts required')
    return True


def finalizer_ready(identity):
    if owner.resources.same(identity):return False
    if (FINALIZER/'failure.json').exists() or not (FINALIZER/'complete.json').is_file():
        raise ValueError('Exact final CPU export must complete before full endpoint assessments')
    done=json.loads((FINALIZER/'complete.json').read_text())
    if (done.get('complete') is not True or done.get('completed_updates')!=3721
            or done.get('actual_counts',{}).get('total_unique')!=984399690
            or done.get('all_actual_exported_parameters_and_complete_native_probes_cold_load_bit_exact') is not True
            or done.get('real_word_logits_exact_through_fold_and_cold_native') is not True
            or done.get('old_checkpoint_actor_optimizer_or_reference_not_needed_for_inference') is not True):
        raise ValueError('Actual complete standalone text/photo/word/parameter checks required')
    return True


def worker(endpoint,gpu):
    plan=legacy.registered('QA-'+endpoint+'-'+str(gpu),gpu)
    import torch
    from slm_lab.dense_foundation_inference_v1 import DenseNativeView
    from slm_lab.full_held_QA_native_v1 import FullHeldQA,evaluate_shard
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85,0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name(0)!='NVIDIA L40':raise ValueError('Only the original owned physical L40 is permitted')
    guard=FileBindings(plan['source_files_sha256']);guard.check(full=True)
    spec=plan['endpoints'][endpoint]
    native,metadata=load_native(spec,device='cuda:0')
    actual_count=sum(p.numel() for p in native.parameters())
    signature=[(id(p),p._version,p.data_ptr()) for p in native.parameters()]
    population=FullHeldQA(ROOT)
    if metadata['tokenizer_sha256']!=population.tokenizer_sha256:raise ValueError('The own endpoint and complete held population must share exactly one tokenizer')
    folder=REPORTS/endpoint;folder.mkdir(exist_ok=True);journal=folder/f'QA-shard-{gpu}-actual-deliveries.jsonl'
    with journal.open('x') as output:
        count=0
        def record(row):
            nonlocal count
            output.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n');output.flush();count+=1
            if count%100==0:
                os.fsync(output.fileno());atomic_json(folder/f'QA-shard-{gpu}-progress.json',dict(actual_questions=count,
                    original_selected_questions=len(population.cases[gpu::4]),last_original_id=row['audit_id'],model_updates=0,goal_achieved=False))
        with inference_cache():
            report=evaluate_shard(native,population.tokenizer,population,folder/f'QA-shard-{gpu}.json',rank=gpu,world=4,
                checkpoint_sha256=spec['checkpoint_sha256'],checked=lambda:owner.owned(plan),record=record)
        os.fsync(output.fileno())
    if signature!=[(id(p),p._version,p.data_ptr()) for p in native.parameters()] or any(p.grad is not None for p in native.parameters()):
        raise ValueError('Evaluation must preserve every actual native model value and zero gradients')
    owner.owned(plan);guard.check(full=True)
    atomic_json(folder/f'QA-shard-{gpu}-complete.json',dict(physical_GPU=gpu,endpoint=endpoint,
        actual_original_questions=len(report['results']),actual_student_values=actual_count,
        actual_delivery_journal_sha256=file_hash(journal),all_loaded_values_unchanged=True,model_updates=0,goal_achieved=False))


def paired(before,after):
    for key in ['original_population_sha256','fixed_generation_budgets','inference_device_type','inference_precision']:
        if before[key]!=after[key]:raise ValueError('Every native QA comparison must use the same full protocol and population')
    if len(before['results'])!=16961 or len(after['results'])!=16961:raise ValueError('Every original QA outcome required')
    groups={}
    for a,b in zip(before['results'],after['results']):
        for key in ['audit_id','benchmark','original_native_prompt_ids','generation_budget','inspected_exact_match_scopes']:
            if a[key]!=b[key]:raise ValueError('Every paired original question, input token and exposure stratum must match')
        item=groups.setdefault(a['benchmark'],dict(questions=0,before_correct=0,after_correct=0,gained=[],lost=[]))
        item['questions']+=1;item['before_correct']+=a['verdict']['correct'];item['after_correct']+=b['verdict']['correct']
        if not a['verdict']['correct'] and b['verdict']['correct']:item['gained'].append(a['audit_id'])
        if a['verdict']['correct'] and not b['verdict']['correct']:item['lost'].append(a['audit_id'])
    return dict(by_original_benchmark=groups,candidate_promoted=False,goal_achieved=False)


def parent():
    if '16 passed' not in (ROOT/'reports/public-pretraining-lineage-drivers-v1-final2-tests.log').read_text():
        raise ValueError('Actual complete revised public-pretraining lineage and every original benchmark endpoint must pass')

    if '24 passed' not in (ROOT/'reports/terminal-waiter-pipeline-drivers-v1-tests.log').read_text() or '3 passed' not in (ROOT/'reports/terminal-waiter-gpu-lease-v2-tests.log').read_text():raise ValueError('Actual revised pipeline and terminal-waiter kernel-lease controls required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or REPORTS.exists() or QUEUE.exists():raise ValueError('Fresh CPU-hidden evaluation queue required')
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.full_held_QA_native_v1 import FullHeldQA,reconcile
    population=FullHeldQA(ROOT);preparation=json.loads(PREPARATION.read_text())
    if preparation['original_population_sha256']!=population.identity_sha256:raise ValueError('Exact complete16961-question preparation required')
    previous=json.loads(PREVIOUS_QUEUE.read_text());finalizer=json.loads(FINALIZER_QUEUE.read_text());me=owner.resources.identity(os.getpid())
    files=dict(population.bindings)
    extra=[PREPARATION,PREVIOUS_QUEUE,FINALIZER_QUEUE,ROOT/'reports/full-held-QA-native-v1-final-tests.log',ROOT/'reports/state-code-first-full-assessment-drivers-v1-tests.log']
    extra+=python_sources(ROOT,[Path(__file__),ROOT/'tests/test_full_held_QA_pipeline_v4.py'])
    files.update({str(p):file_hash(p) for p in extra});bind_tests(files,ROOT);guard=FileBindings(files);guard.check(full=True)
    expires=time.time()+14*24*3600
    atomic_json(QUEUE,dict(status='waiting for exact complete code-v4 epochs, native assessments and restoration',owner=me,
        predecessor=previous['owner'],exact_finalizer_owner=finalizer['owner'],endpoints=descriptors(ROOT),full_QA_questions_each_endpoint=16961,
        all_nine_endpoints_predeclared_not_score_selected=True,source_files_sha256=files,expires_at_unix=expires,
        automatic_retry=False,model_updates=0,candidate_promoted=False,goal_achieved=False))
    while not (previous_ready(previous['owner']) and finalizer_ready(finalizer['owner'])):
        if time.time()>expires:raise ValueError('Declared evaluation queue lifetime exceeded; no automatic retry')
        guard.check(full=False);time.sleep(2)
    specs=descriptors(ROOT)
    for spec in specs.values():
        spec['checkpoint_sha256']=file_hash(spec['checkpoint']);spec['plan_sha256']=file_hash(spec['plan'])
        files[spec['checkpoint']]=spec['checkpoint_sha256'];files[spec['plan']]=spec['plan_sha256']
    for path in [PREVIOUS/'complete.json',PREVIOUS/'restored.json',FINALIZER/'plan.json',FINALIZER/'complete.json']:files[str(path)]=file_hash(path)
    files.update(owner.lease_sources())
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);owner.idle_original()
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(used)!=4 or any(int(value)>128 for value in used):raise ValueError('All four original GPUs must be idle before acquisition')
        if shutil.disk_usage(ROOT).free<27*1024**3:raise ValueError('Preserve26GiB persistent reserve and full evaluation report budget')
        bind_tests(files,ROOT);guard=FileBindings(files);guard.check(full=True);REPORTS.mkdir()
        plan=dict(owner=me,waiter=owner.WAITER,terminal_waiter_receipt_sha256=owner.terminal_receipt_sha256(),endpoints=specs,source_files_sha256=files,expires_at_unix=time.time()+72*3600,
            full_QA_questions_each_endpoint=16961,full_population_sha256=population.identity_sha256,
            all_four_L40s_each_endpoint_in_sequence=True,automatic_retry=False,model_updates=0,candidate_promoted=False,goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[])
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned full native QA assessment interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        watcher=None;driver=str(Path(__file__).resolve())
        try:
            owner.hold_original(plan)
            watcher=subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,
                stdout=(REPORTS/'watchdog.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            reports={}
            for endpoint in ENDPOINTS:
                children=[owner.launch(plan,['-u',driver,'--worker',endpoint,'--gpu',str(gpu)],'QA-'+endpoint+'-'+str(gpu),str(gpu)) for gpu in range(4)]
                owner.wait_children(plan,children)
                reports[endpoint]=reconcile(population,[REPORTS/endpoint/f'QA-shard-{gpu}.json' for gpu in range(4)],REPORTS/endpoint/'full-QA.json')
            comparisons={a+'__to__'+b:paired(reports[a],reports[b]) for a,b in COMPARISONS}
            atomic_json(REPORTS/'paired-comparisons.json',comparisons);guard.check(full=True)
            atomic_json(REPORTS/'complete.json',dict(all_nine_predeclared_endpoints_assessed=True,original_questions_each_endpoint=16961,
                total_original_question_endpoint_assessments=len(ENDPOINTS)*16961,full_original_inputs_and_budgets_preserved=True,
                all_raw_deliveries_and_question_verdicts_retained=True,automatic_selection_or_promotion=False,
                full_independent_output_readback_and_broader_multimodal_testing_still_required=True,model_updates=0,goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS/'failure.json',dict(error=repr(error),automatic_retry=False,model_updates=0,goal_achieved=False));raise
        finally:
            owner.restore_owned(plan,'Full native QA assessment ended; original work restored')
            if watcher is not None:watcher.wait(timeout=20)


if __name__=='__main__':
    configure();parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watchdog',action='store_true')
    parser.add_argument('--worker',choices=ENDPOINTS);parser.add_argument('--gpu',type=int,choices=range(4));args=parser.parse_args()
    if args.watchdog:owner.watchdog()
    elif args.worker:worker(args.worker,args.gpu)
    else:parent()
