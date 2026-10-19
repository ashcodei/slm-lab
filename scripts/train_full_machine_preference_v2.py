"""Matched full1620pair execution-preference epochs on the four original L40s.

Preserves the entire existing eight-endpoint pipeline. Runs only after its
last GPU owner and all workers finish with complete assessment/restoration
receipts. Both methods start from the same complete own code/RL actor. A
fresh capacity/recovery/cold-load gate precedes either production epoch.
"""
import argparse
from datetime import timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import train_whole_human_SFT_v1 as legacy
import own_training_gpu_lease_v2 as owner
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.public_pretraining_lineage_tests_v1 import bind_tests
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/full-machine-preference-v2'
OUTPUT=ROOT/'checkpoints/full-machine-preference-v2'
QUEUE=ROOT/'reports/full-machine-preference-v2-queue.json'
PREVIOUS=ROOT/'reports/full-original-MBPP-pipeline-v5'
PREVIOUS_QUEUE=ROOT/'reports/full-original-MBPP-pipeline-v5-queue.json'
CODE=ROOT/'reports/broad-anchored-code-v7'
CODE_QUEUE=ROOT/'reports/broad-anchored-code-v7-queue.json'
ARMS=('chosen_CE','machine_DPO')
SUITES=('code','broad','preferences','QA','AI2D','MBPP')
MAX_STATE=256*1024**2
RESERVE=26*1024**3
legacy.owner=owner


def configure():
    owner.REPORTS=REPORTS;owner.resources.REPORTS=REPORTS
    legacy.REPORTS=REPORTS;legacy.OUTPUT=OUTPUT


def terminal_stage(folder, identity):
    if owner.resources.same(identity):return False
    children=folder/'children.json'
    if children.exists() and any(owner.resources.same(p) for p in json.loads(children.read_text())):return False
    if (folder/'failure.json').exists() or any(not (folder/name).is_file() for name in ['plan.json','complete.json','restored.json']):
        raise ValueError('Exact predecessor must complete successfully with every worker terminal and restoration retained')
    if json.loads((folder/'plan.json').read_text())['owner']!=identity:
        raise ValueError('The completed stage must have the exact predeclared owner birth')
    return True


def previous_ready(identity):
    if not terminal_stage(PREVIOUS,identity):return False
    from slm_lab.public_pretraining_endpoints_v1 import ENDPOINTS
    plan=json.loads((PREVIOUS/'plan.json').read_text());done=json.loads((PREVIOUS/'complete.json').read_text())
    if (set(plan.get('endpoints',{}))!=set(ENDPOINTS) or done.get('all_nine_fixed_endpoints_assessed') is not True
            or done.get('original_problems_each_endpoint')!=600 or done.get('primary_publisher_test_problems')!=500
            or done.get('total_actual_original_problem_endpoint_deliveries')!=len(ENDPOINTS)*600
            or done.get('all_original_cases_and_generated_actions_and_raw_values_retained') is not True
            or done.get('every_raw_value_verdict_rederived_without_source_eligibility_exclusions') is not True):
        raise ValueError('All eight original complete MBPP endpoints and retained native actions must finish first')
    for endpoint in ENDPOINTS:
        if not (PREVIOUS/endpoint/'full-original-MBPP.json').is_file():raise ValueError('Every original MBPP endpoint report required')
        for rank in range(4):
            row=json.loads((PREVIOUS/endpoint/f'MBPP-shard-{rank}-complete.json').read_text())
            if (row.get('endpoint')!=endpoint or row.get('physical_GPU')!=rank
                    or row.get('actual_original_problems')!=150 or row.get('model_updates')!=0):
                raise ValueError('Every original four-worker MBPP receipt must reconcile')
    return True


def code_ready(identity):
    if not terminal_stage(CODE,identity):return False
    plan=json.loads((CODE/'plan.json').read_text());done=json.loads((CODE/'complete.json').read_text())
    if (plan.get('original_coding_sources')!=3721 or plan.get('planned_steps')!=3721
            or done.get('all_3721_original_coding_sources_once_each_arm') is not True
            or done.get('matched_native_code_broad_human_preference_and_held_APPS_assessments_complete') is not True):
        raise ValueError('Both complete original3721source code/RL epochs and their native retention assessments required')
    for arm in ['anchor_CE','anchor_execution']:
        trained=json.loads((CODE/(arm+'-training-complete.json')).read_text())
        receipt=json.loads((CODE/('train-'+arm+'-resume-distributed-state.json')).read_text())
        if (trained.get('actual_updates')!=3721 or trained.get('all_3721_original_coding_sources_once') is not True
                or receipt.get('completed')!=3721 or {p['rank'] for p in receipt['all_four_private_states']}!=set(range(4))
                or any(p['actual_completed']!=3721 for p in receipt['all_four_private_states'])):
            raise ValueError('All four private optimizer/cursor states of each complete code arm are required')
    return True


def training_options():
    return dict(seed=991429,global_batch_size=16,learning_rate=1e-6,
        betas=[.9,.95],weight_decay=.01,gradient_clip=1.,beta=.1,chosen_CE_weight=.1)


def extreme_batches(data):
    from slm_lab.machine_code_preference_stage_v1 import pair_information
    rows=[pair_information(data,p) for p in range(len(data))];batches=[]
    for field in ['chosen','rejected']:
        for measure in ['length','targets']:
            order=sorted(range(len(data)),key=lambda p:(len(data.sequence(p,field)[0]) if measure=='length'
                else rows[p][field+'_response_EOS_targets'],p),reverse=True)[:16]
            if order not in batches:batches.append(order)
    return batches


def distributed(mode,arm):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1,2,3' or int(os.environ['WORLD_SIZE'])!=4:
        raise ValueError('All four owned original local GPUs are required')
    rank=int(os.environ['RANK']);plan=owner.registered(mode+'-'+arm+'-rank-'+str(rank))
    import copy
    import random
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint,capture_rng
    from slm_lab.execution_contrast_data_v1 import ExecutionContrastData
    from slm_lab.machine_preference_policy_v2 import fresh,make_reference,options,sources,export
    from slm_lab.parallel_machine_preference_v1 import ParallelMachinePreferenceStage,local_gradients
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1);device=torch.device('cuda',rank);torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.85,device);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name(device)!='NVIDIA L40':raise ValueError('Only the four original local L40s may be used')
    dist.init_process_group('nccl',timeout=timedelta(minutes=40))
    control=dist.new_group(backend='gloo',timeout=timedelta(minutes=40));started=time.monotonic()
    try:
        digest=file_hash(REPORTS/'plan.json');guard=FileBindings(plan['source_files_sha256'])
        model,_=fresh(plan,device);reference=make_reference(model,plan)
        data=ExecutionContrastData(ROOT,purpose='optimization');setting=options(plan,arm)
        parameters=[p for _,p in model.named_selected(('actor',))]
        optimizer=torch.optim.AdamW(parameters,lr=setting['learning_rate'],betas=tuple(setting['betas']),
                                    weight_decay=setting['weight_decay'],foreach=False)
        amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16)
        stage=ParallelMachinePreferenceStage(model,optimizer,data,reference,setting,sources(plan,digest),
            world=4,rank=rank,gradient_group=dist.group.WORLD,control_group=control,amp=amp)
        if len(data)!=1620 or stage.planned!=102 or stage.CPU_order is not None:
            raise ValueError('Every complete original1620pair source epoch is required on production GPUs')
        random.seed(setting['seed']+rank);np.random.seed(setting['seed']+rank);torch.manual_seed(setting['seed']+rank)
        frozen=tensor_digest(model.base.named_parameters());critic=model.parts['critic'].weight.detach().clone()
        initial=tensor_digest(model.parts.named_parameters())
        def adam():return tensor_digest((str(i)+'.'+k,state[k]) for i,state in enumerate(optimizer.state.values())
                                       for k in ['step','exp_avg','exp_avg_sq'])
        def save_private(label):
            path=OUTPUT/f'{mode}-{arm}-{label}-rank-{rank}.pt'
            atomic_checkpoint(path,stage.state(),MAX_STATE,RESERVE)
            proof=dict(rank=rank,path=str(path),sha256=file_hash(path),actual_completed=stage.completed)
            proofs=stage.gather(proof)
            if rank==0:atomic_json(REPORTS/f'{mode}-{arm}-{label}-distributed-state.json',dict(
                all_four_private_states=proofs,completed=stage.completed,
                complete_distributed_state_requires_every_private_file=True,goal_achieved=False))
            return path
        def step():owner.owned(plan);return stage.update()
        extremes=[];cold_proof=None
        if mode=='capacity':
            first=step();saved_path=save_private('step1');saved_hash=file_hash(saved_path)
            def draws():return [random.random(),float(np.random.random()),float(torch.rand(())),float(torch.rand((),device=device))]
            expected_draws=draws();expected=step();learned=tensor_digest(model.parts.named_parameters());moments=adam()
            payload=torch.load(saved_path,map_location='cpu',weights_only=True,mmap=True);bad=copy.deepcopy(payload)
            bad['core']['core']['rng']['cuda']=torch.zeros(4,dtype=torch.uint8)
            RNG=capture_rng(device);versions=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in model.parameters()]
            try:stage.resume(bad)
            except ValueError:pass
            else:raise ValueError('Invalid private CUDA RNG must reject')
            if (versions!=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in model.parameters()]
                    or moments!=adam() or not torch.equal(RNG['torch'],torch.get_rng_state())
                    or not torch.equal(RNG['cuda'],torch.cuda.get_rng_state(device))):
                raise ValueError('Rejected recovery mutated actor, Adam or actual private RNG')
            stage.resume(payload)
            if draws()!=expected_draws or step()!=expected or learned!=tensor_digest(model.parts.named_parameters()) or moments!=adam():
                raise ValueError('Actual next global machine-source update, actor/Adam and all private RNG must recover exactly')
            if file_hash(saved_path)!=saved_hash:
                raise ValueError('The original private checkpoint bytes must remain intact')
            save_private('step2')
            for positions in plan['capacity_original_pair_batches']:
                owner.owned(plan);optimizer.zero_grad(set_to_none=True)
                rows=local_gradients(model,data,positions,reference,setting,rank=rank,world=4,device=device,amp=amp)
                if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters)
                        or any(p.grad is not None for p in stage.fixed)):
                    raise ValueError('Whole longest chosen/rejected programs need finite actor-only gradients on every GPU')
                extremes.append(dict(global_original_positions=positions,actual_local_complete_pair_records=rows,
                                     backward_only_no_optimizer_commit=True));optimizer.zero_grad(set_to_none=True)
            # Reload a new complete own model from immutable ancestry, then import
            # only the actual trained student actor. No private reference is needed.
            cold,_=fresh(plan,device);cold.import_export(model.export(sources(plan,digest)),expected_sources=sources(plan,digest))
            model.eval();cold.eval()
            long_position=plan['capacity_longest_chosen_position'];batch=data.batch([long_position],'chosen',device)
            inputs={k:v for k,v in batch.items() if k!='labels'}
            with torch.no_grad(),amp():a=model(**inputs)['logits'];b=cold(**inputs)['logits']
            if (not torch.equal(a,b) or tensor_digest(model.named_parameters())!=tensor_digest(cold.named_parameters())):
                raise ValueError('All actual student values and the whole longest native output must cold-load exactly')
            cold_proof=dict(all_exported_student_parameter_values_exact=True,whole_longest_native_output_exact=True,
                complete_context_tokens=batch['input_ids'].shape[1],source_position=long_position,
                private_training_reference_not_used_for_cold_inference=True)
            del cold,a,b,batch,inputs,payload,bad
        else:
            if rank==0 and arm=='chosen_CE':atomic_checkpoint(OUTPUT/'initial-actor.pt',export(model,plan,digest,'initial',0),96*1024**2,RESERVE)
            dist.barrier(group=control);journal=(REPORTS/(arm+'-metrics.jsonl')).open('x') if rank==0 else None
            try:
                while stage.completed<stage.planned:
                    tick=time.monotonic();record=step()
                    if journal:
                        journal.write(json.dumps(record)+'\n');journal.flush()
                        if stage.completed==1 or stage.completed%10==0 or stage.completed==stage.planned:
                            atomic_json(REPORTS/(arm+'-progress.json'),dict(actual_updates=stage.completed,planned_updates=stage.planned,
                                latest=record,last_update_seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-started,goal_achieved=False))
                    # Immutable per-generation states preserve earlier committed
                    # all-worker recovery while a new generation is being saved.
                    if stage.completed%25==0 or stage.completed==stage.planned:save_private('step'+str(stage.completed))
                if rank==0:
                    atomic_checkpoint(OUTPUT/(arm+'-actor.pt'),export(model,plan,digest,arm,stage.completed),96*1024**2,RESERVE)
                    atomic_json(REPORTS/(arm+'-training-complete.json'),dict(actual_updates=stage.completed,
                        original_machine_pairs=len(data),all_original_machine_pairs_once=True,rounded_repeats=0,
                        human_preferences=False,held_quality_pending=True,candidate_promoted=False,goal_achieved=False))
            finally:
                if journal:journal.close()
        if (frozen!=tensor_digest(model.base.named_parameters()) or not torch.equal(critic,model.parts['critic'].weight)
                or initial==tensor_digest(model.parts.named_parameters())):
            raise ValueError('Actor must learn with every own frozen dense/vision/word and unused critic value exact')
        reference.assert_bound(model);guard.check(full=True);data.guard.check(full=True)
        if mode=='capacity':atomic_json(REPORTS/f'capacity-{arm}-{rank}.json',dict(physical_GPU=rank,actual_counts=model.counts(),
            actual_Adam_states=len(optimizer.state),actual_logical_updates=2,physical_optimizer_calls=3,
            exact_next_global_update_weights_Adam_private_RNG=True,invalid_private_CUDA_RNG_rejected_before_mutation=True,
            complete_source_extremes=extremes,actual_cold_load=cold_proof,resident_inventory=reference.resident_inventory(model),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            seconds=time.monotonic()-started,production_updates=0,goal_achieved=False))
        dist.barrier(group=control)
    finally:dist.destroy_process_group()


def assess(arm,gpu,suite):
    plan=legacy.registered('assess-'+arm+'-'+suite+'-'+str(gpu),gpu)
    import torch
    from slm_lab.machine_preference_policy_v2 import load
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85,0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name(0)!='NVIDIA L40':raise ValueError('An original isolated physical L40 is required')
    checkpoint=OUTPUT/(arm+'-actor.pt');guard=FileBindings(plan['source_files_sha256'])
    model,metadata=load(checkpoint,REPORTS/'plan.json',arm,expected_checkpoint_sha256=file_hash(checkpoint),
        expected_plan_sha256=file_hash(REPORTS/'plan.json'),device='cuda:0')
    folder=REPORTS/arm;folder.mkdir(exist_ok=True)
    versions=[(id(p),p._version,p.data_ptr()) for p in model.parameters()]
    if suite=='preferences':
        from slm_lab.held_human_shards_v1 import evaluate_shard
        from slm_lab.whole_human_data_v2 import WholeHumanData
        data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','valid','evaluation')
        provenance=dict(checkpoint_sha256=file_hash(checkpoint),run_plan_sha256=file_hash(REPORTS/'plan.json'),
            human_manifest_sha256=plan['human_manifest_sha256'],frozen_dense_sha256=plan['frozen_dense_sha256'],
            inference_device_type='cuda',inference_precision='float32')
        evaluate_shard(model,data,folder/f'held-preferences-shard-{gpu}.json',rank=gpu,world=4,
            provenance=provenance,beta=plan['training_options']['beta'],checked=lambda:owner.owned(plan))
    elif suite in ['QA','AI2D','MBPP']:
        if suite=='QA':
            from slm_lab.full_held_QA_native_v1 import FullHeldQA,evaluate_shard
            population=FullHeldQA(ROOT)
        elif suite=='AI2D':
            from slm_lab.ai2d_full_native_evaluation_v1 import FullAI2D,evaluate_shard
            population=FullAI2D(ROOT)
        else:
            from slm_lab.MBPP_original_population_v2 import FullOriginalMBPP
            from slm_lab.MBPP_all_native_evaluation_v1 import evaluate_shard
            population=FullOriginalMBPP(ROOT)
        if metadata['tokenizer_sha256']!=population.tokenizer_sha256:raise ValueError('The exact own tokenizer must match the original full assessment population')
        journal=folder/f'{suite}-shard-{gpu}-actual-deliveries.jsonl'
        with journal.open('x') as stream:
            count=0
            def record(row):
                nonlocal count
                stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n');stream.flush();count+=1
                if count%25==0:atomic_json(folder/f'{suite}-shard-{gpu}-progress.json',dict(actual_original_questions=count,model_updates=0,goal_achieved=False))
            with inference_cache():
                args=(model.policy,population,folder/f'{suite}-shard-{gpu}.json') if suite=='MBPP' else (
                    model.policy,population.tokenizer,population,folder/f'{suite}-shard-{gpu}.json')
                report=evaluate_shard(*args,rank=gpu,world=4,checkpoint_sha256=file_hash(checkpoint),
                    checked=lambda:owner.owned(plan),record=record)
            os.fsync(stream.fileno())
        atomic_json(folder/f'{suite}-shard-{gpu}-complete.json',dict(arm=arm,physical_GPU=gpu,
            actual_original_questions=len(report['results']),actual_delivery_journal_sha256=file_hash(journal),model_updates=0,goal_achieved=False))
    else:
        def evaluate(module,name,data=None,**kwargs):
            original=module.load_checkpoint;module.load_checkpoint=lambda *a,**kw:(model.policy,metadata)
            try:
                owner.owned(plan)
                if data is None:module.evaluate(checkpoint,folder/name,device='cuda:0',**kwargs)
                else:module.evaluate(checkpoint,data,folder/name,device='cuda:0',**kwargs)
            finally:module.load_checkpoint=original
        original_threads=torch.set_num_threads;torch.set_num_threads=lambda count:None
        try:
            with inference_cache():
                if suite=='code':
                    from slm_lab.native_code_shards_v1 import evaluate_shard
                    for frame in ['completion','instruction']:evaluate_shard(model.policy,metadata,checkpoint,folder,rank=gpu,world=4,frame=frame)
                elif gpu==0:
                    from slm_lab import knowledge_evaluation
                    evaluate(knowledge_evaluation,'knowledge.json',Path(plan['pulse_sources']['knowledge']),max_examples=56,max_new_tokens=32)
                elif gpu==1:
                    from slm_lab import math_evaluation
                    evaluate(math_evaluation,'math.json',ROOT/'data/math-reasoning-v2/test.jsonl',max_examples=16,max_new_tokens=256)
                elif gpu==2:
                    from slm_lab import ocr_evaluation,object_evaluation
                    evaluate(ocr_evaluation,'ocr.json',Path(plan['pulse_sources']['ocr']),max_examples=32,max_new_tokens=128,image_size=448)
                    evaluate(object_evaluation,'objects.json',Path(plan['pulse_sources']['objects']),max_examples=32,max_new_tokens=32)
                else:
                    from slm_lab import vision_evaluation
                    evaluate(vision_evaluation,'vision.json',Path(plan['pulse_sources']['vision']),max_examples=32,image_size=448)
        finally:torch.set_num_threads=original_threads
    if versions!=[(id(p),p._version,p.data_ptr()) for p in model.parameters()] or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Every assessment must preserve all student values and absent gradients')
    guard.check(full=True);owner.owned(plan)
    atomic_json(folder/f'assessment-{suite}-{gpu}-complete.json',dict(physical_GPU=gpu,actual_counts=model.counts(),
        checkpoint_sha256=file_hash(checkpoint),all_loaded_values_unchanged=True,model_updates=0,goal_achieved=False))


def parent():
    if '16 passed' not in (ROOT/'reports/public-pretraining-lineage-drivers-v1-final2-tests.log').read_text():
        raise ValueError('Actual complete revised public-pretraining lineage and every original benchmark endpoint must pass')

    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(p.exists() for p in [REPORTS,OUTPUT,QUEUE]):
        raise ValueError('A fresh GPU-hidden CPU queue must preserve all earlier attempts')
    from slm_lab.execution_contrast_data_v1 import ExecutionContrastData
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.full_held_QA_native_v1 import FullHeldQA
    from slm_lab.ai2d_full_native_evaluation_v1 import FullAI2D
    from slm_lab.MBPP_original_population_v2 import FullOriginalMBPP
    data=ExecutionContrastData(ROOT,purpose='optimization');previous=json.loads(PREVIOUS_QUEUE.read_text())
    code=json.loads(CODE_QUEUE.read_text());me=owner.resources.identity(os.getpid());setting=training_options()
    if len(data)!=1620 or math.ceil(len(data)/setting['global_batch_size'])!=102:raise ValueError('The complete fixed original source epoch is required')
    tests={'parallel-machine-preference-v1-tests':7,'machine-preference-policy-v1-tests':2,
           'full-machine-preference-v1-driver-tests':7,'terminal-waiter-gpu-lease-v2-tests':3}
    proof_path=ROOT/'reports/parallel-machine-preference-four-CPU-v2/complete.json';proof=json.loads(proof_path.read_text())
    if not proof['all_four_CPU_workers_passed'] or not proof['both_matched_arms']:raise ValueError('Actual four-worker matched whole-source recovery proof is required')
    files={str(p):digest for p,digest in data.guard.files.items()}
    for name,count in tests.items():
        path=ROOT/'reports'/(name+'.log')
        if str(count)+' passed' not in path.read_text():raise ValueError('Actual objective, cold-load, full-epoch and lease tests must pass before queueing')
        files[str(path)]=file_hash(path)
    for worker in proof['proofs']:
        FileBindings(worker['source_files_sha256']).check(full=True);files.update(worker['source_files_sha256'])
    populations=[FullHeldQA(ROOT),FullAI2D(ROOT),FullOriginalMBPP(ROOT)]
    for population in populations:files.update(population.bindings)
    seeds=[Path(__file__),ROOT/'tests/test_full_machine_preference_v1_driver.py',ROOT/'tests/test_public_pretraining_machine_policy_v2.py']
    extra=[PREVIOUS_QUEUE,CODE_QUEUE,proof_path,data.folder/'admission.json',data.folder/'manifest.json',data.folder/'pairs.jsonl',data.tokenizer_path,
           *python_sources(ROOT,seeds)]
    files.update({str(p):file_hash(p) for p in extra});bind_tests(files,ROOT);guard=FileBindings(files);expires=time.time()+21*86400
    atomic_json(QUEUE,dict(owner=me,exact_predecessor=previous['owner'],exact_starting_code_owner=code['owner'],
        status='waiting after the intact original eight-endpoint GPU pipeline',source_files_sha256=files,
        matched_arms=list(ARMS),original_machine_pairs_each_arm=1620,planned_updates_each_arm=102,
        fixed_training_options=setting,own_nonzero_starting_reference=True,human_preferences=False,
        full_QA_questions_each_endpoint=16961,full_AI2D_questions_each_endpoint=3088,
        AI2D_image_conditions=['proper_image','gray_image','wrong_image'],full_MBPP_problems_each_endpoint=600,
        held_human_pairs_each_endpoint=4642,all_existing_nine_endpoint_populations_preserved=True,
        fresh_four_L40_capacity_required=True,production_updates=0,expires_at_unix=expires,
        automatic_retry=False,candidate_promoted=False,goal_achieved=False))
    del populations
    while not previous_ready(previous['owner']):
        if time.time()>expires:raise ValueError('Declared queue lifetime ended without stopping or restarting original work')
        guard.check();time.sleep(5)
    if not code_ready(code['owner']):raise ValueError('Original complete code/RL actor must be terminal before use')
    import torch
    from slm_lab.broad_anchored_policy_v3 import load as load_code
    from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference
    from slm_lab.machine_preference_policy_v2 import reference_sources
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1);source_plan=CODE/'plan.json';checkpoint=ROOT/'checkpoints/broad-anchored-code-v7/anchor_execution-actor.pt'
    inputs=dict(checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),plan=str(source_plan),plan_sha256=file_hash(source_plan))
    code_plan=json.loads(source_plan.read_text());files.update(code_plan['source_files_sha256'])
    model,_=load_code(checkpoint,source_plan,'anchor_execution',expected_checkpoint_sha256=inputs['checkpoint_sha256'],
                     expected_plan_sha256=inputs['plan_sha256'],device='cpu')
    if model.counts()['actual_unique']!=999706442:raise ValueError('The actual full own sub1B actor is required')
    for path in [source_plan,checkpoint,CODE/'complete.json',CODE/'restored.json',PREVIOUS/'complete.json',PREVIOUS/'restored.json']:
        files[str(path)]=file_hash(path)
    files.update(owner.lease_sources())
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);owner.idle_original()
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(used)!=4 or any(int(v)>128 for v in used):raise ValueError('Every original L40 must be idle before ownership')
        if shutil.disk_usage(ROOT).free<RESERVE+64*MAX_STATE+2*1024**3:
            raise ValueError('All private capacity/recovery generations, complete reports and persistent reserve must fit')
        bind_tests(files,ROOT);guard=FileBindings(files);guard.check(full=True);REPORTS.mkdir();OUTPUT.mkdir()
        plan=dict(starting_policy_inputs=inputs,actual_counts=model.counts(),actor_settings=model.settings,
            frozen_dense_sha256=tensor_digest(model.base.named_parameters()),starting_actor_parts_sha256=tensor_digest(model.parts.named_parameters()),
            original_machine_pairs=len(data),planned_steps=102,training_options=setting,
            machine_admission_sha256=file_hash(data.folder/'admission.json'),machine_manifest_sha256=file_hash(data.folder/'manifest.json'),
            human_manifest_sha256=code_plan['human_manifest_sha256'],pulse_sources=code_plan['pulse_sources'],
            tokenizer_sha256=data.provenance['tokenizer_sha256'],capacity_original_pair_batches=extreme_batches(data),
            capacity_longest_chosen_position=max(range(len(data)),key=lambda p:len(data.sequence(p,'chosen')[0])),
            owner=me,waiter=owner.WAITER,terminal_waiter_receipt_sha256=owner.terminal_receipt_sha256(),
            source_files_sha256=files,expires_at_unix=time.time()+7*86400,arms=list(ARMS),
            all_original_machine_pairs_once_each_arm=True,rounded_repeats=0,human_preferences=False,
            rejected_programs_never_positive_CE_targets=True,each_arm_uses_all_four_GPUs_in_sequence=True,
            training_precision='bf16_autocast_with_FP32_actor_and_Adam',
            original_nine_endpoint_pipeline_unchanged=True,full_original_QA_AI2D_MBPP_and_held_human_assessments_required=True,
            automatic_retry=False,candidate_promoted=False,goal_achieved=False)
        plan['own_reference_binding']=FrozenOwnActorReference(model,sources=reference_sources(plan)).binding()
        del model,data
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[])
        driver=str(Path(__file__).resolve());watcher=None
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned full machine preference comparison interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        def run(mode,arm):
            child=owner.launch(plan,['-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',driver,
                '--distributed',mode,'--arm',arm],mode+'-'+arm+'-launcher','0,1,2,3');owner.wait_children(plan,[child])
        try:
            owner.hold_original(plan)
            watcher=subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,
                stdout=(REPORTS/'watchdog.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            for arm in ARMS:
                run('capacity',arm)
                proofs=[json.loads((REPORTS/f'capacity-{arm}-{rank}.json').read_text()) for rank in range(4)]
                if any(p['physical_GPU']!=rank or p['actual_counts']!=plan['actual_counts'] or p['actual_Adam_states']!=561
                        or not p['exact_next_global_update_weights_Adam_private_RNG']
                        or not p['actual_cold_load']['all_exported_student_parameter_values_exact']
                        or not p['actual_cold_load']['whole_longest_native_output_exact'] for rank,p in enumerate(proofs)):
                    raise ValueError('Every actual full-size four-L40 capacity, recovery and cold-load check must pass')
            atomic_json(REPORTS/'capacity.json',dict(passed=True,arms=list(ARMS),production_updates=0,goal_achieved=False))
            for arm in ARMS:run('train',arm)
            atomic_json(REPORTS/'training-complete.json',dict(all1620original_machine_pairs_once_each_arm=True,
                actual_updates_each_arm=102,human_preferences=False,held_assessments_pending=True,goal_achieved=False))
            for arm in ['initial',*ARMS]:
                for suite in SUITES:
                    children=[owner.launch(plan,['-u',driver,'--assess',arm,'--suite',suite,'--gpu',str(gpu)],
                        'assess-'+arm+'-'+suite+'-'+str(gpu),str(gpu)) for gpu in range(4)]
                    owner.wait_children(plan,children)
                    reconcile_assessment(arm,suite)
            guard.check(full=True)
            atomic_json(REPORTS/'complete.json',dict(all1620original_machine_pairs_once_each_arm=True,
                full_original_QA16961_AI2D3088times3_MBPP600_and_human4642_each_endpoint_complete=True,
                matched_initial_and_both_training_arms_assessed=True,original_nine_endpoint_pipeline_preserved=True,
                independent_generated_output_readback_and_new_standalone_finalization_pending=True,
                candidate_promoted=False,goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS/'failure.json',dict(error=repr(error),automatic_retry=False,goal_achieved=False));raise
        finally:
            owner.restore_owned(plan,'Full machine preference comparison ended; owned workers released')
            if watcher is not None:watcher.wait(timeout=20)


def reconcile_assessment(arm,suite):
    folder=REPORTS/arm
    if suite=='code':
        from slm_lab.native_code_shards_v1 import reconcile
        for frame in ['completion','instruction']:reconcile([folder/f'code-{frame}-shard-{rank}.json' for rank in range(4)],folder/f'code-{frame}.json')
    elif suite=='preferences':
        from slm_lab.held_human_shards_v1 import reconcile
        from slm_lab.whole_human_data_v2 import WholeHumanData
        reconcile(WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','valid','evaluation'),
            [folder/f'held-preferences-shard-{rank}.json' for rank in range(4)],folder/'held-preferences.json')
    elif suite=='QA':
        from slm_lab.full_held_QA_native_v1 import FullHeldQA,reconcile
        reconcile(FullHeldQA(ROOT),[folder/f'QA-shard-{rank}.json' for rank in range(4)],folder/'full-QA.json')
    elif suite=='AI2D':
        from slm_lab.ai2d_full_native_evaluation_v1 import FullAI2D,reconcile
        reconcile(FullAI2D(ROOT),[folder/f'AI2D-shard-{rank}.json' for rank in range(4)],folder/'full-AI2D.json')
    elif suite=='MBPP':
        from slm_lab.MBPP_original_population_v2 import FullOriginalMBPP
        from slm_lab.MBPP_all_native_evaluation_v1 import reconcile
        reconcile(FullOriginalMBPP(ROOT),[folder/f'MBPP-shard-{rank}.json' for rank in range(4)],folder/'full-original-MBPP.json')
    elif suite!='broad':raise ValueError('Only the complete predeclared assessment suites are allowed')


if __name__=='__main__':
    configure();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watchdog',action='store_true');parser.add_argument('--distributed',choices=['capacity','train'])
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--assess',choices=['initial',*ARMS])
    parser.add_argument('--suite',choices=SUITES);parser.add_argument('--gpu',type=int,choices=range(4));args=parser.parse_args()
    if args.watchdog:owner.watchdog()
    elif args.distributed:distributed(args.distributed,args.arm)
    elif args.assess:assess(args.assess,args.gpu,args.suite)
    else:parent()
