"""Code-first complete new-code/output/state epoch before human and execution RL.

Starts from the complete own main checkpoint with no actor-fold arithmetic.
Defers full final assessments until post-training; keeps quick native controls.
Fresh four-L40 capacity/recovery gates the actual complete new-source epoch.
"""
import argparse
from datetime import datetime,timezone,timedelta
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

import train_phase_joint_v3 as joint
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/code-first-continuation-v2'
OUTPUT=ROOT/'checkpoints/code-first-continuation-v2'
QUEUE=ROOT/'reports/code-first-continuation-v2-queue.json'
INITIALIZER_REPORT=ROOT/'reports/code-first-continuation-v2-initializer'
MAIN=ROOT/'reports/phase-joint-v3'
HANDOFF=ROOT/'reports/code-first-state-v2-handoff/complete.json'
AUXILIARY=ROOT/'reports/public-execution-targets-v1-readback-admission'
STATES=ROOT/'reports/public-state-traces-full-v1-readback-admission'
MAX_STATE=14*1024**3
MAX_MODEL=5*1024**3
RESERVE=26*1024**3
owner=joint.owner


def configure():
    joint.REPORTS=REPORTS;joint.OUTPUT=OUTPUT;joint.configure()


def dependencies_ready(dependencies):
    if not HANDOFF.exists():return False
    handoff=json.loads(HANDOFF.read_text())
    if (handoff.get('old_unstarted_controllers_all_terminal') is not True
            or handoff.get('original_main_owner')!=dependencies['main']):
        raise ValueError('Explicit complete code-first handoff for the exact current main owner required')
    for name,folder in [('main',MAIN),('auxiliary',AUXILIARY),('states',STATES)]:
        if owner.resources.same(dependencies[name]):return False
        if (folder/'failure.json').exists() or not (folder/'complete.json').exists():
            raise ValueError('Every exact original '+name+' job must complete without substitution')
    if any(owner.resources.same(r) for r in json.loads((MAIN/'children.json').read_text())):return False
    if not (MAIN/'restored.json').exists():raise ValueError('Every original main worker must finish and restore original work')
    main_plan=json.loads((MAIN/'plan.json').read_text());trained=json.loads((MAIN/'training-complete.json').read_text())
    done=json.loads((MAIN/'complete.json').read_text());auxiliary=json.loads((AUXILIARY/'complete.json').read_text());states=json.loads((STATES/'complete.json').read_text())
    if (main_plan.get('planned_steps')!=35284 or main_plan.get('all_selected_family_records_once') is not True
            or trained.get('actual_updates')!=35284 or done.get('native_development_assessments_complete') is not True
            or auxiliary.get('every_original_reference_disposition_and_candidate_actual_native_target_rebuilt') is not True
            or auxiliary.get('dataset_admitted_to_auxiliary_CE_training') is not True
            or states.get('dataset_admitted_to_auxiliary_CE_training') is not True
            or states.get('every_original_source_and_every_state_target_and_mask_rebuilt') is not True):
        raise ValueError('Complete main35284source epoch and every admitted original public execution target required')
    return True


def prepare_initializer(files):
    import torch
    from slm_lab.fast_joint_inference_v2 import load
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint
    from slm_lab.focused_code_continuation_data_v3 import FocusedCodeContinuationData,FocusedContinuationSchedule
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    checkpoint=ROOT/'checkpoints/phase-joint-v3/trained/model.pt';source_plan=MAIN/'plan.json'
    inputs=dict(checkpoint=str(checkpoint),plan=str(source_plan),checkpoint_sha256=file_hash(checkpoint),plan_sha256=file_hash(source_plan))
    source=json.loads(source_plan.read_text());raw=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if (source.get('planned_steps')!=35284 or raw.get('arm')!='trained' or raw.get('completed_updates')!=35284
            or raw.get('whole_source_epochs_completed')!=1 or source.get('all_selected_family_records_once') is not True):
        raise ValueError('Only the exact complete own main35284checkpoint may initialize code-first learning')
    del raw
    files.update({str(checkpoint):inputs['checkpoint_sha256'],str(source_plan):inputs['plan_sha256']});guard=FileBindings(files)
    model,metadata=load(checkpoint,source_plan,'trained',expected_checkpoint_sha256=inputs['checkpoint_sha256'],
        expected_plan_sha256=inputs['plan_sha256'],device='cpu')
    if model.counts()['total_unique']!=984399690 or metadata['actual_completed_updates']!=35284 or torch.cuda.is_initialized():
        raise ValueError('Actual complete own984399690-value main model on CPU required')
    data=FocusedCodeContinuationData(ROOT,model.config,purpose='optimization');schedule=FocusedContinuationSchedule(data,seed=972317,width=16)
    source_digest=tensor_digest(model.named_parameters());OUTPUT.mkdir();INITIALIZER_REPORT.mkdir()
    sources=dict(own_complete_main_checkpoint_sha256=inputs['checkpoint_sha256'],own_complete_main_plan_sha256=inputs['plan_sha256'])
    initializer=OUTPUT/'own-complete-main-initializer.pt';atomic_checkpoint(initializer,model.export(sources),MAX_MODEL,RESERVE)
    settings=dict(initializer=str(initializer),initializer_sha256=file_hash(initializer),initializer_sources=sources,
        actual_counts=model.counts(),tokenizer_sha256=data.code.provenance['tokenizer_sha256'])
    binding=dict(path=str(initializer),checkpoint_sha256=settings['initializer_sha256'],parameter_sha256=source_digest,foundation_completed_updates=0)
    if source_digest!=tensor_digest(model.named_parameters()):raise ValueError('Copying the exact whole starting model must not change parameters')
    guard.check(full=True)
    receipt=dict(kind='copy_complete_own_dense_main_without_new_actor_fold',actual_source_updates=35284,
        actual_unique_source_values=984399690,actual_unique_initializer_values=984399690,no_new_parameter_arithmetic=True)
    atomic_json(INITIALIZER_REPORT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
        starting_main_inputs=inputs,actual_initialization_receipt=receipt,initial_dense_settings=settings,
        starting_policy_reference_binding=binding,old_or_partial_or_post_RL_policy_not_used_as_initializer=True,
        fresh_four_L40_capacity_and_cold_reload_still_required=True,model_updates=0,goal_achieved=False,
        source_files_sha256={str(checkpoint):inputs['checkpoint_sha256'],str(source_plan):inputs['plan_sha256'],str(initializer):settings['initializer_sha256']}))
    files.update(data.source_files_sha256);files[str(initializer)]=settings['initializer_sha256']
    files[str(INITIALIZER_REPORT/'complete.json')]=file_hash(INITIALIZER_REPORT/'complete.json')
    del model
    return settings,binding,data,schedule


def distributed(mode):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1,2,3' or int(os.environ['WORLD_SIZE'])!=4:raise ValueError('All four original L40s required')
    rank=int(os.environ['RANK']);plan=owner.registered(mode+'-rank-'+str(rank));digest=file_hash(REPORTS/'plan.json')
    import copy
    from concurrent.futures import ThreadPoolExecutor
    import random
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.codenet_checkpoint_v1 import save_collective,capture_rng,atomic_checkpoint
    from slm_lab.fast_joint_inference_v2 import fresh,export
    from slm_lab.fast_joint_stage_v1 import make_optimizer,parameter_groups
    from slm_lab.focused_code_continuation_data_v3 import FocusedCodeContinuationData
    from slm_lab.focused_continuation_run_contract_v2 import schedule_for,make_state,restore_state
    from slm_lab.packed_validation_v1 import evaluate as packed_loss
    from slm_lab.phase_joint_update_v1 import update
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1);device=torch.device('cuda',rank);torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.85,device);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name(device)!='NVIDIA L40':raise ValueError('Only the original local L40s are admitted')
    dist.init_process_group('nccl',timeout=timedelta(minutes=40));control=dist.new_group(backend='gloo',timeout=timedelta(minutes=40))
    try:
        guard=FileBindings(plan['source_files_sha256']);model,_=fresh(plan,device)
        data=FocusedCodeContinuationData(ROOT,model.config,purpose='optimization');schedule=schedule_for(data,plan,mode)
        reference=copy.deepcopy(model).eval().requires_grad_(False);binding=plan['starting_policy_reference_binding']
        if tensor_digest(reference.named_parameters())!=binding['parameter_sha256']:raise ValueError('Exact complete own initial frozen reference required')
        options=plan['training_options'];optimizer=make_optimizer(model,options,device);records=[]
        groups=parameter_groups(model);selected={id(p) for rows in groups.values() for _,p in rows}
        fixed=tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in selected)
        random.seed(plan['seed']+rank);np.random.seed(plan['seed']+rank);torch.manual_seed(plan['seed']+rank)
        amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16)
        def step(i,prepared=None):
            owner.owned(plan)
            row=update(model,optimizer,data,schedule,i,options,reference=reference,device=device,rank=rank,world=4,amp=amp,prepared=prepared)
            records.append(row);return row
        def save_state(path,completed):
            save_collective(path,lambda rngs:make_state(model,optimizer,plan,digest,completed,records,rngs,mode,
                schedule=schedule,reference_binding=binding,reference=reference),device,4,rank,MAX_STATE,RESERVE)
            dist.barrier(group=control)
        def adam():return tensor_digest((str(i)+'.'+k,s[k]) for i,s in enumerate(optimizer.state.values()) for k in ('step','exp_avg','exp_avg_sq'))
        def equal(a,b):
            if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
            if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a==b
        def draws():return [random.random(),float(np.random.random()),float(torch.rand(())),float(torch.rand((),device=device))]
        def recover(path,completed,next_step):
            nonlocal records
            save_state(path,completed);expected_draws=draws();expected=step(next_step)
            expected_rng=capture_rng(device);weights=tensor_digest(model.named_parameters());moments=adam()
            saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            for kind in ['source','CUDA_RNG']:
                core=dict(saved['core']);bad=dict(saved,core=core)
                if kind=='source':
                    core['metrics']=[dict(r) for r in core['metrics']];core['metrics'][0]['global_original_positions']=[-1]
                else:
                    core['rngs']=[dict(r) for r in core['rngs']];core['rngs'][rank]['cuda']=torch.zeros(4,dtype=torch.uint8)
                rng=capture_rng(device);signature=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in model.parameters()]
                try:restore_state(model,optimizer,bad,plan,digest,device,rank,4,mode,schedule=schedule,reference_binding=binding,reference=reference)
                except ValueError:pass
                else:raise ValueError('Corrupt actual source cursor or private CUDA RNG must reject')
                if (signature!=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in model.parameters()] or moments!=adam()
                        or not equal(rng,capture_rng(device))):
                    raise ValueError('Rejected recovery changed actual live model, Adam or RNG')
            count,records=restore_state(model,optimizer,saved,plan,digest,device,rank,4,mode,
                schedule=schedule,reference_binding=binding,reference=reference)
            actual_draws=draws();actual=step(next_step)
            if (count!=completed or actual_draws!=expected_draws or actual!=expected
                    or weights!=tensor_digest(model.named_parameters()) or moments!=adam() or not equal(expected_rng,capture_rng(device))):
                raise ValueError('Actual next source update, complete weights, Adam and private RNG must recover exactly')
        def save_model(arm,completed):
            if rank==0:
                folder=OUTPUT/arm;folder.mkdir();shutil.copyfile(data.code.tokenizer_path,folder/'tokenizer.json')
                atomic_checkpoint(folder/'model.pt',export(model,plan,digest,arm,completed),MAX_MODEL,RESERVE)
            dist.barrier(group=control)
        def validate(label):
            row=packed_loss(model,data.replay.packed,device=device,rank=rank,world=4,amp=amp)
            if rank==0:atomic_json(REPORTS/('held-packed-'+label+'.json'),row)
        def standalone(folder_name,completed):
            from slm_lab.own_dense_standalone_v1 import export as standalone_export,load as standalone_load
            receipt=json.loads((INITIALIZER_REPORT/'complete.json').read_text())['actual_initialization_receipt']
            sources=dict(code_first_continuation_plan_sha256=digest,own_complete_main_initializer_sha256=plan['initial_dense_settings']['initializer_sha256'])
            folder=OUTPUT/folder_name;path=folder/'model.pt'
            if rank==0:
                folder.mkdir();shutil.copyfile(data.code.tokenizer_path,folder/'tokenizer.json')
                atomic_checkpoint(path,standalone_export(model,sources=sources,provenance=receipt,completed_updates=completed,tokenizer_sha256=plan['tokenizer_sha256']),MAX_MODEL,RESERVE)
            dist.barrier(group=control)
            model.eval();code_position=plan['capacity_global_batches'][0][1][0]
            code=data.batch('new:CodeContests_complete_code',[code_position],device)
            word_position=next(i for i,r in enumerate(data.replay.words.rows) if r['byte_family']=='real_word')
            word=data.batch('words',[word_position],device)
            native={k:code[k] for k in ('input_ids','attention_mask','pixel_values') if k in code}
            with torch.no_grad(),amp():
                before=model(**native)['logits'].cpu();before_word=model.word_logits(word['word_pixels']).cpu()
            cold,metadata=standalone_load(path,expected_checkpoint_sha256=file_hash(path),expected_sources=sources)
            cold=cold.to(device).eval()
            with torch.no_grad(),amp():
                after=cold(**native)['logits'].cpu();after_word=cold.word_logits(word['word_pixels']).cpu()
            if (not torch.equal(before,after) or not torch.equal(before_word,after_word)
                    or tensor_digest(cold.named_parameters())!=tensor_digest(model.named_parameters())
                    or metadata['actual_counts']!=plan['actual_counts']):
                raise ValueError('Self-contained export must cold-load every trained value, full4096logits and real-word byte outputs exactly')
            del cold,code,word,native,before,after,before_word,after_word
            torch.cuda.empty_cache()
            atomic_json(REPORTS/f'{folder_name}-cold-reload-{rank}.json',dict(physical_GPU=rank,
                actual_unique_student_values=984399690,complete_native4096logits_and_real_word_outputs_bit_exact=True,
                every_exported_parameter_bit_exact=True,no_old_base_actor_optimizer_or_reference_loaded_for_inference=True,
                actual_completed_updates=completed,checkpoint_sha256=file_hash(path),model_updates=0,goal_achieved=False))
            dist.barrier(group=control)
        started=time.monotonic()
        if mode=='capacity':
            step(0);recover(OUTPUT/'capacity-after1.pt',1,1)
            for i in range(2,7):step(i)
            language=tensor_digest(groups['language']);recover(OUTPUT/'capacity-after7.pt',7,7)
            if language!=tensor_digest(groups['language']):raise ValueError('Actual complete word update must keep the language decoder exact')
            if len(optimizer.state)!=445 or records[1]['full_context_reference_KL_sum']<=0:raise ValueError('All full Adam groups and actual trained reference KL required')
            standalone('capacity-standalone',8)
            atomic_json(REPORTS/f'capacity-{rank}.json',dict(rank=rank,physical_GPU=rank,actual_counts=model.counts(),
                actual_Adam_states=len(optimizer.state),logical_updates=8,physical_optimizer_calls=10,actual_records=records,
                actual_next_source_updates_weights_Adam_all_worker_RNG_exact=True,
                invalid_source_and_private_CUDA_RNG_rejected_before_mutation=True,decoder_exact_during_word_update=True,
                complete_standalone_native_cold4096_and_word_outputs_and_every_value_exact=True,
                whole4096_code_output_state_raw_education_and_real448photo_word_inputs=True,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                seconds=time.monotonic()-started,production_updates=0,goal_achieved=False))
        else:
            save_model('initial',0);validate('initial')
            journal=(REPORTS/'training-metrics.jsonl').open('x') if rank==0 else None
            def prepare(i):
                name,positions=schedule.at(i);local=positions[rank::4]
                if not local:return []
                prefix=model.config.image_token_count(448) if data.kinds[name] in ('image','word') else 0
                width=max(1,min(options['maximum_microbatch'],options['maximum_microbatch_tokens']//(max(data.lengths[name][p] for p in local)+prefix)))
                return [{k:(v.pin_memory() if isinstance(v,torch.Tensor) else v) for k,v in data.batch(name,local[a:a+width],'cpu').items()}
                    for a in range(0,len(local),width)]
            with ThreadPoolExecutor(max_workers=1) as reader:
                pending=reader.submit(prepare,0)
                for i in range(len(schedule)):
                    if i%32==0:guard.check()
                    prepared=pending.result()
                    if i+1<len(schedule):pending=reader.submit(prepare,i+1)
                    tick=time.monotonic();row=step(i,prepared)
                    if journal:
                        journal.write(json.dumps(row)+'\n');journal.flush()
                        if i==0 or (i+1)%25==0:atomic_json(REPORTS/'progress.json',dict(actual_updates=i+1,planned_updates=len(schedule),
                            latest=row,last_update_seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-started,goal_achieved=False))
                    if (i+1)%2048==0 or i+1==len(schedule):
                        validate(str(i+1));save_state(OUTPUT/'resume.pt',i+1)
                        save_model('trained' if i+1==len(schedule) else f'step-{i+1}',i+1)
            if journal:journal.close()
            standalone('standalone',len(schedule))
            if rank==0:atomic_json(REPORTS/'training-complete.json',dict(actual_updates=len(records),
                all58488new_code_and_all_admitted_public_execution_targets_once=True,
                all_admitted_public_state_examples_once=True,
                actual_selected_replay_source_budget=schedule.selected_budget(data),
                actual_counts=model.counts(),native_assessments_and_independent_state_readback_pending=True,goal_achieved=False))
        if (fixed!=tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in selected)
                or tensor_digest(reference.named_parameters())!=binding['parameter_sha256']):raise ValueError('Unused own heads or starting-policy reference changed')
        guard.check(full=True);dist.barrier(group=control)
    finally:dist.destroy_process_group()


def assess_full(arm,gpu,suite):
    plan=joint.legacy.registered('full-'+suite+'-'+arm+'-'+str(gpu),gpu)
    import torch
    from slm_lab.dense_foundation_inference_v1 import DenseNativeView
    from slm_lab.fast_joint_inference_v2 import load
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85,0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name(0)!='NVIDIA L40':raise ValueError('Only the original isolated L40 is admitted')
    checkpoint=OUTPUT/arm/'model.pt';bindings_path=REPORTS/'native-assessment-inputs.json'
    inputs=json.loads(bindings_path.read_text())[arm]
    model,metadata=load(checkpoint,REPORTS/'plan.json',arm,expected_checkpoint_sha256=inputs['checkpoint_sha256'],
        expected_plan_sha256=inputs['run_plan_sha256'],device='cuda:0')
    native=DenseNativeView(model).eval();signature=[(id(p),p._version,p.data_ptr()) for p in native.parameters()]
    if sum(p.numel() for p in native.parameters())!=984399690:raise ValueError('Every actual own dense student value required')
    if suite=='QA':from slm_lab.full_held_QA_native_v1 import FullHeldQA as Population,evaluate_shard
    else:from slm_lab.ai2d_full_native_evaluation_v1 import FullAI2D as Population,evaluate_shard
    population=Population(ROOT)
    if population.identity_sha256!=plan['full_assessment_populations'][suite] or metadata['tokenizer_sha256']!=population.tokenizer_sha256:
        raise ValueError('Exact full original held population and own BPE required')
    folder=REPORTS/arm;folder.mkdir(exist_ok=True);journal=folder/f'full-{suite}-shard-{gpu}-actual-deliveries.jsonl'
    with journal.open('x') as stream:
        def record(row):stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n');stream.flush()
        with inference_cache():report=evaluate_shard(native,population.tokenizer,population,folder/f'full-{suite}-shard-{gpu}.json',
            rank=gpu,world=4,checkpoint_sha256=inputs['checkpoint_sha256'],checked=lambda:owner.owned(plan),record=record)
        os.fsync(stream.fileno())
    if signature!=[(id(p),p._version,p.data_ptr()) for p in native.parameters()] or any(p.grad is not None for p in native.parameters()):
        raise ValueError('Full native assessment must preserve every actual model value')
    owner.owned(plan);FileBindings(plan['source_files_sha256']).check(full=True)
    atomic_json(folder/f'full-{suite}-shard-{gpu}-complete.json',dict(physical_GPU=gpu,endpoint=arm,
        actual_original_questions=len(report['results']),actual_student_values=984399690,
        actual_delivery_journal_sha256=file_hash(journal),all_loaded_values_unchanged=True,model_updates=0,goal_achieved=False))


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(p.exists() for p in [REPORTS,OUTPUT,QUEUE,INITIALIZER_REPORT]):
        raise ValueError('Fresh CPU-hidden continuation queue required')
    from slm_lab.ai2d_full_native_evaluation_v1 import FullAI2D,reconcile as image_reconcile
    from slm_lab.full_held_QA_native_v1 import FullHeldQA,reconcile as QA_reconcile
    from slm_lab.focused_continuation_run_contract_v2 import capacity_batches
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.evaluation_provenance import capture
    tests={'state-code-first-data-and-run-contract-v1':3,'state-code-first-training-drivers-v1-final':11,'code-first-driver-and-standalone-v1':4}
    test_files=[]
    for name,count in tests.items():
        path=ROOT/'reports'/(name+'-tests.log')
        if str(count)+' passed' not in path.read_text():raise ValueError('Actual continuation, source-completeness and fold controls required')
        test_files.append(path)
    prep_path=ROOT/'reports/focused-code-continuation-sources-v3-preparation/complete.json';prep=json.loads(prep_path.read_text())
    fold_readback=ROOT/'reports/full-actor-fold-dense-update-CPU-v1-readback/complete.json'
    if (not prep['all12actual_replay_family_extremes_rebuilt'] or not prep['all58488_CODE_example_batches_identical_to_preceding_full_actual_rebuild']
            or prep.get('all_admitted_output_and_state_examples_rebuilt_in_actual_complete_batches') is not True
            or prep.get('all_existing_CODE_output_and_replay_exposures_exactly_preserved') is not True
            or json.loads(fold_readback.read_text())['actual_changed_dense_language_tensors']!=317):
        raise ValueError('Actual complete new source preparation and full-size dense-gradient mechanism required')
    QA=FullHeldQA(ROOT);images=FullAI2D(ROOT);me=owner.resources.identity(os.getpid())
    dependencies={name:json.loads((folder/'plan.json').read_text())['owner'] for name,folder in [('main',MAIN),('auxiliary',AUXILIARY),('states',STATES)]}
    files={**QA.bindings,**images.bindings}
    extras=[prep_path,fold_readback,*test_files,ROOT/'tests/test_focused_continuation_run_contract_v2.py',
        ROOT/'tests/test_code_first_continuation_driver_v2.py',ROOT/'tests/test_own_dense_standalone_v1.py',ROOT/'reports/code-first-driver-and-standalone-v1-tests.log',
        MAIN/'plan.json',AUXILIARY/'plan.json',STATES/'plan.json',*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_code_first_continuation_driver_v2.py'])]
    if '4 passed' not in (ROOT/'reports/code-first-driver-and-standalone-v1-tests.log').read_text():raise ValueError('Actual exact-owner, complete-source and standalone native controls required')
    files.update({str(p):file_hash(p) for p in extras});files.update({str(ROOT/p):h for p,h in capture()['files_sha256'].items()})
    guard=FileBindings(files);expires=time.time()+14*86400
    atomic_json(QUEUE,dict(at=datetime.now(timezone.utc).isoformat(),owner=me,exact_dependencies=dependencies,
        status='waiting for complete original main, admitted public outputs and explicit code-first handoff',
        predeclared_starting_policy='complete own phase-joint-v3 main35284, copied without actor folding',
        planned_new_code_presentations=58488,all12broad_and_raw_replay_families=True,
        all_admitted_public_execution_targets_required=True,all_admitted_public_state_targets_required=True,full_QA_questions_each_endpoint=16961,
        original_AI2D_questions_each_endpoint=3088,original_AI2D_conditions=['proper_image','gray_image','wrong_image'],
        source_files_sha256=files,expires_at_unix=expires,automatic_retry=False,production_updates=0,goal_achieved=False))
    while not dependencies_ready(dependencies):
        if time.time()>expires:raise ValueError('Queue observation lifetime exceeded; original jobs left intact')
        guard.check();time.sleep(5)
    for path in [MAIN/'complete.json',MAIN/'restored.json',AUXILIARY/'complete.json',STATES/'complete.json',HANDOFF]:
        files[str(path)]=file_hash(path)
    settings,binding,data,schedule=prepare_initializer(files)
    source_main_path=ROOT/'reports/phase-joint-v3/plan.json';source_main=json.loads(source_main_path.read_text())
    control_path=ROOT/'reports/phase-joint-full-CPU-v2/plan.json';control=json.loads(control_path.read_text())
    for path in [source_main_path,control_path,*[Path(p) for p in source_main['pulse_sources'].values()]]:
        files[str(path)]=file_hash(path)
    rates=dict(language=1e-5,vision=1e-4,word=3e-4)
    options=dict(control['training_options'],learning_rates=rates,phase_learning_rates=dict(instruction_and_vision=rates))
    capacity=capacity_batches(data);limits=dict(foundation=[0,0],instruction_and_vision=[0,len(schedule)])
    required=RESERVE+4*MAX_STATE+(len(schedule)//2048+5)*MAX_MODEL
    if shutil.disk_usage(ROOT).free<required:raise ValueError('Every full private/native snapshot and atomic reserve must fit')
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);owner.idle_original()
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(used)!=4 or any(int(x)>128 for x in used):raise ValueError('Every original L40 must be idle before ownership')
        guard=FileBindings(files);guard.check(full=True);REPORTS.mkdir()
        plan=dict(owner=me,waiter=owner.WAITER,source_files_sha256=files,expires_at_unix=time.time()+7*86400,
            initial_dense_settings=settings,actual_counts=settings['actual_counts'],tokenizer_sha256=settings['tokenizer_sha256'],
            starting_main_inputs=json.loads((INITIALIZER_REPORT/'complete.json').read_text())['starting_main_inputs'],
            source_start_is_complete_main=True,full_assessments_deferred_until_after_human_and_execution_training=True,
            starting_policy_reference_binding=binding,seed=972317,global_batch_width=16,
            planned_steps=len(schedule),source_schedule_sha256=schedule.identity_sha256,phase_bounds=limits,
            capacity_global_batches=capacity,capacity_phase_bounds=dict(foundation=[0,0],instruction_and_vision=[0,8]),
            complete_source_budget=data.budget(),selected_source_budget=schedule.selected_budget(data),training_options=options,
            pulse_sources=source_main['pulse_sources'],full_assessment_populations=dict(QA=QA.identity_sha256,AI2D=images.identity_sha256),
            raw_replay_intended_to_reduce_measured_instruction_retention_drift_not_already_proved=True,
            complete_new_code_and_execution_source_epoch_once=True,complete_public_state_source_epoch_once=True,all_original_replay_sources_once=False,
            no_external_pretrained_model_or_teacher=True,automatic_retry=False,candidate_promoted=False,goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[]);driver=str(Path(__file__).resolve())
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned whole-source continuation interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        watcher=None
        try:
            owner.idle_original();os.kill(owner.WAITER['pid'],signal.SIGSTOP);deadline=time.monotonic()+5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in ['T','t']:
                if time.monotonic()>deadline:raise ValueError('Exact original idle waiter hold failed')
                time.sleep(.05)
            watcher=subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,stdout=(REPORTS/'watchdog.log').open('x'),
                stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            def run(mode):
                child=owner.launch(plan,['-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',driver,'--distributed',mode],mode+'-launcher','0,1,2,3')
                owner.wait_children(plan,[child])
            run('capacity');proofs=[json.loads((REPORTS/f'capacity-{rank}.json').read_text()) for rank in range(4)]
            for rank,p in enumerate(proofs):
                if (p['physical_GPU']!=rank or p['actual_counts']!=plan['actual_counts'] or p['actual_Adam_states']!=445
                        or p['logical_updates']!=8 or p['physical_optimizer_calls']!=10
                        or not p['actual_next_source_updates_weights_Adam_all_worker_RNG_exact']
                        or not p['invalid_source_and_private_CUDA_RNG_rejected_before_mutation']
                        or not p['complete_standalone_native_cold4096_and_word_outputs_and_every_value_exact']):
                    raise ValueError('Every actual new full-profile four-L40 recovery proof required')
            atomic_json(REPORTS/'capacity.json',dict(passed=True,proofs=proofs,production_updates=0,goal_achieved=False))
            run('train')
            inputs={arm:dict(checkpoint_sha256=file_hash(OUTPUT/arm/'model.pt'),run_plan_sha256=file_hash(REPORTS/'plan.json')) for arm in ['initial','trained']}
            atomic_json(REPORTS/'native-assessment-inputs.json',inputs)
            for arm in ['initial','trained']:
                for suite in ['code','broad']:
                    children=[owner.launch(plan,['-u',driver,'--assess',arm,'--suite',suite,'--gpu',str(gpu)],'assess-'+arm+'-'+suite+'-'+str(gpu),str(gpu)) for gpu in range(4)]
                    owner.wait_children(plan,children)
                from slm_lab.native_code_shards_v1 import reconcile
                for frame in ['completion','instruction']:reconcile([REPORTS/arm/f'code-{frame}-shard-{r}.json' for r in range(4)],REPORTS/arm/f'code-{frame}.json')
            guard.check(full=True)
            atomic_json(REPORTS/'complete.json',dict(all58488new_code_and_all_admitted_execution_examples_once=True,
                all_admitted_public_state_examples_once=True,
                all12selected_replay_families_present=True,all_four_L40_capacity_and_actual_recovery_passed=True,
                original_local_code_and_broad_assessments_complete=True,
                full_QA_AI2D_and_MBPP_assessments_deferred_to_complete_post_training_pipeline=True,
                all_native_deliveries_and_fixed_primary_populations_retained=True,
                standalone_model_and_tokenizer_with_no_older_checkpoint_dependency=True,
                independent_full_state_readback_human_retention_and_broader_held_code_benchmarks_still_required=True,
                candidate_promoted=False,goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False));raise
        finally:
            owner.restore_owned(plan,'Whole-source code continuation ended; original work restored')
            if watcher is not None:watcher.wait(timeout=20)


if __name__=='__main__':
    configure();p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watchdog',action='store_true')
    p.add_argument('--distributed',choices=['capacity','train']);p.add_argument('--assess',choices=['initial','trained'])
    p.add_argument('--full-assess',choices=['initial','trained']);p.add_argument('--suite',choices=['code','broad','QA','AI2D'])
    p.add_argument('--gpu',type=int,choices=range(4));a=p.parse_args()
    if a.watchdog:owner.watchdog()
    elif a.distributed:distributed(a.distributed)
    elif a.full_assess:assess_full(a.full_assess,a.gpu,a.suite)
    elif a.assess:joint.assess(a.assess,a.gpu,a.suite)
    else:
        try:parent()
        except BaseException as error:
            atomic_json(ROOT/'reports/code-first-continuation-v2-parent-failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
                automatic_retry=False,goal_achieved=False));raise
