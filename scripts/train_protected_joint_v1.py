"""One owned four-L40 full-weight code/general/image/word curriculum.

Replaces unstarted multi-arm alignment studies after an explicit handover.
All original train sources are covered once, with no final-batch repeats.
Short matched development pulses accompany learning; full baseline/final native
comparisons run afterward, so baseline assessment does not delay first learning.
"""
import argparse
from datetime import datetime,timezone,timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time
from types import SimpleNamespace

import train_whole_human_SFT_v1 as legacy
from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint
from slm_lab.parallel_file_bindings_v1 import ParallelFileBindings as FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

owner=legacy.owner
ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/protected-joint-v1'
OUTPUT=ROOT/'checkpoints/protected-joint-v1'
QUEUE=ROOT/'reports/protected-joint-v1-queue.json'
CPU=ROOT/'reports/fast-joint-full-CPU-v2'
PREVIOUS=ROOT/'reports/code-initialized-alignment-v1'
HANDOVER=ROOT/'reports/fast-joint-v1-handover'
MAX_STATE=16*1024**3
MAX_MODEL=5*1024**3
RESERVE=26*1024**3


def configure():
    owner.REPORTS=REPORTS;owner.resources.REPORTS=REPORTS;legacy.REPORTS=REPORTS;legacy.OUTPUT=OUTPUT


class RuntimeGuard:
    def __init__(self,files):
        self.full=FileBindings(files)
        self.runtime={p:d for p,d in self.full.files.items() if p.suffix=='.py'}
    def check(self,full=False):
        if full:self.full.check(full=True);return
        for p,state in self.full.states.items():
            if self.full._state(p)!=state:raise ValueError('Immutable admitted source changed: '+str(p))
        for p,d in self.runtime.items():
            if file_hash(p)!=d:raise ValueError('Bound neural/runtime source changed: '+str(p))


def pulse(model,metadata,plan,arm,rank,device,guard):
    """Bounded matched development checks, explicitly not full benchmark scores."""
    from slm_lab.dense_native_view_v2 import CompleteDenseNativeView as DenseNativeView
    view=DenseNativeView(model);checkpoint=OUTPUT/arm/'model.pt';destination=REPORTS/arm
    def evaluate(module,name,data=None,**kwargs):
        original=module.load_checkpoint
        module.load_checkpoint=lambda path,requested,training_state=False:(view,metadata)
        try:
            owner.owned(plan)
            if data is None:result=module.evaluate(checkpoint,destination/name,device=str(device),**kwargs)
            else:result=module.evaluate(checkpoint,data,destination/name,device=str(device),**kwargs)
        finally:module.load_checkpoint=original
        return result
    model.eval()
    if rank==0:
        from slm_lab import knowledge_evaluation
        evaluate(knowledge_evaluation,'pulse-knowledge.json',Path(plan['pulse_sources']['knowledge']),max_examples=56,max_new_tokens=32)
    elif rank==1:
        from slm_lab import code_quality
        evaluate(code_quality,'pulse-code-completion.json',max_new_tokens=128,prompt_mode='completion')
    elif rank==2:
        from slm_lab import ocr_evaluation,object_evaluation,vision_evaluation
        evaluate(ocr_evaluation,'pulse-ocr.json',Path(plan['pulse_sources']['ocr']),max_examples=32,max_new_tokens=128,image_size=448)
        evaluate(object_evaluation,'pulse-objects.json',Path(plan['pulse_sources']['objects']),max_examples=32,max_new_tokens=32)
        evaluate(vision_evaluation,'pulse-vision.json',Path(plan['pulse_sources']['vision']),max_examples=32,image_size=448)
    else:
        from slm_lab import math_evaluation
        evaluate(math_evaluation,'pulse-math.json',ROOT/'data/math-reasoning-v2/test.jsonl',max_examples=16,max_new_tokens=256)
    guard.check()


def isolated_pulse(plan,arm,rank,local):
    """An optional development measurement may time out; training must not hang."""
    owner.owned(plan);subject=f'pulse-{arm}-{rank}'
    child=owner.launch(plan,['-u',str(Path(__file__).resolve()),'--pulse-child',str(rank),'--arm',arm],subject,str(local))
    actor=next(a for a in json.loads((REPORTS/'children.json').read_text()) if a['pid']==child.pid and a['subject']==subject)
    start=time.monotonic();reason=None
    try:
        status=child.wait(timeout=plan['development_pulse_wall_seconds'])
        if status!=0:reason='development child exited '+str(status)
    except subprocess.TimeoutExpired:
        reason='declared development measurement wall budget exhausted'
        owner.resources.stop_child(actor);child.wait(timeout=10)
    if owner.resources.same(actor):raise ValueError('Exact development child remains live')
    owner.owned(plan)
    if reason:
        atomic_json(REPORTS/arm/f'pulse-incomplete-rank-{rank}.json',dict(reason=reason,
            seconds=time.monotonic()-start,physical_GPU=local,child_birth=actor,measurement_complete=False,
            automatic_retry=False,full_endpoint_evaluation_still_required=True,goal_achieved=False))


def pulse_child(arm,rank):
    """No distributed context or thread-pool reconfiguration inside generation."""
    plan=legacy.registered(f'pulse-{arm}-{rank}',rank)
    import faulthandler
    import torch
    from slm_lab.fast_joint_inference_v2 import load
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    faulthandler.enable();faulthandler.register(signal.SIGUSR1,all_threads=True)
    faulthandler.dump_traceback_later(150,repeat=False)
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.35,0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checkpoint=OUTPUT/arm/'model.pt'
    bindings={p:d for p,d in plan['source_files_sha256'].items() if Path(p).suffix=='.py'}
    for p in [REPORTS/'plan.json',checkpoint,OUTPUT/arm/'tokenizer.json',*map(Path,plan['pulse_sources'].values())]:
        bindings[str(p)]=file_hash(p)
    guard=RuntimeGuard(bindings);started=time.monotonic()
    model,metadata=load(checkpoint,REPORTS/'plan.json',arm,expected_checkpoint_sha256=bindings[str(checkpoint)],
        expected_plan_sha256=bindings[str(REPORTS/'plan.json')],device='cuda:0')
    original_threads=torch.set_num_threads
    # Evaluators request 1/2 threads. One fixed thread avoids changing the
    # active pool between OCR and object/caption checks. This affects execution
    # scheduling only; development budgets are separately declared above.
    torch.set_num_threads=lambda count:None
    try:
        with inference_cache() as cache:
            pulse(model,metadata,plan,arm,rank,torch.device('cuda',0),guard)
            counts=dict(cache)
        atomic_json(REPORTS/arm/f'pulse-complete-rank-{rank}.json',dict(physical_GPU=rank,
            checkpoint_sha256=bindings[str(checkpoint)],seconds=time.monotonic()-started,
            CPU_threads=1,RoPE_table_cache=counts,development_only=True,
            code_budget=128,math_budget=256,full_endpoint_budget_unchanged=512,goal_achieved=False))
    finally:
        torch.set_num_threads=original_threads;faulthandler.cancel_dump_traceback_later()


def distributed(mode):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1,2,3':raise ValueError('Exact original four local GPUs required')
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=4:raise ValueError('Exactly four data-parallel workers required')
    plan=owner.registered(mode+'-rank-'+str(rank));digest=file_hash(REPORTS/'plan.json')
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.codenet_checkpoint_v1 import save_collective,capture_rng
    from slm_lab.fast_joint_checkpoint_v2 import bundle,restore
    from slm_lab.fast_joint_data_v1 import FastJointSources,CompleteBucketSchedule
    from slm_lab.fast_joint_stage_v1 import make_optimizer,parameter_groups
    from slm_lab.protected_joint_update_v1 import update as protected_update
    from functools import partial
    from slm_lab.fast_joint_inference_v2 import fresh,export
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1);torch.cuda.set_device(local);device=torch.device('cuda',local)
    if torch.cuda.get_device_name(local)!='NVIDIA L40':raise ValueError('Actual L40 profile required')
    torch.cuda.set_per_process_memory_fraction(.85,local);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(minutes=40))
    control=dist.new_group(backend='gloo',timeout=timedelta(minutes=40))
    import faulthandler
    faulthandler.enable();faulthandler.register(signal.SIGUSR1,all_threads=True)
    from slm_lab.deterministic_positions_v1 import check_operator
    atomic_json(REPORTS/f'position-operator-{rank}.json',dict(physical_GPU=local,checks=check_operator(device),goal_achieved=False))
    guard=RuntimeGuard(plan['source_files_sha256']);model,metadata=fresh(plan,device);data=FastJointSources(ROOT,model.config)
    if data.budget()!=plan['complete_source_budget']:raise ValueError('Every actual complete train source must match the bound epoch')
    schedule=CompleteBucketSchedule(data.lengths,plan['global_family_batch_widths'],plan['seed'])
    if len(schedule)!=plan['planned_steps']:raise ValueError('Exact whole-source curriculum length required')
    options=plan['training_options'];optimizer=make_optimizer(model,options,device)
    reference,_=fresh(plan,device);reference.eval().requires_grad_(False)
    if reference.counts()!=model.counts():raise ValueError('Exact separately counted own initial reference required')
    update=partial(protected_update,reference=reference)
    reference_digest=tensor_digest(reference.named_parameters())
    groups=parameter_groups(model);selected={id(p) for g in groups.values() for _,p in g}
    fixed=tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in selected)
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16)
    random.seed(plan['seed']+rank);np.random.seed(plan['seed']+rank);torch.manual_seed(plan['seed']+rank)
    metadata=dict(metadata,tokenizer_sha256=plan['tokenizer_sha256'],training_options=dict(image_size=448))
    records=[];started=time.monotonic()
    def save_state(path,completed):
        guard.check();save_collective(path,lambda rngs:bundle(model,optimizer,plan,digest,completed,records,rngs,mode),
            device,world,rank,MAX_STATE,RESERVE);dist.barrier(group=control)
    def save_model(arm,completed):
        if rank==0:
            (OUTPUT/arm).mkdir(exist_ok=True);(REPORTS/arm).mkdir(exist_ok=True)
            shutil.copyfile(ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json',OUTPUT/arm/'tokenizer.json')
            atomic_checkpoint(OUTPUT/arm/'model.pt',export(model,plan,digest,arm,completed),MAX_MODEL,RESERVE)
        dist.barrier(group=control);torch.cuda.empty_cache()
    try:
        if mode=='capacity':
            schedule.entries=[(name,positions) for name,positions in plan['capacity_global_batches']]
            records.append(update(model,optimizer,data,schedule,0,options,device=device,rank=rank,world=world,amp=amp))
            records.append(update(model,optimizer,data,schedule,1,options,device=device,rank=rank,world=world,amp=amp))
            save_state(OUTPUT/'capacity-resume-step2.pt',2)
            def draws():return [random.random(),float(np.random.random()),float(torch.rand(())),float(torch.rand((),device=device))]
            expected_draws=draws()
            expected=update(model,optimizer,data,schedule,2,options,device=device,rank=rank,world=world,amp=amp)
            if expected['full_context_reference_KL_sum']<=0:raise ValueError('Actual full-length positive own-reference KL required')
            expected_weights=tensor_digest(model.named_parameters())
            def adam_digest():return tensor_digest((str(i)+'.'+k,v[k]) for i,v in enumerate(optimizer.state.values()) for k in ('step','exp_avg','exp_avg_sq'))
            expected_adam=adam_digest();payload=torch.load(OUTPUT/'capacity-resume-step2.pt',map_location='cpu',weights_only=True,mmap=True)
            bad_rngs=[dict(r) for r in payload['rngs']];bad_rngs[rank]['cuda']=torch.zeros(4,dtype=torch.uint8)
            bad=dict(payload,rngs=bad_rngs);before_rng=capture_rng(device)
            try:restore(model,optimizer,bad,plan,digest,device,rank,world,'capacity')
            except ValueError:pass
            else:raise ValueError('Invalid actual private CUDA RNG must reject')
            if (expected_weights!=tensor_digest(model.named_parameters()) or expected_adam!=adam_digest()
                    or not torch.equal(before_rng['torch'],torch.get_rng_state()) or not torch.equal(before_rng['cuda'],torch.cuda.get_rng_state(device))):
                raise ValueError('Invalid recovery mutated live weights/Adam or caller RNG')
            completed,records=restore(model,optimizer,payload,plan,digest,device,rank,world,'capacity')
            if draws()!=expected_draws:raise ValueError('Actual Python/NumPy/CPU/private CUDA next draws must restore')
            repeated=update(model,optimizer,data,schedule,2,options,device=device,rank=rank,world=world,amp=amp)
            if expected!=repeated or expected_weights!=tensor_digest(model.named_parameters()) or expected_adam!=adam_digest():
                raise ValueError('Actual protected four-GPU next update and all weights/Adam must recover exactly')
            records.append(repeated)
            language_before_image=tensor_digest(groups['language'])
            records.append(update(model,optimizer,data,schedule,3,options,device=device,rank=rank,world=world,amp=amp))
            if language_before_image!=tensor_digest(groups['language']):raise ValueError('Protected image update changed language weights')
            if fixed!=tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in selected):raise ValueError('Unused heads changed')
            if reference_digest!=tensor_digest(reference.named_parameters()):raise ValueError('Frozen own reference changed')
            result=dict(rank=rank,physical_GPU=local,full_configured_word_book_image_capacity_passed=True,
                exact_actual_distributed_next_update_weights_and_Adam_recovery=True,
                actual_parameter_counts=model.counts(),training_only_reference_values=reference.counts()['total_unique'],
                actual_selected_values={n:sum(p.numel() for _,p in g) for n,g in groups.items()},
                actual_Adam_states=len(optimizer.state),actual_CPU_and_private_CUDA_RNG_restored=True,
                actual_positive_full_context_KL=repeated['full_context_reference_KL_sum'],
                actual_language_exact_during_image_update=True,actual_training_only_reference_bytes_exact=True,
                disposable_logical_updates=4,physical_optimizer_calls=5,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(local),peak_reserved_bytes=torch.cuda.max_memory_reserved(local),
                seconds=time.monotonic()-started,production_updates=0,goal_achieved=False)
            atomic_json(REPORTS/f'capacity-{rank}.json',result)
        else:
            # The matched initial artifact is durable before the first update;
            # development generation is deferred until learning has begun.
            save_model('initial',0)
            journal=(REPORTS/'training-metrics.jsonl').open('x') if rank==0 else None
            from concurrent.futures import ThreadPoolExecutor
            def prepare(step):
                name,positions=schedule.rank_positions(step,rank,world)
                if not positions:return []
                prefix=model.config.image_token_count(448) if data.kinds[name] in ('word','image') else 0
                width=max(1,min(options['maximum_microbatch'],options['maximum_microbatch_tokens']//(max(data.lengths[name][p] for p in positions)+prefix)))
                batches=[data.batch(name,positions[i:i+width],'cpu') for i in range(0,len(positions),width)]
                return [{k:(v.pin_memory() if isinstance(v,torch.Tensor) else v) for k,v in b.items()} for b in batches]
            with ThreadPoolExecutor(max_workers=1) as reader:
                pending=reader.submit(prepare,0)
                for step in range(len(schedule)):
                    owner.owned(plan)
                    if step%32==0:guard.check()
                    prepared=pending.result()
                    if step+1<len(schedule):pending=reader.submit(prepare,step+1)
                    tick=time.monotonic();record=update(model,optimizer,data,schedule,step,options,device=device,rank=rank,world=world,amp=amp,prepared=prepared)
                    records.append(record)
                    if rank==0:
                        journal.write(json.dumps(record)+'\n');journal.flush()
                        if (step+1)%25==0 or step==0:
                            atomic_json(REPORTS/'progress.json',dict(actual_step=step+1,planned_steps=len(schedule),latest=record,
                                last_update_seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-started,goal_achieved=False))
                            print(json.dumps(dict(step=step+1,family=record['family'],token_mean_CE=record['token_mean_CE'],step_seconds=time.monotonic()-tick)),flush=True)
                    if (step+1)%1024==0 or step+1==len(schedule):
                        save_state(OUTPUT/'resume.pt',step+1)
                        arm='trained' if step+1==len(schedule) else f'step-{step+1}'
                        save_model(arm,step+1)
                        if step+1==1024:isolated_pulse(plan,'initial',rank,local);dist.barrier(group=control)
                        isolated_pulse(plan,arm,rank,local);dist.barrier(group=control)
            if journal:journal.close()
            if fixed!=tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in selected):raise ValueError('Unused heads changed during full joint curriculum')
            if rank==0:atomic_json(REPORTS/'training-complete.json',dict(actual_updates=len(records),planned_updates=len(schedule),
                full_source_epoch_complete=True,all_original_targets_and_partial_batches_preserved=True,
                checkpoint_sha256=file_hash(OUTPUT/'trained/model.pt'),all_other_heads_exact=True,
                seconds=time.monotonic()-started,actual_counts=model.counts(),candidate_promoted=False,goal_achieved=False))
        if reference_digest!=tensor_digest(reference.named_parameters()):raise ValueError('The own training-only reference must remain exact')
        guard.check(full=True);dist.barrier(group=control)
    finally:dist.destroy_process_group()


def assess(arm,suite,gpu,shard):
    subject='assess-'+arm+'-'+suite+'-'+str(shard if shard is not None else gpu)
    plan=legacy.registered(subject,gpu);torch,guard,amp=legacy.runtime(plan)
    from slm_lab.fast_joint_inference_v2 import load
    from slm_lab.dense_native_view_v2 import CompleteDenseNativeView as DenseNativeView
    model,metadata=load(OUTPUT/arm/'model.pt',REPORTS/'plan.json',arm,expected_checkpoint_sha256=file_hash(OUTPUT/arm/'model.pt'),
        expected_plan_sha256=file_hash(REPORTS/'plan.json'),device='cuda:0')
    legacy.native_suite(plan,SimpleNamespace(aligned=DenseNativeView(model)),metadata,arm,suite,gpu,guard,shard)


def make_pulse_sources():
    folder=ROOT/'data/protected-joint-v1-pulses'
    if folder.exists():raise ValueError('Preserve previous bounded development source selection')
    folder.mkdir();paths={};origins={}
    specs=[('knowledge','data/knowledge-eval-v1/questions.jsonl','domain',4),
        ('ocr','data/alignment-curriculum-v1/images.valid.jsonl','task',8),
        ('objects','data/object-qa-v1/valid.jsonl',None,15),
        ('vision','data/alignment-curriculum-v1/contrastive.valid.jsonl',None,32)]
    for name,relative,field,per_group in specs:
        source=ROOT/relative;rows=list(map(json.loads,source.open()));chosen=[]
        if field:
            for value in sorted({r[field] for r in rows}):chosen.extend([r for r in rows if r[field]==value][:per_group])
        elif name=='objects':
            from slm_lab.object_evaluation import select
            chosen=select(rows,per_group)
        else:chosen=rows[:per_group]
        if name=='objects':
            from slm_lab.ocr_evaluation import normalized
            if any(not any(other['task']==r['task'] and other['group_id']!=r['group_id']
                    and normalized(other['response']).casefold()!=normalized(r['response']).casefold()
                    for other in chosen) for r in chosen):raise ValueError('Every object development row needs an independent differently labelled photo control')
        if name=='vision' and len({r['group_id'] for r in chosen})!=len(chosen):raise ValueError('Unique caption image groups required')
        for row in chosen:
            if row.get('image'):
                p=Path(row['image']);row['image']=str((p if p.is_absolute() else source.parent/p).resolve())
        output=folder/(name+'.jsonl');output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen));paths[name]=str(output)
        origins[name]=dict(original_path=str(source),original_sha256=file_hash(source),subset_sha256=file_hash(output),examples=len(chosen),role='evaluation_only')
        if name=='knowledge':
            original_audit=source.with_suffix('.audit.json')
            atomic_json(output.with_suffix('.audit.json'),dict(clean_questions_sha256=file_hash(output),inherited_original_all_question_audit=str(original_audit),
                inherited_original_all_question_audit_sha256=file_hash(original_audit),source_subset_only=True,semantic_independence_not_certified=True))
    atomic_json(folder/'manifest.json',dict(sources=origins,development_subsets_not_full_benchmark_scores=True,never_training_data=True,goal_achieved=False))
    return paths


def previous_ready(previous,prior):
    if owner.resources.same(prior['owner']):return False
    if any(owner.resources.same(a) for a in json.loads((previous/'children.json').read_text())):return False
    if not (previous/'restored.json').exists():raise ValueError('Exact previous owner/workers must end and restore original work')
    retirement=ROOT/'reports/protected-joint-v1-handover.json'
    if retirement.exists():
        r=json.loads(retirement.read_text());identity=prior['owner']
        if (any(str(r['retired_exact_owner'][k])!=str(identity[k]) for k in ['pid','ticks'])
                or r['retained_completed_updates']!=2048 or r['partial_epoch_and_full_endpoints_explicitly_incomplete'] is not True
                or r['initial_and_all_learned_checkpoints_preserved'] is not True):raise ValueError('Exact explicit coding-priority handover required')
        for path,digest in r['matched_reports_sha256'].items():
            if file_hash(path)!=digest:raise ValueError('Actual handover score evidence changed')
        from monitor_fast_joint_regression_v1 import scores
        base=scores(previous/'initial');point=scores(previous/'step-2048')
        if (base['protocol']!=point['protocol'] or point['syntax']>base['syntax']-5 or point['correct']>=base['correct']):
            raise ValueError('Actual persistent large coding regression required for this manual research handover')
        return True
    completed=previous/'training-complete.json'
    if not completed.exists() or not json.loads(completed.read_text())['full_source_epoch_complete']:
        raise ValueError('Previous run must finish its epoch or have an explicit preserved quality handover')
    return True


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Waiting controller must hide all GPUs')
    if REPORTS.exists() or OUTPUT.exists() or QUEUE.exists():raise ValueError('Preserve previous protected attempts')
    previous=ROOT/'reports/fast-joint-v3';prior=json.loads((previous/'plan.json').read_text())
    control=ROOT/'reports/protected-joint-full-CPU-v2';proof=json.loads((control/'complete.json').read_text())
    control_plan=json.loads((control/'plan.json').read_text())
    if not (proof['language_bytes_exact_during_visual_update'] and proof['reference_and_all_unused_heads_exact']
            and proof['actual_next_model_Adam_CPU_RNG_and_metrics_recovery_exact']
            and proof['actual_trained_text_image_word_cold_reload_exact']):raise ValueError('Actual complete-size protected CPU learning/recovery/cold outputs required')
    positive=ROOT/'reports/protected-positive-KL-full-CPU-v1.json'
    if not json.loads(positive.read_text())['actual_nonzero_KL_and_all317_language_gradients_finite']:
        raise ValueError('Actual full-size positive-reference-KL gradients required')
    tests={'reports/protected-joint-stage-v1-tests.log':'2 passed','reports/protected-joint-update-v1-tests.log':'1 passed',
        'reports/protected-joint-driver-v1-tests.log':'2 passed',
        'reports/deterministic-positions-v1-tests.log':'3 passed','reports/fast-joint-checkpoint-v2-tests.log':'1 passed'}
    for path,passed in tests.items():
        if passed not in (ROOT/path).read_text():raise ValueError('Actual protected invariance/update/position/recovery tests required')
    files=dict(control_plan['source_files_sha256'])
    additions=['scripts/train_protected_joint_v1.py','src/slm_lab/protected_joint_update_v1.py',
        'src/slm_lab/protected_joint_stage_v1.py','src/slm_lab/dense_native_view_v2.py',
        'scripts/monitor_fast_joint_regression_v1.py','tests/test_protected_joint_stage_v1.py',
        'tests/test_protected_joint_update_v1.py','tests/test_protected_joint_driver_v1.py',
        'reports/fast-joint-caption-CPU-v2/complete.json','reports/fast-joint-caption-CPU-v2/initial-complete.json',
        'reports/fast-joint-caption-CPU-v2/step-1024-complete.json',str(positive.relative_to(ROOT)),*tests]
    for name in additions:files[str(ROOT/name)]=file_hash(ROOT/name)
    for name in ['plan.json','complete.json','resume-step1.pt','model.pt']:files[str(control/name)]=file_hash(control/name)
    pulse_sources=make_pulse_sources()
    for path in (ROOT/'data/protected-joint-v1-pulses').iterdir():files[str(path)]=file_hash(path)
    me=owner.resources.identity(os.getpid());guard=RuntimeGuard(files);until=time.monotonic()+24*3600
    while not previous_ready(previous,prior):
        guard.check()
        if time.monotonic()>until:raise ValueError('Previous exact job still live; expire without preempting it')
        atomic_json(QUEUE,dict(status='waiting for exact preceding experiment and preserved coding-priority handover or full epoch completion',
            owner=me,predecessor=prior['owner'],source_files_sha256=files,production_updates=0,goal_achieved=False));time.sleep(5)
    settings={key:prior[key] for key in ['initial_dense_settings','actual_counts','planned_steps','seed','complete_source_budget',
        'global_family_batch_widths','tokenizer_sha256','all_original_sources_once','no_rounding_repeats']}
    settings['training_options']=control_plan['training_options']
    if (settings['initial_dense_settings']!=control_plan['initial_dense_settings']
            or settings['complete_source_budget']!=control_plan['available_full_original_source_budget']):
        raise ValueError('Exact same own initial student and full original source curriculum required')
    word,book,image=prior['capacity_global_batches']
    settings.update(capacity_global_batches=[book,word,book,image],pulse_sources=pulse_sources,
        development_pulse_wall_seconds=180,development_code_generation_budget=128,development_math_generation_budget=256,
        initial_development_pulse_deferred_until_update=1024,CPU_control_barriers='gloo',
        actual_training_only_reference_parameters=settings['actual_counts']['total_unique'],
        exported_policy_parameters=settings['actual_counts']['total_unique'],
        extra_reference_is_own_initial_policy_not_external_teacher=True,
        previous_partial_exposures_not_new_raw_data=True,configured_GPU_capacity_must_be_reproved=True,
        preference_and_execution_RL_deferred_until_skills_are_measured=True)
    for path in [previous/'plan.json',previous/'restored.json',ROOT/'reports/protected-joint-v1-handover.json']:
        if path.exists():files[str(path)]=file_hash(path)
    guard=RuntimeGuard(files)
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);owner.idle_original()
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(used)!=4 or any(int(v)>128 for v in used):raise ValueError('All four local GPUs must be idle before ownership')
        required=RESERVE+4*MAX_STATE+(settings['planned_steps']//1024+2)*MAX_MODEL
        if shutil.disk_usage(ROOT).free<required:raise ValueError('All planned native snapshots, complete Adam/atomic states and reserve must fit')
        guard.check(full=True);REPORTS.mkdir();OUTPUT.mkdir()
        plan=dict(**settings,owner=me,waiter=owner.WAITER,source_files_sha256=files,created_at=datetime.now(timezone.utc).isoformat(),
            expires_at_unix=time.time()+48*3600,automatic_retry=False,candidate_promoted=False,goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[])
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned protected experiment interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        driver=str(Path(__file__).resolve());watcher=subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,
            stdout=(REPORTS/'watchdog.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        def tasks(items):owner.wait_children(plan,[owner.launch(plan,['-u',driver,*args],subject,str(gpu)) for args,subject,gpu in items])
        def distributed_task(mode):
            child=owner.launch(plan,['-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',driver,'--distributed',mode],mode+'-launcher','0,1,2,3')
            owner.wait_children(plan,[child])
        try:
            owner.idle_original();os.kill(owner.WAITER['pid'],signal.SIGSTOP);limit=time.monotonic()+5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in ('T','t'):
                if time.monotonic()>limit:raise ValueError('Exact original sleeping waiter hold failed')
                time.sleep(.05)
            if (Path('/proc')/str(owner.WAITER['pid'])/'task'/str(owner.WAITER['pid'])/'children').read_text().strip():raise ValueError('Never preempt original active work')
            distributed_task('capacity');proofs=[json.loads((REPORTS/f'capacity-{i}.json').read_text()) for i in range(4)]
            for i,p in enumerate(proofs):
                if (p['physical_GPU']!=i or p['actual_parameter_counts']!=plan['actual_counts'] or p['actual_Adam_states']!=445
                        or p['training_only_reference_values']!=plan['actual_training_only_reference_parameters']
                        or not p['exact_actual_distributed_next_update_weights_and_Adam_recovery']
                        or not p['actual_language_exact_during_image_update'] or not p['actual_training_only_reference_bytes_exact']
                        or p['actual_positive_full_context_KL']<=0):raise ValueError('Every actual full-profile four-GPU protected/reference capacity and exact recovery must pass')
            atomic_json(REPORTS/'capacity.json',dict(passed=True,proofs=proofs,old_capacity_not_reused=True,production_updates=0,goal_achieved=False))
            distributed_task('train')
            for arm in ['initial','trained']:
                tasks([(['--assess',suite,'--arm',arm,'--gpu',str(i)],'assess-'+arm+'-'+suite+'-'+str(i),i) for i,suite in enumerate(['knowledge','code','cpp','vision'])])
                for suite in ['public','math']:
                    tasks([(['--assess',suite,'--arm',arm,'--gpu',str(i),'--shard',str(i)],'assess-'+arm+'-'+suite+'-'+str(i),i) for i in range(4)])
                from slm_lab.full_math_shards_v1 import reconcile
                reconcile([REPORTS/arm/f'math-{i}.json' for i in range(4)],ROOT/'data/math-reasoning-v2/test.jsonl',OUTPUT/arm/'tokenizer.json',REPORTS/arm/'math-complete.json')
            guard.check(full=True)
            atomic_json(REPORTS/'complete.json',dict(status='Protected full-source training and unchanged native endpoints finished; independent comparison still required',
                independent_comparison_pending=True,candidate_promoted=False,goal_achieved=False))
        except BaseException as error:atomic_json(REPORTS/'failure.json',dict(error=repr(error),automatic_retry=False,goal_achieved=False));raise
        finally:owner.restore_owned(plan,'Protected owned experiment ended; original work restored');watcher.wait(timeout=20)

if __name__=='__main__':
    configure();p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watchdog',action='store_true')
    p.add_argument('--distributed',choices=['capacity','train']);p.add_argument('--assess',choices=['knowledge','code','cpp','vision','public','math'])
    p.add_argument('--pulse-child',type=int,choices=range(4));p.add_argument('--arm',default='initial');p.add_argument('--gpu',type=int,choices=range(4),default=0);p.add_argument('--shard',type=int,choices=range(4))
    args=p.parse_args()
    if args.watchdog:owner.watchdog()
    elif args.pulse_child is not None:pulse_child(args.arm,args.pulse_child)
    elif args.distributed:distributed(args.distributed)
    elif args.assess:assess(args.arm,args.assess,args.gpu,args.shard)
    else:parent()
