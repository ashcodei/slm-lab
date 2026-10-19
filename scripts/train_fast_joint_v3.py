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
REPORTS=ROOT/'reports/fast-joint-v3'
OUTPUT=ROOT/'checkpoints/fast-joint-v3'
QUEUE=ROOT/'reports/fast-joint-v3-queue.json'
CPU=ROOT/'reports/fast-joint-full-CPU-v2'
PREVIOUS=ROOT/'reports/code-initialized-alignment-v1'
HANDOVER=ROOT/'reports/fast-joint-v1-handover'
MAX_STATE=16*1024**3
MAX_MODEL=5*1024**3
RESERVE=26*1024**3


def configure():
    owner.REPORTS=REPORTS;owner.resources.REPORTS=REPORTS;legacy.REPORTS=REPORTS;legacy.OUTPUT=OUTPUT


def previous_ended(identity):
    if owner.resources.same(identity):return False
    path=PREVIOUS/'children.json'
    if path.exists() and any(owner.resources.same(a) for a in json.loads(path.read_text())):return False
    if not (PREVIOUS/'restored.json').exists():raise ValueError('Old exact GPU workers must end and original work must be restored')
    retirement=HANDOVER/'code-evaluation-retirement.json'
    if retirement.exists():
        r=json.loads(retirement.read_text())
        if (r.get('retired_owner')!=dict(pid=identity['pid'],ticks=str(identity['ticks']))
                or r.get('both_training_arms_completed_before_signal') is not True
                or r.get('remaining_evaluations_explicitly_incomplete') is not True):
            raise ValueError('Explicit completed-training evaluation handover receipt required')
        for p,digest in r['retained_terminal_training_receipts_sha256'].items():
            if file_hash(p)!=digest:raise ValueError('Completed preceding training evidence changed')
        return True
    if not all((PREVIOUS/n).exists() for n in ('complete.json','independent-comparison.json')) or (PREVIOUS/'failure.json').exists():
        raise ValueError('Predecessor needs successful completion or an explicit authorized evaluation retirement')
    return True


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
    from slm_lab.dense_foundation_inference_v1 import DenseNativeView
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


def validate_inherited_capacity(plan,previous,capacity):
    """Reuse actual hardware evidence only when the complete learning graph agrees."""
    keys=['initial_dense_settings','actual_counts','planned_steps','seed','complete_source_budget',
          'global_family_batch_widths','training_options','tokenizer_sha256','capacity_global_batches',
          'all_original_sources_once','no_rounding_repeats']
    if any(plan[k]!=previous[k] for k in keys):raise ValueError('Inherited full training profile changed')
    proofs=capacity.get('proofs',[])
    if (capacity.get('passed') is not True or len(proofs)!=4
            or {p['physical_GPU'] for p in proofs}!={0,1,2,3}):
        raise ValueError('Every actual original GPU capacity receipt required')
    for p in proofs:
        if (p['actual_parameter_counts']!=plan['actual_counts']
                or p['full_configured_word_book_image_capacity_passed'] is not True
                or p['exact_actual_distributed_next_update_weights_and_Adam_recovery'] is not True
                or p['actual_CPU_and_private_CUDA_RNG_restored'] is not True):
            raise ValueError('Full configured graph and exact actual recovery required')
    # All runtime modules admitted by the prior CPU/GPU proofs must be byte
    # identical. Only this new outer controller, its additional admission
    # implementation, and an inference-only cache are new.
    for path,digest in previous['source_files_sha256'].items():
        p=Path(path)
        if p.suffix=='.py' and p.is_relative_to(ROOT/'src/slm_lab'):
            if plan['source_files_sha256'].get(path)!=digest or file_hash(path)!=digest:
                raise ValueError('Previously proved neural/training graph changed: '+path)
    return True


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
    from slm_lab.fast_joint_stage_v1 import make_optimizer,parameter_groups,update
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
        dist.barrier(group=control)
    try:
        if mode=='capacity':
            schedule.entries=[(name,positions) for name,positions in plan['capacity_global_batches']]
            records.append(update(model,optimizer,data,schedule,0,options,device=device,rank=rank,world=world,amp=amp))
            save_state(OUTPUT/'capacity-resume-step1.pt',1)
            def draws():return [random.random(),float(np.random.random()),float(torch.rand(())),float(torch.rand((),device=device))]
            expected_draws=draws()
            expected=update(model,optimizer,data,schedule,1,options,device=device,rank=rank,world=world,amp=amp)
            expected_weights=tensor_digest(model.named_parameters())
            def adam_digest():return tensor_digest((str(i)+'.'+k,v[k]) for i,v in enumerate(optimizer.state.values()) for k in ('step','exp_avg','exp_avg_sq'))
            expected_adam=adam_digest();payload=torch.load(OUTPUT/'capacity-resume-step1.pt',map_location='cpu',weights_only=True,mmap=True)
            bad_rngs=[dict(r) for r in payload['rngs']];bad_rngs[rank]['cuda']=torch.zeros(4,dtype=torch.uint8)
            bad=dict(payload,rngs=bad_rngs);before_rng=capture_rng(device)
            try:restore(model,optimizer,bad,plan,digest,device,rank,world,'capacity')
            except ValueError:pass
            else:raise ValueError('Invalid actual private CUDA RNG must reject')
            if (expected_weights!=tensor_digest(model.named_parameters()) or expected_adam!=adam_digest()
                    or not torch.equal(before_rng['torch'],torch.get_rng_state()) or not torch.equal(before_rng['cuda'],torch.cuda.get_rng_state(device))):
                raise ValueError('Invalid recovery mutated live weights/Adam or caller RNG')
            completed,records=restore(model,optimizer,payload,plan,digest,device,rank,world,'capacity')
            if draws()!=expected_draws:raise ValueError('Actual Python/NumPy/CPU/private-CUDA next draws must recover exactly')
            repeated=update(model,optimizer,data,schedule,1,options,device=device,rank=rank,world=world,amp=amp)
            if expected!=repeated or expected_weights!=tensor_digest(model.named_parameters()) or expected_adam!=adam_digest():
                raise ValueError('Actual four-GPU next update/weights/all Adam states must recover exactly')
            records.append(repeated)
            # The longest actual image family uses its configured complete
            # global batch and dynamic microbatch, with all Adam states live.
            records.append(update(model,optimizer,data,schedule,2,options,device=device,rank=rank,world=world,amp=amp))
            if fixed!=tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in selected):raise ValueError('Unused counted heads changed')
            result=dict(rank=rank,physical_GPU=local,full_configured_word_book_image_capacity_passed=True,
                exact_actual_distributed_next_update_weights_and_Adam_recovery=True,
                actual_parameter_counts=model.counts(),actual_selected_values={n:sum(p.numel() for _,p in g) for n,g in groups.items()},
                actual_CPU_and_private_CUDA_RNG_restored=True,disposable_logical_updates=3,physical_optimizer_calls=4,
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
        guard.check(full=True);dist.barrier(group=control)
    finally:dist.destroy_process_group()


def assess(arm,suite,gpu,shard):
    subject='assess-'+arm+'-'+suite+'-'+str(shard if shard is not None else gpu)
    plan=legacy.registered(subject,gpu);torch,guard,amp=legacy.runtime(plan)
    from slm_lab.fast_joint_inference_v2 import load,DenseNativeView
    model,metadata=load(OUTPUT/arm/'model.pt',REPORTS/'plan.json',arm,expected_checkpoint_sha256=file_hash(OUTPUT/arm/'model.pt'),
        expected_plan_sha256=file_hash(REPORTS/'plan.json'),device='cuda:0')
    legacy.native_suite(plan,SimpleNamespace(aligned=DenseNativeView(model)),metadata,arm,suite,gpu,guard,shard)


def make_pulse_sources():
    folder=ROOT/'data/fast-joint-v3-pulses'
    if folder.exists():raise ValueError('Preserve previous bounded development source selection')
    folder.mkdir();paths={};origins={}
    specs=[('knowledge','data/knowledge-eval-v1/questions.jsonl','domain',4),
        ('ocr','data/alignment-curriculum-v1/images.valid.jsonl','task',8),
        ('objects','data/object-qa-v1/valid.jsonl',None,32),
        ('vision','data/alignment-curriculum-v1/contrastive.valid.jsonl',None,32)]
    for name,relative,field,per_group in specs:
        source=ROOT/relative;rows=list(map(json.loads,source.open()));chosen=[]
        if field:
            for value in sorted({r[field] for r in rows}):chosen.extend([r for r in rows if r[field]==value][:per_group])
        else:chosen=rows[:per_group]
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


def parent():
    if REPORTS.exists() or OUTPUT.exists() or QUEUE.exists():raise ValueError('Preserve prior joint production attempts')
    proof=json.loads((CPU/'complete.json').read_text());cpu_plan=json.loads((CPU/'plan.json').read_text())
    if not (proof['all_language_vision_word_groups_changed'] and proof['actual_trained_native_text_image_word_cold_reload_exact']):
        raise ValueError('Actual full-size joint CPU learning/cold loading must pass')
    if '1 passed' not in (ROOT/'reports/fast-joint-checkpoint-v2-tests.log').read_text():raise ValueError('Actual complete checkpoint recovery checks required')
    if '2 passed' not in (ROOT/'reports/fast-joint-driver-v1-tests.log').read_text():raise ValueError('Exact original-owner/worker handover checks required')
    if '3 passed' not in (ROOT/'reports/deterministic-positions-v1-tests.log').read_text():raise ValueError('Exact position-grid forward/gradient and full native model integration checks required')
    if '3 passed' not in (ROOT/'reports/fast-joint-driver-v3-tests.log').read_text():raise ValueError('Exact isolated development timeout and unchanged inherited training profile checks required')
    previous_joint=ROOT/'reports/fast-joint-v1'
    previous_plan=json.loads((previous_joint/'plan.json').read_text())
    if owner.resources.same(previous_plan['owner']) or any(owner.resources.same(a) for a in json.loads((previous_joint/'children.json').read_text())) or not (previous_joint/'restored.json').exists():raise ValueError('Previous failed capacity owner and every exact worker must be terminal and restored')
    files=dict(cpu_plan['source_files_sha256'])
    for name in ['plan.json','failure.json','restored.json']:files[str(previous_joint/name)]=file_hash(previous_joint/name)
    for name in ['scripts/train_fast_joint_v3.py','tests/test_fast_joint_driver_v3.py','reports/fast-joint-driver-v3-tests.log','src/slm_lab/parallel_file_bindings_v1.py','tests/test_parallel_file_bindings_v1.py','src/slm_lab/inference_rotary_cache_v1.py','tests/test_inference_rotary_cache_v1.py','reports/inference-rotary-cache-v1-tests.log','reports/parallel-file-bindings-v1-tests.log','src/slm_lab/fast_joint_checkpoint_v2.py','src/slm_lab/fast_joint_inference_v2.py','src/slm_lab/deterministic_positions_v1.py','tests/test_deterministic_positions_v1.py','reports/deterministic-positions-v1-tests.log','tests/test_fast_joint_checkpoint_v2.py',
        'reports/fast-joint-checkpoint-v2-tests.log','tests/test_fast_joint_driver_v1.py','reports/fast-joint-driver-v1-tests.log']:
        files[str(ROOT/name)]=file_hash(ROOT/name)
    for name in ['plan.json','complete.json','disposable-trained-model.pt']:files[str(CPU/name)]=file_hash(CPU/name)
    pulse_sources=make_pulse_sources()
    for p in (ROOT/'data/fast-joint-v3-pulses').iterdir():files[str(p)]=file_hash(p)
    predecessor=json.loads((ROOT/'reports/code-initialized-alignment-v1-queue.json').read_text())['owner'];me=owner.resources.identity(os.getpid())
    guard=RuntimeGuard(files);deadline=time.monotonic()+24*3600
    while not previous_ended(predecessor):
        guard.check()
        if time.monotonic()>deadline:raise ValueError('Expire without preempting the original active work')
        atomic_json(QUEUE,dict(status='waiting for exact completed-training evaluator handover and original GPU restoration',owner=me,
            predecessor=predecessor,source_files_sha256=files,observed_at=datetime.now(timezone.utc).isoformat(),production_updates=0,goal_achieved=False));time.sleep(5)
    previous_joint2=ROOT/'reports/fast-joint-v2'
    inherited=json.loads((previous_joint2/'plan.json').read_text())
    prior_owner=inherited['owner'];limit=time.monotonic()+3600
    while owner.resources.same(prior_owner) or any(owner.resources.same(a) for a in json.loads((previous_joint2/'children.json').read_text())):
        if time.monotonic()>limit:raise ValueError('Previous exact trainer still live; no overlapping GPU ownership')
        atomic_json(QUEUE,dict(status='waiting for exact previous trainer/evaluation isolation handover',owner=me,predecessor=prior_owner,production_updates=0,goal_achieved=False))
        time.sleep(2)
    if not (previous_joint2/'restored.json').exists():raise ValueError('Previous original work restoration required')
    capacity=json.loads((previous_joint2/'capacity.json').read_text())
    if not capacity['passed'] or len(capacity['proofs'])!=4 or any(not p['exact_actual_distributed_next_update_weights_and_Adam_recovery'] for p in capacity['proofs']):raise ValueError('Inherited actual four-GPU exact capacity/recovery proof required')
    settings={key:inherited[key] for key in ['initial_dense_settings','actual_counts','planned_steps','seed','complete_source_budget','global_family_batch_widths','training_options','tokenizer_sha256','capacity_global_batches','all_original_sources_once','no_rounding_repeats']}
    settings.update(pulse_sources=pulse_sources,preference_and_execution_RL_deferred_until_skills_are_measured=True,
        development_pulse_wall_seconds=180,development_code_generation_budget=128,development_math_generation_budget=256,
        initial_development_pulse_deferred_until_update=1024,CPU_control_barriers='gloo',
        inherited_capacity_plan_sha256=file_hash(previous_joint2/'plan.json'),inherited_capacity_sha256=file_hash(previous_joint2/'capacity.json'),
        GPU_capacity_inherited_only_for_identical_training_graph=True)
    if settings['complete_source_budget']!=cpu_plan['complete_source_budget'] or settings['training_options']!=cpu_plan['training_options']:raise ValueError('Whole-source and actual training profile must remain unchanged')
    for name in ['plan.json','capacity.json','restored.json']:
        files[str(previous_joint2/name)]=file_hash(previous_joint2/name)
    for i in range(4):files[str(previous_joint2/f'capacity-{i}.json')]=file_hash(previous_joint2/f'capacity-{i}.json')
    if (HANDOVER/'code-evaluation-retirement.json').exists():files[str(HANDOVER/'code-evaluation-retirement.json')]=file_hash(HANDOVER/'code-evaluation-retirement.json')
    files[str(PREVIOUS/'restored.json')]=file_hash(PREVIOUS/'restored.json');guard=RuntimeGuard(files)
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);owner.idle_original()
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(used)!=4 or any(int(v)>128 for v in used):raise ValueError('All four GPUs must be idle before exclusive ownership')
        if shutil.disk_usage(ROOT).free<RESERVE+4*MAX_STATE:raise ValueError('Complete full-model/Adam atomic state and reserve required')
        guard.check(full=True);REPORTS.mkdir();OUTPUT.mkdir()
        plan=dict(**settings,owner=me,waiter=owner.WAITER,source_files_sha256=files,created_at=datetime.now(timezone.utc).isoformat(),
            expires_at_unix=time.time()+48*3600,automatic_retry=False,candidate_promoted=False,goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[])
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned joint training interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        driver=str(Path(__file__).resolve());watcher=subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,
            stdout=(REPORTS/'watchdog.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        def tasks(items):owner.wait_children(plan,[owner.launch(plan,['-u',driver,*args],subject,str(gpu)) for args,subject,gpu in items])
        def distributed_task(mode):
            child=owner.launch(plan,['-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',driver,'--distributed',mode],mode+'-launcher','0,1,2,3')
            owner.wait_children(plan,[child])
        try:
            owner.idle_original();os.kill(owner.WAITER['pid'],signal.SIGSTOP);until=time.monotonic()+5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in ('T','t'):
                if time.monotonic()>until:raise ValueError('Original exact idle waiter hold failed')
                time.sleep(.05)
            if (Path('/proc')/str(owner.WAITER['pid'])/'task'/str(owner.WAITER['pid'])/'children').read_text().strip():raise ValueError('Never preempt original real work')
            validate_inherited_capacity(plan,inherited,capacity)
            atomic_json(REPORTS/'capacity.json',dict(passed=True,inherited_from='reports/fast-joint-v2/capacity.json',proofs=capacity['proofs'],identical_training_graph_and_full_profile=True,production_updates=0,goal_achieved=False))
            distributed_task('train')
            for arm in ['initial','trained']:
                tasks([(['--assess',suite,'--arm',arm,'--gpu',str(i)],'assess-'+arm+'-'+suite+'-'+str(i),i) for i,suite in enumerate(['knowledge','code','cpp','vision'])])
                for suite in ['public','math']:
                    tasks([(['--assess',suite,'--arm',arm,'--gpu',str(i),'--shard',str(i)],'assess-'+arm+'-'+suite+'-'+str(i),i) for i in range(4)])
                from slm_lab.full_math_shards_v1 import reconcile
                reconcile([REPORTS/arm/f'math-{i}.json' for i in range(4)],ROOT/'data/math-reasoning-v2/test.jsonl',OUTPUT/arm/'tokenizer.json',REPORTS/arm/'math-complete.json')
            guard.check(full=True)
            atomic_json(REPORTS/'complete.json',dict(status='Joint full-weight curriculum and matched native endpoints finished; independent comparison still required',
                independent_comparison_pending=True,candidate_promoted=False,goal_achieved=False))
        except BaseException as e:atomic_json(REPORTS/'failure.json',dict(error=repr(e),automatic_retry=False,goal_achieved=False));raise
        finally:owner.restore_owned(plan,'Joint owned run ended; original work restored');watcher.wait(timeout=20)


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
