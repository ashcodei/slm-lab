"""Complete expanded foundation/instruction epoch after the current exact job.

No current work is preempted. Raw foundation runs without a model reference;
the completed own foundation becomes the later text anchor. Fresh actual full
four-L40 capacity/recovery gates production. Every original source is retained.
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

import train_whole_human_SFT_v1 as legacy
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/phase-joint-v2'
OUTPUT=ROOT/'checkpoints/phase-joint-v2'
QUEUE=ROOT/'reports/phase-joint-v2-queue.json'
PREVIOUS=ROOT/'reports/execution-single-update-recovery-v3'
MAX_STATE=14*1024**3
MAX_MODEL=5*1024**3
RESERVE=26*1024**3
owner=legacy.owner


def configure():
    owner.REPORTS=REPORTS;owner.resources.REPORTS=REPORTS
    legacy.REPORTS=REPORTS;legacy.OUTPUT=OUTPUT


def previous_ready(prior):
    if owner.resources.same(prior['owner']):return False
    if any(owner.resources.same(a) for a in json.loads((PREVIOUS/'children.json').read_text())):return False
    if not (PREVIOUS/'restored.json').exists():raise ValueError('Every preceding exact worker must end and restore original work')
    done=PREVIOUS/'training-complete.json'
    if (not done.exists() or (PREVIOUS/'failure.json').exists()
            or json.loads(done.read_text()).get('matched_native_development_assessments_finished') is not True):
        raise ValueError('The preceding controlled experiment and native assessments must actually complete; no automatic retirement')
    return True


def distributed(mode):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1,2,3' or int(os.environ['WORLD_SIZE'])!=4:raise ValueError('All four owned original GPUs required')
    rank=int(os.environ['RANK']);plan=owner.registered(mode+'-rank-'+str(rank));digest=file_hash(REPORTS/'plan.json')
    import copy
    from concurrent.futures import ThreadPoolExecutor
    import random
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.codenet_checkpoint_v1 import save_collective,capture_rng
    from slm_lab.expanded_joint_sources_v1 import ExpandedJointSources,CompletePhaseSchedule
    from slm_lab.fast_joint_inference_v2 import fresh,export
    from slm_lab.fast_joint_stage_v1 import make_optimizer,parameter_groups
    from slm_lab.phase_joint_update_v1 import update
    from slm_lab.phase_joint_checkpoint_v1 import make_state,restore_state
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint
    torch.set_num_threads(1);device=torch.device('cuda',rank);torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.85,device);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name(device)!='NVIDIA L40':raise ValueError('Only original local L40s admitted')
    dist.init_process_group('nccl',timeout=timedelta(minutes=40));control=dist.new_group(backend='gloo',timeout=timedelta(minutes=40))
    started=time.monotonic()
    try:
        guard=FileBindings(plan['source_files_sha256']);model,_=fresh(plan,device);data=ExpandedJointSources(ROOT,model.config)
        if data.budget()!=plan['complete_source_budget']:raise ValueError('Every original broad/code/QA/image family must match')
        schedule=CompletePhaseSchedule(data,plan['global_family_batch_widths'],plan['seed'])
        if len(schedule)!=plan['planned_steps'] or schedule.phase_bounds!=plan['phase_bounds']:raise ValueError('Exact full expanded source phases required')
        if mode=='capacity':
            schedule.entries=[(name,positions) for name,positions in plan['capacity_global_batches']]
            schedule.phase_for_step=['foundation']*2+['instruction_and_vision']*4;schedule.phase_bounds=plan['capacity_phase_bounds']
        options=plan['training_options'];optimizer=make_optimizer(model,options,device);reference=None;binding=None;records=[]
        groups=parameter_groups(model);selected={id(p) for rows in groups.values() for _,p in rows}
        unused=tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in selected)
        random.seed(plan['seed']+rank);np.random.seed(plan['seed']+rank);torch.manual_seed(plan['seed']+rank)
        amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16)
        def save_model(arm,completed):
            if rank==0:
                folder=OUTPUT/arm;folder.mkdir()
                shutil.copyfile(ROOT/'data/tokenizer-corpus/tokenizer.json',folder/'tokenizer.json')
                atomic_checkpoint(folder/'model.pt',export(model,plan,digest,arm,completed),MAX_MODEL,RESERVE)
            dist.barrier(group=control)
        def make_reference(boundary):
            nonlocal reference,binding
            arm=mode+'-foundation-reference';save_model(arm,boundary)
            reference=copy.deepcopy(model).eval().requires_grad_(False)
            binding=dict(path=str(OUTPUT/arm/'model.pt'),checkpoint_sha256=file_hash(OUTPUT/arm/'model.pt'),
                parameter_sha256=tensor_digest(reference.named_parameters()),foundation_completed_updates=boundary)
            gathered=[None]*4;dist.all_gather_object(gathered,binding,group=control)
            if any(row!=binding for row in gathered):raise ValueError('Every worker must use identical newly trained own foundation weights')
            if rank==0:atomic_json(REPORTS/(mode+'-reference-binding.json'),binding)
        def save_state(path,completed):
            save_collective(path,lambda rngs:make_state(model,optimizer,plan,digest,completed,records,rngs,mode,
                reference_binding=binding,reference=reference),device,4,rank,MAX_STATE,RESERVE)
            dist.barrier(group=control)
        def step(i,prepared=None):
            owner.owned(plan)
            record=update(model,optimizer,data,schedule,i,options,reference=reference,device=device,rank=rank,world=4,amp=amp,prepared=prepared)
            records.append(record);return record
        def adam():return tensor_digest((str(i)+'.'+k,s[k]) for i,s in enumerate(optimizer.state.values()) for k in ('step','exp_avg','exp_avg_sq'))
        def recover(path,completed,next_step):
            nonlocal records
            save_state(path,completed);expected=step(next_step);weights=tensor_digest(model.named_parameters());moments=adam()
            saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            bad=dict(saved,core=dict(saved['core'],rngs=[dict(r) for r in saved['core']['rngs']]))
            bad['core']['rngs'][rank]['cuda']=torch.zeros(4,dtype=torch.uint8);RNG=capture_rng(device)
            versions=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in model.parameters()]
            try:restore_state(model,optimizer,bad,plan,digest,device,rank,4,mode,reference_binding=binding,reference=reference)
            except ValueError:pass
            else:raise ValueError('Invalid actual private CUDA RNG must reject')
            if (versions!=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in model.parameters()] or moments!=adam()
                    or not torch.equal(RNG['torch'],torch.get_rng_state()) or not torch.equal(RNG['cuda'],torch.cuda.get_rng_state(device))):
                raise ValueError('Rejected state changed actual policy/Adam/CPU/CUDA RNG')
            count,records=restore_state(model,optimizer,saved,plan,digest,device,rank,4,mode,reference_binding=binding,reference=reference)
            if count!=completed or step(next_step)!=expected or weights!=tensor_digest(model.named_parameters()) or moments!=adam():
                raise ValueError('Actual next source/phase/weights/Adam/private RNG must recover exactly')
        if mode=='capacity':
            step(0);recover(OUTPUT/'capacity-raw-step1.pt',1,1);make_reference(2)
            step(2);step(3)
            if records[-1]['full_context_reference_KL_sum']<=0:raise ValueError('Actual trained own-reference KL must be positive')
            recover(OUTPUT/'capacity-post-step4.pt',4,4);language=tensor_digest(groups['language']);step(5)
            if language!=tensor_digest(groups['language']):raise ValueError('Actual word update changed protected decoder')
            proof=dict(rank=rank,physical_GPU=rank,actual_counts=model.counts(),actual_Adam_states=len(optimizer.state),
                logical_control_updates=6,physical_optimizer_calls=8,actual_records=records,
                exact_raw_and_later_next_updates_weights_Adam_all_worker_RNG=True,invalid_private_CUDA_RNG_rejected_before_mutation=True,
                actual_positive_own_foundation_reference_KL=True,actual_reference_binding=binding,
                own_reference_values=reference.counts()['total_unique'],exported_policy_values=model.counts()['total_unique'],
                actual_full_source_extremes_not_truncated=True,decoder_exact_during_word_update=True,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                seconds=time.monotonic()-started,production_updates=0,goal_achieved=False)
            atomic_json(REPORTS/f'capacity-{rank}.json',proof)
        else:
            from slm_lab.packed_validation_v1 import evaluate as held_packed
            def validate_packed(label):
                report=held_packed(model,data.packed,device=device,rank=rank,world=4,amp=amp)
                if rank==0:atomic_json(REPORTS/('held-packed-'+label+'.json'),report)
            save_model('initial',0);validate_packed('initial')
            journal=(REPORTS/'training-metrics.jsonl').open('x') if rank==0 else None
            def prepare(i):
                name,positions=schedule.rank_positions(i,rank,4)
                if not positions:return []
                prefix=model.config.image_token_count(448) if data.kinds[name] in ('word','image') else 0
                width=max(1,min(options['maximum_microbatch'],options['maximum_microbatch_tokens']//(max(data.lengths[name][p] for p in positions)+prefix)))
                return [{k:(v.pin_memory() if isinstance(v,torch.Tensor) else v) for k,v in data.batch(name,positions[a:a+width],'cpu').items()}
                    for a in range(0,len(positions),width)]
            with ThreadPoolExecutor(max_workers=1) as reader:
                pending=reader.submit(prepare,0)
                for i in range(len(schedule)):
                    if i%32==0:guard.check()
                    prepared=pending.result()
                    if i+1<len(schedule):pending=reader.submit(prepare,i+1)
                    tick=time.monotonic();record=step(i,prepared)
                    if journal:
                        journal.write(json.dumps(record)+'\n');journal.flush()
                        if i==0 or (i+1)%25==0:atomic_json(REPORTS/'progress.json',dict(actual_updates=i+1,planned_updates=len(schedule),
                            phase=record['phase'],latest=record,last_update_seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-started,goal_achieved=False))
                    boundary=i+1==schedule.phase_bounds['foundation'][1]
                    if (i+1)%2048==0 or boundary or i+1==len(schedule):validate_packed(str(i+1))
                    if boundary:make_reference(i+1)
                    if (i+1)%2048==0 or boundary or i+1==len(schedule):
                        save_state(OUTPUT/'resume.pt',i+1)
                        save_model('trained' if i+1==len(schedule) else f'step-{i+1}',i+1)
            if journal:journal.close()
            if rank==0:atomic_json(REPORTS/'training-complete.json',dict(actual_updates=len(records),full_selected_family_source_epoch_complete=True,
                complete_source_budget=data.budget(),actual_counts=model.counts(),own_reference_binding=binding,
                native_quality_and_independent_state_reconciliation_pending=True,goal_achieved=False))
        if (unused!=tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in selected)
                or binding['parameter_sha256']!=tensor_digest(reference.named_parameters())):raise ValueError('Unused heads/critic or own frozen reference changed')
        guard.check(full=True);dist.barrier(group=control)
    finally:dist.destroy_process_group()


def assess(arm,gpu,suite):
    plan=legacy.registered('assess-'+arm+'-'+suite+'-'+str(gpu),gpu)
    import torch
    from slm_lab.fast_joint_inference_v2 import load
    from slm_lab.dense_native_view_v2 import CompleteDenseNativeView
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85,0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checkpoint=OUTPUT/arm/'model.pt';model,metadata=load(checkpoint,REPORTS/'plan.json',arm,
        expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=file_hash(REPORTS/'plan.json'),device='cuda:0')
    view=CompleteDenseNativeView(model);destination=REPORTS/arm;destination.mkdir(exist_ok=True)
    old_threads=torch.set_num_threads;torch.set_num_threads=lambda count:None
    def evaluate(module,name,data=None,**kwargs):
        original=module.load_checkpoint;module.load_checkpoint=lambda *a,**kw:(view,metadata)
        try:
            if data is None:module.evaluate(checkpoint,destination/name,device='cuda:0',**kwargs)
            else:module.evaluate(checkpoint,data,destination/name,device='cuda:0',**kwargs)
        finally:module.load_checkpoint=original
    try:
        with inference_cache():
            if suite=='code':
                from slm_lab.native_code_shards_v1 import evaluate_shard
                for frame in ['completion','instruction']:evaluate_shard(view,metadata,checkpoint,destination,rank=gpu,world=4,frame=frame)
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
        owner.owned(plan);FileBindings(plan['source_files_sha256']).check(full=True)
    finally:torch.set_num_threads=old_threads


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU queue must hide all GPUs')
    if REPORTS.exists() or OUTPUT.exists() or QUEUE.exists():raise ValueError('Preserve each previous queued/partial attempt')
    tests={'phase-joint-update-v1':2,'phase-joint-checkpoint-v1':1,'expanded-joint-sources-v1':2,'native-code-shards-v1':2,'phase-joint-driver-v2':2,'packed-validation-v1':1}
    for name,count in tests.items():
        if str(count)+' passed' not in (ROOT/'reports'/(name+'-tests.log')).read_text():raise ValueError('Actual phase/source/checkpoint/resource/sharding tests required')
    cpu=ROOT/'reports/phase-joint-full-CPU-v2';proof=json.loads((cpu/'complete.json').read_text());control=json.loads((cpu/'plan.json').read_text())
    if not proof['exact_next_phase_weights_Adam_RNG_and_metrics'] or not proof['trained_native_text_image_word_cold_reload_exact']:
        raise ValueError('Actual full-size new phase learning/recovery/cold-load controls required')
    sources=json.loads((ROOT/'reports/expanded-joint-sources-v1/complete.json').read_text())
    prior=json.loads((PREVIOUS/'plan.json').read_text());files=dict(control['source_files_sha256'])
    additions=[Path(__file__),ROOT/'src/slm_lab/phase_joint_checkpoint_v1.py',ROOT/'src/slm_lab/native_code_shards_v1.py',ROOT/'src/slm_lab/packed_validation_v1.py',
        ROOT/'src/slm_lab/phase_joint_update_v1.py',ROOT/'src/slm_lab/expanded_joint_sources_v1.py',cpu/'complete.json',
        ROOT/'reports/expanded-joint-sources-v1/complete.json']
    for name in tests:additions.extend([ROOT/'reports'/(name+'-tests.log'),ROOT/'tests'/('test_'+name.replace('-','_')+'.py')])
    files.update(sources['source_files_sha256'])
    for p in additions:files[str(p)]=file_hash(p)
    for p in prior['pulse_sources'].values():files[p]=file_hash(p)
    me=owner.resources.identity(os.getpid());guard=FileBindings(files);deadline=time.monotonic()+24*3600
    while not previous_ready(prior):
        guard.check()
        if time.monotonic()>deadline:raise ValueError('Waiting lease expired without preempting the original job')
        atomic_json(QUEUE,dict(status='waiting for exact current execution trial and native assessments to finish',owner=me,predecessor=prior['owner'],
            planned_steps=sources['planned_optimizer_updates'],planned_targets=sources['complete_causal_response_EOS_targets'],source_files_sha256=files,production_updates=0,goal_achieved=False));time.sleep(5)
    base=json.loads((ROOT/'reports/protected-joint-v1/plan.json').read_text())
    from slm_lab.expanded_joint_sources_v1 import ExpandedJointSources
    from slm_lab.model import ModelConfig
    import torch
    torch.set_num_threads(1);initial=torch.load(base['initial_dense_settings']['initializer'],map_location='cpu',weights_only=True,mmap=True)
    data=ExpandedJointSources(ROOT,ModelConfig(**initial['config']));del initial
    packed='packed:original_packed_code';post=[n for n in data.phases()['instruction_and_vision'] if data.kinds[n]=='text']
    def longest(names,targets=False):
        name=max(names,key=lambda n:max(data.targets[n] if targets else data.lengths[n]))
        return name,sorted(range(len(data.lengths[name])),key=lambda p:((data.targets if targets else data.lengths)[name][p],p),reverse=True)[:16]
    photo='images:additional_photo_captions';word='words'
    capacity=[[packed,list(range(16))],[packed,list(range(16,32))],longest(post),longest(post,True),
        [photo,sorted(range(len(data.lengths[photo])),key=lambda p:(data.lengths[photo][p],p),reverse=True)[:32]],
        [word,sorted(range(len(data.lengths[word])),key=lambda p:(data.lengths[word][p],p),reverse=True)[:16]]]
    files[str(PREVIOUS/'plan.json')]=file_hash(PREVIOUS/'plan.json');files[str(PREVIOUS/'restored.json')]=file_hash(PREVIOUS/'restored.json')
    rates=dict(foundation=dict(language=1e-5,vision=1e-4,word=3e-4),instruction_and_vision=dict(language=1e-5,vision=1e-4,word=3e-4))
    options=dict(control['training_options'],phase_learning_rates=rates,learning_rates=rates['foundation'])
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);owner.idle_original()
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(used)!=4 or any(int(x)>128 for x in used):raise ValueError('Every original L40 must be idle before ownership')
        required=RESERVE+4*MAX_STATE+(sources['planned_optimizer_updates']//2048+6)*MAX_MODEL
        if shutil.disk_usage(ROOT).free<required:raise ValueError('All planned complete optimizer/native snapshots and atomic reserve must fit')
        guard=FileBindings(files);guard.check(full=True);REPORTS.mkdir();OUTPUT.mkdir()
        plan=dict(initial_dense_settings=base['initial_dense_settings'],actual_counts=base['actual_counts'],training_options=options,
            complete_source_budget=sources['complete_source_budget'],planned_steps=sources['planned_optimizer_updates'],phase_bounds=sources['phase_bounds'],
            global_family_batch_widths=sources['global_family_widths'],seed=958117,capacity_global_batches=capacity,
            capacity_phase_bounds=dict(foundation=[0,2],instruction_and_vision=[2,6]),pulse_sources=prior['pulse_sources'],
            tokenizer_sha256=base['tokenizer_sha256'],owner=me,waiter=owner.WAITER,source_files_sha256=files,expires_at_unix=time.time()+48*3600,
            all_original_sources_once=False,all_selected_family_records_once=True,declared_original_exercise_repetitions=4,
            cross_family_semantic_independence_not_established=True,no_rounding_repeats=True,raw_pretraining_has_no_reference=True,
            later_reference_is_own_completed_foundation=True,code_benchmarks_not_a_mid_pretraining_stop_rule=True,
            genuine_human_preference_training_and_final_independent_benchmarks_still_required=True,automatic_retry=False,candidate_promoted=False,goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[]);driver=str(Path(__file__).resolve())
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned complete phase experiment interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        watcher=subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,stdout=(REPORTS/'watchdog.log').open('x'),
            stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        def run(mode):
            child=owner.launch(plan,['-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',driver,'--distributed',mode],mode+'-launcher','0,1,2,3')
            owner.wait_children(plan,[child])
        try:
            owner.idle_original();os.kill(owner.WAITER['pid'],signal.SIGSTOP)
            deadline=time.monotonic()+5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in ('T','t'):
                if time.monotonic()>deadline:raise ValueError('Exact original idle waiter hold failed')
                time.sleep(.05)
            run('capacity');proofs=[json.loads((REPORTS/f'capacity-{rank}.json').read_text()) for rank in range(4)]
            for rank,p in enumerate(proofs):
                if (p['physical_GPU']!=rank or p['actual_counts']!=plan['actual_counts'] or p['actual_Adam_states']!=445
                        or not p['exact_raw_and_later_next_updates_weights_Adam_all_worker_RNG'] or not p['actual_positive_own_foundation_reference_KL']):
                    raise ValueError('Every actual new full-profile four-L40 phase/reference admission must pass')
            atomic_json(REPORTS/'capacity.json',dict(passed=True,proofs=proofs,production_updates=0,goal_achieved=False));run('train')
            for arm in ['initial','trained']:
                for suite in ['code','broad']:
                    children=[owner.launch(plan,['-u',driver,'--assess',arm,'--suite',suite,'--gpu',str(gpu)],'assess-'+arm+'-'+suite+'-'+str(gpu),str(gpu)) for gpu in range(4)]
                    owner.wait_children(plan,children)
                from slm_lab.native_code_shards_v1 import reconcile
                for frame in ['completion','instruction']:reconcile([REPORTS/arm/f'code-{frame}-shard-{rank}.json' for rank in range(4)],REPORTS/arm/f'code-{frame}.json')
            guard.check(full=True);atomic_json(REPORTS/'complete.json',dict(full_selected_expanded_family_source_epoch_complete=True,
                native_development_assessments_complete=True,independent_state_quality_readback_and_full_final_benchmarks_pending=True,
                genuine_human_preference_policy_training_still_required=True,candidate_promoted=False,goal_achieved=False))
        except BaseException as error:atomic_json(REPORTS/'failure.json',dict(error=repr(error),automatic_retry=False,goal_achieved=False));raise
        finally:owner.restore_owned(plan,'Complete phase experiment ended; original work restored');watcher.wait(timeout=20)


if __name__=='__main__':
    configure();p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watchdog',action='store_true');p.add_argument('--distributed',choices=['capacity','train']);p.add_argument('--assess',choices=['initial','trained']);p.add_argument('--suite',choices=['code','broad']);p.add_argument('--gpu',type=int,choices=range(4));args=p.parse_args()
    if args.watchdog:owner.watchdog()
    elif args.distributed:distributed(args.distributed)
    elif args.assess:assess(args.assess,args.gpu,args.suite)
    else:parent()
