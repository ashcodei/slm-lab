"""Own full public raw/function pretraining with exact multimodal retention.

The GPU lease additionally requires an explicit completed lineage handoff. A
waiting queue alone does not stop another job or admit incomplete public data.
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

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.pretraining_operational_recovery_v1 import math_for_runtime,validate_interruption
from slm_lab.diverse_images import atomic_json
import own_training_gpu_lease_v2 as owner

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/public-pretraining-v1'
OUTPUT=ROOT/'checkpoints/public-pretraining-v1'
QUEUE=ROOT/'reports/public-pretraining-recovery-v1-queue.json'
RUNTIME=ROOT/'reports/public-pretraining-recovery-v1'
MATH_PLAN=REPORTS/'plan.json'
INTERRUPTION=ROOT/'reports/instance-interruption-20261018-v1/complete.json'
PREVIOUS=ROOT/'reports/code-first-continuation-v3'
PREPARATION=ROOT/'reports/public-pretraining-sources-CPU-v4'
HANDOFF=ROOT/'reports/public-pretraining-lineage-handoff-v1/complete.json'
RAW_ADMISSION=ROOT/'data/public-billion-causal-windows-v1/admission.json'
FUNCTION_ADMISSION=ROOT/'data/direct-algorithm-function-contracts-admitted-v1/admission.json'
MAX_STATE=14*1024**3
MAX_MODEL=5*1024**3
RESERVE=26*1024**3


def configure():
    owner.REPORTS=RUNTIME;owner.resources.REPORTS=RUNTIME


def distributed(mode):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1,2,3' or int(os.environ['WORLD_SIZE'])!=4:
        raise ValueError('All four owned original L40s required')
    rank=int(os.environ['RANK']);runtime=owner.registered(mode+'-rank-'+str(rank))
    plan=math_for_runtime(runtime);digest=file_hash(MATH_PLAN)
    from concurrent.futures import ThreadPoolExecutor
    import random
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.codenet_checkpoint_v1 import collect_worker_rng,capture_rng,atomic_checkpoint
    from slm_lab.async_durable_checkpoint_v1 import AsyncCheckpoint
    from slm_lab.public_pretraining_native_v1 import fresh,export,load
    from slm_lab.public_pretraining_data_v2 import PublicPretrainingData
    from slm_lab.public_pretraining_run_contract_v1 import schedule_for
    from slm_lab.public_pretraining_checkpoint_v1 import make_state,restore_state
    from slm_lab.public_pretraining_update_v1 import update
    from slm_lab.fast_joint_stage_v1 import make_optimizer,parameter_groups
    from slm_lab.packed_validation_v1 import evaluate as packed_loss
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1);device=torch.device('cuda',rank);torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.85,device);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name(device)!='NVIDIA L40':raise ValueError('Only the original local L40s are eligible')
    dist.init_process_group('nccl',timeout=timedelta(minutes=40));control=dist.new_group(backend='gloo',timeout=timedelta(minutes=40))
    try:
        guard=FileBindings(runtime['source_files_sha256']);model,_=fresh(plan,device)
        data=PublicPretrainingData(ROOT,model.config);schedule=schedule_for(data,plan,mode)
        options=plan['training_options'];optimizer=make_optimizer(model,options,device);records=[]
        groups=parameter_groups(model);selected={id(p) for rows in groups.values() for _,p in rows}
        fixed=tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in selected)
        random.seed(plan['seed']+rank);np.random.seed(plan['seed']+rank);torch.manual_seed(plan['seed']+rank)
        amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16)
        scratch=Path(runtime['volatile_capacity_directory'])
        writer=AsyncCheckpoint() if rank==0 else None
        def writer_status(wait=False):
            outcome=[None]
            if rank==0:
                try:
                    if wait:writer.flush()
                    else:writer.check()
                except BaseException as error:outcome[0]=repr(error)
            dist.broadcast_object_list(outcome,src=0,group=control)
            if outcome[0] is not None:raise ValueError('Durable background checkpoint failed: '+outcome[0])
        def step(i,prepared=None):
            owner.owned(runtime);writer_status()
            row=update(model,optimizer,data,schedule,i,options,device=device,rank=rank,world=4,amp=amp,prepared=prepared)
            records.append(row);return row
        def save_state(path,completed):
            writer_status(wait=True)
            reserve=0 if path.is_relative_to(scratch) else RESERVE
            rngs=collect_worker_rng(device,4);outcome=[None]
            if rank==0:
                try:
                    writer.submit(path,make_state(model,optimizer,plan,digest,completed,records,rngs,mode,
                        schedule=schedule,data=data),MAX_STATE,reserve)
                except BaseException as error:outcome[0]=repr(error)
            dist.broadcast_object_list(outcome,src=0,group=control)
            if outcome[0] is not None:raise ValueError('Immutable CPU checkpoint snapshot failed: '+outcome[0])
            # All GPU reads/copies have completed; only private CPU bytes remain.
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
            writer_status(wait=True)
            saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            for kind in ['source','CUDA_RNG']:
                core=dict(saved['core']);bad=dict(saved,core=core)
                if kind=='source':
                    core['metrics']=[dict(r) for r in core['metrics']];core['metrics'][0]['global_original_positions']=[-1]
                else:
                    core['rngs']=[dict(r) for r in core['rngs']];core['rngs'][(rank+1)%4]['cuda']=torch.zeros(4,dtype=torch.uint8)
                rng=capture_rng(device);signature=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in model.parameters()]
                try:restore_state(model,optimizer,bad,plan,digest,device,rank,4,mode,schedule=schedule,data=data)
                except ValueError:pass
                else:raise ValueError('Corrupt actual source cursor or another private CUDA RNG must reject')
                if (signature!=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in model.parameters()] or moments!=adam()
                        or not equal(rng,capture_rng(device))):raise ValueError('Rejected recovery mutated complete model, Adam or RNG')
            count,records=restore_state(model,optimizer,saved,plan,digest,device,rank,4,mode,schedule=schedule,data=data)
            if (count!=completed or draws()!=expected_draws or step(next_step)!=expected
                    or weights!=tensor_digest(model.named_parameters()) or moments!=adam() or not equal(expected_rng,capture_rng(device))):
                raise ValueError('Actual next source update, every weight/moment and all worker RNG must recover exactly')
        def native(arm,completed,capacity=False):
            folder=(scratch/'native') if capacity else OUTPUT/arm
            path=folder/'model.pt'
            if rank==0:
                folder.mkdir();shutil.copyfile(data.code.tokenizer_path,folder/'tokenizer.json')
                atomic_checkpoint(path,export(model,plan,digest,arm,completed),MAX_MODEL,0 if capacity else RESERVE)
            dist.barrier(group=control)
            if not capacity:return
            model.eval()
            raw=data.batch(plan['capacity_global_batches'][0][0],[plan['capacity_global_batches'][0][1][0]],device)
            photo_name,photo_positions=plan['capacity_global_batches'][-2]
            photo=data.batch(photo_name,[photo_positions[0]],device)
            word_position=next(i for i,r in enumerate(data.retention.replay.words.rows) if r['byte_family']=='real_word')
            word=data.batch('words',[word_position],device)
            def logits(batch):return model(**{k:batch[k] for k in ('input_ids','attention_mask','pixel_values') if k in batch})['logits'].cpu()
            with torch.no_grad(),amp():before=[logits(raw),logits(photo),model.word_logits(word['word_pixels']).cpu()]
            cold,_=load(path,REPORTS/'plan.json','capacity',expected_checkpoint_sha256=file_hash(path),expected_plan_sha256=digest,device=device)
            with torch.no_grad(),amp():
                after=[cold(**{k:b[k] for k in ('input_ids','attention_mask','pixel_values') if k in b})['logits'].cpu() for b in [raw,photo]]
                after.append(cold.word_logits(word['word_pixels']).cpu())
            if (any(not torch.equal(a,b) for a,b in zip(before,after))
                    or tensor_digest(cold.named_parameters())!=tensor_digest(model.named_parameters())
                    or raw['input_ids'].shape[1]!=4096 or photo['pixel_values'].shape[-2:]!=(448,448)):
                raise ValueError('Every exported value and actual4096text/448photo/real-word output must cold-load bit-exactly')
            del cold,raw,photo,word,before,after;torch.cuda.empty_cache()
            atomic_json(RUNTIME/f'capacity-cold-{rank}.json',dict(physical_GPU=rank,every_exported_value_and_all_three_native_outputs_exact=True,
                checkpoint_sha256=file_hash(path),actual_completed_updates=completed,capacity_file_volatile=True,production_updates=0,goal_achieved=False))
            dist.barrier(group=control)
        started=time.monotonic()
        if mode=='capacity':
            for i in range(7):step(i)
            language=tensor_digest(groups['language']);recover(scratch/'after7.pt',7,7)
            if language!=tensor_digest(groups['language']):raise ValueError('Image update must keep the language decoder exact')
            recover(scratch/'after8.pt',8,8)
            if language!=tensor_digest(groups['language']) or len(optimizer.state)!=445:
                raise ValueError('All445actual Adam leaves and frozen language during image/word steps required')
            native('capacity',9,capacity=True)
            atomic_json(RUNTIME/f'capacity-{rank}.json',dict(rank=rank,physical_GPU=rank,actual_counts=model.counts(),actual_Adam_states=len(optimizer.state),
                logical_updates=9,physical_optimizer_calls=11,actual_records=records,all_nine_declared_actual_capacity_batches=True,
                actual_next_source_updates_weights_Adam_all_worker_RNG_exact=True,
                invalid_source_and_other_worker_CUDA_RNG_rejected_before_mutation=True,
                every_exported_value_and4096text448photo_real_word_outputs_exact=True,
                training_only_reference_values=0,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),seconds=time.monotonic()-started,production_updates=0,goal_achieved=False))
        else:
            # The original initial export is durable and immutable, not recreated.
            initial,_=load(OUTPUT/'initial/model.pt',MATH_PLAN,'initial',
                expected_checkpoint_sha256=runtime['initial_checkpoint_sha256'],expected_plan_sha256=digest,device=device)
            if tensor_digest(initial.named_parameters())!=tensor_digest(model.named_parameters()):
                raise ValueError('Durable initial export must match every original source value')
            del initial;torch.cuda.empty_cache()
            start=0
            if (OUTPUT/'resume.pt').exists():
                saved=torch.load(OUTPUT/'resume.pt',map_location='cpu',weights_only=True,mmap=True)
                start,records=restore_state(model,optimizer,saved,plan,digest,device,rank,4,'train',schedule=schedule,data=data)
                del saved
            if rank==0:atomic_json(RUNTIME/'recovered-cursor.json',dict(actual_durable_updates=start,
                interrupted_unrecoverable_updates=runtime['interrupted_observed_updates'],
                original_math_plan_sha256=digest,original_horizon_unchanged=True,goal_achieved=False))
            def validate(label):
                result=packed_loss(model,data.retention.replay.packed,device=device,rank=rank,world=4,amp=amp)
                if rank==0:atomic_json(RUNTIME/('held-packed-'+label+'.json'),result)
            validate('recovered-'+str(start));journal=(RUNTIME/'training-metrics.jsonl').open('x') if rank==0 else None
            def prepare(i):
                name,positions=schedule.at(i);local=positions[rank::4]
                if not local:return []
                prefix=model.config.image_token_count(448) if data.kinds[name] in ('image','word') else 0
                width=max(1,min(options['maximum_microbatch'],options['maximum_microbatch_tokens']//(max(data.lengths[name][p] for p in local)+prefix)))
                return [{k:(v.pin_memory() if isinstance(v,torch.Tensor) else v) for k,v in data.batch(name,local[a:a+width],'cpu').items()}
                    for a in range(0,len(local),width)]
            with ThreadPoolExecutor(max_workers=1) as reader:
                pending=reader.submit(prepare,start) if start<len(schedule) else None
                for i in range(start,len(schedule)):
                    if i%32==0:guard.check()
                    prepared=pending.result()
                    if i+1<len(schedule):pending=reader.submit(prepare,i+1)
                    tick=time.monotonic();row=step(i,prepared)
                    if journal:
                        journal.write(json.dumps(row)+'\n');journal.flush()
                        if i==0 or (i+1)%25==0:atomic_json(RUNTIME/'progress.json',dict(actual_updates=i+1,planned_updates=len(schedule),
                            latest=row,last_update_seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-started,goal_achieved=False))
                    if i==start or (i+1)%64==0 or i+1==len(schedule):save_state(OUTPUT/'resume.pt',i+1)
                    if (i+1)%2048==0 or i+1==len(schedule):validate(str(i+1))
            if journal:journal.close()
            writer_status(wait=True)
            native('trained',len(schedule))
            if rank==0:atomic_json(REPORTS/'training-complete.json',dict(actual_updates=len(records),
                all_new_admitted_raw_train_windows_once=True,all826verified_train_functions_once=True,
                every_previous_selected_code_state_and12family_retention_presentation_preserved=True,
                actual_selected_source_budget=schedule.selected_budget(data),actual_counts=model.counts(),
                original_native_assessments_and_independent_state_readback_pending=True,goal_achieved=False))
        if fixed!=tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in selected):raise ValueError('Unused critic/auxiliary heads must remain exact')
        writer_status(wait=True)
        if writer is not None:writer.close()
        guard.check(full=True);dist.barrier(group=control)
    finally:dist.destroy_process_group()


def assess(arm,gpu,suite):
    plan=owner.registered('assess-'+arm+'-'+suite+'-'+str(gpu))
    import torch
    from slm_lab.public_pretraining_native_v1 import load
    from slm_lab.dense_native_view_v2 import CompleteDenseNativeView
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    from slm_lab.native_greedy_observer_v1 import observe
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85,0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inputs=json.loads((REPORTS/'native-assessment-inputs.json').read_text())[arm];checkpoint=OUTPUT/arm/'model.pt'
    model,metadata=load(checkpoint,REPORTS/'plan.json',arm,expected_checkpoint_sha256=inputs['checkpoint_sha256'],
        expected_plan_sha256=file_hash(REPORTS/'plan.json'),device='cuda:0')
    view=CompleteDenseNativeView(model).eval();destination=REPORTS/arm;destination.mkdir(exist_ok=True)
    old_threads=torch.set_num_threads;torch.set_num_threads=lambda count:None
    def evaluate(module,name,data=None,**kwargs):
        original=module.load_checkpoint;module.load_checkpoint=lambda *a,**kw:(view,metadata)
        try:
            if data is None:module.evaluate(checkpoint,destination/name,device='cuda:0',**kwargs)
            else:module.evaluate(checkpoint,data,destination/name,device='cuda:0',**kwargs)
        finally:module.load_checkpoint=original
    try:
        with (destination/f'{suite}-actual-native-actions-{gpu}.jsonl').open('x') as actions:
            def record(event):actions.write(json.dumps(event)+'\n');actions.flush()
            with inference_cache(),observe(view,record):
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
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or RUNTIME.exists() or QUEUE.exists():
        raise ValueError('Fresh GPU-hidden operational recovery required')
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.public_pretraining_run_contract_v1 import stage_ready,scratch_memory_safe
    interruption=json.loads(INTERRUPTION.read_text());plan=json.loads(MATH_PLAN.read_text())
    validate_interruption(plan,json.loads((REPORTS/'children.json').read_text()),interruption,owner.resources.same)
    if any((REPORTS/name).exists() for name in ['failure.json','training-complete.json','complete.json','restored.json']):
        raise ValueError('Never overwrite a terminal scientific result')
    for actor in [plan['owner'],*json.loads((REPORTS/'children.json').read_text())]:
        if owner.resources.same(actor):raise ValueError('Old production workers must be absent before recovery')
    for folder,gpu in [(PREVIOUS,True),(PREPARATION,False)]:
        birth=json.loads((folder/'plan.json').read_text())['owner']
        if not stage_ready(folder,birth,owner.resources.same,gpu_stage=gpu):raise ValueError('Original complete source stage required')
    tests=ROOT/'reports/async-durable-pretraining-recovery-v1-final-tests.log'
    if '5 passed' not in tests.read_text():raise ValueError('Actual async isolation, atomic failure and recovery routing tests required')
    lineage_test=ROOT/'reports/instance-lineage-recovery-v1-tests.log'
    if '1 passed' not in lineage_test.read_text():raise ValueError('All ten unchanged scientific controllers must pass recovery routing checks')
    mapping=ROOT/'reports/instance-lineage-recovery-v1-file-map.json'
    routes=json.loads(mapping.read_text())
    files=dict(plan['source_files_sha256'])
    for path in [lineage_test,mapping,ROOT/'tests/test_instance_lineage_recovery_v1.py',*[ROOT/'scripts'/r['recovery_script'] for r in routes['stages']]]:
        files[str(path)]=file_hash(path)
    for path in [MATH_PLAN,INTERRUPTION,tests,*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_async_durable_checkpoint_v1.py',ROOT/'tests/test_pretraining_operational_recovery_v1.py'])]:
        files[str(path)]=file_hash(path)
    me=owner.resources.identity(os.getpid())
    scratch=Path('/dev/shm')/('slm-pretraining-recovery-'+str(me['pid'])+'-'+me['ticks'])
    stat=dict((k,int(v)) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines()))
    memory=scratch_memory_safe(stat,int(Path('/sys/fs/cgroup/memory.max').read_text()))
    runtime=dict(owner=me,waiter=owner.WAITER,terminal_waiter_receipt_sha256=owner.terminal_receipt_sha256(),
        source_files_sha256=files,expires_at_unix=time.time()+14*86400,volatile_capacity_directory=str(scratch),
        volatile_capacity_memory_check=memory,original_neural_plan=str(MATH_PLAN),original_neural_plan_sha256=file_hash(MATH_PLAN),
        initial_checkpoint_sha256=file_hash(OUTPUT/'initial/model.pt'),
        interrupted_observed_updates=interruption['observed_public_updates'],checkpoint_interval=64,checkpoint_after_first_update=True,
        immutable_CPU_snapshot_background_atomic_IO=True,neural_options_unchanged=True,pulse_sources=plan['pulse_sources'],
        no_original_scientific_files_overwritten=True,goal_achieved=False)
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);owner.idle_original()
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(used)!=4 or any(int(x)>128 for x in used):raise ValueError('All four original L40s must be idle')
        if shutil.disk_usage(ROOT).free<RESERVE+2*MAX_STATE+MAX_MODEL:
            raise ValueError('Both durable checkpoint generations, final model and reserve must fit')
        FileBindings(files).check(full=True);RUNTIME.mkdir();scratch.mkdir(mode=0o700)
        atomic_json(RUNTIME/'plan.json',runtime);atomic_json(RUNTIME/'children.json',[])
        atomic_json(QUEUE,dict(owner=me,source_files_sha256=files,original_neural_plan_sha256=file_hash(MATH_PLAN),
            status='recovering same full public pretraining after confirmed process interruption',planned_updates=plan['planned_steps'],
            original_source_schedule_sha256=plan['source_schedule_sha256'],goal_achieved=False))
        driver=str(Path(__file__).resolve());watcher=None;success=False
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned operational recovery interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        try:
            owner.hold_original(runtime)
            watcher=subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,stdout=(RUNTIME/'watchdog.log').open('x'),
                stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            def run(mode):
                child=owner.launch(runtime,['-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',driver,'--distributed',mode],mode+'-launcher','0,1,2,3')
                owner.wait_children(runtime,[child])
            run('capacity');proofs=[json.loads((RUNTIME/f'capacity-{r}.json').read_text()) for r in range(4)]
            for rank,p in enumerate(proofs):
                if (p['physical_GPU']!=rank or p['actual_counts']!=plan['actual_counts'] or p['actual_Adam_states']!=445
                        or p['logical_updates']!=9 or p['physical_optimizer_calls']!=11
                        or not p['actual_next_source_updates_weights_Adam_all_worker_RNG_exact']
                        or not p['invalid_source_and_other_worker_CUDA_RNG_rejected_before_mutation']
                        or not p['every_exported_value_and4096text448photo_real_word_outputs_exact']):
                    raise ValueError('Fresh actual all-worker async recovery and native proof required')
            atomic_json(RUNTIME/'capacity.json',dict(passed=True,proofs=proofs,
                snapshots_volatile_not_production_resume=True,background_snapshot_actual_next_updates_exact=True,goal_achieved=False))
            run('train')
            atomic_json(REPORTS/'native-assessment-inputs.json',{arm:dict(checkpoint_sha256=file_hash(OUTPUT/arm/'model.pt')) for arm in ['initial','trained']})
            for arm in ['initial','trained']:
                for suite in ['code','broad']:
                    children=[owner.launch(runtime,['-u',driver,'--assess',arm,'--suite',suite,'--gpu',str(gpu)],'assess-'+arm+'-'+suite+'-'+str(gpu),str(gpu)) for gpu in range(4)]
                    owner.wait_children(runtime,children)
                from slm_lab.native_code_shards_v1 import reconcile
                for frame in ['completion','instruction']:reconcile([REPORTS/arm/f'code-{frame}-shard-{r}.json' for r in range(4)],REPORTS/arm/f'code-{frame}.json')
            FileBindings(files).check(full=True)
            atomic_json(REPORTS/'complete.json',dict(complete=True,full_public_pretraining_source_epoch_complete=True,
                all_original_raw_train_windows_and826functions_once=True,every_previous_selected_source_exposure_retained=True,
                all_four_actual_L40_capacity_recovery_and_cold_exports_passed=True,original_local_code_and_broad_assessments_complete=True,
                self_contained_own_student_and_tokenizer=True,full_human_alignment_and_execution_RL_and_independent_assessments_still_required=True,
                operational_runtime=str(RUNTIME),operational_runtime_plan_sha256=file_hash(RUNTIME/'plan.json'),
                original_interrupted_run_receipt=str(INTERRUPTION),candidate_promoted=False,goal_achieved=False))
            success=True
        except BaseException as error:
            atomic_json(RUNTIME/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False));raise
        finally:
            owner.restore_owned(runtime,'Operational recovery ended; every exact registered worker terminal')
            if watcher is not None:watcher.wait(timeout=20)
        if success:
            atomic_json(REPORTS/'restored.json',dict(at=datetime.now(timezone.utc).isoformat(),
                reason='Original mathematical job completed under explicitly separate recovered runtime',
                original_owner=plan['owner'],actual_recovery_owner=me,actual_recovery_children=json.loads((RUNTIME/'children.json').read_text()),
                all_registered_point_actors_terminal=True,operational_restored_sha256=file_hash(RUNTIME/'restored.json'),goal_achieved=False))
            atomic_json(RUNTIME/'complete.json',dict(complete=True,original_scientific_job_complete=True,
                original_neural_plan_sha256=file_hash(MATH_PLAN),goal_achieved=False))


if __name__=='__main__':
    configure();parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watchdog',action='store_true')
    parser.add_argument('--distributed',choices=['capacity','train']);parser.add_argument('--assess',choices=['initial','trained'])
    parser.add_argument('--suite',choices=['code','broad']);parser.add_argument('--gpu',type=int,choices=range(4));args=parser.parse_args()
    if args.watchdog:owner.watchdog()
    elif args.distributed:distributed(args.distributed)
    elif args.assess:assess(args.assess,args.gpu,args.suite)
    else:
        try:parent()
        except BaseException as error:
            atomic_json(ROOT/'reports/public-pretraining-recovery-v1-parent-failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False));raise
