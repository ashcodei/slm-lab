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
from slm_lab.diverse_images import atomic_json
import own_training_gpu_lease_v2 as owner

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/public-pretraining-v1'
OUTPUT=ROOT/'checkpoints/public-pretraining-v1'
QUEUE=ROOT/'reports/public-pretraining-v1-queue.json'
PREVIOUS=ROOT/'reports/code-first-continuation-v3'
PREPARATION=ROOT/'reports/public-pretraining-sources-CPU-v4'
HANDOFF=ROOT/'reports/public-pretraining-lineage-handoff-v1/complete.json'
RAW_ADMISSION=ROOT/'data/public-billion-causal-windows-v1/admission.json'
FUNCTION_ADMISSION=ROOT/'data/direct-algorithm-function-contracts-admitted-v1/admission.json'
MAX_STATE=14*1024**3
MAX_MODEL=5*1024**3
RESERVE=26*1024**3


def configure():
    owner.REPORTS=REPORTS;owner.resources.REPORTS=REPORTS


def dependencies_ready(dependencies):
    from slm_lab.public_pretraining_run_contract_v1 import stage_ready
    for key,folder,gpu in [('source',PREPARATION,False),('code_first',PREVIOUS,True)]:
        if not stage_ready(folder,dependencies[key],owner.resources.same,gpu_stage=gpu):return False
    prepared=json.loads((PREPARATION/'complete.json').read_text())
    trained=json.loads((PREVIOUS/'training-complete.json').read_text())
    prior=json.loads((PREVIOUS/'plan.json').read_text());done=json.loads((PREVIOUS/'complete.json').read_text())
    if (prepared.get('complete') is not True or prepared.get('all_new_admitted_raw_train_windows_once') is not True
            or prepared.get('all826independently_admitted_source_train_functions_once') is not True
            or prepared.get('all_previous_selected_source_presentations_exactly_preserved') is not True
            or trained.get('actual_updates')!=prior['planned_steps']
            or done.get('original_local_code_and_broad_assessments_complete') is not True
            or done.get('all58488new_code_and_all_admitted_execution_examples_once') is not True
            or done.get('all_admitted_public_state_examples_once') is not True):
        raise ValueError('Every complete original code epoch, native controls and full admitted public source must finish')
    if not HANDOFF.exists():return False
    handoff=json.loads(HANDOFF.read_text())
    if (handoff.get('complete') is not True or handoff.get('old_unstarted_controllers_all_terminal') is not True
            or handoff.get('new_public_pretraining_owner')!=dependencies['self']
            or handoff.get('original_code_first_owner')!=dependencies['code_first']
            or handoff.get('replacement_full_pipeline_tested_and_queued') is not True):
        raise ValueError('Exact tested complete replacement lineage and preserved original code owner required')
    for birth in handoff['retired_unstarted_controllers']:
        if owner.resources.same(birth):raise ValueError('No retired original controller may still own the old lineage')
    FileBindings(handoff['source_files_sha256']).check()
    return True


def distributed(mode):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1,2,3' or int(os.environ['WORLD_SIZE'])!=4:
        raise ValueError('All four owned original L40s required')
    rank=int(os.environ['RANK']);plan=owner.registered(mode+'-rank-'+str(rank));digest=file_hash(REPORTS/'plan.json')
    from concurrent.futures import ThreadPoolExecutor
    import random
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.codenet_checkpoint_v1 import save_collective,capture_rng,atomic_checkpoint
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
        guard=FileBindings(plan['source_files_sha256']);model,_=fresh(plan,device)
        data=PublicPretrainingData(ROOT,model.config);schedule=schedule_for(data,plan,mode)
        options=plan['training_options'];optimizer=make_optimizer(model,options,device);records=[]
        groups=parameter_groups(model);selected={id(p) for rows in groups.values() for _,p in rows}
        fixed=tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in selected)
        random.seed(plan['seed']+rank);np.random.seed(plan['seed']+rank);torch.manual_seed(plan['seed']+rank)
        amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16)
        scratch=Path(plan['volatile_capacity_directory'])
        def step(i,prepared=None):
            owner.owned(plan)
            row=update(model,optimizer,data,schedule,i,options,device=device,rank=rank,world=4,amp=amp,prepared=prepared)
            records.append(row);return row
        def save_state(path,completed):
            reserve=0 if path.is_relative_to(scratch) else RESERVE
            save_collective(path,lambda rngs:make_state(model,optimizer,plan,digest,completed,records,rngs,mode,
                schedule=schedule,data=data),device,4,rank,MAX_STATE,reserve)
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
            atomic_json(REPORTS/f'capacity-cold-{rank}.json',dict(physical_GPU=rank,every_exported_value_and_all_three_native_outputs_exact=True,
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
            atomic_json(REPORTS/f'capacity-{rank}.json',dict(rank=rank,physical_GPU=rank,actual_counts=model.counts(),actual_Adam_states=len(optimizer.state),
                logical_updates=9,physical_optimizer_calls=11,actual_records=records,all_nine_declared_actual_capacity_batches=True,
                actual_next_source_updates_weights_Adam_all_worker_RNG_exact=True,
                invalid_source_and_other_worker_CUDA_RNG_rejected_before_mutation=True,
                every_exported_value_and4096text448photo_real_word_outputs_exact=True,
                training_only_reference_values=0,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),seconds=time.monotonic()-started,production_updates=0,goal_achieved=False))
        else:
            native('initial',0)
            def validate(label):
                result=packed_loss(model,data.retention.replay.packed,device=device,rank=rank,world=4,amp=amp)
                if rank==0:atomic_json(REPORTS/('held-packed-'+label+'.json'),result)
            validate('initial');journal=(REPORTS/'training-metrics.jsonl').open('x') if rank==0 else None
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
                    if (i+1)%512==0 or i+1==len(schedule):save_state(OUTPUT/'resume.pt',i+1)
                    if (i+1)%2048==0 or i+1==len(schedule):validate(str(i+1))
            if journal:journal.close()
            native('trained',len(schedule))
            if rank==0:atomic_json(REPORTS/'training-complete.json',dict(actual_updates=len(records),
                all_new_admitted_raw_train_windows_once=True,all826verified_train_functions_once=True,
                every_previous_selected_code_state_and12family_retention_presentation_preserved=True,
                actual_selected_source_budget=schedule.selected_budget(data),actual_counts=model.counts(),
                original_native_assessments_and_independent_state_readback_pending=True,goal_achieved=False))
        if fixed!=tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in selected):raise ValueError('Unused critic/auxiliary heads must remain exact')
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
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(p.exists() for p in [REPORTS,OUTPUT,QUEUE]):
        raise ValueError('Fresh GPU-hidden queue required; preserve every existing attempt')
    from slm_lab.local_source_closure_v2 import python_sources
    tests={'public-pretraining-data-update-checkpoint-v1-tests':7,'public-pretraining-data-v2-tests':2,
        'public-pretraining-run-contract-v1-final-tests':3,'public-pretraining-native-v1-final-tests':2,
        'public-pretraining-driver-v1-tests':3}
    paths=[Path(__file__),PREPARATION/'plan.json',PREVIOUS/'plan.json']
    for name,count in tests.items():
        p=ROOT/'reports'/(name+'.log')
        if str(count)+' passed' not in p.read_text():raise ValueError('Actual full-source recovery, factory, schedule and driver controls must pass')
        paths.append(p)
    proof=ROOT/'reports/public-pretraining-four-CPU-v1/complete.json'
    if not json.loads(proof.read_text())['all_four_actual_CPU_workers_passed']:raise ValueError('Actual four-replica empty-tail and recovery proof required')
    paths.extend([proof,*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_public_pretraining_driver_v1.py'])])
    files={str(p):file_hash(p) for p in paths};guard=FileBindings(files);me=owner.resources.identity(os.getpid())
    dependencies=dict(self=me,source=json.loads((PREPARATION/'plan.json').read_text())['owner'],
        code_first=json.loads((PREVIOUS/'plan.json').read_text())['owner'])
    atomic_json(QUEUE,dict(at=datetime.now(timezone.utc).isoformat(),owner=me,exact_dependencies=dependencies,source_files_sha256=files,
        status='waiting for complete original code epoch, all admitted public raw/function sources and explicit tested lineage handoff',
        no_original_jobs_stopped_or_restarted_by_this_queue=True,production_updates=0,goal_achieved=False))
    while not dependencies_ready(dependencies):guard.check();time.sleep(5)
    import torch
    from slm_lab.model import ModelConfig
    from slm_lab.public_pretraining_data_v2 import PublicPretrainingData
    from slm_lab.public_pretraining_run_contract_v1 import CAPACITY_FAMILIES,schedule_for,scratch_memory_safe
    from slm_lab.public_pretraining_native_v1 import validate_start
    torch.set_num_threads(1)
    preparation=json.loads((PREPARATION/'complete.json').read_text());source_plan=PREVIOUS/'plan.json'
    checkpoint=ROOT/'checkpoints/code-first-continuation-v3/standalone/model.pt'
    inputs=dict(checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),plan=str(source_plan),plan_sha256=file_hash(source_plan))
    raw=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True);prior=json.loads(source_plan.read_text());validate_start(prior,raw)
    config=ModelConfig(**raw['complete_dense']['config']);counts=raw['actual_counts'];del raw
    data=PublicPretrainingData(ROOT,config)
    stat=dict((k,int(v)) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines()))
    memory=scratch_memory_safe(stat,int(Path('/sys/fs/cgroup/memory.max').read_text()))
    scratch=Path('/dev/shm')/('slm-public-pretraining-v1-'+str(me['pid'])+'-'+me['ticks'])
    if scratch.exists():raise ValueError('A fresh owned volatile capacity directory is required')
    if shutil.disk_usage(ROOT).free<RESERVE+2*MAX_STATE+2*MAX_MODEL:
        raise ValueError('Complete current+atomic resume, initial/final natives and persistent reserve must fit')
    capacity=[[name,preparation['actual_capacity_global_batches'][name]] for name in CAPACITY_FAMILIES]
    options=dict(learning_rates=dict(language=1e-4,vision=1e-4,word=3e-4),betas=[.9,.95],weight_decay=.01,
        gradient_clip=1.,warmup_fraction=.02,cosine_floor=.1,maximum_microbatch=4,maximum_microbatch_tokens=4096,
        ctc_weight=1.,word_native_CE_weight=1.,gradient_bucket_values=16*1024**2,image_size=448)
    files.update(preparation['source_files_sha256']);files.update(data.source_files_sha256);files.update(owner.lease_sources())
    for p in [PREPARATION/'complete.json',PREVIOUS/'complete.json',PREVIOUS/'training-complete.json',PREVIOUS/'restored.json',
            checkpoint,source_plan,HANDOFF,RAW_ADMISSION,FUNCTION_ADMISSION]:files[str(p)]=file_hash(p)
    plan=dict(owner=me,waiter=owner.WAITER,terminal_waiter_receipt_sha256=owner.terminal_receipt_sha256(),source_files_sha256=files,
        expires_at_unix=time.time()+14*86400,complete_code_first_inputs=inputs,actual_counts=counts,tokenizer_sha256=data.code.provenance['tokenizer_sha256'],
        initial_dense_settings=dict(initializer_sha256=inputs['checkpoint_sha256']),seed=976319,planned_steps=preparation['planned_steps'],
        source_schedule_sha256=preparation['source_schedule_sha256'],actual_schedule_settings=preparation['actual_schedule_settings'],
        complete_source_budget=preparation['complete_source_budget'],selected_source_budget=preparation['selected_source_budget'],
        capacity_global_batches=capacity,training_options=options,public_raw_admission_sha256=file_hash(RAW_ADMISSION),
        public_function_admission_sha256=file_hash(FUNCTION_ADMISSION),volatile_capacity_directory=str(scratch),volatile_capacity_memory_check=memory,
        pulse_sources=prior['pulse_sources'],all_original_raw_train_windows_once=True,all826verified_original_train_functions_once=True,
        all_previous_selected_source_presentations_preserved=True,whole_function_auxiliary_EOS_weight=0.,training_only_reference_parameters=0,
        no_external_pretrained_model_or_live_teacher=True,automatic_retry=False,candidate_promoted=False,goal_achieved=False)
    schedule_for(data,plan,'train');schedule_for(data,plan,'capacity');del data
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);owner.idle_original()
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(used)!=4 or any(int(x)>128 for x in used):raise ValueError('All four original L40s must be idle before ownership')
        FileBindings(files).check(full=True);REPORTS.mkdir();OUTPUT.mkdir();scratch.mkdir(mode=0o700)
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[]);driver=str(Path(__file__).resolve())
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned public pretraining interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        watcher=None
        try:
            owner.hold_original(plan)
            watcher=subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,stdout=(REPORTS/'watchdog.log').open('x'),
                stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            def run(mode):
                child=owner.launch(plan,['-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',driver,'--distributed',mode],mode+'-launcher','0,1,2,3')
                owner.wait_children(plan,[child])
            run('capacity');proofs=[json.loads((REPORTS/f'capacity-{r}.json').read_text()) for r in range(4)]
            for rank,p in enumerate(proofs):
                if (p['physical_GPU']!=rank or p['actual_counts']!=counts or p['actual_Adam_states']!=445
                        or p['logical_updates']!=9 or p['physical_optimizer_calls']!=11
                        or not p['actual_next_source_updates_weights_Adam_all_worker_RNG_exact']
                        or not p['invalid_source_and_other_worker_CUDA_RNG_rejected_before_mutation']
                        or not p['every_exported_value_and4096text448photo_real_word_outputs_exact']):
                    raise ValueError('Every actual full-profile four-L40 capacity/recovery/native proof required')
            capacity_files={str(p):file_hash(p) for p in scratch.rglob('*') if p.is_file()}
            atomic_json(REPORTS/'capacity.json',dict(passed=True,proofs=proofs,volatile_capacity_file_sha256=capacity_files,
                capacity_files_disposable_not_durable_resume=True,production_updates=0,goal_achieved=False))
            run('train');atomic_json(REPORTS/'native-assessment-inputs.json',{arm:dict(checkpoint_sha256=file_hash(OUTPUT/arm/'model.pt')) for arm in ['initial','trained']})
            for arm in ['initial','trained']:
                for suite in ['code','broad']:
                    children=[owner.launch(plan,['-u',driver,'--assess',arm,'--suite',suite,'--gpu',str(gpu)],'assess-'+arm+'-'+suite+'-'+str(gpu),str(gpu)) for gpu in range(4)]
                    owner.wait_children(plan,children)
                from slm_lab.native_code_shards_v1 import reconcile
                for frame in ['completion','instruction']:reconcile([REPORTS/arm/f'code-{frame}-shard-{r}.json' for r in range(4)],REPORTS/arm/f'code-{frame}.json')
            FileBindings(files).check(full=True)
            atomic_json(REPORTS/'complete.json',dict(complete=True,full_public_pretraining_source_epoch_complete=True,
                all_original_raw_train_windows_and826functions_once=True,every_previous_selected_source_exposure_retained=True,
                all_four_actual_L40_capacity_recovery_and_cold_exports_passed=True,original_local_code_and_broad_assessments_complete=True,
                self_contained_own_student_and_tokenizer=True,full_human_alignment_and_execution_RL_and_independent_assessments_still_required=True,
                candidate_promoted=False,goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False));raise
        finally:
            owner.restore_owned(plan,'Owned public pretraining ended; exact registered workers terminal')
            if watcher is not None:watcher.wait(timeout=20)


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
            atomic_json(ROOT/'reports/public-pretraining-v1-parent-failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False));raise
