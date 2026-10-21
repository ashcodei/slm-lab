"""Four complete dense-code learning-rate comparisons with persistent resume.

All four L40s train one common policy at a time. Only one complete Adam/model
checkpoint slot is needed; atomic replacement retains its previous state.
The central1e-5 arm is last so its final Adam/RNG survives the whole study.
"""
import argparse
from datetime import datetime, timezone, timedelta
import fcntl
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import train_recovered_dense_code_GPU_v1 as resource

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/durable-distributed-code-v1'
MODELS = ROOT/'checkpoints/durable-distributed-code-v1'
SCRATCH = Path('/tmp/slm-durable-distributed-code-v1')
SCRIPT = Path(__file__).resolve()
RATES = {'rate3e6': 3e-6, 'rate3e5': 3e-5, 'rate1e4': 1e-4, 'rate1e5': 1e-5}
GIB = 1024**3
read, write, sha, identity, live = resource.read, resource.write, resource.sha, resource.identity, resource.live


def configure(attempt):
    resource.OUT = OUT/'attempts'/str(attempt)
    resource.SCRATCH = SCRATCH
    resource.SCRIPT = SCRIPT
    return resource.OUT


def sources(study, profile, purpose):
    return dict(initial_native_tensor_sha256=study['initial_native_tensor_sha256'],
        original_public_manifest_sha256=study['manifest_sha256'], original_public_admission_sha256=study['admission_sha256'],
        study_plan_sha256=sha(OUT/'study.json'),
        profile_sha256=hashlib.sha256(profile.encode()).hexdigest(), purpose_sha256=hashlib.sha256(purpose.encode()).hexdigest())


def assessment(model, data, checkpoint, folder, plan, rank, group):
    import torch
    from torch import distributed as dist
    from slm_lab.codenet_token_pack_v1 import CompleteCodePack
    from slm_lab.complete_response_stopping_loss_v1 import complete_response_CE_sums
    from slm_lab.distributed_dense_code_step_v1 import gather
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    from slm_lab.native_code_shards_v1 import evaluate_shard, reconcile
    from slm_lab.native_greedy_observer_v1 import observe, signature
    checked = lambda: resource.checked(plan)
    folder.mkdir(parents=True, exist_ok=True)
    model.eval();model.network.gradient_checkpointing=False
    before=signature(model)
    held=CompleteCodePack(data.folder,'validation',purpose='audit');records=[]
    with torch.no_grad():
        for i in range(rank,len(held),4):
            checked();ids,labels=held.sequence(i)
            inputs=torch.tensor([ids],device='cuda:0');targets=torch.tensor([labels],device='cuda:0')
            mask=torch.ones_like(inputs,dtype=torch.bool)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(inputs)['logits']
            loss=complete_response_CE_sums(logits.float(),targets,mask,profile='plain_tokens')
            records.append(dict(reserved_local_position=i,original_pack_position=int(held.positions[i]),
                complete_input_token_ids=ids,original_response_EOS_labels=labels,
                CE_sum=float(loss['original_CE_sums']),EOS_CE_sum=float(loss['EOS_CE_sums']),
                response_EOS_targets=sum(x!=-100 for x in labels)))
    parts=gather(records,group)
    if rank==0:
        rows=sorted([r for p in parts for r in p],key=lambda r:r['reserved_local_position'])
        total=sum(r['response_EOS_targets'] for r in rows)
        if [r['reserved_local_position'] for r in rows]!=list(range(425)) or total!=367945:
            raise ValueError('Every original held source and response/EOS target is required')
        write(folder/'held-code-NLL.json',dict(complete=True,reserved_views=425,response_EOS_targets=total,
            token_mean_NLL=sum(r['CE_sum'] for r in rows)/total,EOS_token_mean_NLL=sum(r['EOS_CE_sum'] for r in rows)/425,
            records=rows,reserved_optimization_updates=0,all_original_masks_and_targets_preserved=True,goal_achieved=False))
    metadata=dict(tokenizer_sha256=sha(data.folder/'tokenizer.json'),training_options={'image_size':448})
    setter=torch.set_num_threads;torch.set_num_threads=lambda _:None
    try:
        with inference_cache():
            for frame in ['completion','instruction']:
                with (folder/f'code-{frame}-rank-{rank}-actual-actions.jsonl').open('x') as stream:
                    def record(row):
                        checked();stream.write(json.dumps(row)+'\n');stream.flush()
                    with observe(model,record) as delivered:
                        envelope=evaluate_shard(model,metadata,checkpoint,folder,rank=rank,world=4,frame=frame,device='cuda:0')
                    if len(delivered)!=6:raise ValueError('Every unchanged assigned native action delivery is required')
                    os.fsync(stream.fileno())
                envelopes=gather(str(envelope),group)
                if rank==0:reconcile(envelopes,folder/('code-'+frame+'.json'))
            tasks=[('knowledge',0,dict(max_examples=56,max_new_tokens=32)),
                ('math',1,dict(max_examples=16,max_new_tokens=256)),
                ('ocr',2,dict(max_examples=32,max_new_tokens=128,image_size=448)),
                ('objects',2,dict(max_examples=32,max_new_tokens=32)),
                ('vision',3,dict(max_examples=32,image_size=448))]
            for name,owner,options in tasks:
                if rank!=owner:continue
                module=importlib.import_module('slm_lab.'+{'objects':'object'}.get(name,name)+'_evaluation')
                loader=module.load_checkpoint;module.load_checkpoint=lambda *a,**kw:(model,metadata)
                try:
                    source=ROOT/'data/math-reasoning-v2/test.jsonl' if name=='math' else Path(plan['pulse_sources'][name])
                    checked();module.evaluate(checkpoint,source,folder/(name+'.json'),device='cuda:0',**options)
                finally:module.load_checkpoint=loader
    finally:torch.set_num_threads=setter
    if signature(model)!=before or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Assessment must preserve every full-native value')
    data.audit_provenance();dist.barrier(group=group)
    if rank==0:write(folder/'complete.json',dict(complete=True,all_425held_48code_and_bounded_multimodal_assessments=True,
        checkpoint_sha256=sha(checkpoint),fresh_independent_program_execution_still_required=True,goal_achieved=False))


def worker(rank,attempt):
    folder=configure(attempt);plan=read(folder/'plan.json');study=read(OUT/'study.json');me=identity(os.getpid())
    while me not in [r['actor'] for r in read(folder/'children.json')]:
        if not live(plan['owner']):raise ValueError('A live registered owner is required before CUDA')
        time.sleep(.05)
    resource.checked(plan)
    import random
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint
    from slm_lab.distributed_dense_code_step_v1 import update, gather, agree
    from slm_lab.distributed_dense_code_state_v1 import make, restore
    from slm_lab.own_dense_standalone_v1 import export, load as load_native
    from slm_lab.parallel_function_repair_stage_v1 import value_digest
    from slm_lab.recovered_dense_base_v1 import recover
    from slm_lab.recovered_reasoned_code_data_v1 import RecoveredReasonedCodeSFTData
    from slm_lab.serial_dense_code_epoch_v2 import schedule
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[(rank*2)%len(allowed)]})
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(28*GIB/torch.cuda.get_device_properties(0).total_memory,0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',init_method='file://'+str((folder/'rendezvous').resolve()),rank=rank,world_size=4,
        timeout=timedelta(minutes=30))
    group=dist.new_group(backend='gloo',timeout=timedelta(minutes=30))
    checked=lambda:resource.checked(plan)
    data=RecoveredReasonedCodeSFTData(ROOT,purpose='optimization');entries=schedule(data)
    def fresh(rate):
        model,evidence=recover(ROOT);model.to('cuda:0')
        optimizer=torch.optim.AdamW([p for _,p in model.language_named_parameters()],lr=rate,
            betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False,fused=False)
        return model,optimizer,evidence
    def seed():
        random.seed(1009901+rank);np.random.seed(1009901+rank);torch.manual_seed(1009901+rank);torch.cuda.manual_seed(1009901+rank)
    def save(state,path,maximum=12*GIB):
        error=None
        if rank==0:
            try:atomic_checkpoint(path,state,maximum,8*GIB if str(path).startswith(str(ROOT)) else 40*GIB)
            except Exception as exc:error=repr(exc)
        agree(error,group)
    # Capacity is a separate discarded four-update experiment on the real graph.
    write(folder/f'rank-{rank}-phase.json',dict(phase='actual_four_rank_full_native_capacity'))
    model,optimizer,evidence=fresh(1e-5);seed();cap_sources=sources(study,'capacity','discarded_four_rank_capacity')
    groups=[entries[i] for i in [137,61,95,229]];rows=[]
    rows.append(update(model,optimizer,data,groups[0],checked=checked,object_group=group))
    state=make(model,optimizer,sources=cap_sources,completed=1,planned=4,metrics=rows,device='cuda:0',object_group=group)
    save(state,SCRATCH/'capacity-first.pt');del state
    rows.append(update(model,optimizer,data,groups[1],checked=checked,object_group=group))
    state=make(model,optimizer,sources=cap_sources,completed=2,planned=4,metrics=rows,device='cuda:0',object_group=group)
    expected=value_digest(state) if rank==0 else None;del state,model,optimizer;gc.collect();torch.cuda.empty_cache()
    model,optimizer,_=fresh(1e-5)
    state=torch.load(SCRATCH/'capacity-first.pt',map_location='cpu',weights_only=True,mmap=True)
    _,rows=restore(model,optimizer,state,sources=cap_sources,planned=4,device='cuda:0',object_group=group);del state
    rows.append(update(model,optimizer,data,groups[1],checked=checked,object_group=group))
    state=make(model,optimizer,sources=cap_sources,completed=2,planned=4,metrics=rows,device='cuda:0',object_group=group)
    agree('Fresh common Adam/four private RNG next-update mismatch' if rank==0 and value_digest(state)!=expected else None,group)
    del state
    for positions in groups[2:]:rows.append(update(model,optimizer,data,positions,checked=checked,object_group=group))
    memory=gather(torch.cuda.max_memory_reserved(),group)
    if rank==0:write(folder/'capacity-complete.json',dict(complete=True,actual_parameters_per_replica=984399690,
        language_values=931244544,Adam_states=317,world_size=4,logical_updates=4,physical_collective_updates=5,
        original_batch_indices=[137,61,95,229],actual_tail_rank_counts=rows[-1]['rank_source_counts'],
        exact_fresh_model_common_Adam_all_four_private_RNG_recovery=True,peak_reserved_bytes_per_rank=memory,
        all_capacity_learning_discarded=True,production_updates=0))
    del model,optimizer;gc.collect();torch.cuda.empty_cache()
    resume_path=MODELS/'training-state.pt'
    for profile,rate in RATES.items():
        destination=MODELS/profile;report=OUT/profile
        destination.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
        bound=sources(study,profile,'complete_distributed_full_language_SFT')
        if (report/'complete.json').exists():
            complete=read(report/'complete.json')
            if not complete['complete'] or sha(destination/'model.pt')!=complete['checkpoint_sha256']:
                raise ValueError('Every previously completed profile must retain its exact final model')
            continue
        write(folder/f'rank-{rank}-phase.json',dict(phase='loading_original_full_native',profile=profile))
        model,optimizer,evidence=fresh(rate);seed();rows=[];start=0
        if resume_path.exists():
            state=torch.load(resume_path,map_location='cpu',weights_only=True,mmap=True)
            if state['common']['model']['sources']==bound:
                start,rows=restore(model,optimizer,state,sources=bound,planned=230,device='cuda:0',object_group=group)
            elif state['common']['completed']!=230:
                raise ValueError('An incomplete different profile cannot be overwritten')
            del state
        if rank==0:write(report/f'attempt-{attempt}-resume.json',dict(restored_updates=start,planned_updates=230,
            original_order_and_full_Adam_and_all_rank_RNG_preserved=True))
        with (report/f'actual-updates-attempt-{attempt}-rank-{rank}.jsonl').open('x') as journal:
            write(folder/f'rank-{rank}-phase.json',dict(phase='full_language_training',profile=profile,restored_updates=start))
            for i in range(start,len(entries)):
                row=update(model,optimizer,data,entries[i],checked=checked,object_group=group);rows.append(row)
                journal.write(json.dumps(dict(step=i+1,**row))+'\n');journal.flush();os.fsync(journal.fileno())
                if rank==0:write(report/'progress.json',dict(phase='full_language_training',actual_complete_updates=i+1,
                    planned_updates=230,last_training_token_NLL=row['original_token_mean_NLL'],world_size=4,
                    learning_rate=rate,full_language_values=931244544,goal_achieved=False))
                if i==0 or (i+1)%16==0 or i+1==len(entries):
                    state=make(model,optimizer,sources=bound,completed=i+1,planned=230,metrics=rows,device='cuda:0',object_group=group)
                    save(state,resume_path);del state
                    if rank==0:write(OUT/'durable-state.json',dict(profile=profile,completed=i+1,checkpoint=str(resume_path),
                        sha256=sha(resume_path),complete_native_common_Adam_and_four_private_RNG_included=True))
                    dist.barrier(group=group)
        if ([p for row in rows for p in row['actual_positions']]!=[p for entry in entries for p in entry]
                or len(rows)!=230 or [r['actual_Adam_step'] for r in rows]!=list(range(1,231))):
            raise ValueError('Every original source, whole response/EOS and common Adam update is required')
        final_tensor_digest=tensor_digest(model.named_parameters())
        if len(set(gather(final_tensor_digest,group)))!=1:
            raise ValueError('Every final replica must retain exactly the same complete learned values')
        if rank==0:
            shutil.copyfile(data.folder/'tokenizer.json',destination/'tokenizer.json')
            payload=export(model,sources=bound,provenance=dict(study='four-rate durable data-parallel full-language SFT',
                own_scratch_origin_native=evidence,existing_admitted_corpus_recovered=True,
                missing_historical_execution_witnesses_not_revalidated=True,
                inherited_reward_feature_compatibility_after_language_updates_not_certified=True),
                completed_updates=230,tokenizer_sha256=sha(data.folder/'tokenizer.json'))
        else:payload=None
        path=destination/'model.pt';save(payload,path,4*GIB);del payload,optimizer;gc.collect();torch.cuda.empty_cache()
        # Each actual GPU receives a fresh CPU-loaded final graph before evaluation.
        digest=sha(path);del model;gc.collect();torch.cuda.empty_cache()
        model,_=load_native(path,expected_checkpoint_sha256=digest,expected_sources=bound);model.to('cuda:0')
        if tensor_digest(model.named_parameters())!=final_tensor_digest:
            raise ValueError('Every complete final native value must survive actual fresh-graph recovery')
        if rank==0:write(report/'training-complete.json',dict(complete=True,updates=230,views=3675,
            response_EOS_targets=3038462,whole_context_positions=4797843,tail=11,world_size=4,
            checkpoint_sha256=digest,all_capacity_learning_discarded=True,candidate_promoted=False,goal_achieved=False))
        assessment(model,data,path,report/f'assessment-attempt-{attempt}',plan,rank,group)
        if rank==0:write(report/'complete.json',dict(complete=True,checkpoint_sha256=digest,updates=230,
            matched_bounded_assessment=str(report/f'assessment-attempt-{attempt}'),
            full_final_benchmarks_and_compatible_human_RLHF_still_required=True,candidate_promoted=False,goal_achieved=False))
        del model;gc.collect();torch.cuda.empty_cache();dist.barrier(group=group)
    # A matched unmodified full-native baseline uses the same sharding/precision.
    if not (OUT/'initial/complete.json').exists():
        model,optimizer,evidence=fresh(1e-5);del optimizer
        initial=SCRATCH/'initial';initial.mkdir(exist_ok=True)
        if rank==0:
            shutil.copyfile(data.folder/'tokenizer.json',initial/'tokenizer.json')
            payload=export(model,sources=sources(study,'initial','unmodified_baseline'),provenance=dict(original=evidence),
                completed_updates=0,tokenizer_sha256=sha(data.folder/'tokenizer.json'))
        else:payload=None
        save(payload,initial/'model.pt',4*GIB);del payload
        assessment(model,data,initial/'model.pt',OUT/f'initial/assessment-attempt-{attempt}',plan,rank,group)
        if rank==0:write(OUT/'initial/complete.json',dict(complete=True,updates=0,
            original_native_tensor_sha256=evidence['recovered_native_tensor_sha256'],goal_achieved=False))
    dist.destroy_process_group()


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'torch' in sys.modules or os.nice(0)!=0:
        raise ValueError('A normal-priority GPU-hidden non-neural owner is required')
    tests=ROOT/'reports/distributed-dense-code-v1-tests.log'
    if '1 passed' not in tests.read_text():raise ValueError('The actual four-process serial/recovery controls must pass')
    for pending in MODELS.glob('**/*.pending'):
        raise ValueError('Inspect interrupted persistent checkpoint before resume: '+str(pending))
    gpus,apps=resource.snapshot()
    if len(gpus)!=4 or [g['uuid'] for g in gpus]!=resource.UUIDS or apps:
        raise ValueError('All four exact existing L40s must be idle for ownership')
    retained=sum(p.stat().st_size for p in [MODELS/'training-state.pt',*[MODELS/name/'model.pt' for name in RATES]] if p.exists())
    if (shutil.disk_usage(ROOT).free+retained<48*GIB+256*1024**2
            or shutil.disk_usage(ROOT).free<20*GIB or shutil.disk_usage('/tmp').free<64*GIB):
        raise ValueError('Four4GiB final models, two12GiB atomic states and8GiB durable reserve must fit')
    from slm_lab.local_source_closure_v2 import python_sources
    files={str(p):sha(p) for p in [*python_sources(ROOT,[SCRIPT]),Path(resource.__file__),tests,
        ROOT/'tests/test_distributed_dense_code_v1.py',ROOT/'data/public-reasoned-code-SFT-v1/manifest.json',
        ROOT/'data/public-reasoned-code-SFT-v1/admission.json']}
    old=read(ROOT/'reports/broad-anchored-code-v7/plan.json');proof=read(resource.PROOF)
    if not proof['complete'] or not proof['exact_model']['original_full_native_values_reconstructed_exactly']:
        raise ValueError('The exact full-native recovery proof is required')
    OUT.mkdir(exist_ok=True);MODELS.mkdir(exist_ok=True);SCRATCH.mkdir(exist_ok=True)
    study=dict(format='durable-four-rate-complete-dense-code-study-v1',source_files_sha256=files,rates=RATES,
        initial_native_tensor_sha256=proof['exact_model']['recovered_native_tensor_sha256'],
        manifest_sha256='379665024ed1e7cd70c31653ade14070ab7ac10d95ab9630ddaf072353c8bb28',
        admission_sha256='8b370b9bcce35831c15c106040a2cc6ae1b9a925dcc8b0a3fcafd1eee8d82db4',
        views_per_arm=3675,updates_per_arm=230,context_positions_per_arm=4797843,response_EOS_targets_per_arm=3038462,
        complete_source_epoch_per_arm=True,world_size=4,one_atomic_durable_common_state_with_four_private_RNGs=True,
        predeclared_central_arm_last_to_retain_its_final_optimizer=True,
        completed_noncentral_optimizer_slot_replaced_only_after_final_model_and_assessments=True,
        missing_original_raw_execution_witnesses_not_revalidated=True,original_interrupted_scopes_still_required=True,
        full_final_independent_benchmarks_and_human_RLHF_still_required=True,candidate_promoted=False,goal_achieved=False)
    if (OUT/'study.json').exists():
        if read(OUT/'study.json')!=study:raise ValueError('Resume requires every original source/study byte unchanged')
    else:write(OUT/'study.json',study)
    attempt=len(list((OUT/'attempts').glob('*')))+1;folder=configure(attempt);folder.mkdir(parents=True)
    plan=dict(owner=identity(os.getpid()),at=datetime.now(timezone.utc).isoformat(),expires_at_unix=time.time()+24*3600,
        study_sha256=sha(OUT/'study.json'),pulse_sources=old['pulse_sources'],source_files_sha256=files)
    write(folder/'plan.json',plan);write(folder/'children.json',[])
    with resource.LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);resource.checked(plan)
        watcher=subprocess.Popen([sys.executable,'-u',str(SCRIPT),'--watchdog','--attempt',str(attempt)],cwd=ROOT,
            stdout=(folder/'watchdog.log').open('x'),stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        children=[];processes=[]
        try:
            for rank in range(4):
                process=subprocess.Popen([sys.executable,'-u',str(SCRIPT),'--gpu-child','--rank',str(rank),'--attempt',str(attempt)],
                    cwd=ROOT,stdout=(folder/f'rank-{rank}.log').open('x'),stderr=subprocess.STDOUT,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(rank),'CUBLAS_WORKSPACE_CONFIG':':4096:8',
                        'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'})
                children.append(dict(actor=identity(process.pid),gpu=rank));write(folder/'children.json',children);processes.append(process)
            while any(p.poll() is None for p in processes):
                resource.checked(plan)
                if any(p.returncode not in [None,0] for p in processes):raise ValueError('An owned rank failed; retain state without automatic retry')
                time.sleep(1)
            if any(p.returncode!=0 for p in processes):raise ValueError('Every actual rank must complete')
            for name,digest in files.items():
                if sha(name)!=digest:raise ValueError('An original bound source changed during the run')
            write(OUT/'complete.json',dict(complete=True,all_four_rates_all_230updates_and_bounded_assessments=True,
                all_final_models_persistent=True,central_final_full_Adam_four_RNG_persistent=True,
                original_full_final_benchmarks_and_human_RLHF_still_required=True,candidate_promoted=False,goal_achieved=False))
        except BaseException as exc:
            write(folder/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),partial_outputs_preserved=True,goal_achieved=False));raise
        finally:
            for child in children:resource.stop(child['actor'],plan)
            write(folder/'restored.json',dict(own_children_terminal=all(not live(c['actor']) for c in children),external_processes_signalled=0))
            watcher.wait(timeout=10)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--gpu-child',action='store_true');parser.add_argument('--rank',type=int)
    parser.add_argument('--watchdog',action='store_true');parser.add_argument('--attempt',type=int)
    args=parser.parse_args()
    if args.watchdog:configure(args.attempt);resource.watchdog()
    elif args.gpu_child:worker(args.rank,args.attempt)
    elif args.run:parent()
    else:parser.error('An explicit owned run, rank or watchdog mode is required')
