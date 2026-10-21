"""Four bounded full-language SFT comparisons from the exact recovered native.

All four existing L40s participate, one fixed learning rate per device. Each
worker first proves real whole-source capacity and fresh Adam/RNG recovery;
probe weights are discarded before its complete230-update training epoch.
The central1e-5arm also keeps an atomic durable full-state checkpoint. All
four standalone final models and all observed metrics are durable.
"""
import argparse
from datetime import datetime,timezone
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

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/recovered-dense-code-v1'
MODELS=ROOT/'checkpoints/recovered-dense-code-v1'
SCRATCH=Path('/tmp/slm-recovered-dense-code-v1')
PROOF=ROOT/'reports/post-reset-recovery-v1/complete.json'
LOCK=ROOT/'reports/code-residual-production-v4.gpu.lock'
SCRIPT=Path(__file__).resolve()
RATES={'rate3e6':3e-6,'rate1e5':1e-5,'rate3e5':3e-5,'rate1e4':1e-4}
UUIDS=['GPU-76bcbccd-4cd1-4b1e-fd4a-c35193c69274','GPU-dd0acfbe-d00f-7764-aa40-c74357fffca2',
    'GPU-a7cf786b-0235-967a-3df6-10c55ac4261b','GPU-61eabc56-b315-d708-227e-1d74f7d5a958']
GIB=1024**3


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def identity(pid):
    try:s=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return None
    return None if s[0] in ['Z','X'] else dict(pid=pid,birth=s[19])


def live(a):return identity(a['pid'])==a


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('w') as f:
        f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(pending,path)


def snapshot():
    fields=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free','--format=csv,noheader,nounits'],text=True)
    gpus=[]
    for line in fields.splitlines():
        index,uuid,name,free=[x.strip() for x in line.split(',')]
        gpus.append(dict(index=int(index),uuid=uuid,name=name,free_MiB=int(free)))
    fields=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    apps=[]
    for line in fields.splitlines():
        uuid,pid,used=[x.strip() for x in line.split(',')];apps.append(dict(uuid=uuid,pid=int(pid),used_MiB=int(used)))
    return gpus,apps


def lease_owner():
    s=LOCK.stat();device=(os.major(s.st_dev),os.minor(s.st_dev),s.st_ino);owners=[]
    for line in Path('/proc/locks').read_text().splitlines():
        f=line.split()
        if len(f)<8 or f[1:4]!=['FLOCK','ADVISORY','WRITE']:continue
        a,b,c=f[5].split(':')
        if (int(a,16),int(b,16),int(c))==device:owners.append(int(f[4]))
    if len(owners)!=1:raise ValueError('One actual exclusive global GPU lease is required')
    return owners[0]


_last_check=0.
def checked(plan):
    global _last_check
    if not live(plan['owner']) or time.time()>plan['expires_at_unix'] or lease_owner()!=plan['owner']['pid']:
        raise ValueError('Exact live owner, lease and bounded time are required')
    if shutil.disk_usage(ROOT).free<8*GIB or shutil.disk_usage(SCRATCH).free<40*GIB:
        raise ValueError('The declared durable8GiB and scratch40GiB reserves must remain funded')
    if time.monotonic()-_last_check<1:return
    gpus,apps=snapshot();children=read(OUT/'children.json');known={r['actor']['pid']:r for r in children}
    if len(gpus)!=4 or [g['uuid'] for g in gpus]!=UUIDS or any(g['name']!='NVIDIA L40' or g['free_MiB']<12*1024 for g in gpus):
        raise ValueError('The exact four L40s and12GiB measured per-device free margin must remain')
    for app in apps:
        entry=known.get(app['pid'])
        if (entry is None or entry['gpu'] is None or app['uuid']!=UUIDS[entry['gpu']]
                or identity(app['pid']) not in [None,entry['actor']]):
            raise ValueError('Only exact registered own CUDA contexts on assigned devices are allowed')
    _last_check=time.monotonic()


def stop(actor,plan):
    from probe_supplemental_public_code_GPU_v1 import stop_new
    stop_new(actor,protected={plan['owner']['pid']},script=SCRIPT)


def watchdog():
    plan=read(OUT/'plan.json')
    while not (OUT/'restored.json').exists():
        if not live(plan['owner']) or time.time()>plan['expires_at_unix']:
            for entry in read(OUT/'children.json'):stop(entry['actor'],plan)
            write(OUT/'watchdog-cleanup.json',dict(own_registered_children_only=True,external_processes_signalled=0));return
        time.sleep(.2)


def source_map(plan,profile,purpose):
    proof=read(PROOF)
    return dict(donor_model_sha256=proof['exact_model']['source_checkpoint_sha256'],
        initial_native_tensor_sha256=proof['exact_model']['recovered_native_tensor_sha256'],
        original_public_manifest_sha256=plan['manifest_sha256'],original_public_admission_sha256=plan['admission_sha256'],
        recovery_proof_sha256=sha(PROOF),run_plan_sha256=sha(OUT/'plan.json'),
        profile_sha256=hashlib.sha256(profile.encode()).hexdigest(),purpose_sha256=hashlib.sha256(purpose.encode()).hexdigest())


def worker(profile,gpu):
    plan=read(OUT/'plan.json');me=identity(os.getpid())
    while me not in [r['actor'] for r in read(OUT/'children.json')]:
        if not live(plan['owner']):raise ValueError('Unregistered worker cannot allocate CUDA')
        time.sleep(.05)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('Only the registered physical GPU may be visible')
    checked(plan)
    import random
    import numpy as np
    import torch
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint
    from slm_lab.own_dense_checkpoint_v1 import make_state,restore_state
    from slm_lab.own_dense_standalone_v1 import export
    from slm_lab.parallel_function_repair_stage_v1 import value_digest
    from slm_lab.recovered_dense_base_v1 import recover
    from slm_lab.recovered_reasoned_code_data_v1 import RecoveredReasonedCodeSFTData
    from slm_lab.serial_dense_code_bf16_step_v2 import update
    from slm_lab.serial_dense_code_epoch_v2 import train
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1);torch.set_num_interop_threads(1);os.sched_setaffinity(0,{[20,22,5,7][gpu]})
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(28*GIB/torch.cuda.get_device_properties(0).total_memory,0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name(0)!='NVIDIA L40':raise ValueError('The assigned actual L40 is required')
    data=RecoveredReasonedCodeSFTData(ROOT,purpose='optimization');entries=data.complete_schedule()
    folder=OUT/profile;work=SCRATCH/profile;folder.mkdir();work.mkdir();records=[]
    def fresh():
        model,evidence=recover(ROOT);model.to('cuda:0')
        optimizer=torch.optim.AdamW([p for _,p in model.language_named_parameters()],lr=RATES[profile],
            betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False,fused=False)
        return model,optimizer,evidence
    model,optimizer,evidence=fresh();initial=tensor_digest(model.language_named_parameters())
    names={n for n,_ in model.language_named_parameters()}
    def fixed(m):return value_digest(dict(network={n:t.detach().cpu() for n,t in m.network.state_dict().items()
        if n not in names and n!='lm_head.weight'},heads={n:t.detach().cpu() for n,t in m.heads.state_dict().items()}))
    original_fixed=fixed(model);selected=[137,61,95,229];groups=[entries[i] for i in selected]
    if gpu==0:
        initial_folder=SCRATCH/'initial';initial_folder.mkdir();(OUT/'initial').mkdir()
        shutil.copyfile(data.folder/'tokenizer.json',initial_folder/'tokenizer.json')
        initial_sources=source_map(plan,'initial','unmodified_native_baseline')
        initial_payload=export(model,sources=initial_sources,provenance=dict(exact_recovered_original_native=evidence,
            baseline_before_any_capacity_or_training_update=True),completed_updates=0,
            tokenizer_sha256=sha(data.folder/'tokenizer.json'))
        initial_path=initial_folder/'model.pt';atomic_checkpoint(initial_path,initial_payload,4*GIB,40*GIB);del initial_payload
        write(folder/'progress.json',dict(phase='unmodified_full_native_baseline_assessment'))
        assess(model,data,initial_path,OUT/'initial/assessment',plan)
        write(OUT/'initial/complete.json',dict(complete=True,actual_native_parameters=984399690,
            unchanged_original_tensor_sha256=evidence['recovered_native_tensor_sha256'],actual_model_updates=0,
            all_original_425held_48code_and_bounded_broad_assessments_complete=True,goal_achieved=False))
    random.seed(1009901);np.random.seed(1009901);torch.manual_seed(1009901);torch.cuda.manual_seed(1009901)
    bound=source_map(plan,profile,'discarded_capacity')
    def cap_step(i):return update(model,optimizer,data,groups[i],'plain_tokens',checked=lambda:checked(plan))
    def cap_state(i):return make_state(model,optimizer,sources=bound,completed=i,planned=4,
        cursor={'next_capacity_group':i},metrics=records,device='cuda:0')
    write(folder/'progress.json',dict(phase='actual_full_native_capacity',actual_registered_parameters=model.counts()['total_unique']))
    records.append(cap_step(0));state=cap_state(1);atomic_checkpoint(work/'capacity-first.pt',state,12*GIB,40*GIB);first=value_digest(state);del state
    records.append(cap_step(1));state=cap_state(2);expected=value_digest(state);expected_rows=records.copy();del state
    del model,optimizer;gc.collect();torch.cuda.empty_cache();checked(plan)
    model,optimizer,_=fresh();state=torch.load(work/'capacity-first.pt',map_location='cpu',weights_only=True,mmap=True)
    if value_digest(state)!=first:raise ValueError('Every original saved capacity value must remain exact')
    _,records=restore_state(model,optimizer,state,expected_sources=bound,planned=4,
        expected_cursor={'next_capacity_group':1},device='cuda:0');del state
    records.append(cap_step(1));state=cap_state(2)
    if value_digest(state)!=expected or records!=expected_rows:raise ValueError('Full317Adam/model/RNG next update failed exact fresh recovery')
    del state
    for i in [2,3]:records.append(cap_step(i))
    if fixed(model)!=original_fixed or len(optimizer.state)!=317:raise ValueError('Only every dense-language Adam tensor may learn')
    write(folder/'capacity-complete.json',dict(complete=True,actual_unique_parameters=984399690,selected_language_values=931244544,
        selected_Adam_states=317,logical_updates=4,physical_updates=5,original_global_batch_indices=selected,
        actual_tail=len(groups[-1]),exact_fresh_model_Adam_private_RNG_recovery=True,all_fixed_values_unchanged=True,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),hard_allocator_limit_bytes=28*GIB,
        all_capacity_learning_discarded=True,production_updates=0))
    del model,optimizer;gc.collect();torch.cuda.empty_cache()
    model,optimizer,evidence=fresh()
    if tensor_digest(model.language_named_parameters())!=initial or optimizer.state:
        raise ValueError('Production must restart from exact own original values, never probe learning')
    bound=source_map(plan,profile,'complete_full_language_SFT');random.seed(1009901);np.random.seed(1009901);torch.manual_seed(1009901);torch.cuda.manual_seed(1009901)
    journal=folder/'actual-updates.jsonl'
    with journal.open('x') as stream:
        def record(step,row):
            stream.write(json.dumps(dict(step=step,**row),allow_nan=False)+'\n');stream.flush()
            write(folder/'progress.json',dict(phase='full_language_training',actual_complete_updates=step,planned_updates=230,
                last_token_NLL=row['original_token_mean_NLL'],selected_full_language_values=931244544,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),goal_achieved=False))
        def checkpoint(step,order,rows):
            state=make_state(model,optimizer,sources=bound,completed=step,planned=230,
                cursor={'next_global_batch':step},metrics=rows,device='cuda:0')
            path=work/('final-state.pt' if step==230 else 'resume.pt')
            atomic_checkpoint(path,state,12*GIB,40*GIB)
            if profile=='rate1e5' and (step==1 or step%64==0 or step==230):
                durable=MODELS/profile/'training-state.pt';durable.parent.mkdir(parents=True,exist_ok=True)
                with (OUT/'durable-writer.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX);atomic_checkpoint(durable,state,12*GIB,8*GIB)
                write(folder/'durable-state.json',dict(checkpoint=str(durable),sha256=sha(durable),completed=step,
                    complete_Adam_RNG_and_native_included=True))
            write(folder/'latest-state.json',dict(checkpoint=str(path),sha256=sha(path),completed=step))
        result=train(model,optimizer,data,checked=lambda:checked(plan),record=record,checkpoint=checkpoint)
        os.fsync(stream.fileno())
    if fixed(model)!=original_fixed:raise ValueError('All original vision/critic/auxiliary values must remain exact')
    model.eval();destination=MODELS/profile;destination.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(data.folder/'tokenizer.json',destination/'tokenizer.json')
    payload=export(model,sources=bound,provenance=dict(study='post-scratch-loss exact-native full-language SFT',
        original_model_recovery=evidence,existing_admitted_native_packet_reused=True,
        original_missing_source_witnesses_not_revalidated=True,no_capacity_learning_loaded=True,
        inherited_human_reward_feature_compatibility_after_language_updates_not_certified=True),
        completed_updates=230,tokenizer_sha256=sha(data.folder/'tokenizer.json'))
    path=destination/'model.pt'
    with (OUT/'durable-writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);atomic_checkpoint(path,payload,4*GIB,8*GIB)
    cold=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    for section in ['network','heads']:
        a=payload['complete_dense'][section];b=cold['complete_dense'][section]
        if a.keys()!=b.keys() or any(not torch.equal(a[k],b[k]) for k in a):raise ValueError('Every saved final native tensor must round-trip exactly')
    del payload,cold
    result.update(actual_native_parameters=984399690,full_language_parameters_updated=931244544,actual_full_Adam_states=len(optimizer.state),
        learning_rate=RATES[profile],initial_language_tensor_sha256=initial,final_language_tensor_sha256=tensor_digest(model.language_named_parameters()),
        final_checkpoint=str(path),final_checkpoint_sha256=sha(path),complete_saved_tensor_roundtrip=True,
        source_files_sha256=plan['source_files_sha256'],sources=bound,all_capacity_learning_discarded=True,
        raw_historical_witness_revalidation_not_claimed=True,original_public_data_scope_retained=True,
        compatible_reward_and_new_human_PPO_training_not_certified=True,matched_assessment_pending=True)
    write(folder/'training-complete.json',result);del optimizer;gc.collect();torch.cuda.empty_cache()
    assess(model,data,path,folder/'assessment',plan)
    write(folder/'complete.json',dict(complete=True,profile=profile,training_updates=230,
        native_parameters=984399690,full_language_values_updated=931244544,persistent_standalone=str(path),
        persistent_checkpoint_sha256=sha(path),matched_original_bounded_assessments_complete=True,
        final_independent_full_benchmarks_and_human_RLHF_still_required=True,candidate_promoted=False,goal_achieved=False))


def assess(model,data,checkpoint,folder,plan):
    import torch
    from slm_lab.codenet_token_pack_v1 import CompleteCodePack
    from slm_lab.complete_response_stopping_loss_v1 import complete_response_CE_sums
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    from slm_lab.native_code_shards_v1 import evaluate_shard,reconcile
    from slm_lab.native_greedy_observer_v1 import observe,signature
    folder.mkdir();model.eval();model.network.gradient_checkpointing=False;before=signature(model)
    held=CompleteCodePack(data.folder,'validation',purpose='audit');records=[]
    with torch.no_grad():
        for i in range(len(held)):
            checked(plan);ids,labels=held.sequence(i);inputs=torch.tensor([ids],device='cuda:0');targets=torch.tensor([labels],device='cuda:0')
            mask=torch.ones_like(inputs,dtype=torch.bool)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(inputs)['logits']
            loss=complete_response_CE_sums(logits.float(),targets,mask,profile='plain_tokens')
            records.append(dict(reserved_local_position=i,original_pack_position=int(held.positions[i]),complete_input_token_ids=ids,
                original_response_EOS_labels=labels,CE_sum=float(loss['original_CE_sums']),EOS_CE_sum=float(loss['EOS_CE_sums']),
                response_EOS_targets=sum(x!=-100 for x in labels)))
    total=sum(r['response_EOS_targets'] for r in records)
    if len(records)!=425 or total!=367945:raise ValueError('Every original reserved view/target must remain evaluation-only')
    write(folder/'held-code-NLL.json',dict(complete=True,reserved_views=425,response_EOS_targets=total,
        token_mean_NLL=sum(r['CE_sum'] for r in records)/total,EOS_token_mean_NLL=sum(r['EOS_CE_sum'] for r in records)/425,
        records=records,reserved_optimization_updates=0,all_original_masks_and_targets_preserved=True,goal_achieved=False))
    metadata=dict(tokenizer_sha256=sha(data.folder/'tokenizer.json'),training_options={'image_size':448})
    setter=torch.set_num_threads;torch.set_num_threads=lambda _:None
    try:
        with inference_cache():
            for frame in ['completion','instruction']:
                with (folder/('code-'+frame+'-actual-actions.jsonl')).open('x') as stream:
                    def record(row):checked(plan);stream.write(json.dumps(row)+'\n');stream.flush()
                    with observe(model,record) as delivered:
                        envelope=evaluate_shard(model,metadata,checkpoint,folder,rank=0,world=1,frame=frame,device='cuda:0')
                    if len(delivered)!=24:raise ValueError('Every complete original native code/action delivery must remain')
                    os.fsync(stream.fileno())
                reconcile([envelope],folder/('code-'+frame+'.json'))
            for name,options in [('knowledge',dict(max_examples=56,max_new_tokens=32)),('math',dict(max_examples=16,max_new_tokens=256)),
                    ('ocr',dict(max_examples=32,max_new_tokens=128,image_size=448)),('objects',dict(max_examples=32,max_new_tokens=32)),
                    ('vision',dict(max_examples=32,image_size=448))]:
                module=importlib.import_module('slm_lab.'+{'objects':'object'}.get(name,name)+'_evaluation');loader=module.load_checkpoint
                module.load_checkpoint=lambda *a,**kw:(model,metadata)
                try:
                    source=ROOT/'data/math-reasoning-v2/test.jsonl' if name=='math' else Path(plan['pulse_sources'][name])
                    checked(plan);module.evaluate(checkpoint,source,folder/(name+'.json'),device='cuda:0',**options)
                finally:module.load_checkpoint=loader
    finally:torch.set_num_threads=setter
    if signature(model)!=before or any(p.grad is not None for p in model.parameters()):raise ValueError('Assessment must preserve every actual native parameter')
    data.audit_provenance()


def parent():
    if (OUT.exists() or MODELS.exists() or SCRATCH.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!=''
            or 'torch' in sys.modules or os.nice(0)!=0):raise ValueError('Fresh normal-priority CPU owner and fresh output paths required')
    from slm_lab.local_source_closure_v2 import python_sources
    tests=ROOT/'reports/serial-dense-recovered-code-v2-tests.log';proof=read(PROOF)
    resources=ROOT/'reports/recovered-dense-code-resources-v1-tests.log'
    if ('6 passed' not in tests.read_text() or '2 passed' not in resources.read_text() or not proof['complete'] or not proof['no_GPU_initialized']
            or not proof['exact_model']['original_full_native_values_reconstructed_exactly']
            or proof['original_full_epoch_budget']['source_presentations']!=3675):
        raise ValueError('Actual full recovered model/data proof and six operator/epoch checks required')
    if shutil.disk_usage(ROOT).free<48*GIB+256*1024**2 or shutil.disk_usage(SCRATCH.parent).free<360*GIB:
        raise ValueError('Four4GiB exports, central two12GiB checkpoint generations,8GiB durable reserve and full scratch budget must fit')
    gpus,apps=snapshot()
    if len(gpus)!=4 or [g['uuid'] for g in gpus]!=UUIDS or apps:raise ValueError('All four actual existing L40s must be idle before exclusive ownership')
    old=read(ROOT/'reports/broad-anchored-code-v7/plan.json')
    files={str(p):sha(p) for p in [*python_sources(ROOT,[SCRIPT]),tests,resources,PROOF,ROOT/'tests/test_recovered_dense_code_resources_v1.py',ROOT/'tests/test_serial_dense_code_bf16_step_v2.py',
        ROOT/'tests/test_serial_dense_code_epoch_v2.py',ROOT/'data/public-reasoned-code-SFT-v1/manifest.json',
        ROOT/'data/public-reasoned-code-SFT-v1/admission.json',*map(Path,old['pulse_sources'].values()),ROOT/'data/math-reasoning-v2/test.jsonl']}
    from slm_lab.evaluation_provenance import SOURCE_NAMES
    for name in SOURCE_NAMES:files[str(ROOT/'src/slm_lab'/(name+'.py'))]=sha(ROOT/'src/slm_lab'/(name+'.py'))
    OUT.mkdir();SCRATCH.mkdir();write(OUT/'children.json',[])
    plan=dict(owner=identity(os.getpid()),at=datetime.now(timezone.utc).isoformat(),expires_at_unix=time.time()+24*3600,
        source_files_sha256=files,rates=RATES,full_source_views_per_arm=3675,full_source_updates_per_arm=230,
        whole_contexts_per_arm=4797843,response_EOS_targets_per_arm=3038462,actual_tail=11,
        actual_native_parameters=984399690,full_language_values=931244544,full_language_tensors=317,
        hard_own_allocator_bytes=28*GIB,minimum_GPU_free_bytes=12*GIB,durable_reserve_bytes=8*GIB,
        all_four_standalones_durable=True,predeclared_central_durable_Adam_RNG_arm='rate1e5',
        other_private_optimizer_states_on_volatile_scratch=True,missing_original_raw_witnesses_not_revalidated=True,
        existing_exact_admitted_corpus_only_no_new_source_admission=True,all_capacity_weights_discarded=True,
        manifest_sha256='379665024ed1e7cd70c31653ade14070ab7ac10d95ab9630ddaf072353c8bb28',
        admission_sha256='8b370b9bcce35831c15c106040a2cc6ae1b9a925dcc8b0a3fcafd1eee8d82db4',pulse_sources=old['pulse_sources'],
        original_interrupted_jobs_not_marked_complete=True,candidate_promoted=False,goal_achieved=False)
    write(OUT/'plan.json',plan)
    with LOCK.open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);checked(plan)
        watcher=subprocess.Popen([sys.executable,'-u',str(SCRIPT),'--watchdog'],cwd=ROOT,
            stdout=(OUT/'watchdog.log').open('x'),stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        processes=[];children=[]
        try:
            for gpu,profile in enumerate(RATES):
                process=subprocess.Popen([sys.executable,'-u',str(SCRIPT),'--gpu-child',profile,'--gpu',str(gpu)],cwd=ROOT,
                    stdout=(OUT/(profile+'.log')).open('x'),stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),
                    'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                    'TOKENIZERS_PARALLELISM':'false'})
                children.append(dict(actor=identity(process.pid),gpu=gpu,profile=profile));write(OUT/'children.json',children);processes.append(process)
            while any(p.poll() is None for p in processes):
                checked(plan)
                if any(p.returncode not in [None,0] for p in processes):raise ValueError('Owned capacity/training worker failed; preserve outputs without automatic retry')
                time.sleep(1)
            if any(p.returncode!=0 for p in processes):raise ValueError('Every original complete worker must finish successfully')
            for profile in RATES:
                if not read(OUT/profile/'complete.json')['complete']:raise ValueError('Every complete epoch and bounded assessment is required')
            if any(sha(p)!=h for p,h in files.items()):raise ValueError('A bound actual source/data/implementation byte changed')
            write(OUT/'complete.json',dict(complete=True,profiles=list(RATES),complete_updates_per_arm=230,
                complete_full_language_learning_and_bounded_assessments=True,all_four_models_durably_exported=True,
                no_external_pretrained_weights=True,new_human_or_PPO_updates=0,
                full_original_independent_final_benchmark_and_human_multimodal_scope_still_required=True,
                candidate_promoted=False,goal_achieved=False))
        except BaseException as error:
            write(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),partial_outputs_preserved=True,
                no_automatic_retry=True,external_processes_signalled=0,goal_achieved=False));raise
        finally:
            for entry in children:stop(entry['actor'],plan)
            write(OUT/'restored.json',dict(own_children_terminal=all(not live(r['actor']) for r in children),external_processes_signalled=0))
            watcher.wait(timeout=10)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--watchdog',action='store_true');parser.add_argument('--gpu-child',choices=list(RATES));parser.add_argument('--gpu',type=int)
    args=parser.parse_args()
    if args.watchdog:watchdog()
    elif args.gpu_child:worker(args.gpu_child,args.gpu)
    elif args.run:parent()
    else:parser.error('Explicit owned run, worker or watchdog mode required')
