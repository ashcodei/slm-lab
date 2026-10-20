"""Own the full public-code CE/EOS comparison after exact original delivery.

The source-waiting parent loads no Torch, model or GPU lease. Both arms retain
one completed own-policy initializer, its nonzero critic, frozen human reward,
all admitted train views and every original full assessment population.
"""
import argparse
from datetime import timedelta
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
SUITES=('code','broad','preferences','QA','AI2D','MBPP','words')
EXTRA=('reasoning','native_math','cpp','public_code')
PROFILES=('plain_tokens','eos_weight4')


def read(path):return json.loads(Path(path).read_text())


def normal():
    if (os.nice(0)!=0 or os.sched_getscheduler(0)!=os.SCHED_OTHER
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('A fresh normal nice0 one-thread owned process is required')


def configure(workspace):
    global WORKSPACE,OUT,DEST,owner
    WORKSPACE=Path(workspace).resolve();OUT=WORKSPACE/'reports';DEST=WORKSPACE/'checkpoints'
    import own_training_gpu_lease_v2 as owner
    owner.REPORTS=OUT;owner.resources.REPORTS=OUT


def memory_available():
    stat={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
    return int(Path('/sys/fs/cgroup/memory.max').read_text())-sum(stat[k] for k in ['anon','shmem','kernel'])


def preflight(plan):
    normal();owner.owned(plan)
    storage=plan['storage_budget']
    if (shutil.disk_usage(ROOT).free<storage['minimum_home_free_bytes']
            or shutil.disk_usage(WORKSPACE).free<storage['minimum_scratch_free_bytes']
            or memory_available()<112*1024**3):
        raise ValueError('The complete working memory and original other-writer reserves must remain available')


def gpu(rank,visible):
    normal()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=visible:raise ValueError('Exact owned physical GPU visibility required')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_device(rank)
    device=torch.device('cuda',rank);torch.cuda.set_per_process_memory_fraction(.85,device)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name(device)!='NVIDIA L40':raise ValueError('Only an original physical L40 is eligible')
    return device


def initializer(plan,device='cpu'):
    from slm_lab.own_policy_reward_standalone_v1 import load
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    spec=plan['initializer']
    bundle,meta=load(spec['checkpoint'],expected_checkpoint_sha256=spec['checkpoint_sha256'],expected_sources=spec['sources'],device=device)
    if (meta['completed_updates']!=10724 or meta['actual_registered_parameters']!=999706442
            or meta['tokenizer_sha256']!=spec['tokenizer_sha256']
            or tensor_digest(bundle.actor.parts['critic'].named_parameters())!=plan['initial_policy_critic_sha256']):
        raise ValueError('Only the exact completed own unfused policy, inherited critic and shared reward may initialize')
    return bundle


def sources(plan,profile):
    from slm_lab.parallel_function_repair_stage_v1 import value_digest
    return {**plan['initializer']['sources'],
        'initial_policy_package_sha256':plan['initializer']['checkpoint_sha256'],
        'initial_policy_critic_sha256':plan['initial_policy_critic_sha256'],
        'public_code_manifest_sha256':plan['public_code_manifest_sha256'],
        'public_code_admission_sha256':plan['public_code_admission_sha256'],
        'complete_capacity_plan_sha256':plan['complete_capacity_plan_sha256'],
        'new_training_recipe_sha256':plan['training_recipe']['sha256'],'training_profile_sha256':value_digest(profile)}


def distributed(profile,phase):
    import random
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.parallel_public_reasoned_code_SFT_stage_v1 import ParallelPublicReasonedCodeSFTStage
    from slm_lab.public_reasoned_code_capacity_controller_v2 import run_capacity
    from slm_lab.public_reasoned_code_training_controller_v2 import read_capacity,run_training
    from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
    rank=int(os.environ['RANK'])
    if os.environ.get('WORLD_SIZE')!='4' or int(os.environ['LOCAL_RANK'])!=rank:
        raise ValueError('All four original local replicas are required')
    plan=owner.registered(profile+'-'+phase+'-rank-'+str(rank));preflight(plan)
    device=gpu(rank,'0,1,2,3');dist.init_process_group('gloo',timeout=timedelta(minutes=45))
    gradients=dist.new_group(backend='nccl',timeout=timedelta(minutes=45))
    try:
        guard=FileBindings(plan['source_files_sha256']);guard.check(full=True)
        mode='train' if phase=='train' else 'capacity'
        data=PublicReasonedCodeSFTData(ROOT,purpose='optimization' if mode=='train' else 'capacity',views='both')
        if (data.manifest_sha256!=plan['public_code_manifest_sha256'] or data.admission_sha256!=plan['public_code_admission_sha256']
                or file_hash(data.folder/'tokenizer.json')!=plan['initializer']['tokenizer_sha256']):
            raise ValueError('Actual admitted train package and own-policy tokenizer must remain exact')
        capacity=read(plan['complete_capacity_plan_path']);options=plan['stage_options'];bundle=initializer(plan,device)
        setting=plan['optimizer']
        adam=torch.optim.AdamW([p for _,p in bundle.actor.named_selected(('actor',))],lr=options['learning_rate'],
            betas=tuple(setting['betas']),eps=setting['eps'],weight_decay=setting['weight_decay'],foreach=False,fused=False)
        seed=options['seed']+rank;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
        entries=data.complete_schedule(**{k:options[k] for k in ['epochs','width','seed']}) if mode=='train' else capacity['capacity_entries']
        stage=ParallelPublicReasonedCodeSFTStage(bundle.actor,adam,data,entries,profile,options,sources(plan,profile),
            mode=mode,rank=rank,world=4,gradient_group=gradients)
        directory=DEST/profile/(mode+'-state');folder=OUT/profile/phase;folder.mkdir(parents=True,exist_ok=True)
        inputs=torch.load(plan['cold_inputs']['path'],map_location='cpu',weights_only=True,mmap=True)
        def independent():return initializer(plan,device).actor
        kwargs=dict(capacity_plan=capacity,independent_actor=independent,cold_inputs=inputs)
        storage=plan['storage_budget']
        if phase=='capacity':
            result=run_capacity(stage,directory,maximum_bytes=storage['maximum_state_bytes'],
                minimum_free_bytes=storage['minimum_scratch_free_bytes'],**kwargs)
        else:
            reports=[read(OUT/profile/'capacity'/('rank-'+str(r)+'.json')) for r in range(4)]
            if phase=='readback':result=read_capacity(stage,directory,reports=reports,**kwargs)
            else:
                readbacks=[read(OUT/profile/'readback'/('rank-'+str(r)+'.json')) for r in range(4)]
                def progress(row,elapsed):
                    atomic_json(OUT/'progress.json',dict(profile=profile,phase='whole public reasoning-code SFT',
                        actual_updates=stage.completed,planned_updates=len(entries),actual_last_update_seconds=elapsed,goal_achieved=False))
                result=run_training(stage,directory,folder/'complete-actual-updates.jsonl.gz',
                    capacity_plan=capacity,reports=reports,readbacks=readbacks,checkpoint_every=plan['checkpoint_every'],
                    maximum_state_bytes=storage['maximum_state_bytes'],minimum_free_bytes=storage['minimum_scratch_free_bytes'],
                    maximum_journal_raw_bytes=storage['maximum_journal_raw_bytes'],preflight=lambda:preflight(plan),progress=progress)
                if result['original_complete_training_budget']!=plan['complete_training_budget']:
                    raise ValueError('The full original admitted train population cannot be narrowed')
        data.audit_provenance();guard.check(full=True);preflight(plan)
        atomic_json(folder/('rank-'+str(rank)+'.json'),result)
    finally:dist.destroy_process_group()


def export_candidate(profile):
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.own_policy_reward_standalone_v1 import load
    from slm_lab.parallel_repair_durable_state_v1 import read_latest
    from slm_lab.public_reasoned_code_policy_export_v1 import export
    from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    normal()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Full self-contained export is CPU-only')
    plan=owner.registered(profile+'-CPU-export');preflight(plan);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);guard=FileBindings(plan['source_files_sha256']);guard.check(full=True)
    proofs=[read(OUT/profile/'train'/('rank-'+str(r)+'.json')) for r in range(4)]
    if (any(not p['complete'] or p['original_complete_training_budget']!=plan['complete_training_budget'] for p in proofs)
            or len({p['actual_pointer_sha256'] for p in proofs})!=1 or len({p['actual_checkpoint_sha256'] for p in proofs})!=1):
        raise ValueError('All four complete original training receipts must agree')
    state,receipt,_=read_latest(DEST/profile/'train-state',expected_pointer_sha256=proofs[0]['actual_pointer_sha256'])
    if receipt['checkpoint_sha256']!=proofs[0]['actual_checkpoint_sha256']:raise ValueError('Exact final shared checkpoint required')
    bundle=initializer(plan);bound=sources(plan,profile);bundle.actor.import_export(state['model'],expected_sources=bound);bundle.actor.eval()
    data=PublicReasonedCodeSFTData(ROOT,views='both')
    packet,provenance=export(bundle.actor,bundle.reward,state,proofs[0],data=data,sources=bound,
        tokenizer_sha256=plan['initializer']['tokenizer_sha256'])
    folder=DEST/profile/'package';folder.mkdir(parents=True);shutil.copyfile(plan['initializer']['tokenizer'],folder/'tokenizer.json')
    checkpoint=folder/'model.pt';atomic_checkpoint(checkpoint,packet,plan['storage_budget']['maximum_package_bytes'],40*1024**3);del packet
    cold,meta=load(checkpoint,expected_checkpoint_sha256=file_hash(checkpoint),expected_sources=bound)
    if tensor_digest(cold.actor.named_parameters())!=tensor_digest(bundle.actor.named_parameters()):
        raise ValueError('Every actual native/policy/critic/reward/auxiliary parameter must cold-load exactly')
    inputs=torch.load(plan['cold_inputs']['path'],map_location='cpu',weights_only=True,mmap=True)
    with torch.no_grad():
        for kind in ['text','photo']:
            a,b=bundle.actor(**inputs[kind]),cold(**inputs[kind])
            if any(not torch.equal(a[k],b[k]) for k in ['logits','values']):raise ValueError('Full original policy text/photo/critic cold outputs differ')
        a,b=bundle.actor.base.word_logits(inputs['word']['word_pixels']),cold.word_logits(inputs['word']['word_pixels'])
        if list(a.shape)!=[16,140,257] or not torch.equal(a,b):raise ValueError('All original16word/140frame outputs must remain exact')
        ids=torch.tensor([data.sequence(0)[0]],dtype=torch.long)
        if not torch.equal(bundle.reward.score(ids),cold.preference_score(ids)):raise ValueError('Shared frozen human reward score changed during cold export')
    data.audit_provenance();guard.check(full=True);preflight(plan)
    atomic_json(OUT/profile/'package-export.json',dict(complete=True,checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),
        sources=bound,tokenizer_sha256=plan['initializer']['tokenizer_sha256'],completed_updates=state['completed'],
        actual_registered_parameters=meta['actual_registered_parameters'],all_full_original_CPUcold_outputs_exact=True,
        policy_critic_reward_and_all_auxiliary_values_preserved=True,policy_factors_not_merged=True,provenance=provenance,
        candidate_promoted=False,goal_achieved=False))


def load_candidate(plan,device='cpu'):
    from slm_lab.own_policy_reward_standalone_v1 import load
    spec=plan['standalone'];bundle,meta=load(spec['checkpoint'],expected_checkpoint_sha256=spec['checkpoint_sha256'],expected_sources=spec['sources'],device=device)
    expected=10724 if plan['endpoint']=='initial' else plan['complete_training_budget']['updates']
    if (meta['completed_updates']!=expected or meta['actual_registered_parameters']!=999706442
            or meta['tokenizer_sha256']!=plan['initializer']['tokenizer_sha256']):
        raise ValueError('Exact complete fixed baseline or complete trained unfused policy required')
    return bundle.actor,bundle.actor.policy,meta


def package_cold(profile,physical):
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings,capture_rng
    from slm_lab.diverse_images import atomic_json
    from slm_lab.own_policy_reward_standalone_v1 import load
    from slm_lab.parallel_function_repair_stage_v1 import value_digest
    from slm_lab.parallel_repair_durable_state_v1 import read_latest
    from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    plan=owner.registered(profile+'-package-GPUcold-'+str(physical));preflight(plan);device=gpu(0,str(physical))
    guard=FileBindings(plan['source_files_sha256']);guard.check(full=True);before=value_digest(capture_rng(device))
    original=initializer(plan,device);artifact=read(OUT/profile/'package-export.json');bound=sources(plan,profile)
    state,receipt,_=read_latest(DEST/profile/'train-state')
    if receipt['checkpoint_sha256']!=read(OUT/profile/'train/rank-0.json')['actual_checkpoint_sha256']:
        raise ValueError('The exact final complete shared source state is required')
    original.actor.import_export(state['model'],expected_sources=bound);original.actor.eval()
    cold,meta=load(artifact['checkpoint'],expected_checkpoint_sha256=artifact['checkpoint_sha256'],expected_sources=bound,device=device)
    actual=dict(cold.actor.named_parameters())
    if (list(actual)!=[n for n,_ in original.actor.named_parameters()]
            or any(not torch.equal(p,actual[n]) for n,p in original.actor.named_parameters())):
        raise ValueError('Every final source policy/critic/reward/auxiliary value must GPUcold-load exactly')
    raw=torch.load(plan['cold_inputs']['path'],map_location='cpu',weights_only=True,mmap=True)
    outputs={}
    with torch.no_grad(),torch.autocast('cuda',enabled=False):
        for kind in ['text','photo']:
            batch={k:v.to(device) for k,v in raw[kind].items()}
            a,b=original.actor(**batch),cold(**batch)
            if any(not torch.equal(a[k],b[k]) for k in ['logits','values']):raise ValueError('Every full native policy/critic GPUcold output must remain exact')
            outputs[kind]={key:tensor_digest([(key,b[key])]) for key in ['logits','values']}
        words=raw['word']['word_pixels'].to(device);a,b=original.actor.base.word_logits(words),cold.word_logits(words)
        if list(a.shape)!=[16,140,257] or not torch.equal(a,b):raise ValueError('Every full16word/140frame GPUcold output must remain exact')
        outputs['word']=tensor_digest([('logits',b)])
        data=PublicReasonedCodeSFTData(ROOT,views='both');ids=torch.tensor([data.sequence(0)[0]],device=device)
        if not torch.equal(original.reward.score(ids),cold.preference_score(ids)):raise ValueError('The trained shared human reward GPUcold score differs')
    if value_digest(capture_rng(device))!=before:raise ValueError('Complete cold reconstruction and native probes must preserve private CPU/CUDA RNG')
    guard.check(full=True);preflight(plan)
    atomic_json(OUT/profile/('package-cold-'+str(physical)+'.json'),dict(complete=True,physical_GPU=physical,
        actual_registered_parameters=meta['actual_registered_parameters'],checkpoint_sha256=artifact['checkpoint_sha256'],
        every_actual_source_parameter_equal=True,all_original_full4096text448photo16word_critic_reward_GPUcold_exact=True,
        output_digests=outputs,private_RNG_unchanged=True,new_optimizer_updates=0,goal_achieved=False))


def assessment(endpoint,suite,physical):
    plan=owner.registered(endpoint+'-'+suite+'-'+str(physical));preflight(plan)
    folder=OUT/endpoint/'assessment';ap=read(folder/'plan.json')
    if suite not in ['preferences',*EXTRA]:
        import assess_machine_standalone_full_v2_assessment_recovery_gate_v1 as evaluator
        from slm_lab.unfused_policy_assessment_binding_v1 import bind
        with bind(evaluator,reports=folder,plan=ap,load_student=load_candidate,
                register=lambda *a:owner.registered(endpoint+'-'+suite+'-'+str(physical))) as functions:functions[0](suite,physical)
        return
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.diverse_images import atomic_json
    gpu(0,str(physical));model,native,meta=load_candidate(ap,'cuda:0');guard=FileBindings(ap['source_files_sha256']);guard.check(full=True)
    if suite=='preferences':
        from slm_lab.unfused_policy_held_human_v1 import CachedHeldReference,evaluate_shard
        from slm_lab.whole_human_data_v2 import WholeHumanData
        data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','valid','evaluation');spec=ap['cached_own_reference']
        cache=CachedHeldReference(data,spec['report'],expected_sha256=spec['report_sha256'],expected_profile=spec['profile'])
        provenance=dict(checkpoint_sha256=ap['standalone']['checkpoint_sha256'],run_plan_sha256=sha(folder/'plan.json'),
            human_manifest_sha256=spec['profile']['human_manifest_sha256'],frozen_dense_sha256=spec['profile']['frozen_dense_sha256'],
            inference_device_type='cuda',inference_precision='float32')
        evaluate_shard(model,data,folder/'standalone'/('held-preferences-shard-'+str(physical)+'.json'),cache,rank=physical,world=4,
            provenance=provenance,beta=spec['profile']['DPO_beta'],checked=lambda:preflight(plan))
    elif suite=='reasoning':
        from slm_lab.public_reasoning_reserved_evaluation_v1 import PublicReasoningReserved
        from slm_lab.reasoning_reserved_shards_v1 import evaluate
        data=PublicReasoningReserved(ROOT/'data/public-reasoning-curriculum-v1',ROOT/'reports/public-reasoning-curriculum-v1-readback-admission/complete.json')
        with (folder/('reasoning-shard-'+str(physical)+'-actual-actions.jsonl')).open('x') as stream:
            def record(row):stream.write(json.dumps(row)+'\n');stream.flush()
            evaluate(native,data,folder/('reasoning-shard-'+str(physical)+'.json'),rank=physical,world=4,
                checkpoint_sha256=ap['standalone']['checkpoint_sha256'],checked=lambda:preflight(plan),record=record)
            os.fsync(stream.fileno())
    else:
        from slm_lab import full_math_shards_v1,cpp_evaluation,public_code_benchmark
        module={'native_math':full_math_shards_v1,'cpp':cpp_evaluation,'public_code':public_code_benchmark}[suite]
        previous=module.load_checkpoint;module.load_checkpoint=lambda *a,**k:(native,meta)
        try:
            checkpoint=ap['standalone']['checkpoint']
            if suite=='native_math':module.evaluate_shard(checkpoint,ROOT/'data/math-reasoning-v2/test.jsonl',
                folder/('native-math-shard-'+str(physical)+'.json'),shard=physical,shards=4,device='cuda:0',max_new_tokens=512,checked=lambda:preflight(plan))
            elif physical==0:
                if suite=='cpp':module.evaluate(checkpoint,folder/'cpp.json',device='cuda:0',max_new_tokens=512)
                else:module.evaluate(checkpoint,ROOT/'data/public-code-assessment-v1/questions.jsonl',folder/'public-code.json',device='cuda:0',max_new_tokens=512)
        finally:module.load_checkpoint=previous
    guard.check(full=True);preflight(plan)
    atomic_json(folder/(suite+'-rank-'+str(physical)+'-complete.json'),dict(complete=True,physical_GPU=physical,actual_policy_values=999706442,model_updates=0))


def sha(path):
    from slm_lab.public_reasoned_code_owned_recipe_v1 import sha as compute
    return compute(path)


def parent(args):
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.public_reasoned_code_owned_recipe_v1 import prepare,read,sha,live,STORAGE
    from slm_lab.reserved_atomic_metadata_v1 import write_json
    normal();workspace=args.workspace.absolute();data_plan=args.data_plan.resolve();delivery=args.delivery.resolve()
    if workspace.exists() or workspace.resolve()!=workspace or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'torch' in sys.modules:
        raise ValueError('Fresh normal non-neural CUDA-hidden source observer required')
    captured_data=read(data_plan/'owner-plan.json')['owner'];captured_delivery=read(delivery/'waiter-plan.json')['owner']
    paths=python_sources(ROOT,[Path(__file__),ROOT/'src/slm_lab/public_reasoned_code_owned_recipe_v1.py'])
    files={str(p):sha(p) for p in paths}
    for path in [data_plan/'owner-plan.json',delivery/'waiter-plan.json']:files[str(path)]=sha(path)
    workspace.mkdir();out=workspace/'reports';out.mkdir();deadline=time.monotonic()+35*86400
    def guard():
        normal()
        if any(sha(p)!=h for p,h in files.items()):raise ValueError('Every exact captured source/predecessor byte must remain unchanged')
    def write(name,value):return write_json(out/name,value,root=workspace,minimum_free_bytes=40*1024**3,
        maximum_metadata_bytes=64*1024**2,guard=guard)
    me=dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    write('waiter-plan.json',dict(owner=me,source_files_sha256=files,exact_data_plan_owner=captured_data,exact_delivery_owner=captured_delivery,
        GPU_contexts=0,GPU_leases=0,new_training_updates=0,goal_achieved=False))
    recipe=None
    while recipe is None:
        guard();recipe=prepare(ROOT,data_plan=data_plan,delivery=delivery,expected_data_owner=captured_data,
            expected_delivery_owner=captured_delivery,runtime_sources=files)
        if recipe is not None:break
        write('waiting-progress.json',dict(waiting_for='exact full public-data plan and completed original unfused policy/reward delivery',
            neural_work_started=False,GPU_contexts=0,GPU_leases=0,new_training_updates=0))
        if time.monotonic()>deadline:raise ValueError('Source waiting lifetime expired; original work is not relabelled stopped')
        time.sleep(30)
    while (shutil.disk_usage(workspace).free<STORAGE['required_scratch_free_bytes']
            or shutil.disk_usage(ROOT).free<STORAGE['required_home_free_bytes'] or memory_available()<112*1024**3):
        guard();write('waiting-progress.json',dict(waiting_for='complete96GiBscratch/60GiBhome/112GiBhost budget',neural_work_started=False))
        if time.monotonic()>deadline:raise ValueError('Full experiment funding wait expired; preserve all original workers')
        time.sleep(30)
    configure(workspace)
    with owner.LEASE.open('a+') as lease:
        while True:
            guard();owner.idle_original()
            try:fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:available=False
            else:
                used=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
                if len(used)!=4 or any(not line.startswith('NVIDIA L40,') for line in used):
                    raise ValueError('Every original physical L40 must remain present')
                available=all(int(line.split(',')[1])<=128 for line in used)
                if not available:fcntl.flock(lease.fileno(),fcntl.LOCK_UN)
            if available:break
            write('waiting-progress.json',dict(waiting_for='all four idle original L40s and the actual exclusive kernel flock',neural_work_started=False))
            if time.monotonic()>deadline:raise ValueError('Owned GPU funding wait expired without changing any original worker')
            time.sleep(30)
        import torch
        from slm_lab.codenet_checkpoint_v1 import FileBindings
        from slm_lab.diverse_images import atomic_json
        from slm_lab.own_policy_reward_standalone_v1 import inspect_package
        from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
        spec=recipe['initializer'];inspection=inspect_package(spec['checkpoint'],expected_checkpoint_sha256=spec['checkpoint_sha256'],expected_sources=spec['sources'])
        if inspection['actual_counts']['actual_unique']!=999706442 or inspection['completed_updates']!=10724:raise ValueError('Actual complete fixed original own-policy package required')
        raw=torch.load(spec['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
        critic=tensor_digest([('weight',raw['policy_adapter']['parts']['critic.weight'])]);del raw
        bound={**recipe['source_files_sha256'],**owner.lease_sources()};FileBindings(bound).check(full=True)
        DEST.mkdir();plan=dict(recipe,owner=owner.resources.identity(os.getpid()),waiter=owner.WAITER,
            terminal_waiter_receipt_sha256=owner.terminal_receipt_sha256(),expires_at_unix=time.time()+7*86400,
            source_files_sha256=bound,initial_policy_critic_sha256=critic,suites=list(SUITES),extra_suites=list(EXTRA),
            expected_student_values=999706442,expected_native_base_values=984399690,policy_factors_not_merged=True)
        recipe_path=OUT/'training-recipe.json';atomic_json(recipe_path,dict(stage_options=plan['stage_options'],optimizer=plan['optimizer'],
            complete_training_budget=plan['complete_training_budget'],profiles=list(PROFILES),initial_policy_critic_sha256=critic))
        plan['training_recipe']=dict(path=str(recipe_path),sha256=sha(recipe_path));bound[str(recipe_path)]=sha(recipe_path)
        atomic_json(OUT/'plan.json',plan);atomic_json(OUT/'children.json',[])
        driver=str(Path(__file__).resolve());watcher=None;success=False
        def interrupted(*a):raise KeyboardInterrupt('Owned public-code comparison interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        try:
            watcher=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',driver,'--workspace',str(WORKSPACE),'--watchdog'],cwd=ROOT,
                stdout=(OUT/'watchdog.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            def launch(arguments,subject,visible):
                child=owner.launch(plan,['-u',driver,'--workspace',str(WORKSPACE),*arguments],subject,visible);owner.wait_children(plan,[child])
            def assess(endpoint,standalone,suites):
                from slm_lab.unfused_policy_assessment_binding_v1 import bind
                folder=OUT/endpoint/'assessment';folder.mkdir(parents=True,exist_ok=True);(folder/'standalone').mkdir(exist_ok=True)
                ap=dict(plan,endpoint=endpoint,standalone=standalone,source_files_sha256={**bound,
                    standalone['checkpoint']:standalone['checkpoint_sha256'],str(Path(standalone['checkpoint']).parent/'tokenizer.json'):spec['tokenizer_sha256']})
                if not (folder/'plan.json').exists():atomic_json(folder/'plan.json',ap)
                elif read(folder/'plan.json')!=ap:raise ValueError('The same exact endpoint and full source plan must remain')
                import assess_machine_standalone_full_v2_assessment_recovery_gate_v1 as evaluator
                for suite in suites:
                    children=[owner.launch(plan,['-u',driver,'--workspace',str(WORKSPACE),'--assess',suite,'--endpoint',endpoint,'--gpu',str(r)],
                        endpoint+'-'+suite+'-'+str(r),str(r)) for r in range(4)]
                    owner.wait_children(plan,children)
                    if suite=='preferences':
                        from slm_lab.unfused_policy_held_human_v1 import CachedHeldReference,reconcile
                        from slm_lab.whole_human_data_v2 import WholeHumanData
                        data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','valid','evaluation');ref=plan['cached_own_reference']
                        cache=CachedHeldReference(data,ref['report'],expected_sha256=ref['report_sha256'],expected_profile=ref['profile'])
                        reconcile(data,[folder/'standalone'/('held-preferences-shard-'+str(r)+'.json') for r in range(4)],folder/'standalone/held-preferences.json',cache)
                    elif suite=='reasoning':
                        from slm_lab.public_reasoning_reserved_evaluation_v1 import PublicReasoningReserved
                        from slm_lab.reasoning_reserved_shards_v1 import reconcile
                        reconcile(PublicReasoningReserved(ROOT/'data/public-reasoning-curriculum-v1',ROOT/'reports/public-reasoning-curriculum-v1-readback-admission/complete.json'),
                            [folder/('reasoning-shard-'+str(r)+'.json') for r in range(4)],folder/'all-reserved-reasoning.json')
                    elif suite=='native_math':
                        from slm_lab.full_math_shards_v1 import reconcile
                        reconcile([folder/('native-math-shard-'+str(r)+'.json') for r in range(4)],ROOT/'data/math-reasoning-v2/test.jsonl',
                            Path(standalone['checkpoint']).parent/'tokenizer.json',folder/'full-native-math.json')
                    elif suite not in ['cpp','public_code']:
                        with bind(evaluator,reports=folder,plan=ap,load_student=load_candidate,register=lambda *a:plan) as functions:functions[1](suite,ap)
                    if sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())>STORAGE['maximum_assessment_bytes_per_endpoint']:
                        raise ValueError('Complete endpoint assessments exceed their declared budget; preserve all results')
            # Return early coding evidence before the slower all-domain suites;
            # every full original endpoint population remains required below.
            assess('initial',spec,['code'])
            candidates={}
            for profile in PROFILES:
                for phase in ['capacity','readback','train']:
                    child=owner.launch(plan,['-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',driver,
                        '--workspace',str(WORKSPACE),'--distributed',phase,'--profile',profile],profile+'-'+phase+'-launcher','0,1,2,3')
                    owner.wait_children(plan,[child])
                    if any(read(OUT/profile/phase/('rank-'+str(r)+'.json')).get('complete') is not True for r in range(4)):
                        raise ValueError('Every actual full-source phase replica must finish')
                launch(['--export','--profile',profile],profile+'-CPU-export','')
                candidates[profile]=read(OUT/profile/'package-export.json')
                cold_children=[owner.launch(plan,['-u',driver,'--workspace',str(WORKSPACE),'--cold','--profile',profile,'--gpu',str(r)],
                    profile+'-package-cold-'+str(r),str(r)) for r in range(4)]
                owner.wait_children(plan,cold_children)
                if any(read(OUT/profile/('package-cold-'+str(r)+'.json')).get('all_original_full4096text448photo16word_critic_reward_GPUcold_exact') is not True for r in range(4)):
                    raise ValueError('Every original full-package GPUcold proof is required before candidate assessment')
                assess(profile,candidates[profile],['code'])
            for endpoint,standalone in [('initial',spec),*candidates.items()]:assess(endpoint,standalone,[*SUITES[1:],*EXTRA])
            for endpoint in ['initial',*PROFILES]:
                folder=OUT/endpoint/'assessment'
                for name,count in [('standalone/full-QA.json',16961),('standalone/full-AI2D.json',3088),('standalone/full-original-MBPP.json',600),
                        ('standalone/held-preferences.json',4642),('standalone/all-held-words.json',3429),('all-reserved-reasoning.json',289),
                        ('full-native-math.json',1319),('cpp.json',12),('public-code.json',175)]:
                    if len(read(folder/name).get('results',[]))!=count:raise ValueError('Every complete original endpoint population must remain')
            # Keep each complete candidate and tokenizer self-contained on the
            # persistent volume. Scratch states and all observations stay intact.
            publication=ROOT/'checkpoints/public-reasoned-code-stopping-v1';pending=publication.with_name(publication.name+'.pending')
            if publication.exists() or pending.exists():raise ValueError('Preserve prior complete or partial candidate publication')
            pending.mkdir();published={}
            for profile,candidate in candidates.items():
                folder=pending/profile;folder.mkdir();published[profile]={}
                for name in ['model.pt','tokenizer.json']:
                    source=Path(candidate['checkpoint']).parent/name;preflight(plan)
                    if shutil.disk_usage(ROOT).free-source.stat().st_size<STORAGE['minimum_home_free_bytes']:
                        raise ValueError('Both complete persistent packages must preserve the original other-writer reserve')
                    with source.open('rb') as a,(folder/name).open('xb') as b:
                        while block:=a.read(1024**2):
                            if shutil.disk_usage(ROOT).free-len(block)<STORAGE['minimum_home_free_bytes']:
                                raise ValueError('Complete persistent package copy must preserve every other-writer reserve')
                            b.write(block)
                        b.flush();os.fsync(b.fileno())
                    if sha(folder/name)!=sha(source):raise ValueError('Complete persistent candidate bytes differ')
                    published[profile][name]=sha(folder/name)
            atomic_json(pending/'manifest.json',dict(format='own-complete-public-reasoned-code-two-profile-policy-reward-v1',
                profiles=published,source_recipe_sha256=plan['training_recipe']['sha256'],actual_policy_reward_values=999706442,
                policy_factors_not_merged=True,all_original_full_assessments_completed=True,candidate_promoted=False,goal_achieved=False))
            with (pending/'manifest.json').open('rb') as stream:os.fsync(stream.fileno())
            pending.rename(publication)
            directory_fd=os.open(publication.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(directory_fd)
            finally:os.close(directory_fd)
            FileBindings(bound).check(full=True);success=True
        except BaseException as exc:
            atomic_json(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False));raise
        finally:
            owner.restore_owned(plan,'Owned full public-code comparison ended; only its registered children released')
            if watcher is not None:watcher.wait(timeout=20)
        if success:atomic_json(OUT/'complete.json',dict(complete=True,profiles=list(PROFILES),all_original_train_views_and_EOS_targets_retained=True,
            all_actual_four_GPU_capacity_replay_fresh_readback_and_training_complete=True,
            all_original_full_code_broad_QA_AI2D_MBPP_human_words_reasoning_math_cpp_public_code_populations_complete=True,
            published_folder=str(publication),policy_critic_and_frozen_reward_basis_preserved=True,
            independent_final_neural_score_program_reexecution_and_quality_readbacks_pending=True,candidate_promoted=False,goal_achieved=False))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path('/tmp/slm-public-reasoned-code-stopping-owned-GPU-v1'))
    parser.add_argument('--data-plan',type=Path,default=ROOT/'reports/public-reasoned-code-stopping-plan-v1')
    parser.add_argument('--delivery',type=Path,default=Path('/tmp/slm-native-policy-reward-delivery-CPU-v2/reports'))
    parser.add_argument('--run',action='store_true');parser.add_argument('--watchdog',action='store_true')
    parser.add_argument('--distributed',choices=['capacity','readback','train']);parser.add_argument('--profile',choices=PROFILES)
    parser.add_argument('--export',action='store_true');parser.add_argument('--cold',action='store_true');parser.add_argument('--assess',choices=[*SUITES,*EXTRA])
    parser.add_argument('--endpoint',choices=['initial',*PROFILES]);parser.add_argument('--gpu',type=int,choices=range(4))
    args=parser.parse_args(argv)
    if args.run:return parent(args)
    configure(args.workspace)
    if args.watchdog:return owner.watchdog()
    if args.distributed and args.profile:return distributed(args.profile,args.distributed)
    if args.export and args.profile:return export_candidate(args.profile)
    if args.cold and args.profile and args.gpu is not None:return package_cold(args.profile,args.gpu)
    if args.assess and args.endpoint and args.gpu is not None:return assessment(args.endpoint,args.assess,args.gpu)
    parser.error('Use explicit source-waiting --run or a registered owned child mode')


if __name__=='__main__':main()
