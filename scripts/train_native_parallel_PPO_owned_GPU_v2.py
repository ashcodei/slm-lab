"""Owned four-L40 full-prompt PPO after the complete current-native reward.

The source observer is non-neural and takes no GPU lease while its exact
predecessor lives. Actual four-rank full-horizon capacity/private replay and
standalone full native GPU colds precede the original whole prompt epoch.
Both endpoints retain every original code/broad/multimodal/human assessment.
"""
import argparse
from datetime import timedelta
import fcntl
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
RESERVE=40*1024**3


def read(p):return json.loads(Path(p).read_text())


def normal():
    if (os.nice(0)!=0 or os.sched_getscheduler(0)!=os.SCHED_OTHER
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Only a normal nice0 one-thread owned PPO context is eligible')


def configure(workspace):
    global WORKSPACE,OUT,DEST,owner
    WORKSPACE=Path(workspace).resolve();OUT=WORKSPACE/'reports';DEST=WORKSPACE/'checkpoints'
    import own_training_gpu_lease_v2 as owner
    owner.REPORTS=OUT;owner.resources.REPORTS=OUT


def available_memory():
    values={k:int(v) for k,v in (s.split() for s in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
    return int(Path('/sys/fs/cgroup/memory.max').read_text())-sum(values[k] for k in ['anon','shmem','kernel'])


def preflight(plan):
    normal();owner.owned(plan)
    if min(shutil.disk_usage(ROOT).free,shutil.disk_usage(WORKSPACE).free)<RESERVE or available_memory()<64*1024**3:
        raise ValueError('Every original40GiBwriter reserve and64GiBother-worker host headroom must remain')


def device_setup(rank=None):
    import torch
    normal();torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if rank is None:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('CPU export must keep CUDA hidden')
        return torch.device('cpu')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ['0,1,2,3',str(rank)]:raise ValueError('Only the exact original physical L40visibility is admitted')
    index=rank if os.environ['CUDA_VISIBLE_DEVICES']=='0,1,2,3' else 0
    torch.cuda.set_device(index);torch.cuda.set_per_process_memory_fraction(.85,index)
    if torch.cuda.get_device_name(index)!='NVIDIA L40':raise ValueError('An actual original L40 is required')
    return torch.device('cuda',index)


def build(plan,device):
    import torch
    from slm_lab.native_reward_bundle_v1 import attach
    from slm_lab.native_shared_ppo_v1 import NativeSharedPPOStage
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.own_dense_standalone_v1 import load
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.corpus import file_hash
    native=plan['initializer'];base,metadata=load(native['checkpoint'],expected_checkpoint_sha256=native['checkpoint_sha256'],expected_sources=native['sources'])
    if base.counts()['total_unique']!=984399690 or metadata['completed_updates']!=5362:
        raise ValueError('The actual complete original joint native is the fixed policy/reward feature basis')
    base.to(device).eval();spec=plan['reward_bundle']
    reward=attach(base,spec['path'],expected_sha256=spec['sha256'],expected_sources=spec['sources'],expected_options=spec['options'])
    model=OwnDenseActor(base,**plan['actor_settings']).eval();options=plan['local_options']
    opt=torch.optim.AdamW([p for _,p in model.named_selected(('actor','critic'))],lr=options['learning_rate'],
        betas=tuple(options['betas']),weight_decay=options['weight_decay'],foreach=False)
    sources={**spec['sources'],'human_reward_bundle_sha256':spec['sha256'],'tokenizer_sha256':plan['tokenizer_sha256'],
        'run_plan_sha256':file_hash(OUT/'plan.json')}
    data=WholeHumanData(plan['data'],'preferences','train','optimization')
    return NativeSharedPPOStage(model,opt,data,reward,options,sources,device)


def admitted_capacities(plan):
    rows=[read(OUT/'capacity'/f'rank-{r}.json') for r in range(4)]
    for rank,row in enumerate(rows):
        if (row.get('complete') is not True or row.get('rank')!=rank or row.get('world_size')!=4
                or row.get('actual_native_values')!=984399690 or row.get('actual_policy_reward_values')!=999706442
                or row.get('actual_full_horizon_capacity',{}).get('full_size_original_L40_horizon_capacity_passed') is not True
                or row.get('all_actual_private_weights_Adam_actions_and_RNG_replayed') is not True
                or row.get('one_peer_corrupt_Adam_rejected_by_every_rank_before_mutation') is not True
                or row.get('reward_bundle_sha256')!=plan['reward_bundle']['sha256']):
            raise ValueError('Every actual original L40 must prove the complete original512horizon and exact private replay')
        cold=read(OUT/'capacity'/f'cold-{rank}.json')
        if cold.get('complete') is not True or cold.get('physical_GPU')!=rank or cold.get('all_full4096text448photo140word_native_outputs_exact') is not True:
            raise ValueError('All four actual original full native GPU colds are required before production PPO')
    return rows


def distributed(mode):
    import copy
    import random
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint,capture_rng
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.native_parallel_ppo_v2 import Collective,ParallelNativePPO
    from slm_lab.native_parallel_ppo_capacity_v2 import stress
    from slm_lab.native_parallel_ppo_checkpoint_v2 import make_state,restore_state,finish
    from slm_lab.parallel_function_repair_stage_v1 import value_digest
    if os.environ.get('WORLD_SIZE')!='4':raise ValueError('All four actual original PPO ranks are required')
    rank=int(os.environ['RANK']);plan=owner.registered(mode+'-rank-'+str(rank));preflight(plan);device=device_setup(rank)
    dist.init_process_group('nccl',timeout=timedelta(minutes=45));control=dist.new_group(backend='gloo',timeout=timedelta(minutes=45))
    try:
        kernel=build(plan,device);stage=ParallelNativePPO(kernel,Collective(device,control_group=control))
        guard=FileBindings(plan['source_files_sha256']);folder=DEST/mode/f'rank-{rank}';folder.mkdir()
        seed=plan['private_seed']+rank;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        stage.start_history(folder/'history',plan['history_budget'],checked=lambda:preflight(plan))
        capacity=None;recovery=False;corruption=False
        if mode=='capacity':
            error=None
            try:capacity=stress(stage,checked=lambda:preflight(plan))
            except Exception as exc:error=repr(exc)
            stage.collective.agree(error)
            stage.update();saved=make_state(stage);first=folder/'first.pt'
            atomic_checkpoint(first,saved,plan['storage_budget']['maximum_private_state_bytes'],RESERVE)
            stage.update();expected=make_state(stage);private=value_digest(capture_rng(device))
            broken=copy.deepcopy(saved)
            if rank==2:broken['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))
            before=value_digest(kernel.model.export(kernel.sources));before_adam=value_digest(kernel.optimizer.state_dict())
            try:
                restore_state(stage,broken,expected_common_sha256=saved['common_sha256'],
                    expected_private_rng_sha256=saved['private_RNG_sha256'],expected_completed=1)
            except ValueError:corruption=True
            if (not corruption or before!=value_digest(kernel.model.export(kernel.sources))
                    or before_adam!=value_digest(kernel.optimizer.state_dict()) or private!=value_digest(capture_rng(device))):
                raise ValueError('The actual peer corruption must reject before changing any weight, Adam or private generator')
            restore_state(stage,saved,expected_common_sha256=saved['common_sha256'],
                expected_private_rng_sha256=saved['private_RNG_sha256'],expected_completed=1,checked=lambda:preflight(plan))
            stage.start_history(folder/'replayed-history',plan['history_budget'],checked=lambda:preflight(plan))
            stage.update();actual=make_state(stage)
            if actual['common_sha256']!=expected['common_sha256'] or actual['private_RNG_sha256']!=expected['private_RNG_sha256']:
                raise ValueError('Every actual full-size own action, weight, Adam value and private generator must replay exactly')
            recovery=True
        else:
            admitted_capacities(plan)
            while stage.completed<stage.options['steps']:
                preflight(plan);row=stage.update()
                if stage.completed==1 or stage.completed%16==0 or stage.completed==stage.options['steps']:
                    atomic_json(OUT/mode/f'progress-{rank}.json',dict(actual_completed_updates=stage.completed,
                        planned_updates=stage.options['steps'],actual_original_prompt_draws=stage.cursor(stage.completed)['actual_prompt_draws'],
                        last_global_record_sha256=row['global_record_sha256'],actual_global_action_tokens=row['actual_action_tokens'],goal_achieved=False))
                if stage.completed==1 or stage.completed%plan['checkpoint_every']==0:
                    saved=make_state(stage);slot=(stage.completed//plan['checkpoint_every'])%2
                    atomic_checkpoint(folder/f'resume-slot-{slot}.pt',saved,plan['storage_budget']['maximum_private_state_bytes'],RESERVE)
            finish(stage)
        final=make_state(stage);path=folder/'final.pt'
        atomic_checkpoint(path,final,plan['storage_budget']['maximum_private_state_bytes'],RESERVE)
        guard.check(full=True);preflight(plan)
        atomic_json(OUT/mode/f'rank-{rank}.json',dict(complete=True,rank=rank,world_size=4,mode=mode,
            actual_native_values=984399690,actual_policy_reward_values=999706442,
            actual_updates=stage.completed,actual_physical_Adam_calls=3 if mode=='capacity' else stage.completed,
            actual_original_prompt_draws=stage.cursor(stage.completed)['actual_prompt_draws'],
            actual_full_horizon_capacity=capacity,all_actual_private_weights_Adam_actions_and_RNG_replayed=recovery,
            one_peer_corrupt_Adam_rejected_by_every_rank_before_mutation=corruption,
            reward_bundle_sha256=plan['reward_bundle']['sha256'],artifact=dict(path=str(path),sha256=file_hash(path),
                common_sha256=final['common_sha256'],private_RNG_sha256=final['private_RNG_sha256']),
            full85786prompt512action_epoch_complete=mode=='train' and stage.completed==10724,
            useful_own_output_alignment_proved=False,goal_achieved=False))
        dist.barrier(group=control)
    finally:dist.destroy_process_group()


def reconstructed(plan,mode):
    from slm_lab.native_parallel_ppo_export_v2 import materialize
    kernel=build(plan,'cpu');rows=[read(OUT/mode/f'rank-{r}.json') for r in range(4)]
    if any(r.get('complete') is not True or r.get('rank')!=i or r.get('mode')!=mode for i,r in enumerate(rows)):
        raise ValueError('All four actual closed private producer artifacts must remain complete')
    proof=materialize(kernel,[r['artifact'] for r in rows],expected_completed=2 if mode=='capacity' else 10724,checked=lambda:preflight(plan))
    return kernel,proof


def export_native(mode):
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.own_dense_actor_fold_v1 import fold_actor
    from slm_lab.own_dense_standalone_v1 import export,load
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    plan=owner.registered(mode+'-CPU-export');preflight(plan);device_setup()
    kernel,source_proof=reconstructed(plan,mode)
    native,fold=fold_actor(kernel.model.eval(),sources=kernel.sources,working_budget_bytes=16*1024**3)
    folder=DEST/mode/'native';folder.mkdir();shutil.copyfile(plan['initializer']['tokenizer'],folder/'tokenizer.json')
    path=folder/'model.pt';steps=2 if mode=='capacity' else 10724
    atomic_checkpoint(path,export(native,sources=kernel.sources,provenance=dict(kind='own-four-L40-shared-native-human-PPO-v2',
        mode=mode,full85786prompt512action_epoch_complete=mode=='train',actual_world_size=4,
        full_original_joint_body_updates=5362,complete_original_human_head_updates=2681,
        own_output_human_quality_not_proven=True),completed_updates=steps,tokenizer_sha256=plan['tokenizer_sha256']),
        plan['storage_budget']['maximum_native_bytes'],RESERVE)
    cold,_=load(path,expected_checkpoint_sha256=file_hash(path),expected_sources=kernel.sources)
    if tensor_digest(native.named_parameters())!=tensor_digest(cold.named_parameters()):raise ValueError('Every complete native folded value must independently cold-load exactly')
    raw=torch.load(plan['cold_inputs']['path'],map_location='cpu',weights_only=True,mmap=True)
    with torch.no_grad():
        for kind in ['text','photo']:
            a=native(**raw[kind]);b=cold(**raw[kind])
            if any(not torch.equal(a[k],b[k]) for k in ['logits','values']):raise ValueError('Every actual full original text/photo native cold output must match exactly')
        if not torch.equal(native.word_logits(raw['word']['word_pixels']),cold.word_logits(raw['word']['word_pixels'])):
            raise ValueError('Every actual whole140frame word native output must independently cold-load exactly')
    preflight(plan);FileBindings(plan['source_files_sha256']).check(full=True)
    atomic_json(OUT/mode/'native-export.json',dict(complete=True,mode=mode,checkpoint=str(path),checkpoint_sha256=file_hash(path),
        tokenizer=str(folder/'tokenizer.json'),tokenizer_sha256=plan['tokenizer_sha256'],sources=kernel.sources,
        completed_updates=steps,actual_native_values=984399690,full_source_reconstruction=source_proof,
        every_complete_native_value_and_full_text_photo_word_CPUcold_exact=True,fold_report=fold,
        full85786prompt512action_epoch_complete=mode=='train',goal_achieved=False))


def native_cold(mode,physical):
    import torch
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.own_dense_actor_fold_v1 import fold_actor
    from slm_lab.own_dense_standalone_v1 import load
    plan=owner.registered(mode+'-GPUcold-'+str(physical));preflight(plan)
    # Reconstruct the source on CPU without replaying any private CUDA state.
    torch.set_num_threads(1);torch.set_num_interop_threads(1);kernel,proof=reconstructed(plan,mode)
    expected,_=fold_actor(kernel.model.eval(),sources=kernel.sources,working_budget_bytes=16*1024**3)
    artifact=read(OUT/mode/'native-export.json');candidate,_=load(artifact['checkpoint'],
        expected_checkpoint_sha256=artifact['checkpoint_sha256'],expected_sources=kernel.sources)
    del kernel
    device=device_setup(physical);expected.to(device).eval();candidate.to(device).eval()
    raw=torch.load(plan['cold_inputs']['path'],map_location='cpu',weights_only=True,mmap=True)
    with torch.no_grad():
        for kind in ['text','photo']:
            batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in raw[kind].items()}
            a,b=expected(**batch),candidate(**batch)
            if any(not torch.equal(a[k],b[k]) for k in ['logits','values']):raise ValueError('Full original native GPUtext/photo cold output differs')
        pixels=raw['word']['word_pixels'].to(device)
        a,b=expected.word_logits(pixels),candidate.word_logits(pixels)
        if not torch.equal(a,b):raise ValueError('Full original native GPUword cold output differs')
        if (raw['text']['input_ids'].shape[1]!=4096 or tuple(raw['photo']['pixel_values'].shape[-2:])!=(448,448)
                or tuple(a.shape)!=(16,140,257)):
            raise ValueError('The complete original4096text/448photo/16by140word source shapes are required')
    preflight(plan)
    atomic_json(OUT/mode/f'cold-{physical}.json',dict(complete=True,physical_GPU=physical,actual_native_values=984399690,
        checkpoint_sha256=artifact['checkpoint_sha256'],all_full4096text448photo140word_native_outputs_exact=True,
        actual_text_tokens=raw['text']['input_ids'].shape[1],actual_photo_shape=list(raw['photo']['pixel_values'].shape),
        full_source_reconstruction=proof,goal_achieved=False))


def load_candidate(plan,device='cpu'):
    from slm_lab.dense_foundation_inference_v1 import DenseNativeView
    from slm_lab.own_dense_standalone_v1 import load
    spec=plan['standalone'];model,metadata=load(spec['checkpoint'],expected_checkpoint_sha256=spec['checkpoint_sha256'],expected_sources=spec['sources'])
    expected=2681 if plan['endpoint']=='initial' else 10724
    if (model.counts()['total_unique']!=984399690 or metadata['completed_updates']!=expected
            or metadata['tokenizer_sha256']!=plan['tokenizer_sha256']
            or plan['endpoint']!='initial' and metadata['provenance'].get('full85786prompt512action_epoch_complete') is not True):
        raise ValueError('Only the exact complete fixed baseline or full original PPO native may enter unchanged full assessments')
    model.to(device).eval();return model,DenseNativeView(model).eval(),metadata


def assessment(endpoint,suite,physical):
    import train_native_code_suffix_aligned_GPU_v4 as original
    previous=original.OUT,original.load_candidate,getattr(original,'owner',None)
    original.OUT,original.load_candidate,original.owner=OUT,load_candidate,owner
    try:original.assessment(endpoint,suite,physical)
    finally:
        original.OUT,original.load_candidate=previous[:2]
        if previous[2] is None:del original.owner
        else:original.owner=previous[2]


def parent(args):
    from slm_lab.native_parallel_ppo_owned_recipe_v2 import prepare,sha,BUDGET,SUITES
    from slm_lab.full_experiment_launch_queue_v1 import identity
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.reserved_atomic_metadata_v1 import write_json
    normal();workspace=Path(args.workspace).absolute();reward=Path(args.reward).resolve()
    if workspace.exists() or workspace.resolve()!=workspace or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'torch' in sys.modules:
        raise ValueError('A fresh normal non-neural CUDA-hidden observer must precede the complete owned PPO run')
    captured=read(reward/'waiter-plan.json')['owner'];files={str(p):sha(p) for p in python_sources(ROOT,[Path(__file__)])}
    files[str(reward/'waiter-plan.json')]=sha(reward/'waiter-plan.json');workspace.mkdir();reports=workspace/'reports';reports.mkdir()
    def guard():
        normal()
        if any(sha(p)!=h for p,h in files.items()):raise ValueError('Every exact original head owner and full PPO runtime byte must remain')
    def write(name,value):return write_json(reports/name,value,root=workspace,minimum_free_bytes=RESERVE,
        maximum_metadata_bytes=BUDGET['maximum_metadata_bytes'],guard=guard)
    write('waiter-plan.json',dict(owner=identity(os.getpid()),source_files_sha256=files,exact_original_reward_owner=captured,
        all_original_experiments_and_full_assessments_preserved=True,GPU_contexts=0,GPU_leases=0,production_updates=0,goal_achieved=False))
    deadline=time.monotonic()+35*86400;recipe=None
    while recipe is None:
        guard();recipe=prepare(ROOT,reward=reward,expected_owner=captured,joint=args.joint,source_recipe=args.source_recipe,runtime_sources=files)
        if recipe is not None:break
        write('waiting-progress.json',dict(waiting_for='exact full original current-native reward owner,whole epoch and publication',
            GPU_work_started=False,production_PPO_updates=0,goal_achieved=False))
        if time.monotonic()>deadline:raise ValueError('Source observation expired; no original owner is relabelled stopped')
        time.sleep(30)
    while (shutil.disk_usage(workspace).free<BUDGET['required_scratch_free_bytes']
            or shutil.disk_usage(ROOT).free<BUDGET['required_home_free_bytes'] or available_memory()<112*1024**3):
        guard();write('waiting-progress.json',dict(waiting_for='full152GiBscratch,48GiBhome and112GiBhost free budget',GPU_work_started=False))
        if time.monotonic()>deadline:raise ValueError('Full physical funding observation expired without altering original workers')
        time.sleep(30)
    configure(workspace)
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.diverse_images import atomic_json
    with owner.LEASE.open('a+') as lease:
        while True:
            try:fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                guard()
                if time.monotonic()>deadline:raise ValueError('Exclusive GPUlease observation expired without preemption')
                time.sleep(30)
        owner.idle_original();guard()
        used=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(used)!=4 or any(r.rsplit(',',1)[0].strip()!='NVIDIA L40' or int(r.rsplit(',',1)[1])>128 for r in used):
            raise ValueError('All four actual original L40s must be idle before new owned PPO work')
        published=ROOT/'checkpoints/native-shared-human-PPO-owned-v2'
        if published.exists() or published.with_name(published.name+'.pending').exists():raise ValueError('Preserve every earlier complete or partial PPO publication')
        bound={**recipe['source_files_sha256'],**owner.lease_sources()}
        source_guard=FileBindings(bound);source_guard.check(full=True)
        plan=dict(recipe,owner=identity(os.getpid()),source_files_sha256=bound,waiter=owner.WAITER,
            terminal_waiter_receipt_sha256=owner.terminal_receipt_sha256(),expires_at_unix=time.time()+14*86400,
            candidate_promoted=False,goal_achieved=False)
        atomic_json(OUT/'plan.json',plan);atomic_json(OUT/'children.json',[])
        for mode in ['capacity','train']:(OUT/mode).mkdir();(DEST/mode).mkdir(parents=True)
        driver=str(Path(__file__).resolve());base=['-u',driver,'--workspace',str(workspace)]
        watcher=subprocess.Popen([str(ROOT/'.venv/bin/python'),*base,'--watchdog'],cwd=ROOT,
            stdout=(OUT/'watchdog.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned full shared-native PPO interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        def group(mode):
            child=owner.launch(plan,['-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                driver,'--workspace',str(workspace),'--distributed',mode],mode+'-torchrun','0,1,2,3');owner.wait_children(plan,[child])
        def export_and_cold(mode):
            child=owner.launch(plan,[*base,'--export',mode],mode+'-CPU-export','');owner.wait_children(plan,[child])
            children=[owner.launch(plan,[*base,'--cold',mode,'--gpu',str(r)],mode+'-GPUcold-'+str(r),str(r)) for r in range(4)]
            owner.wait_children(plan,children)
        def assess(endpoint,standalone):
            from slm_lab.visual_native_assessment_binding_v1 import bind
            from slm_lab.public_reasoning_reserved_evaluation_v1 import PublicReasoningReserved
            from slm_lab.reasoning_reserved_shards_v1 import reconcile
            import assess_machine_standalone_full_v2_assessment_recovery_gate_v1 as evaluator
            folder=OUT/endpoint/'assessment';folder.mkdir(parents=True)
            ap=dict(plan,endpoint=endpoint,standalone=standalone,
                source_files_sha256={**bound,standalone['checkpoint']:standalone['checkpoint_sha256']})
            atomic_json(folder/'plan.json',ap)
            for suite in [*SUITES,'reasoning']:
                children=[owner.launch(plan,[*base,'--assess',suite,'--endpoint',endpoint,'--gpu',str(r)],
                    endpoint+'-'+suite+'-'+str(r),str(r)) for r in range(4)];owner.wait_children(plan,children)
                if suite=='reasoning':
                    population=PublicReasoningReserved(ROOT/'data/public-reasoning-curriculum-v1',ROOT/'reports/public-reasoning-curriculum-v1-readback-admission/complete.json')
                    reconcile(population,[folder/f'reasoning-shard-{r}.json' for r in range(4)],folder/'all-reserved-reasoning.json')
                else:
                    with bind(evaluator,reports=folder,plan=ap,load_student=load_candidate,register=lambda *a:plan) as functions:functions[1](suite,ap)
                if sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())>BUDGET['maximum_each_complete_assessment_bytes']:
                    raise ValueError('Preserve every original result; complete assessment exceeded its full physical bound')
        success=False
        try:
            preflight(plan);group('capacity');export_and_cold('capacity');admitted_capacities(plan)
            assess('initial',plan['baseline']);group('train');export_and_cold('train')
            final=read(OUT/'train/native-export.json');assess('PPO',final)
            from compare_visual_native_endpoints_v1 import compare
            compare(OUT/'initial/assessment/standalone',OUT/'PPO/assessment/standalone',OUT/'matched-comparisons.json')
            before,after=[read(OUT/e/'assessment/all-reserved-reasoning.json') for e in ['initial','PPO']]
            keys=['position','source_id','source_group','task','original_prompt_ids','original_native_generation_budget']
            if (len(before.get('results',[]))!=289 or len(after.get('results',[]))!=289
                    or any(any(a[k]!=b[k] for k in keys) for a,b in zip(before['results'],after['results']))):
                raise ValueError('Every original reserved reasoning delivery and full native budget must remain')
            atomic_json(OUT/'matched-reasoning.json',dict(complete=True,before_correct=before['correct'],after_correct=after['correct'],
                original_reserved_questions=289,source_reserved_only_historical_independence_not_certified=True))
            pending=published.with_name(published.name+'.pending');pending.mkdir();total=0;published_files={}
            artifacts=[(Path(final['checkpoint']),'model.pt'),(Path(final['tokenizer']),'tokenizer.json'),
                (OUT/'matched-comparisons.json','matched-comparisons.json'),(OUT/'matched-reasoning.json','matched-reasoning.json')]
            for endpoint in ['initial','PPO']:
                folder=OUT/endpoint/'assessment'
                artifacts += [(p,str(Path(endpoint)/p.relative_to(folder))) for p in folder.rglob('*') if p.is_file()]
            for src,name in artifacts:
                total+=src.stat().st_size
                if total>BUDGET['maximum_published_bytes']:raise ValueError('Preserve complete results; final durable publication exceeded its full budget')
                dest=pending/name;dest.parent.mkdir(parents=True,exist_ok=True)
                with src.open('rb') as a,dest.open('xb') as b:
                    while block:=a.read(1024**2):
                        preflight(plan)
                        if shutil.disk_usage(ROOT).free-len(block)<RESERVE:raise ValueError('Durable native/results publication must preserve the original40GiBfloor')
                        b.write(block)
                    b.flush();os.fsync(b.fileno())
                if sha(src)!=sha(dest):raise ValueError('Every complete durable PPO native and original result byte must remain exact')
                published_files[name]=sha(dest)
            atomic_json(pending/'manifest.json',dict(format='own-four-L40-full-human-PPO-native-and-full-results-v2',
                standalone=final,files_sha256=published_files,all85786original_prompts_once=True,
                fresh_human_judgments_on_own_outputs_not_supplied=True,goal_achieved=False))
            pending.rename(published);source_guard.check(full=True);preflight(plan);success=True
        except BaseException as exc:
            atomic_json(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False));raise
        finally:
            owner.restore_owned(plan,'Owned full shared-native human PPO ended');watcher.wait(timeout=20)
        if success:atomic_json(OUT/'complete.json',dict(complete=True,all10724four_L40_updates_and85786whole_original_prompts_complete=True,
            every_original512action_capacity_private_replay_and_full_native_CPU_GPUcold_complete=True,
            both_full_original_code_broad_QA_AI2D_MBPP_human_words_and_reserved_reasoning_assessments_complete=True,
            actual_native_values=984399690,actual_research_policy_reward_values=999706442,durable_publication=str(published),
            useful_own_output_human_alignment_or_high_end_parity_not_established_by_completion_alone=True,goal_achieved=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path('/tmp/slm-native-parallel-PPO-owned-GPU-v2'))
    parser.add_argument('--reward',type=Path,default=Path('/tmp/slm-native-human-reward-owned-GPU-v1/reports'))
    parser.add_argument('--joint',type=Path,default=Path('/tmp/slm-complete-joint-owned-GPU-v1/reports'))
    parser.add_argument('--source-recipe',type=Path,default=Path('/tmp/slm-native-shared-PPO-source-recipe-CPU-v1/complete.json'))
    parser.add_argument('--run',action='store_true');parser.add_argument('--watchdog',action='store_true')
    parser.add_argument('--distributed',choices=['capacity','train']);parser.add_argument('--export',choices=['capacity','train'])
    parser.add_argument('--cold',choices=['capacity','train']);parser.add_argument('--assess',choices=['code','broad','preferences','QA','AI2D','MBPP','words','reasoning'])
    parser.add_argument('--endpoint',choices=['initial','PPO']);parser.add_argument('--gpu',type=int,choices=range(4))
    args=parser.parse_args()
    if args.run:parent(args)
    else:
        configure(args.workspace)
        if args.watchdog:owner.watchdog()
        elif args.distributed:distributed(args.distributed)
        elif args.export:export_native(args.export)
        elif args.cold and args.gpu is not None:native_cold(args.cold,args.gpu)
        elif args.assess and args.endpoint and args.gpu is not None:assessment(args.endpoint,args.assess,args.gpu)
        else:parser.error('Use explicit source-waiting --run or registered owned worker modes')
