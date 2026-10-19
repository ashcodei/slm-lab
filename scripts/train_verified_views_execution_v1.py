"""Owned exact-source view warm-up, then paired public-target/execution RL.

Waits for the preceding exact owner to finish or an explicit preserved-state
handover. Fresh four-L40 full-profile capacity/recovery gates production.
This is a development experiment, not a final policy or human-feedback RLHF.
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
REPORTS=ROOT/'reports/verified-views-execution-v1'
OUTPUT=ROOT/'checkpoints/verified-views-execution-v1'
QUEUE=ROOT/'reports/verified-views-execution-v1-queue.json'
PREVIOUS=ROOT/'reports/protected-joint-v1'
HANDOVER=ROOT/'reports/verified-views-execution-v1-handover.json'
RESERVE=26*1024**3
MAX_STATE=256*1024**2
owner=legacy.owner


def configure():
    owner.REPORTS=REPORTS;owner.resources.REPORTS=REPORTS
    legacy.REPORTS=REPORTS;legacy.OUTPUT=OUTPUT


def previous_ready(prior):
    if owner.resources.same(prior['owner']):return False
    if any(owner.resources.same(a) for a in json.loads((PREVIOUS/'children.json').read_text())):return False
    if not (PREVIOUS/'restored.json').exists():raise ValueError('Original work must be restored after all exact prior workers end')
    if HANDOVER.exists():
        r=json.loads(HANDOVER.read_text())
        if (any(str(r['retired_exact_owner'][k])!=str(prior['owner'][k]) for k in ['pid','ticks'])
                or r.get('partial_epoch_and_endpoints_explicitly_incomplete') is not True
                or r.get('initial_and_all_evaluated_checkpoints_preserved') is not True):raise ValueError('Exact explicit preserved-state handover required')
        for path,digest in r['retained_evidence_sha256'].items():
            if file_hash(path)!=digest:raise ValueError('Preserved handover evidence changed')
        return True
    if not (PREVIOUS/'complete.json').exists() or (PREVIOUS/'failure.json').exists():
        raise ValueError('Predecessor must finish successfully or have an explicit preserved-state research handover')
    return True


def runtime(mode):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1,2,3' or int(os.environ.get('WORLD_SIZE',0))!=4:raise ValueError('All four owned local GPUs required')
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);plan=owner.registered(mode+'-rank-'+str(rank))
    import torch
    from torch import distributed as dist
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);device=torch.device('cuda',local);torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.85,device);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name(device)!='NVIDIA L40':raise ValueError('Only the four original local L40s are admitted')
    dist.init_process_group('gloo',timeout=timedelta(minutes=30))
    all_gradient=dist.new_group([0,1,2,3],backend='nccl',timeout=timedelta(minutes=20))
    gradients=[dist.new_group(ranks,backend='nccl',timeout=timedelta(minutes=20)) for ranks in [[0,1],[2,3]]]
    controls=[dist.new_group(ranks,backend='gloo',timeout=timedelta(minutes=30)) for ranks in [[0,1],[2,3]]]
    guard=FileBindings({**plan['source_files_sha256'],str(REPORTS/'plan.json'):file_hash(REPORTS/'plan.json')})
    return plan,torch,dist,rank,device,all_gradient,gradients[rank//2],controls[rank//2],guard


def distributed(mode):
    plan,torch,dist,rank,device,all_gradient,gradient,control,guard=runtime(mode)
    import copy
    import random
    import numpy as np
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint,capture_rng
    from slm_lab.execution_group_causal_v3 import CausalExecutionGroup
    from slm_lab.execution_group_capacity_v2 import maximum_prompt,stress
    from slm_lab.fast_joint_inference_v2 import fresh
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.verified_views_warmup_v1 import VerifiedViewsWarmup
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    started=time.monotonic();digest=file_hash(REPORTS/'plan.json');sources=dict(run_plan_sha256=digest,
        own_initial_dense_sha256=plan['initial_dense_settings']['initializer_sha256'],
        verified_views_admission_sha256=file_hash(ROOT/'data/verified-code-views-v1/admission.json'),
        causal_reward_admission_sha256=file_hash(ROOT/'reports/causal-code-rewards-v2/admission.json'))
    def adam(opt):return tensor_digest((str(i)+'.'+k,s[k]) for i,s in enumerate(opt.state.values()) for k in ('step','exp_avg','exp_avg_sq'))
    def make_optimizer(lr):return torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],lr=lr,betas=(.9,.95),weight_decay=.01,foreach=False)
    def save_private(stage,label):
        path=OUTPUT/f'{mode}-{label}-rank-{rank}.pt';atomic_checkpoint(path,stage.state(sources),MAX_STATE,RESERVE)
        proof=dict(rank=rank,path=str(path),sha256=file_hash(path),actual_completed=stage.completed,private_RNG=True)
        proofs=[None]*4;dist.all_gather_object(proofs,proof)
        if rank==0:atomic_json(REPORTS/f'{mode}-{label}-distributed-state.json',dict(all_four_private_states=proofs,
            complete_distributed_state_requires_every_private_file=True,goal_achieved=False))
        return path
    def recover(stage,label):
        first=stage.update();path=save_private(stage,label);expected=stage.update();weights=tensor_digest(model.parts.named_parameters());moments=adam(stage.optimizer)
        payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True);bad=copy.copy(payload)
        name='core' if 'core' in payload else 'private_replica_state';bad[name]=dict(payload[name]);bad[name]['rng']=dict(payload[name]['rng'])
        bad[name]['rng']['cuda']=torch.zeros(4,dtype=torch.uint8);before=capture_rng(device);versions=stage.signature(model.parameters())
        try:stage.resume(bad,sources)
        except ValueError:pass
        else:raise ValueError('Bad actual private CUDA RNG must reject before mutation')
        after=capture_rng(device)
        if (versions!=stage.signature(model.parameters()) or moments!=adam(stage.optimizer)
                or not torch.equal(before['torch'],after['torch']) or not torch.equal(before['cuda'],after['cuda'])):
            raise ValueError('Rejected restore mutated actual live model/Adam/CPU or CUDA RNG')
        stage.resume(payload,sources);repeated=stage.update()
        if expected!=repeated or weights!=tensor_digest(model.parts.named_parameters()) or moments!=adam(stage.optimizer):
            raise ValueError('Actual complete next metrics/actions/rewards/actor/Adam/private RNG must recover exactly')
        return dict(actual_records=[first,expected],logical_updates=2,physical_optimizer_calls=3,
            actual_next_metrics_actions_weights_Adam_and_private_CPU_CUDA_RNG_exact=True,bad_private_CUDA_RNG_rejected_before_mutation=True)
    def journal(stage,label):
        stream=(REPORTS/f'{label}-metrics.jsonl').open('x') if rank==0 or label!='warmup' and rank==2 else None
        try:
            planned=stage.planned if hasattr(stage,'planned') else len(stage.order)
            while stage.completed<planned:
                owner.owned(plan);guard.check();tick=time.monotonic();record=stage.update()
                if stream:
                    stream.write(json.dumps(record)+'\n');stream.flush()
                    atomic_json(REPORTS/f'{label}-progress.json',dict(actual_updates=stage.completed,
                        planned_updates=stage.planned if hasattr(stage,'planned') else len(stage.order),
                        last_update_seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-started,
                        latest=record if label=='warmup' else {k:v for k,v in record.items() if k!='trajectories'},goal_achieved=False))
                # All four ranks enter every durable boundary together.
                if stage.completed%32==0 or stage.completed==(stage.planned if hasattr(stage,'planned') else len(stage.order)):
                    save_private(stage,label+'-'+str(stage.completed))
        finally:
            if stream:stream.close()
    try:
        guard.check(full=True);base,_=fresh(plan,device);model=OwnDenseActor(base)
        if model.counts()['actual_unique']!=999706442:raise ValueError('Actual complete own policy/heads/critic must be strictly below1B')
        frozen=tensor_digest(base.named_parameters());initial_actor=tensor_digest(model.parts.named_parameters());critic=model.parts['critic'].weight.detach().clone()
        gathered=[None]*4;dist.all_gather_object(gathered,initial_actor)
        if len(set(gathered))!=1:raise ValueError('Every worker must start from identical own policy weights')
        random.seed(957813+rank%2);np.random.seed(957813+rank%2);torch.manual_seed(957813+rank%2)
        if mode!='RL':
            optimizer=make_optimizer(plan['warmup_learning_rate'])
            warmup=VerifiedViewsWarmup(model,optimizer,ROOT,method='views',epochs=plan['warmup_epochs'],functions_per_update=8,
                seed=957713,world=4,rank=rank,gradient_group=all_gradient,control_group=None)
            if mode=='capacity':
                from slm_lab.verified_views_capacity_v1 import stress as stress_views,extremes
                warmup_proof=recover(warmup,'warmup-step1')
                warmup_proof['full_actual_source_extremes']=[stress_views(warmup,position) for position in extremes(warmup)]
            else:
                if rank==0:atomic_checkpoint(OUTPUT/'initial-actor.pt',model.export(sources),96*1024**2,RESERVE)
                journal(warmup,'warmup')
                if rank==0:
                    atomic_checkpoint(OUTPUT/'warmup-actor.pt',model.export(sources),96*1024**2,RESERVE)
                    atomic_json(REPORTS/'warmup-training-complete.json',dict(actual_updates=warmup.completed,actual_counts=model.counts(),
                        original_functions=173,epochs=plan['warmup_epochs'],presentations=173*3*plan['warmup_epochs'],
                        missing_repair_reuses_original_completion_not_new_unique_data=True,supervised_not_RL=True,goal_achieved=False))
            del warmup,optimizer
        else:
            payload=torch.load(OUTPUT/'warmup-actor.pt',map_location='cpu',weights_only=True,mmap=True)
            model.import_export(payload,expected_sources=sources);del payload
        if mode=='warmup':
            if frozen!=tensor_digest(base.named_parameters()) or not torch.equal(critic,model.parts['critic'].weight):
                raise ValueError('Every original base/head/critic byte must remain exact during warm-up')
            guard.check(full=True);dist.barrier();return
        # RL has a fresh, separately serialized Adam clock. Both objective arms
        # start from the same actual warm-up policy, with paired private seeds.
        warm_actor=tensor_digest(model.parts.named_parameters());gathered=[None]*4;dist.all_gather_object(gathered,warm_actor)
        if len(set(gathered))!=1:raise ValueError('Paired execution objectives require the exact same warm-up weights')
        optimizer=make_optimizer(plan['RL_learning_rate'])
        method='verified_solution' if rank<2 else 'execution_hybrid'
        random.seed(957913+rank%2);np.random.seed(957913+rank%2);torch.manual_seed(957913+rank%2)
        stage=CausalExecutionGroup(model,optimizer,ROOT,method=method,seed=957117,group_size=plan['local_RL_group_size'],
            max_new_tokens=plan['RL_max_new_tokens'],world=2,rank=rank%2,gradient_group=gradient,control_group=control)
        maximum=maximum_prompt(stage)
        if not maximum['maximum_prompt_and_full_horizon_fit']:raise ValueError('Every complete original prompt plus512 actions must fit')
        if mode=='capacity':
            recovery=recover(stage,'RL-step1');full=stress(stage,maximum['original_train_position'])
            proof=dict(rank=rank,physical_GPU=rank,actual_counts=model.counts(),actual_Adam_states=len(optimizer.state),
                original_functions=173,local_group_size=stage.group_size,global_group_size=stage.group_size*stage.world,
                configured_maximum_new_tokens=stage.max_new_tokens,warmup=warmup_proof,RL_recovery=recovery,
                maximum_original_prompt=maximum,actual_full_horizon_gradient_stress=full,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                seconds=time.monotonic()-started,production_updates=0,goal_achieved=False)
            atomic_json(REPORTS/f'capacity-{rank}.json',proof)
        else:
            journal(stage,method)
            # Each arm leader writes the complete counted shared-base export.
            if rank%2==0:
                atomic_checkpoint(OUTPUT/f'{method}-actor.pt',model.export(sources),96*1024**2,RESERVE)
                atomic_json(REPORTS/f'{method}-training-complete.json',dict(actual_updates=stage.completed,original_functions=173,
                    actual_counts=model.counts(),actual_Adam_states=len(optimizer.state),method=method,
                    genuine_human_preference_RLHF=False,candidate_promoted=False,goal_achieved=False))
        if (frozen!=tensor_digest(base.named_parameters()) or not torch.equal(critic,model.parts['critic'].weight)
                or initial_actor==tensor_digest(model.parts.named_parameters())):raise ValueError('Actual actor must learn while every frozen base/head/critic byte remains exact')
        guard.check(full=True);dist.barrier()
    finally:dist.destroy_process_group()


def assess(arm,gpu):
    plan=legacy.registered('assess-'+arm+'-'+str(gpu),gpu)
    import torch
    from slm_lab.fast_joint_inference_v2 import fresh
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.85,0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,_=fresh(plan,'cuda:0');model=OwnDenseActor(base);checkpoint=OUTPUT/f'{arm}-actor.pt'
    sources=dict(run_plan_sha256=file_hash(REPORTS/'plan.json'),own_initial_dense_sha256=plan['initial_dense_settings']['initializer_sha256'],
        verified_views_admission_sha256=file_hash(ROOT/'data/verified-code-views-v1/admission.json'),
        causal_reward_admission_sha256=file_hash(ROOT/'reports/causal-code-rewards-v2/admission.json'))
    guard=FileBindings({**plan['source_files_sha256'],str(checkpoint):file_hash(checkpoint)})
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True);model.import_export(payload,expected_sources=sources);del payload
    metadata=dict(tokenizer_sha256=plan['initial_dense_settings']['tokenizer_sha256'],training_options=dict(image_size=448))
    destination=REPORTS/arm;destination.mkdir(exist_ok=True);model.eval()
    def evaluate(module,name,data=None,**kwargs):
        saved=module.load_checkpoint;module.load_checkpoint=lambda *args,**kwargs:(model.policy,metadata)
        try:
            owner.owned(plan)
            if data is None:module.evaluate(checkpoint,destination/name,device='cuda:0',**kwargs)
            else:module.evaluate(checkpoint,data,destination/name,device='cuda:0',**kwargs)
        finally:module.load_checkpoint=saved
    threads=torch.set_num_threads;torch.set_num_threads=lambda count:None
    try:
        with inference_cache():
            if gpu==0:
                from slm_lab import code_quality
                for frame in ['completion','instruction']:evaluate(code_quality,'code-'+frame+'.json',max_new_tokens=512,prompt_mode=frame)
            elif gpu==1:
                from slm_lab import knowledge_evaluation,math_evaluation
                evaluate(knowledge_evaluation,'knowledge.json',Path(plan['pulse_sources']['knowledge']),max_examples=56,max_new_tokens=32)
                evaluate(math_evaluation,'math.json',ROOT/'data/math-reasoning-v2/test.jsonl',max_examples=16,max_new_tokens=256)
            elif gpu==2:
                from slm_lab import ocr_evaluation,object_evaluation
                evaluate(ocr_evaluation,'ocr.json',Path(plan['pulse_sources']['ocr']),max_examples=32,max_new_tokens=128,image_size=448)
                evaluate(object_evaluation,'objects.json',Path(plan['pulse_sources']['objects']),max_examples=32,max_new_tokens=32)
            else:
                from slm_lab import vision_evaluation
                evaluate(vision_evaluation,'vision.json',Path(plan['pulse_sources']['vision']),max_examples=32,image_size=448)
        guard.check(full=True);atomic_json(destination/f'assessment-{gpu}-complete.json',dict(physical_GPU=gpu,
            actual_counts=model.counts(),checkpoint_sha256=file_hash(checkpoint),development_comparison_not_final_independent_benchmark=True,goal_achieved=False))
    finally:torch.set_num_threads=threads


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU controller must hide every GPU')
    if REPORTS.exists() or OUTPUT.exists() or QUEUE.exists():raise ValueError('Preserve previous queued/partial/complete attempts')
    prior=json.loads((PREVIOUS/'plan.json').read_text())
    proof=ROOT/'reports/verified-views-warmup-full-CPU-v1/complete.json'
    full=json.loads(proof.read_text())
    if not full['exact_next_views_metrics_weights_Adam_CPU_RNG'] or not full['trained_native_text_image_word_cold_reload_exact']:
        raise ValueError('Actual full-size view learning/recovery/cold-load controls required')
    test_log=ROOT/'reports/verified-views-execution-v1-tests.log'
    if '4 passed' not in test_log.read_text():raise ValueError('Actual no-commit extremes and exact predecessor resource tests required')
    for folder in ['verified-views-warmup-four-CPU-v2','execution-group-causal-four-CPU-v1']:
        if json.loads((ROOT/'reports'/folder/'complete.json').read_text()).get('all_four_CPU_workers_passed') is not True:
            raise ValueError('Both new actual four-CPU kernels must pass')
    files={p:d for p,d in prior['source_files_sha256'].items() if Path(p).suffix=='.py'}
    additions=[Path(__file__),proof,test_log,ROOT/'tests/test_verified_views_execution_v1.py',ROOT/'src/slm_lab/verified_views_warmup_v1.py',ROOT/'src/slm_lab/execution_group_causal_v3.py',
        ROOT/'src/slm_lab/execution_group_controlled_v2.py',ROOT/'src/slm_lab/execution_group_capacity_v2.py',ROOT/'src/slm_lab/verified_views_capacity_v1.py',
        ROOT/'src/slm_lab/causal_code_reward_v2.py',ROOT/'src/slm_lab/causal_code_reward_catalog_v1.py',
        ROOT/'data/verified-code-views-v1/manifest.json',ROOT/'data/verified-code-views-v1/admission.json',
        ROOT/'reports/verified-code-views-v1/independent-reader-v1.json',ROOT/'reports/causal-code-rewards-v2/admission.json',
        ROOT/'reports/causal-code-rewards-v2/independent-reader.json',ROOT/'reports/causal-code-rewards-v2/observations.jsonl']
    for folder in ['verified-views-warmup-four-CPU-v2','execution-group-causal-four-CPU-v1']:additions.append(ROOT/'reports'/folder/'complete.json')
    manifest=json.loads((ROOT/'data/verified-code-views-v1/manifest.json').read_text())
    additions.extend(ROOT/'data/verified-code-views-v1'/name for name in manifest['files_sha256'])
    for path in additions:files[str(path)]=file_hash(path)
    for path in prior['pulse_sources'].values():files[path]=file_hash(path)
    guard=FileBindings(files);me=owner.resources.identity(os.getpid());limit=time.monotonic()+24*3600
    while not previous_ready(prior):
        guard.check()
        if time.monotonic()>limit:raise ValueError('Waiting lease expired without preempting the live original experiment')
        atomic_json(QUEUE,dict(status='waiting for exact preceding protected owner/workers and preserved handover or full completion',
            owner=me,predecessor=prior['owner'],warmup_epochs=4,local_RL_group_size=2,RL_max_new_tokens=512,
            production_updates=0,source_files_sha256=files,goal_achieved=False));time.sleep(5)
    files[str(PREVIOUS/'plan.json')]=file_hash(PREVIOUS/'plan.json');files[str(PREVIOUS/'restored.json')]=file_hash(PREVIOUS/'restored.json')
    if HANDOVER.exists():files[str(HANDOVER)]=file_hash(HANDOVER)
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);owner.idle_original()
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(used)!=4 or any(int(v)>128 for v in used):raise ValueError('All four original GPUs must be idle before ownership')
        if shutil.disk_usage(ROOT).free<RESERVE+52*MAX_STATE:raise ValueError('Complete private checkpoints and atomic writer reserve must fit')
        FileBindings(files).check(full=True);REPORTS.mkdir();OUTPUT.mkdir()
        plan=dict(initial_dense_settings=prior['initial_dense_settings'],actual_counts=999706442,warmup_epochs=4,
            warmup_learning_rate=1e-4,RL_learning_rate=1e-5,local_RL_group_size=2,RL_max_new_tokens=512,
            owner=me,waiter=owner.WAITER,source_files_sha256=files,expires_at_unix=time.time()+12*3600,
            own_initial_model_not_regressing_protected_checkpoint=True,development_experiment_not_final_candidate=True,pulse_sources=prior['pulse_sources'],
            broad_training_and_original_full_endpoints_still_required=True,genuine_human_feedback_training_still_required=True,
            automatic_retry=False,candidate_promoted=False,goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[])
        shutil.copyfile(ROOT/'data/verified-code-views-v1/tokenizer.json',OUTPUT/'tokenizer.json')
        driver=str(Path(__file__).resolve())
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned verified-code experiment interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        watcher=subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,
            stdout=(REPORTS/'watchdog.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        def run(mode):
            child=owner.launch(plan,['-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',driver,'--distributed',mode],mode+'-launcher','0,1,2,3')
            owner.wait_children(plan,[child])
        def assess_arm(arm):
            children=[owner.launch(plan,['-u',driver,'--assess',arm,'--gpu',str(gpu)],'assess-'+arm+'-'+str(gpu),str(gpu)) for gpu in range(4)]
            owner.wait_children(plan,children)
        try:
            owner.idle_original();os.kill(owner.WAITER['pid'],signal.SIGSTOP)
            deadline=time.monotonic()+5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in ('T','t'):
                if time.monotonic()>deadline:raise ValueError('Exact original idle waiter hold failed')
                time.sleep(.05)
            run('capacity');proofs=[json.loads((REPORTS/f'capacity-{rank}.json').read_text()) for rank in range(4)]
            for rank,p in enumerate(proofs):
                if (p['physical_GPU']!=rank or p['actual_counts']['actual_unique']!=999706442 or p['configured_maximum_new_tokens']!=512
                        or not p['RL_recovery']['actual_next_metrics_actions_weights_Adam_and_private_CPU_CUDA_RNG_exact']
                        or not p['actual_full_horizon_gradient_stress']['no_optimizer_commit']):raise ValueError('Every actual configured four-L40 capacity/recovery receipt required')
            atomic_json(REPORTS/'capacity.json',dict(passed=True,proofs=proofs,production_updates=0,goal_achieved=False))
            run('warmup')
            for arm in ['initial','warmup']:assess_arm(arm)
            run('RL')
            for arm in ['verified_solution','execution_hybrid']:assess_arm(arm)
            FileBindings(files).check(full=True)
            atomic_json(REPORTS/'training-complete.json',dict(status='Own exact-source warm-up and both controlled execution arms finished',
                matched_native_development_assessments_finished=True,independent_state_and_quality_readback_pending=True,
                full_broad_final_endpoints_and_genuine_RLHF_still_required=True,candidate_promoted=False,goal_achieved=False))
        except BaseException as error:atomic_json(REPORTS/'failure.json',dict(error=repr(error),automatic_retry=False,goal_achieved=False));raise
        finally:owner.restore_owned(plan,'Verified code-view/execution experiment ended; original work restored');watcher.wait(timeout=20)


if __name__=='__main__':
    configure();p=argparse.ArgumentParser(description=__doc__);p.add_argument('--distributed',choices=['capacity','warmup','RL']);p.add_argument('--watchdog',action='store_true');p.add_argument('--assess',choices=['initial','warmup','verified_solution','execution_hybrid']);p.add_argument('--gpu',type=int,choices=range(4));args=p.parse_args()
    if args.watchdog:owner.watchdog()
    elif args.distributed:distributed(args.distributed)
    elif args.assess:assess(args.assess,args.gpu)
    else:parent()
