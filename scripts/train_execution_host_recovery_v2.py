"""Explicit failed-trial recovery from the complete original 128-update state.

Never preempts a live owner. Keeps the original objective, samples, rewards,
whole prompts and optimizer profile; only gradient transport uses CPU/Gloo.
Fresh four-L40 next-update recovery and full-horizon capacity gate production.
"""
import argparse
from datetime import datetime, timezone, timedelta
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

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/execution-host-recovery-v2'
OUTPUT = ROOT/'checkpoints/execution-host-recovery-v2'
QUEUE = ROOT/'reports/execution-host-recovery-v2-queue.json'
PREVIOUS = ROOT/'reports/verified-views-execution-v1'
MAX_STATE = 256*1024**2
RESERVE = 26*1024**3
owner = legacy.owner


def configure():
    owner.REPORTS = REPORTS
    owner.resources.REPORTS = REPORTS
    legacy.REPORTS = REPORTS
    legacy.OUTPUT = OUTPUT


def previous_ready(prior):
    if owner.resources.same(prior['owner']):
        return False
    if any(owner.resources.same(p) for p in json.loads((PREVIOUS/'children.json').read_text())):
        return False
    if not (PREVIOUS/'restored.json').exists():
        raise ValueError('All preceding exact workers must end and original work must be restored')
    if not (PREVIOUS/'failure.json').exists():
        raise ValueError('Explicit actual terminal failure required; a slow or successful live trial is not restarted')
    return True


def policy_sources(plan, digest):
    return dict(run_plan_sha256=digest, origin_run_plan_sha256=plan['origin_plan_sha256'],
        origin_resume_receipt_sha256=plan['origin_resume_receipt_sha256'],
        causal_reward_admission_sha256=file_hash(ROOT/'reports/causal-code-rewards-v2/admission.json'))


def distributed(mode):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3' or int(os.environ['WORLD_SIZE']) != 4:
        raise ValueError('Exactly four owned original local GPUs required')
    rank = int(os.environ['RANK'])
    plan = owner.registered(mode+'-rank-'+str(rank))
    import copy
    import torch
    from torch import distributed as dist
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint, capture_rng
    from slm_lab.execution_group_host_reduce_v6 import HostReducedCausalExecutionGroup
    from slm_lab.execution_group_host_capacity_v6 import maximum_prompt, stress
    from slm_lab.execution_recovery_semantics_v1 import same_executed_update
    from slm_lab.fast_joint_inference_v2 import fresh
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1)
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.85,device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.get_device_name(device) != 'NVIDIA L40':
        raise ValueError('Only original local L40s are admitted')
    dist.init_process_group('gloo', timeout=timedelta(minutes=30))
    gradients = [dist.new_group(ranks,backend='gloo',timeout=timedelta(minutes=30)) for ranks in [[0,1],[2,3]]]
    controls = [dist.new_group(ranks,backend='gloo',timeout=timedelta(minutes=30)) for ranks in [[0,1],[2,3]]]
    started = time.monotonic()
    try:
        digest = file_hash(REPORTS/'plan.json')
        guard = FileBindings(plan['source_files_sha256'])
        base,_ = fresh(plan,device)
        model = OwnDenseActor(base)
        if model.counts()['actual_unique'] != 999706442:
            raise ValueError('Exact original counted full-size own policy required')
        parameters = [p for _,p in model.named_selected(('actor',))]
        optimizer = torch.optim.AdamW(parameters,lr=plan['RL_learning_rate'],betas=(.9,.95),weight_decay=.01,foreach=False)
        arm = 'verified_solution' if rank < 2 else 'execution_hybrid'
        stage = HostReducedCausalExecutionGroup(model,optimizer,ROOT,method=arm,seed=957117,
            group_size=2,max_new_tokens=512,world=2,rank=rank%2,
            gradient_group=gradients[rank//2],control_group=controls[rank//2])
        path = Path(plan['origin_private_states'][str(rank)]['path'])
        payload = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        stage.resume(payload,plan['origin_sources'])
        if stage.completed != 128:
            raise ValueError('Exact common all-worker128-state required')
        del payload
        sources = policy_sources(plan,digest)
        frozen = tensor_digest(base.named_parameters())
        critic = model.parts['critic'].weight.detach().clone()
        initial = tensor_digest(model.parts.named_parameters())
        def adam():
            return tensor_digest((str(i)+'.'+k,state[k]) for i,state in enumerate(optimizer.state.values())
                                 for k in ('step','exp_avg','exp_avg_sq'))
        def save(label):
            path = OUTPUT/f'{mode}-{label}-rank-{rank}.pt'
            atomic_checkpoint(path,stage.state(sources),MAX_STATE,RESERVE)
            proof = dict(rank=rank,path=str(path),sha256=file_hash(path),actual_completed=stage.completed)
            proofs = [None]*4
            dist.all_gather_object(proofs,proof)
            if rank == 0:
                atomic_json(REPORTS/f'{mode}-{label}-distributed-state.json',dict(all_four_private_states=proofs,
                    complete_distributed_state_requires_every_private_file=True,goal_achieved=False))
            return path
        def step():
            owner.owned(plan)
            return stage.update()
        if mode == 'capacity':
            first = step()
            saved_path = save('step129')
            expected = step()
            weights,moments,RNG = tensor_digest(model.parts.named_parameters()),adam(),capture_rng(device)
            saved = torch.load(saved_path,map_location='cpu',weights_only=True,mmap=True)
            bad = copy.deepcopy(saved)
            bad['private_replica_state']['rng']['cuda'] = torch.zeros(4,dtype=torch.uint8)
            versions = [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in model.parameters()]
            try:
                stage.resume(bad,sources)
            except ValueError:
                pass
            else:
                raise ValueError('Invalid private CUDA RNG must reject')
            if (versions != [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in model.parameters()]
                    or moments != adam() or not torch.equal(RNG['torch'],torch.get_rng_state())
                    or not torch.equal(RNG['cuda'],torch.cuda.get_rng_state(device))):
                raise ValueError('Bad restore changed weights, Adam or actual RNG')
            stage.resume(saved,sources)
            repeated = step()
            if (not same_executed_update(expected,repeated)
                    or weights != tensor_digest(model.parts.named_parameters()) or moments != adam()):
                raise ValueError('Actual next own actions/probabilities/rewards/weights/Adam must recover exactly')
            from slm_lab.execution_group_host_capacity_v6 import equal
            if not equal(RNG,capture_rng(device)):
                raise ValueError('All private Python/NumPy/Torch/actual CUDA RNG streams must recover')
            original = [json.loads(line) for line in (PREVIOUS/(arm+'-metrics.jsonl')).open()]
            for actual in [first,expected]:
                prior = original[actual['step']-1]
                if (actual['source_id'] != prior['source_id'] or actual['global_rewards'] != prior['global_rewards']
                        or any(x[k] != y[k] for x,y in zip(actual['trajectories'],prior['trajectories'])
                               for k in ['response_ids','actual_cached_behavior_logps','own_response'])):
                    raise ValueError('Restored original128state must reproduce the original next two actual behavior groups and rewards')
            maximum = maximum_prompt(stage)
            full = stress(stage,maximum['original_train_position'])
            atomic_json(REPORTS/f'capacity-{rank}.json',dict(physical_GPU=rank,actual_counts=model.counts(),
                actual_Adam_states=len(optimizer.state),inherited_updates=128,logical_disposable_new_updates=2,
                physical_disposable_optimizer_calls=3,exact_next_actions_rewards_weights_Adam_all_private_RNG=True,
                original_actual_step129_and130_behavior_and_rewards_reproduced=True,
                invalid_private_CUDA_RNG_rejected_before_mutation=True,maximum_original_prompt=maximum,
                actual_full_horizon_gradient_stress=full,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),seconds=time.monotonic()-started,
                production_updates=0,goal_achieved=False))
        else:
            journal = (REPORTS/(arm+'-metrics.jsonl')).open('x') if rank in (0,2) else None
            try:
                if journal:
                    for record in stage.metrics:
                        journal.write(json.dumps(record)+'\n')
                    journal.flush()
                while stage.completed < 173:
                    tick = time.monotonic()
                    record = step()
                    if journal:
                        journal.write(json.dumps(record)+'\n')
                        journal.flush()
                        atomic_json(REPORTS/(arm+'-progress.json'),dict(actual_updates=stage.completed,planned_updates=173,
                            inherited_updates=128,new_production_updates=stage.completed-128,
                            last_update_seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-started,
                            latest={k:v for k,v in record.items() if k!='trajectories'},goal_achieved=False))
                    if stage.completed % 32 == 0 or stage.completed == 173:
                        save(arm+'-'+str(stage.completed))
                if rank in (0,2):
                    atomic_checkpoint(OUTPUT/(arm+'-actor.pt'),model.export(sources),96*1024**2,RESERVE)
                    atomic_json(REPORTS/(arm+'-training-complete.json'),dict(actual_updates=173,inherited_updates=128,
                        new_production_updates=45,original_functions=173,actual_counts=model.counts(),
                        method=arm,genuine_human_preference_RLHF=False,goal_achieved=False))
            finally:
                if journal:
                    journal.close()
        if (frozen != tensor_digest(base.named_parameters())
                or not torch.equal(critic,model.parts['critic'].weight)
                or initial == tensor_digest(model.parts.named_parameters())):
            raise ValueError('Actual actor must learn with exact own base and unused critic')
        guard.check(full=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def assess(arm, gpu, suite):
    plan = legacy.registered('assess-'+arm+'-'+suite+'-'+str(gpu),gpu)
    import torch
    from slm_lab.fast_joint_inference_v2 import fresh
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.85,0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base,_ = fresh(plan,'cuda:0')
    model = OwnDenseActor(base)
    checkpoint = OUTPUT/(arm+'-actor.pt')
    payload = torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if not torch.equal(model.parts['critic'].weight.cpu(),payload['parts']['critic.weight']):
        raise ValueError('Separate unused critic must match original own policy')
    model.import_export(payload,expected_sources=policy_sources(plan,file_hash(REPORTS/'plan.json')))
    del payload
    model.eval()
    metadata = dict(tokenizer_sha256=plan['initial_dense_settings']['tokenizer_sha256'],training_options=dict(image_size=448))
    folder = REPORTS/arm
    folder.mkdir(exist_ok=True)
    guard = FileBindings(plan['source_files_sha256'])
    def evaluate(module,name,data=None,**kwargs):
        saved = module.load_checkpoint
        module.load_checkpoint = lambda *a,**kw:(model.policy,metadata)
        try:
            owner.owned(plan)
            if data is None:
                module.evaluate(checkpoint,folder/name,device='cuda:0',**kwargs)
            else:
                module.evaluate(checkpoint,data,folder/name,device='cuda:0',**kwargs)
        finally:
            module.load_checkpoint = saved
    previous_threads = torch.set_num_threads
    torch.set_num_threads = lambda count:None
    try:
        with inference_cache():
            if suite == 'code':
                from slm_lab.native_code_shards_v1 import evaluate_shard
                for frame in ['completion','instruction']:
                    evaluate_shard(model.policy,metadata,checkpoint,folder,rank=gpu,world=4,frame=frame)
            elif gpu == 0:
                from slm_lab import knowledge_evaluation
                evaluate(knowledge_evaluation,'knowledge.json',Path(plan['pulse_sources']['knowledge']),max_examples=56,max_new_tokens=32)
            elif gpu == 1:
                from slm_lab import math_evaluation
                evaluate(math_evaluation,'math.json',ROOT/'data/math-reasoning-v2/test.jsonl',max_examples=16,max_new_tokens=256)
            elif gpu == 2:
                from slm_lab import ocr_evaluation,object_evaluation
                evaluate(ocr_evaluation,'ocr.json',Path(plan['pulse_sources']['ocr']),max_examples=32,max_new_tokens=128,image_size=448)
                evaluate(object_evaluation,'objects.json',Path(plan['pulse_sources']['objects']),max_examples=32,max_new_tokens=32)
            else:
                from slm_lab import vision_evaluation
                evaluate(vision_evaluation,'vision.json',Path(plan['pulse_sources']['vision']),max_examples=32,image_size=448)
    finally:
        torch.set_num_threads = previous_threads
    guard.check(full=True)
    atomic_json(folder/f'assessment-{suite}-{gpu}-complete.json',dict(physical_GPU=gpu,
        actual_counts=model.counts(),checkpoint_sha256=file_hash(checkpoint),
        native_development_comparison_not_final_independent_benchmark=True,goal_achieved=False))


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU owner must hide every GPU')
    if any(path.exists() for path in [REPORTS,OUTPUT,QUEUE]):
        raise ValueError('Preserve earlier complete, partial or failed recovery attempts')
    import torch
    from slm_lab.execution_group_host_capacity_v6 import equal
    torch.set_num_threads(1)
    prior = json.loads((PREVIOUS/'plan.json').read_text())
    proof_path = ROOT/'reports/execution-group-host-four-CPU-v6/complete.json'
    proof = json.loads(proof_path.read_text())
    if not proof['all_four_CPU_workers_passed'] or not proof['exact_baseline_scalar_trajectories_weights_and_Adam']:
        raise ValueError('Actual four-worker original-objective host transport baseline required')
    state_proof = ROOT/'reports/execution-step128-readback-v1/complete.json'
    readback = json.loads(state_proof.read_text())
    FileBindings(readback['source_files_sha256']).check(full=True)
    if (readback['actual_completed_updates_per_arm'] != 128
            or not readback['within_arm_weights_and_Adam_identical']):
        raise ValueError('Independent complete all-worker128state reconciliation required')
    tests = {'execution-host-recovery-v2-driver-tests':2,'execution-recovery-semantics-v1-tests':1,'native-code-shards-v1-tests':2}
    for name,count in tests.items():
        if str(count)+' passed' not in (ROOT/'reports'/(name+'.log')).read_text():
            raise ValueError('Actual resource, semantic replay and complete native code reconciliation tests required')
    files = dict(prior['source_files_sha256'])
    files.update(readback['source_files_sha256'])
    for worker in proof['proofs']:
        FileBindings(worker['source_files_sha256']).check(full=True)
        files.update(worker['source_files_sha256'])
    additions = [Path(__file__),proof_path,state_proof,ROOT/'src/slm_lab/execution_group_host_reduce_v6.py',
        ROOT/'src/slm_lab/execution_group_host_capacity_v6.py',ROOT/'src/slm_lab/execution_recovery_semantics_v1.py',
        ROOT/'tests/test_execution_recovery_semantics_v1.py',ROOT/'tests/test_execution_host_recovery_v2_driver.py',
        ROOT/'src/slm_lab/native_code_shards_v1.py']
    for name in tests:
        additions.append(ROOT/'reports'/(name+'.log'))
    for p in additions:
        files[str(p)] = file_hash(p)
    guard = FileBindings(files)
    me = owner.resources.identity(os.getpid())
    deadline = time.monotonic()+12*3600
    while not previous_ready(prior):
        guard.check()
        if time.monotonic() > deadline:
            raise ValueError('Recovery waiting lease expired without preempting the preceding owner')
        atomic_json(QUEUE,dict(status='waiting for actual failed owner/workers to end and original work to restore',
            owner=me,predecessor=prior['owner'],inherited_updates=128,new_planned_updates_per_arm=45,
            source_files_sha256=files,production_updates=0,goal_achieved=False))
        time.sleep(5)
    origin_digest = file_hash(PREVIOUS/'plan.json')
    origin_sources = dict(run_plan_sha256=origin_digest,
        own_initial_dense_sha256=prior['initial_dense_settings']['initializer_sha256'],
        verified_views_admission_sha256=file_hash(ROOT/'data/verified-code-views-v1/admission.json'),
        causal_reward_admission_sha256=file_hash(ROOT/'reports/causal-code-rewards-v2/admission.json'))
    receipt_path = PREVIOUS/'RL-verified_solution-128-distributed-state.json'
    receipt = json.loads(receipt_path.read_text())
    private = {str(row['rank']):row for row in receipt['all_four_private_states']}
    if set(private) != {'0','1','2','3'}:
        raise ValueError('All four exact original private states are required')
    for row in private.values():
        path = Path(row['path'])
        if file_hash(path) != row['sha256']:
            raise ValueError('Exact source-bound common128-state bytes required')
        raw = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        if raw['private_replica_state']['model']['sources'] != origin_sources:
            raise ValueError('Every state must carry the original live trial model/source bindings')
        del raw
        files[str(path)] = row['sha256']
    for p in [receipt_path,PREVIOUS/'plan.json',PREVIOUS/'failure.json',PREVIOUS/'restored.json']:
        files[str(p)] = file_hash(p)
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        owner.idle_original()
        used = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(used) != 4 or any(int(v) > 128 for v in used):
            raise ValueError('Every preceding GPU context must be gone before recovery ownership')
        if shutil.disk_usage(ROOT).free < RESERVE+24*MAX_STATE+1024**3:
            raise ValueError('Complete capacity/resume/policy snapshots and atomic reserve required')
        guard = FileBindings(files)
        guard.check(full=True)
        REPORTS.mkdir()
        OUTPUT.mkdir()
        plan = dict(initial_dense_settings=prior['initial_dense_settings'],actual_counts=999706442,
            origin_plan=str(PREVIOUS/'plan.json'),origin_plan_sha256=origin_digest,
            origin_resume_receipt=str(receipt_path),origin_resume_receipt_sha256=file_hash(receipt_path),
            origin_private_states=private,origin_sources=origin_sources,
            inherited_updates=128,new_planned_updates_per_arm=45,planned_updates_per_arm=173,
            RL_learning_rate=prior['RL_learning_rate'],local_RL_group_size=2,RL_max_new_tokens=512,
            pulse_sources=prior['pulse_sources'],owner=me,waiter=owner.WAITER,
            source_files_sha256=files,expires_at_unix=time.time()+12*3600,
            gradient_transport='explicit_FP32_CPU_Gloo_SUM_then_copy_to_original_GPU',
            original_rewards_objective_sampler_optimizer_and_global_denominators_unchanged=True,
            original_partial_uncheckpointed_branch_preserved_not_relabelled=True,
            automatic_retry=False,candidate_promoted=False,goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan)
        atomic_json(REPORTS/'children.json',[])
        driver = str(Path(__file__).resolve())
        def interrupted(signum,frame):
            raise KeyboardInterrupt('Owned explicit failed-trial recovery interrupted')
        signal.signal(signal.SIGTERM,interrupted)
        signal.signal(signal.SIGINT,interrupted)
        watcher = subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,
            stdout=(REPORTS/'watchdog.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        def run(mode):
            child = owner.launch(plan,['-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                driver,'--distributed',mode],mode+'-launcher','0,1,2,3')
            owner.wait_children(plan,[child])
        try:
            owner.idle_original()
            os.kill(owner.WAITER['pid'],signal.SIGSTOP)
            stop_deadline = time.monotonic()+5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in ('T','t'):
                if time.monotonic() > stop_deadline:
                    raise ValueError('Exact idle original waiter hold failed')
                time.sleep(.05)
            run('capacity')
            proofs = [json.loads((REPORTS/f'capacity-{rank}.json').read_text()) for rank in range(4)]
            if any(p['physical_GPU'] != rank or p['actual_counts']['actual_unique'] != 999706442
                   or p['actual_Adam_states'] != 561
                   or not p['exact_next_actions_rewards_weights_Adam_all_private_RNG']
                   or not p['original_actual_step129_and130_behavior_and_rewards_reproduced']
                   for rank,p in enumerate(proofs)):
                raise ValueError('Every original full-profile four-L40 host-recovery admission must pass')
            atomic_json(REPORTS/'capacity.json',dict(passed=True,proofs=proofs,production_updates=0,goal_achieved=False))
            run('train')
            for arm in ['verified_solution','execution_hybrid']:
                for suite in ['code','broad']:
                    children = [owner.launch(plan,['-u',driver,'--assess',arm,'--gpu',str(gpu),'--suite',suite],
                        'assess-'+arm+'-'+suite+'-'+str(gpu),str(gpu)) for gpu in range(4)]
                    owner.wait_children(plan,children)
                from slm_lab.native_code_shards_v1 import reconcile
                for frame in ['completion','instruction']:
                    reconcile([REPORTS/arm/f'code-{frame}-shard-{rank}.json' for rank in range(4)],REPORTS/arm/f'code-{frame}.json')
            guard.check(full=True)
            atomic_json(REPORTS/'training-complete.json',dict(matched_native_development_assessments_finished=True,
                inherited_updates_per_arm=128,new_production_updates_per_arm=45,total_updates_per_arm=173,
                original_warmup88_and_original_prefix128_provenance_retained=True,
                held_quality_and_independent_final_state_readback_pending=True,candidate_promoted=False,goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS/'failure.json',dict(error=repr(error),automatic_retry=False,goal_achieved=False))
            raise
        finally:
            owner.restore_owned(plan,'Explicit failed-trial host recovery ended; original work restored')
            watcher.wait(timeout=20)


if __name__ == '__main__':
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watchdog',action='store_true')
    parser.add_argument('--distributed',choices=['capacity','train'])
    parser.add_argument('--assess',choices=['verified_solution','execution_hybrid'])
    parser.add_argument('--gpu',type=int,choices=range(4))
    parser.add_argument('--suite',choices=['code','broad'])
    args = parser.parse_args()
    if args.watchdog:
        owner.watchdog()
    elif args.distributed:
        distributed(args.distributed)
    elif args.assess:
        assess(args.assess,args.gpu,args.suite)
    else:
        parent()
