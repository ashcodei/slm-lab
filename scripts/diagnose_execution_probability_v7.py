"""Disposable reconstruction of the real cached/full probability rejection.

The original numerical guard stays strict. Reconstruct from update128, stop
before attempt139commits, and compare the actual own weights with an FP64
equation reference. No production branch or final-quality claim is made.
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
REPORTS = ROOT/'reports/execution-probability-diagnostic-v7'
OUTPUT = ROOT/'checkpoints/execution-probability-diagnostic-v7'
QUEUE = ROOT/'reports/execution-probability-diagnostic-v7-queue.json'
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
    from slm_lab.execution_group_checked_host_v7 import CheckedHostExecutionGroup
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
        stage = CheckedHostExecutionGroup(model,optimizer,ROOT,method=arm,seed=957117,
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
        from slm_lab.execution_probability_diagnostic_v7 import run
        run(stage,plan,rank,device,REPORTS,OUTPUT,save,step)
        if (frozen != tensor_digest(base.named_parameters())
                or not torch.equal(critic,model.parts['critic'].weight)
                or initial == tensor_digest(model.parts.named_parameters())):
            raise ValueError('Actual actor must learn with exact own base and unused critic')
        guard.check(full=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU owner must hide every GPU')
    if any(path.exists() for path in [REPORTS,OUTPUT,QUEUE]):
        raise ValueError('Preserve earlier complete, partial or failed recovery attempts')
    import torch
    from slm_lab.execution_group_host_capacity_v6 import equal
    torch.set_num_threads(1)
    prior = json.loads((PREVIOUS/'plan.json').read_text())
    proof_path = ROOT/'reports/execution-group-checked-four-CPU-v7/complete.json'
    proof = json.loads(proof_path.read_text())
    if not proof['all_four_CPU_workers_passed'] or not proof['exact_baseline_scalar_trajectories_weights_and_Adam']:
        raise ValueError('Actual four-worker original-objective host transport baseline required')
    state_proof = ROOT/'reports/execution-step128-readback-v1/complete.json'
    readback = json.loads(state_proof.read_text())
    FileBindings(readback['source_files_sha256']).check(full=True)
    if (readback['actual_completed_updates_per_arm'] != 128
            or not readback['within_arm_weights_and_Adam_identical']):
        raise ValueError('Independent complete all-worker128state reconciliation required')
    tests = {'execution-probability-diagnostic-v7-driver-tests':2,'execution-recovery-semantics-v1-tests':1,'native-code-shards-v1-tests':2,'code-probability-numerical-reference-v1-tests':1}
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
        ROOT/'tests/test_execution_recovery_semantics_v1.py',ROOT/'tests/test_execution_probability_diagnostic_v7_driver.py',
        ROOT/'src/slm_lab/native_code_shards_v1.py',ROOT/'src/slm_lab/execution_group_checked_host_v7.py',
        ROOT/'src/slm_lab/execution_probability_diagnostic_v7.py',ROOT/'src/slm_lab/code_probability_numerical_reference_v1.py',
        ROOT/'tests/test_code_probability_numerical_reference_v1.py']
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
            owner=me,predecessor=prior['owner'],inherited_updates=128,new_planned_updates_per_arm=10,
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
            inherited_updates=128,new_planned_updates_per_arm=10,planned_updates_per_arm=173,
            RL_learning_rate=prior['RL_learning_rate'],local_RL_group_size=2,RL_max_new_tokens=512,
            pulse_sources=prior['pulse_sources'],owner=me,waiter=owner.WAITER,
            source_files_sha256=files,expires_at_unix=time.time()+2*3600,diagnostic_only=True,last_attempt139_never_committed=True,
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
            proofs=[json.loads((REPORTS/f'diagnostic-{rank}.json').read_text()) for rank in range(4)]
            if any(p['physical_GPU']!=rank or p['actual_counts']['actual_unique']!=999706442
                   or not p['diagnostic_step139_not_committed'] for rank,p in enumerate(proofs)):
                raise ValueError('Every actual own-policy diagnostic must complete without its rejected-attempt commit')
            atomic_json(REPORTS/'complete.json',dict(disposable_numerical_diagnostic_only=True,proofs=proofs,
                original128state_prefix_reconstruction_and_own_weight_FP64_reference_complete=True,
                full_horizon_training_capacity_and_quality_not_claimed=True,production_updates=0,goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS/'failure.json',dict(error=repr(error),automatic_retry=False,goal_achieved=False))
            raise
        finally:
            owner.restore_owned(plan,'Explicit failed-trial host recovery ended; original work restored')
            watcher.wait(timeout=20)


if __name__ == '__main__':
    configure()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watchdog',action='store_true')
    parser.add_argument('--distributed',choices=['capacity'])
    args=parser.parse_args()
    if args.watchdog:owner.watchdog()
    elif args.distributed:distributed(args.distributed)
    else:parent()
