"""Matched full human DPO/control after successful complete broader foundation.

Sequential four-GPU arms avoid leaving the cheaper control waiting on DPO.
Each arm consumes all public human-ranked training pairs exactly once. Fresh
full-profile capacity and complete private-state recovery gate production.
"""
import argparse
from datetime import datetime, timezone, timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import train_whole_human_SFT_v1 as legacy
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.function_repair_GPU_contract_v2 import study_released, public_ready
from slm_lab.corpus import file_hash
from slm_lab.public_pretraining_lineage_tests_v1 import bind_tests
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/phase-human-preference-v8'
OUTPUT = ROOT/'checkpoints/phase-human-preference-v8'
QUEUE = ROOT/'reports/phase-human-preference-v8-assessment-recovery-gate-v1-queue.json'
PREVIOUS = ROOT/'reports/public-pretraining-v1'
PREVIOUS_QUEUE = ROOT/'reports/public-pretraining-recovery-v1-queue.json'
MAX_STATE = 512*1024**2
RESERVE = 26*1024**3
import own_training_gpu_lease_v2 as owner
legacy.owner=owner


def configure():
    owner.REPORTS = REPORTS
    owner.resources.REPORTS = REPORTS
    legacy.REPORTS = REPORTS
    legacy.OUTPUT = OUTPUT


def previous_ready(identity):
    if not study_released(ROOT):return False
    return public_ready(ROOT,identity)



def distributed(mode, arm):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3' or int(os.environ['WORLD_SIZE']) != 4:
        raise ValueError('Four owned original local GPUs required')
    rank = int(os.environ['RANK'])
    plan = owner.registered(mode+'-'+arm+'-rank-'+str(rank))
    import copy
    import random
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint, capture_rng
    from slm_lab.dense_human_preference_stage_v1 import actual_exposures
    from slm_lab.parallel_human_preference_v2 import ParallelHumanPreferenceStage, local_gradients
    from slm_lab.phase_human_policy_v3 import fresh, options, sources, export
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1)
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.85, device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.get_device_name(device) != 'NVIDIA L40':
        raise ValueError('Only original local L40s may be used')
    dist.init_process_group('nccl', timeout=timedelta(minutes=40))
    control = dist.new_group(backend='gloo', timeout=timedelta(minutes=40))
    started = time.monotonic()
    try:
        digest = file_hash(REPORTS/'plan.json')
        guard = FileBindings(plan['source_files_sha256'])
        model, _ = fresh(plan, device)
        data = WholeHumanData(ROOT/'data/whole-human-alignment-v2', 'preferences', 'train', 'optimization')
        setting = options(plan, arm)
        parameters = [p for _,p in model.named_selected(('actor',))]
        optimizer = torch.optim.AdamW(parameters, lr=setting['learning_rate'],
            betas=tuple(setting['betas']), weight_decay=setting['weight_decay'], foreach=False)
        amp = lambda:torch.autocast('cuda', dtype=torch.bfloat16)
        stage = ParallelHumanPreferenceStage(model, optimizer, data, setting, sources(plan, digest),
            device, amp, world=4, rank=rank, gradient_group=dist.group.WORLD, control_group=control)
        random.seed(setting['seed']+rank)
        np.random.seed(setting['seed']+rank)
        torch.manual_seed(setting['seed']+rank)
        frozen = tensor_digest(model.base.named_parameters())
        critic = model.parts['critic'].weight.detach().clone()
        initial = tensor_digest(model.parts.named_parameters())
        def adam():
            return tensor_digest((str(i)+'.'+k, state[k]) for i,state in enumerate(optimizer.state.values())
                                 for k in ('step','exp_avg','exp_avg_sq'))
        def save_private(label):
            # Keep the last committed all-worker generation while atomically
            # writing the next one. A partial new save cannot erase it.
            generation = math.ceil(stage.completed/512) if label == 'resume' else None
            stem = 'resume-slot-'+str(generation % 2) if generation is not None else label
            path = OUTPUT/f'{mode}-{arm}-{stem}-rank-{rank}.pt'
            atomic_checkpoint(path, stage.state(), MAX_STATE, RESERVE)
            proof = dict(rank=rank, path=str(path), sha256=file_hash(path), actual_completed=stage.completed)
            proofs = [None]*4
            dist.all_gather_object(proofs, proof, group=control)
            if rank == 0:
                atomic_json(REPORTS/f'{mode}-{arm}-{label}-distributed-state.json', dict(
                    all_four_private_states=proofs, complete_distributed_state_requires_every_private_file=True,
                    completed=stage.completed, committed_generation=generation,
                    previous_generation_files_retained=label == 'resume',
                    goal_achieved=False))
            return path
        def step():
            owner.owned(plan)
            return stage.update()
        extremes = []
        if mode == 'capacity':
            first = step()
            saved_path = save_private('step1')
            def draws():
                return [random.random(),float(np.random.random()),float(torch.rand(())),
                        float(torch.rand((),device=device))]
            expected_draws = draws()
            expected = step()
            learned, moments = tensor_digest(model.parts.named_parameters()), adam()
            payload = torch.load(saved_path, map_location='cpu', weights_only=True, mmap=True)
            bad = copy.deepcopy(payload)
            bad['core']['core']['rng']['cuda'] = torch.zeros(4, dtype=torch.uint8)
            RNG = capture_rng(device)
            versions = [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in model.parameters()]
            try:
                stage.resume(bad)
            except ValueError:
                pass
            else:
                raise ValueError('Invalid private CUDA RNG must reject')
            if (versions != [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in model.parameters()]
                    or moments != adam() or not torch.equal(RNG['torch'],torch.get_rng_state())
                    or not torch.equal(RNG['cuda'],torch.cuda.get_rng_state(device))):
                raise ValueError('Invalid recovery mutated policy, Adam or actual RNG')
            stage.resume(payload)
            if (draws() != expected_draws or step() != expected
                    or learned != tensor_digest(model.parts.named_parameters()) or moments != adam()):
                raise ValueError('Actual next global source update, weights, Adam and private RNG must recover exactly')
            save_private('step2')
            for positions in plan['capacity_original_pair_batches']:
                owner.owned(plan)
                optimizer.zero_grad(set_to_none=True)
                rows = local_gradients(model, data, positions, rank=rank, world=4,
                    DPO_weight=setting['DPO_weight'], beta=setting['DPO_beta'],
                    chosen_CE_weight=setting['chosen_CE_weight'], device=device, amp=amp)
                if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters)
                        or any(p.grad is not None for p in stage.fixed)):
                    raise ValueError('Actual longest whole responses need finite actor-only gradients on every replica')
                extremes.append(dict(global_original_positions=positions, actual_local_complete_pair_records=rows,
                    **actual_exposures(data, positions, setting), backward_only_no_optimizer_commit=True))
                optimizer.zero_grad(set_to_none=True)
        else:
            if rank == 0 and arm == 'chosen_CE':
                atomic_checkpoint(OUTPUT/'initial-actor.pt', export(model, plan, digest, 'initial', 0), 96*1024**2, RESERVE)
            dist.barrier(group=control)
            journal = (REPORTS/(arm+'-metrics.jsonl')).open('x') if rank == 0 else None
            try:
                while stage.completed < setting['steps']:
                    tick = time.monotonic()
                    record = step()
                    if journal:
                        journal.write(json.dumps(record)+'\n')
                        journal.flush()
                        if stage.completed == 1 or stage.completed % 25 == 0 or stage.completed == setting['steps']:
                            atomic_json(REPORTS/(arm+'-progress.json'), dict(actual_updates=stage.completed,
                                planned_updates=setting['steps'], latest=record,
                                last_update_seconds=time.monotonic()-tick, elapsed_seconds=time.monotonic()-started,
                                goal_achieved=False))
                    if stage.completed % 512 == 0 or stage.completed == setting['steps']:
                        save_private('resume')
                if rank == 0:
                    atomic_checkpoint(OUTPUT/(arm+'-actor.pt'), export(model, plan, digest, arm, stage.completed),
                                      96*1024**2, RESERVE)
                    atomic_json(REPORTS/(arm+'-training-complete.json'), dict(actual_updates=stage.completed,
                        human_train_pairs=len(data), all_original_human_pairs_once=True,
                        fresh_human_judgments_on_own_outputs=False, held_quality_pending=True, goal_achieved=False))
            finally:
                if journal:
                    journal.close()
        if (frozen != tensor_digest(model.base.named_parameters())
                or not torch.equal(critic, model.parts['critic'].weight)
                or initial == tensor_digest(model.parts.named_parameters())):
            raise ValueError('Policy must learn with exact own frozen foundation and unused critic')
        if mode == 'capacity':
            atomic_json(REPORTS/f'capacity-{arm}-{rank}.json', dict(physical_GPU=rank,
                actual_counts=model.counts(), actual_Adam_states=len(optimizer.state),
                actual_logical_updates=2, physical_optimizer_calls=3,
                exact_next_global_update_weights_Adam_private_RNG=True,
                invalid_private_CUDA_RNG_rejected_before_mutation=True,
                complete_source_extremes=extremes,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                seconds=time.monotonic()-started, production_updates=0, goal_achieved=False))
        guard.check(full=True)
        dist.barrier(group=control)
    finally:
        dist.destroy_process_group()


def assess(arm, gpu, suite):
    plan = legacy.registered('assess-'+arm+'-'+suite+'-'+str(gpu), gpu)
    import torch
    from slm_lab.phase_human_policy_v3 import load
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.85, 0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = OUTPUT/(arm+'-actor.pt')
    model, metadata = load(checkpoint, REPORTS/'plan.json', arm,
        expected_checkpoint_sha256=file_hash(checkpoint),
        expected_plan_sha256=file_hash(REPORTS/'plan.json'), device='cuda:0')
    guard = FileBindings(plan['source_files_sha256'])
    folder = REPORTS/arm
    folder.mkdir(exist_ok=True)
    if suite == 'preferences':
        from slm_lab.held_human_shards_v1 import evaluate_shard
        from slm_lab.whole_human_data_v2 import WholeHumanData
        data = WholeHumanData(ROOT/'data/whole-human-alignment-v2', 'preferences', 'valid', 'evaluation')
        provenance = dict(checkpoint_sha256=file_hash(checkpoint), run_plan_sha256=file_hash(REPORTS/'plan.json'),
            human_manifest_sha256=plan['human_manifest_sha256'], frozen_dense_sha256=plan['frozen_dense_sha256'],
            inference_device_type='cuda', inference_precision='float32')
        evaluate_shard(model, data, folder/f'held-preferences-shard-{gpu}.json', rank=gpu, world=4,
            provenance=provenance, beta=plan['training_options']['DPO_beta'], checked=lambda:owner.owned(plan))
    else:
        def evaluate(module, name, data=None, **kwargs):
            saved = module.load_checkpoint
            module.load_checkpoint = lambda *a,**kw:(model.policy, metadata)
            try:
                owner.owned(plan)
                if data is None:
                    module.evaluate(checkpoint, folder/name, device='cuda:0', **kwargs)
                else:
                    module.evaluate(checkpoint, data, folder/name, device='cuda:0', **kwargs)
            finally:
                module.load_checkpoint = saved
        previous_threads = torch.set_num_threads
        torch.set_num_threads = lambda count:None
        try:
            with inference_cache():
                if suite == 'code':
                    from slm_lab.native_code_shards_v1 import evaluate_shard
                    for frame in ['completion', 'instruction']:
                        evaluate_shard(model.policy, metadata, checkpoint, folder, rank=gpu, world=4, frame=frame)
                elif gpu == 0:
                    from slm_lab import knowledge_evaluation
                    evaluate(knowledge_evaluation, 'knowledge.json', Path(plan['pulse_sources']['knowledge']),
                             max_examples=56, max_new_tokens=32)
                elif gpu == 1:
                    from slm_lab import math_evaluation
                    evaluate(math_evaluation, 'math.json', ROOT/'data/math-reasoning-v2/test.jsonl',
                             max_examples=16, max_new_tokens=256)
                elif gpu == 2:
                    from slm_lab import ocr_evaluation, object_evaluation
                    evaluate(ocr_evaluation, 'ocr.json', Path(plan['pulse_sources']['ocr']),
                             max_examples=32, max_new_tokens=128, image_size=448)
                    evaluate(object_evaluation, 'objects.json', Path(plan['pulse_sources']['objects']),
                             max_examples=32, max_new_tokens=32)
                else:
                    from slm_lab import vision_evaluation
                    evaluate(vision_evaluation, 'vision.json', Path(plan['pulse_sources']['vision']),
                             max_examples=32, image_size=448)
        finally:
            torch.set_num_threads = previous_threads
    guard.check(full=True)
    atomic_json(folder/f'assessment-{suite}-{gpu}-complete.json', dict(physical_GPU=gpu,
        actual_counts=model.counts(), checkpoint_sha256=file_hash(checkpoint),
        development_comparison_not_final_independent_benchmark=True, goal_achieved=False))


def parent():
    if '16 passed' not in (ROOT/'reports/public-pretraining-lineage-drivers-v1-final2-tests.log').read_text():
        raise ValueError('Actual complete revised public-pretraining lineage and every original benchmark endpoint must pass')

    if '24 passed' not in (ROOT/'reports/terminal-waiter-pipeline-drivers-v1-tests.log').read_text() or '3 passed' not in (ROOT/'reports/terminal-waiter-gpu-lease-v2-tests.log').read_text():raise ValueError('Actual revised pipeline and terminal-waiter kernel-lease controls required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Waiting owner must hide every GPU')
    if any(p.exists() for p in [REPORTS, OUTPUT, QUEUE]):
        raise ValueError('Preserve every queued, failed or completed attempt')
    tests = {'parallel-human-preference-v2-tests-v2':4, 'public-pretraining-human-policy-v3-tests':2,
             'held-human-shards-v1-tests':2, 'state-code-first-training-drivers-v1-final-tests':11, 'own-tokenizer-package-v1-tests':2}
    for name, count in tests.items():
        if str(count)+' passed' not in (ROOT/'reports'/(name+'.log')).read_text():
            raise ValueError('Actual global gradients, source loading, held reconciliation and owner gates required')
    proof_path = ROOT/'reports/parallel-human-preference-four-CPU-v2/complete.json'
    proof = json.loads(proof_path.read_text())
    if not proof['all_four_CPU_workers_passed']:
        raise ValueError('Actual four-worker original-human source recovery proof required')
    previous = json.loads(PREVIOUS_QUEUE.read_text())
    files = dict(previous['source_files_sha256'])
    for worker in proof['proofs']:
        FileBindings(worker['source_files_sha256']).check(full=True)
        files.update(worker['source_files_sha256'])
    additions = [Path(__file__), ROOT/'src/slm_lab/function_repair_GPU_contract_v2.py', ROOT/'tests/test_assessment_recovery_lineage_v1.py', ROOT/'reports/assessment-recovery-lineage-v1-tests.log', proof_path, ROOT/'src/slm_lab/parallel_human_preference_v2.py',
        ROOT/'src/slm_lab/phase_human_policy_v3.py', ROOT/'src/slm_lab/held_human_shards_v1.py',
        ROOT/'src/slm_lab/dense_human_preference_evaluation_v1.py', ROOT/'src/slm_lab/native_code_shards_v1.py',
        ROOT/'data/whole-human-alignment-v2/admission.json', ROOT/'data/whole-human-alignment-v2/manifest.json',
        ROOT/'tests/test_parallel_human_preference_v2.py', ROOT/'tests/test_public_pretraining_human_policy_v3.py',
        ROOT/'tests/test_held_human_shards_v1.py',ROOT/'src/slm_lab/own_tokenizer_package_v1.py',
        ROOT/'tests/test_own_tokenizer_package_v1.py', ROOT/'tests/test_phase_human_preference_v7_driver.py']
    for name in tests:
        additions.append(ROOT/'reports'/(name+'.log'))
    for p in additions:
        files[str(p)] = file_hash(p)
    me = owner.resources.identity(os.getpid())
    bind_tests(files,ROOT);guard = FileBindings(files)
    deadline = time.monotonic()+7*24*3600
    while not previous_ready(previous['owner']):
        guard.check()
        if time.monotonic() > deadline:
            raise ValueError('Waiting lease expired without preempting or restarting the foundation')
        atomic_json(QUEUE, dict(status='waiting for exact complete code-first owner, source epoch, native assessments and restoration',
            owner=me, predecessor=previous['owner'], source_files_sha256=files,
            human_train_pairs=85786, global_pairs_per_update=16, planned_updates_per_arm=5362,
            arms=['chosen_CE','human_DPO'], production_updates=0,
            fresh_four_L40_capacity_still_required=True, automatic_retry=False, goal_achieved=False))
        time.sleep(5)
    import torch
    from slm_lab.public_pretraining_native_v1 import load as load_foundation
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1)
    source_plan = PREVIOUS/'plan.json'
    checkpoint = ROOT/'checkpoints/public-pretraining-v1/trained/model.pt'
    inputs = dict(checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
                  plan=str(source_plan), plan_sha256=file_hash(source_plan))
    foundation_plan = json.loads(source_plan.read_text())
    base, _ = load_foundation(checkpoint, source_plan, 'trained',
        expected_checkpoint_sha256=inputs['checkpoint_sha256'],
        expected_plan_sha256=inputs['plan_sha256'], device='cpu')
    model = OwnDenseActor(base)
    data = WholeHumanData(ROOT/'data/whole-human-alignment-v2', 'preferences', 'train', 'optimization')
    setting = dict(seed=959117, epochs=1, steps=math.ceil(len(data)/16), microbatch=1, accumulation=16,
        learning_rate=3e-6, betas=[.9,.95], weight_decay=.01, gradient_clip=1., DPO_beta=.1, chosen_CE_weight=.1)
    extremes = []
    for field in ['chosen','rejected']:
        for measure in ['length','targets']:
            order = sorted(range(len(data)), key=lambda p:(data.rows[p]['native'][field]['length']-
                (data.rows[p]['native'][field]['target_start'] if measure == 'targets' else 0),p), reverse=True)
            if order[:16] not in extremes:
                extremes.append(order[:16])
    for path in [source_plan, checkpoint, PREVIOUS/'training-complete.json', ROOT/'reports/public-pretraining-assessment-recovery-v1/complete.json', ROOT/'reports/public-pretraining-assessment-recovery-v1/restored.json', ROOT/'reports/public-pretraining-complete-epoch-value-readback-CPU-v2/complete.json']:
        files[str(path)] = file_hash(path)
    files.update(owner.lease_sources())
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        owner.idle_original()
        used = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(used) != 4 or any(int(v) > 128 for v in used):
            raise ValueError('Every original GPU must be idle before taking ownership')
        if shutil.disk_usage(ROOT).free < RESERVE+24*MAX_STATE+1024**3:
            raise ValueError('Complete private states, policies and atomic reserve must fit')
        bind_tests(files,ROOT);guard = FileBindings(files)
        guard.check(full=True)
        REPORTS.mkdir()
        OUTPUT.mkdir()
        from slm_lab.own_tokenizer_package_v1 import ensure
        tokenizer_package = ensure(ROOT/'data/tokenizer-corpus/tokenizer.json',OUTPUT,
            expected_sha256=foundation_plan['tokenizer_sha256'],expected_vocab_size=model.config.vocab_size)
        plan = dict(tokenizer_package=tokenizer_package,foundation_inputs=inputs, actual_dense_counts=base.counts(),
            frozen_dense_sha256=tensor_digest(base.named_parameters()), actor_settings=model.settings,
            actual_counts=model.counts(), initial_actor_parts_sha256=tensor_digest(model.parts.named_parameters()),
            human_train_pairs=len(data), human_admission_sha256=file_hash(ROOT/'data/whole-human-alignment-v2/admission.json'),
            human_manifest_sha256=file_hash(ROOT/'data/whole-human-alignment-v2/manifest.json'),
            training_options=setting, capacity_original_pair_batches=extremes,
            pulse_sources=foundation_plan['pulse_sources'], tokenizer_sha256=foundation_plan['tokenizer_sha256'],
            owner=me, waiter=owner.WAITER,terminal_waiter_receipt_sha256=owner.terminal_receipt_sha256(), source_files_sha256=files, expires_at_unix=time.time()+72*3600,
            arms=['chosen_CE','human_DPO'], each_arm_uses_all_four_GPUs_in_sequence=True,
            training_precision='bf16_autocast_with_FP32_policy_and_Adam', native_assessment_precision_by_suite=dict(code='bf16_autocast',broad='bf16_autocast',preferences='float32'),
            all_original_human_training_pairs_once_per_arm=True, rounded_repeats=0,
            reference_is_same_counted_frozen_own_foundation=True, human_ranked_other_model_responses=True,
            fresh_own_output_human_judgments=False, automatic_retry=False, candidate_promoted=False, goal_achieved=False)
        del model, base, data
        atomic_json(REPORTS/'plan.json', plan)
        atomic_json(REPORTS/'children.json', [])
        driver = str(Path(__file__).resolve())
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Owned human preference experiment interrupted')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        watcher = subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,
            stdout=(REPORTS/'watchdog.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        def run(mode, arm):
            child = owner.launch(plan, ['-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                driver,'--distributed',mode,'--arm',arm], mode+'-'+arm+'-launcher','0,1,2,3')
            owner.wait_children(plan, [child])
        try:
            owner.hold_original(plan)
            for arm in plan['arms']:
                run('capacity', arm)
                proofs = [json.loads((REPORTS/f'capacity-{arm}-{rank}.json').read_text()) for rank in range(4)]
                if any(p['physical_GPU'] != rank or p['actual_counts'] != plan['actual_counts']
                       or p['actual_Adam_states'] != 561
                       or not p['exact_next_global_update_weights_Adam_private_RNG'] for rank,p in enumerate(proofs)):
                    raise ValueError('Every full-size configured four-GPU human capacity control must pass')
            atomic_json(REPORTS/'capacity.json',dict(passed=True, arms=plan['arms'], production_updates=0,goal_achieved=False))
            for arm in plan['arms']:
                run('train', arm)
            for arm in ['initial', *plan['arms']]:
                for suite in ['code','broad','preferences']:
                    children = [owner.launch(plan,['-u',driver,'--assess',arm,'--suite',suite,'--gpu',str(gpu)],
                        'assess-'+arm+'-'+suite+'-'+str(gpu),str(gpu)) for gpu in range(4)]
                    owner.wait_children(plan, children)
                from slm_lab.native_code_shards_v1 import reconcile as code_reconcile
                from slm_lab.held_human_shards_v1 import reconcile as human_reconcile
                for frame in ['completion','instruction']:
                    code_reconcile([REPORTS/arm/f'code-{frame}-shard-{rank}.json' for rank in range(4)],REPORTS/arm/f'code-{frame}.json')
                held = WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','valid','evaluation')
                human_reconcile(held,[REPORTS/arm/f'held-preferences-shard-{rank}.json' for rank in range(4)],REPORTS/arm/'held-preferences.json')
            guard.check(full=True)
            atomic_json(REPORTS/'complete.json',dict(all_original_human_train_pairs_once_each_arm=True,
                all_held_human_pairs_and_native_development_assessments_complete=True,
                actual_source_epoch_updates_per_arm=setting['steps'],
                own_output_human_judgments_full_final_benchmarks_and_independent_state_quality_readback_pending=True,
                candidate_promoted=False,goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS/'failure.json',dict(error=repr(error),automatic_retry=False,goal_achieved=False))
            raise
        finally:
            owner.restore_owned(plan,'Complete matched human preference experiment ended; original work restored')
            watcher.wait(timeout=20)


if __name__ == '__main__':
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watchdog',action='store_true')
    parser.add_argument('--distributed',choices=['capacity','train'])
    parser.add_argument('--arm',choices=['chosen_CE','human_DPO'])
    parser.add_argument('--assess',choices=['initial','chosen_CE','human_DPO'])
    parser.add_argument('--suite',choices=['code','broad','preferences'])
    parser.add_argument('--gpu',type=int,choices=range(4))
    args = parser.parse_args()
    if args.watchdog:
        owner.watchdog()
    elif args.distributed:
        distributed(args.distributed,args.arm)
    elif args.assess:
        assess(args.assess,args.gpu,args.suite)
    else:
        parent()
