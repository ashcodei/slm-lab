"""Resume the exact interrupted broad-code epoch after the live dense study.

Runtime ownership is separate from the immutable original scientific plan.
The CE branch resumes all four private states at1536/3721; the execution arm
starts from the original own human-DPO policy and consumes all3721sources.
Both retain the original objectives, source order and full native horizons.
"""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import train_recovered_dense_code_GPU_v1 as resources

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'reports/broad-anchored-code-v7'
OUT = ROOT/'reports/broad-anchored-code-v7-recovery-v3'
MODELS = ROOT/'checkpoints/broad-anchored-code-v7-recovery-v3'
PREP = ROOT/'reports/broad-code-resume-preparation-v1'
PRIOR = ROOT/'reports/durable-distributed-code-v1'
SCRATCH = Path('/tmp/slm-broad-anchored-code-v7-recovery-v3')
SCRIPT = Path(__file__).resolve()
ARMS = ['anchor_CE', 'anchor_execution']
GIB = 1024**3
MAX_PRIVATE = 384*1024**2
read, write, sha, identity, live = resources.read, resources.write, resources.sha, resources.identity, resources.live


def configure():
    resources.OUT = OUT; resources.SCRATCH = SCRATCH; resources.SCRIPT = SCRIPT


def normalize(actor):
    return dict(pid=actor['pid'], birth=str(actor.get('birth', actor.get('ticks'))))


def predecessor_ready(actors):
    if any(live(actor) for actor in actors):
        return False
    if not (PRIOR/'complete.json').exists() or not read(PRIOR/'complete.json').get('all_four_rates_all_230updates_and_bounded_assessments'):
        raise ValueError('The exact preceding full four-rate study must close successfully')
    if not read(PRIOR/'attempts/2/restored.json').get('own_children_terminal'):
        raise ValueError('Every exact preceding registered GPU worker must be terminal')
    return True


def registered(rank):
    configure(); runtime = read(OUT/'plan.json'); me = identity(os.getpid())
    while not any(row['actor'] == me and row['gpu'] == rank for row in read(OUT/'children.json')):
        if not live(runtime['owner']):
            raise ValueError('No CUDA allocation is allowed before exact live ownership/registration')
        time.sleep(.05)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(rank):
        raise ValueError('Only the assigned original physical GPU may be visible')
    resources.checked(runtime)
    return runtime


def worker(rank, arm):
    runtime = registered(rank)
    import random
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.broad_anchored_execution_stage_v1 import BroadAnchoredExecutionStage, semantic_digest
    from slm_lab.broad_anchored_policy_v3 import fresh, make_reference, sources, export
    from slm_lab.broad_collective_resume_v1 import restore
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint, capture_rng, restore_rng
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.distributed_dense_code_step_v1 import agree, gather
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    from slm_lab.parallel_function_repair_stage_v1 import value_digest
    from slm_lab.pretraining_operational_recovery_v1 import math_for_runtime
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    allowed = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, {allowed[rank % len(allowed)]})
    device = torch.device('cuda:0'); torch.cuda.set_device(device); torch.cuda.set_per_process_memory_fraction(.85, device)
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.get_device_name(device) != 'NVIDIA L40':
        raise ValueError('The exact original local L40 is required')
    dist.init_process_group('gloo', init_method='file://'+str(OUT/('rendezvous-'+arm)), rank=rank, world_size=4,
                            timeout=timedelta(minutes=45))
    try:
        guard = FileBindings(runtime['source_files_sha256']); guard.check(full=True)
        plan = math_for_runtime(runtime); digest = runtime['original_neural_plan_sha256']
        model, _ = fresh(plan, device); reference = make_reference(model, plan); corpus = ExpandedExecutionCorpus(ROOT)
        if len(corpus) != 3721 or reference.resident_inventory(model) != plan['resident_training_inventory']:
            raise ValueError('The complete original own policy/reference and all3721sources are required')
        options = plan['training_options']; bound = sources(plan, digest)
        frozen = tensor_digest(model.base.named_parameters()); critic = model.parts['critic'].weight.detach().clone()
        def new_stage():
            optimizer = torch.optim.AdamW([p for _, p in model.named_selected(('actor',))], lr=options['learning_rate'],
                betas=tuple(options['betas']), weight_decay=options['weight_decay'], foreach=False)
            stage = BroadAnchoredExecutionStage(model, optimizer, corpus, reference, root=ROOT, method=arm,
                seed=options['seed'], sources=bound, world=4, rank=rank,
                KL_weight=options['KL_weight'], anchor_weight=options['anchor_weight'],
                cache_absolute_bound=plan['cache_calibration']['bound'], cache_calibration_sha256=plan['cache_calibration']['sha256'])
            return optimizer, stage
        optimizer, stage = new_stage()
        random.seed(options['seed']+rank); np.random.seed(options['seed']+rank); torch.manual_seed(options['seed']+rank)
        initial = model.export(bound); initial_rng = capture_rng(device)
        if rank == 0 and arm == 'anchor_CE':
            atomic_checkpoint(MODELS/'initial-actor.pt', export(model, plan, digest, 'initial', 0), 96*1024**2, 8*GIB)
        def checked_step():
            resources.checked(runtime)
            return stage.update()
        if arm == 'anchor_CE':
            original = runtime['starting_private_states'][rank]
            restore(stage, original['path'], expected_sha256=original['sha256'])
            expected = checked_step(); expected_digest = value_digest(stage.state())
            restore(stage, original['path'], expected_sha256=original['sha256'])
            actual = checked_step()
            if value_digest(stage.state()) != expected_digest or semantic_digest(actual['trajectories']) != semantic_digest(expected['trajectories']):
                raise ValueError('The resumed actual next global update/Adam/actions/allprivateRNG must replay exactly')
            restore(stage, original['path'], expected_sha256=original['sha256'])
        else:
            checked_step(); saved = stage.state(); path = SCRATCH/f'execution-capacity-rank-{rank}.pt'
            atomic_checkpoint(path, saved, MAX_PRIVATE, 40*GIB); del saved
            expected = checked_step(); expected_digest = value_digest(stage.state())
            restore(stage, path, expected_sha256=sha(path)); actual = checked_step()
            if value_digest(stage.state()) != expected_digest or semantic_digest(actual['trajectories']) != semantic_digest(expected['trajectories']):
                raise ValueError('The original fresh execution arm must replay its actual full-profile next update')
            model.import_export(initial, expected_sources=bound); optimizer, stage = new_stage(); restore_rng(initial_rng, device)
        if stage.completed != (1536 if arm == 'anchor_CE' else 0):
            raise ValueError('Capacity learning must be discarded before the original source epoch continues')
        del initial, initial_rng, expected, actual; gc.collect(); torch.cuda.empty_cache()
        write(OUT/f'{arm}-capacity-rank-{rank}.json', dict(complete=True, discarded_capacity_learning=True,
            production_start_cursor=stage.completed, actual_counts=model.counts(), actual_Adam_states=len(optimizer.state),
            exact_actual_next_update_common_values_Adam_actions_private_CPU_CUDA_RNG=True,
            original_complete_native_shape_capacity_reused_from=runtime['original_shape_capacity_receipts'],
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device), no_new_human_labels=True, goal_achieved=False))
        dist.barrier()
        if rank == 0:
            write(OUT/f'{arm}-capacity-complete.json', dict(complete=True, all_four_original_L40s=True,
                production_start_cursor=stage.completed, next_update_replay_passed=True, capacity_updates_discarded=True))
        journal_path = OUT/(arm+'-metrics.jsonl'); journal_hash = __import__('hashlib').sha256(); journal_bytes = 0
        if arm == 'anchor_CE':
            with journal_path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024**2), b''):
                    journal_hash.update(block); journal_bytes += len(block)
            original = runtime['starting_committed_prefix']
            if journal_bytes != original['bytes'] or journal_hash.hexdigest() != original['sha256']:
                raise ValueError('Only the original1536-update committed journal prefix may seed continuation')
        journal = journal_path.open('ab' if arm == 'anchor_CE' else 'xb') if rank == 0 else None
        def checkpoint():
            if journal is not None:
                journal.flush(); os.fsync(journal.fileno())
            dist.barrier(); error = None; proof = None
            try:
                slot = (stage.completed//32) % 2
                path = MODELS/f'{arm}-resume-slot-{slot}-rank-{rank}.pt'
                atomic_checkpoint(path, stage.state(), MAX_PRIVATE, 8*GIB)
                proof = dict(rank=rank, path=str(path), sha256=sha(path), actual_completed=stage.completed,
                    external_journal_prefix_bytes=journal_bytes, external_journal_prefix_sha256=journal_hash.hexdigest())
            except Exception as exc:
                error = repr(exc)
            agree(error); proofs = gather(proof)
            if len({(p['external_journal_prefix_bytes'], p['external_journal_prefix_sha256']) for p in proofs}) != 1:
                raise ValueError('All four complete private states must bind the same durable journal prefix')
            if rank == 0:
                write(OUT/f'{arm}-resume-distributed-state.json', dict(completed=stage.completed, all_four_private_states=proofs,
                    immutable_original_neural_plan_sha256=digest, complete_epoch=stage.completed == 3721, goal_achieved=False))
        try:
            while stage.completed < len(stage.order):
                if stage.completed % 32 == 0:
                    guard.check()
                tick = time.monotonic(); record = checked_step()
                raw = (json.dumps(record, separators=(',', ':'), allow_nan=False)+'\n').encode()
                if journal_bytes+len(raw) > 2*GIB:
                    raise ValueError('Each complete native trajectory journal must fit its2GiB budget')
                journal_hash.update(raw); journal_bytes += len(raw)
                if journal is not None:
                    journal.write(raw); journal.flush()
                    write(OUT/(arm+'-progress.json'), dict(actual_updates=stage.completed, planned_updates=3721,
                        original_committed_start=1536 if arm == 'anchor_CE' else 0, last_source_id=record['source_id'],
                        global_rewards=record['global_rewards'], original_reference_token_mean_CE=record['original_reference_token_mean_CE'],
                        last_update_seconds=time.monotonic()-tick, goal_achieved=False))
                if stage.completed % 32 == 0 or stage.completed == 3721:
                    checkpoint()
            if tensor_digest(model.base.named_parameters()) != frozen or not torch.equal(critic, model.parts['critic'].weight):
                raise ValueError('Every original foundation/vision/head value and unused critic must remain fixed')
            reference.assert_bound(model); guard.check(full=True)
            if len(set(gather(tensor_digest(model.parts.named_parameters())))) != 1:
                raise ValueError('All four complete trained actor replicas must agree before export')
            if rank == 0:
                path = MODELS/(arm+'-actor.pt')
                atomic_checkpoint(path, export(model, plan, digest, arm, 3721), 96*1024**2, 8*GIB)
                write(OUT/(arm+'-training-complete.json'), dict(complete=True, actual_updates=3721,
                    all_3721_original_sources_once=True, original_CE_reference_supervision_every_update=True,
                    original_own_human_DPO_reference_frozen=True, checkpoint_sha256=sha(path),
                    journal_sha256=journal_hash.hexdigest(), journal_bytes=journal_bytes,
                    new_production_updates=3721-(1536 if arm == 'anchor_CE' else 0), capacity_learning_discarded=True,
                    human_preference_labels_added=0, goal_achieved=False))
        finally:
            if journal is not None:
                journal.close()
    finally:
        dist.destroy_process_group()


def assess(rank, arm, suite):
    runtime = registered(rank)
    import importlib
    import torch
    from slm_lab.broad_anchored_policy_v3 import load
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    from slm_lab.pretraining_operational_recovery_v1 import math_for_runtime
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.cuda.set_device(0)
    allowed = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, {allowed[rank % len(allowed)]})
    torch.cuda.set_per_process_memory_fraction(.85, 0); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    plan = math_for_runtime(runtime); checkpoint = MODELS/(arm+'-actor.pt')
    model, metadata = load(checkpoint, OLD/'plan.json', arm, expected_checkpoint_sha256=sha(checkpoint),
                           expected_plan_sha256=runtime['original_neural_plan_sha256'], device='cuda:0')
    guard = FileBindings(runtime['source_files_sha256']); folder = OUT/arm; folder.mkdir(exist_ok=True)
    checked = lambda:resources.checked(runtime)
    with inference_cache():
        if suite == 'preferences':
            from slm_lab.held_human_shards_v1 import evaluate_shard
            from slm_lab.whole_human_data_v2 import WholeHumanData
            data = WholeHumanData(ROOT/'data/whole-human-alignment-v2', 'preferences', 'valid', 'evaluation')
            evaluate_shard(model, data, folder/f'held-preferences-shard-{rank}.json', rank=rank, world=4, beta=.1,
                checked=checked, provenance=dict(checkpoint_sha256=sha(checkpoint), run_plan_sha256=runtime['original_neural_plan_sha256'],
                human_manifest_sha256=plan['human_manifest_sha256'], frozen_dense_sha256=plan['frozen_dense_sha256'],
                inference_device_type='cuda', inference_precision='float32'))
        elif suite == 'apps':
            from tokenizers import Tokenizer
            from slm_lab.held_apps_native_shards_v1 import evaluate_shard
            evaluate_shard(model.policy, Tokenizer.from_file(str(MODELS/'tokenizer.json')), checkpoint,
                ROOT/'data/public-apps-publisher-test-evaluation-v1', Path(runtime['held_APPS_cohort']),
                folder/f'held-APPS-shard-{rank}.json', rank=rank, world=4, device='cuda:0', checked=checked)
        elif suite == 'code':
            from slm_lab.native_code_shards_v1 import evaluate_shard
            setter = torch.set_num_threads; torch.set_num_threads = lambda _:None
            try:
                for frame in ['completion', 'instruction']:
                    evaluate_shard(model.policy, metadata, checkpoint, folder, rank=rank, world=4, frame=frame)
            finally:
                torch.set_num_threads = setter
        else:
            tasks = {0:[('knowledge', dict(max_examples=56, max_new_tokens=32))],
                1:[('math', dict(max_examples=16, max_new_tokens=256))],
                2:[('ocr', dict(max_examples=32, max_new_tokens=128, image_size=448)), ('objects', dict(max_examples=32, max_new_tokens=32))],
                3:[('vision', dict(max_examples=32, image_size=448))]}
            setter = torch.set_num_threads; torch.set_num_threads = lambda _:None
            try:
                for name, options in tasks[rank]:
                    module = importlib.import_module('slm_lab.'+{'objects':'object'}.get(name, name)+'_evaluation')
                    original = module.load_checkpoint; module.load_checkpoint = lambda *a, **kw:(model.policy, metadata)
                    try:
                        checked(); data = ROOT/'data/math-reasoning-v2/test.jsonl' if name == 'math' else Path(plan['pulse_sources'][name])
                        module.evaluate(checkpoint, data, folder/(name+'.json'), device='cuda:0', **options)
                    finally:
                        module.load_checkpoint = original
            finally:
                torch.set_num_threads = setter
    guard.check(full=True)
    write(folder/f'assessment-{suite}-{rank}-complete.json', dict(complete=True, checkpoint_sha256=sha(checkpoint),
        actual_counts=model.counts(), original_development_scope_preserved=True, independent_final_benchmarks_still_required=True))


def parent():
    configure()
    if OUT.exists() or MODELS.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'torch' in sys.modules or os.nice(0) != 0:
        raise ValueError('A fresh GPU-hidden normal-priority non-neural queue owner is required')
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.broad_resume_journal_v1 import copy_prefix
    tests = {ROOT/'reports/broad-resume-journal-v1-tests.log':'4 passed',
             ROOT/'reports/broad-collective-resume-v1-final-tests.log':'1 passed',
             ROOT/'reports/broad-recovery-handoff-v3-tests.log':'2 passed'}
    for path, passed in tests.items():
        if passed not in path.read_text():
            raise ValueError('All committed-prefix and actual four-replica recovery controls must pass')
    prior_plan = read(PRIOR/'attempts/2/plan.json'); prior_children = read(PRIOR/'attempts/2/children.json')
    if len(prior_children) != 4 or sorted(child['gpu'] for child in prior_children) != list(range(4)):
        raise ValueError('Every exact existing four-rate worker must be inventoried before waiting')
    predecessor = [prior_plan['owner'], *[p['actor'] for p in prior_children]]
    prep_actor = read(PREP/'plan.json')['owner']; old = read(OLD/'plan.json')
    files = dict(old['source_files_sha256'])
    for p in [*python_sources(ROOT, [SCRIPT, ROOT/'tests/test_broad_collective_resume_v1.py', ROOT/'tests/test_broad_resume_journal_v1.py']),
              *tests, OLD/'plan.json', PREP/'plan.json']:
        digest = sha(p)
        if str(p) in files and files[str(p)] != digest:
            raise ValueError('Original source bindings conflict with the recovery implementation')
        files[str(p)] = digest
    OUT.mkdir(); MODELS.mkdir(); SCRATCH.mkdir()
    queue = dict(at=datetime.now(timezone.utc).isoformat(), owner=identity(os.getpid()), predecessor=predecessor,
        source_preparation_owner=prep_actor, source_files_sha256=files, original_neural_plan_sha256=sha(OLD/'plan.json'),
        original_full_source_epochs_and_all_assessments_preserved=True, no_new_external_GPU_or_model=True,
        previous_turn_classification='progress', goal_achieved=False)
    write(OUT/'queue.json', queue); deadline = time.monotonic()+14*24*3600
    while not (PREP/'complete.json').exists() or not predecessor_ready(predecessor):
        if time.monotonic() > deadline:
            raise ValueError('The explicit fourteen-day queue horizon expired')
        if not (PREP/'complete.json').exists() and ((PREP/'failure.json').exists() or not live(prep_actor)):
            raise ValueError('The exact full-source preparation ended without successful completion')
        write(OUT/'queue-progress.json', dict(phase='wait_complete_exact_preparation_and_existing_dense_study',
            exact_live_predecessors=[p for p in predecessor if live(p)],
            exact_preparation_live=live(prep_actor), full_preparation_complete=(PREP/'complete.json').exists()))
        time.sleep(5)
    prepared = read(PREP/'complete.json')
    if (not prepared['complete'] or not prepared['whole_original_dataset_and_committed_prompts_actions_masks_reconstructed']
            or prepared['all_original_sources'] != 3721 or prepared['original_committed_updates'] != 1536
            or prepared['original_neural_plan_sha256'] != queue['original_neural_plan_sha256']):
        raise ValueError('Complete original source and private-prefix preparation is required')
    for actor in [old['owner'], *read(OLD/'children.json')]:
        if live(normalize(actor)):
            raise ValueError('An exact old registered broad-code worker remains alive')
    for path, expected in prepared['source_files_sha256'].items():
        if sha(path) != expected:
            raise ValueError('A prepared original source changed before recovery admission')
        files[path] = expected
    for p in [PREP/'complete.json', PREP/'anchor_CE-committed-prefix.jsonl', OUT/'queue.json']:
        files[str(p)] = sha(p)
    capacities = []
    for arm in ARMS:
        for rank in range(4):
            p = OLD/f'capacity-{arm}-{rank}.json'; record = read(p)
            if not record['exact_next_global_update_weights_Adam_actions_private_RNG'] or record['maximum_forced_action_horizon'] != 3059:
                raise ValueError('Every original four-L40 whole native-horizon capacity control is required')
            files[str(p)] = sha(p); capacities.append(str(p))
    cohorts = [p for p in old['source_files_sha256'] if 'held-APPS-cohort' in p]
    if len(cohorts) != 1 or sha(cohorts[0]) != old['source_files_sha256'][cohorts[0]]:
        raise ValueError('The exact preselected original held-APPS cohort must remain fixed')
    if shutil.disk_usage(ROOT).free < 18*GIB or shutil.disk_usage(SCRATCH).free < 48*GIB:
        raise ValueError('Two four-private-state generations, bounded journals/models and8GiB durable reserve must fit')
    prefix = prepared['committed_prefix']
    fork = copy_prefix(prefix['path'], OUT/'anchor_CE-metrics.jsonl', length=prefix['bytes'],
                       expected_sha256=prefix['sha256'], completed=1536, minimum_free_bytes=8*GIB)
    shutil.copyfile(old['tokenizer_package']['path'], MODELS/'tokenizer.json')
    if sha(MODELS/'tokenizer.json') != old['tokenizer_sha256']:
        raise ValueError('The exact original native tokenizer must be packaged without changes')
    files[str(MODELS/'tokenizer.json')] = old['tokenizer_sha256']
    with resources.LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpus, apps = resources.snapshot()
        if [g['uuid'] for g in gpus] != resources.UUIDS or apps:
            raise ValueError('All four exact original L40 contexts must be idle before ownership')
        runtime = dict(owner=identity(os.getpid()), expires_at_unix=time.time()+14*24*3600,
            original_neural_plan=str(OLD/'plan.json'), original_neural_plan_sha256=queue['original_neural_plan_sha256'],
            neural_options_unchanged=True, source_files_sha256=files, held_APPS_cohort=cohorts[0],
            starting_private_states=prepared['private_states'], starting_committed_prefix=fork,
            original_shape_capacity_receipts=capacities, private_checkpoint_interval=32,
            original_source_horizons_and_FP32_TF32_disabled_math_preserved=True)
        write(OUT/'plan.json', runtime); write(OUT/'children.json', []); resources.checked(runtime)
        watcher = subprocess.Popen([sys.executable, '-u', str(SCRIPT), '--watchdog'], cwd=ROOT,
            stdout=(OUT/'watchdog.log').open('x'), stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES':''})
        children = []
        def launch(args, gpu, subject):
            process = subprocess.Popen([sys.executable, '-u', str(SCRIPT), *args], cwd=ROOT,
                stdout=(OUT/(subject+'.log')).open('x'), stderr=subprocess.STDOUT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES':str(gpu), 'CUBLAS_WORKSPACE_CONFIG':':4096:8',
                     'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'TOKENIZERS_PARALLELISM':'false'})
            actor = identity(process.pid)
            if actor is None:
                raise ValueError('A worker ended before exact CUDA registration')
            children.append(dict(actor=actor, gpu=gpu)); write(OUT/'children.json', children)
            return process
        def wait(processes):
            while any(p.poll() is None for p in processes):
                resources.checked(runtime)
                if any(p.poll() not in [None, 0] for p in processes):
                    raise ValueError('An exact recovery worker failed; retain all partial state without automatic retry')
                time.sleep(1)
            if any(p.returncode != 0 for p in processes):
                raise ValueError('Every exact worker must complete')
        try:
            for arm in ARMS:
                wait([launch(['--worker', '--rank', str(rank), '--arm', arm], rank, 'train-'+arm+'-'+str(rank)) for rank in range(4)])
            for arm in ['initial', *ARMS]:
                for suite in ['code', 'broad', 'preferences', 'apps']:
                    wait([launch(['--assess', '--rank', str(rank), '--arm', arm, '--suite', suite], rank,
                                 'assess-'+arm+'-'+suite+'-'+str(rank)) for rank in range(4)])
                from slm_lab.native_code_shards_v1 import reconcile as code_reconcile
                from slm_lab.held_human_shards_v1 import reconcile as human_reconcile
                from slm_lab.held_apps_native_shards_v1 import reconcile as apps_reconcile
                from slm_lab.whole_human_data_v2 import WholeHumanData
                folder = OUT/arm
                for frame in ['completion', 'instruction']:
                    code_reconcile([folder/f'code-{frame}-shard-{r}.json' for r in range(4)], folder/f'code-{frame}.json')
                human_reconcile(WholeHumanData(ROOT/'data/whole-human-alignment-v2', 'preferences', 'valid', 'evaluation'),
                                [folder/f'held-preferences-shard-{r}.json' for r in range(4)], folder/'held-preferences.json')
                apps_reconcile([folder/f'held-APPS-shard-{r}.json' for r in range(4)], Path(cohorts[0]), folder/'held-APPS.json')
            write(OUT/'complete.json', dict(complete=True, all_original3721sources_once_each_arm=True,
                all_original_native_held_human_and_preselected_APPS_assessments_complete=True,
                original_scientific_plan_unchanged=True, original_uncommitted_tail_preserved=True,
                fresh_own_output_human_RLHF_and_full_final_benchmarks_still_required=True, candidate_promoted=False, goal_achieved=False))
        finally:
            for entry in children:
                resources.stop(entry['actor'], runtime)
            write(OUT/'restored.json', dict(own_children_terminal=all(not live(e['actor']) for e in children), external_processes_signalled=0))
            watcher.wait(timeout=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true'); parser.add_argument('--watchdog', action='store_true')
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--assess', action='store_true')
    parser.add_argument('--rank', type=int, choices=range(4)); parser.add_argument('--arm', choices=['initial', *ARMS])
    parser.add_argument('--suite', choices=['code', 'broad', 'preferences', 'apps']); args = parser.parse_args()
    try:
        if args.watchdog:
            configure(); resources.watchdog()
        elif args.worker:
            worker(args.rank, args.arm)
        elif args.assess:
            assess(args.rank, args.arm, args.suite)
        elif args.run:
            parent()
        else:
            parser.error('Choose a complete owned run, registered worker, assessment or watchdog')
    except BaseException as exc:
        if OUT.exists():
            write(OUT/('owner-failure.json' if args.run else f'worker-failure-{os.getpid()}.json'),
                  dict(error=repr(exc), traceback=traceback.format_exc(), partial_artifacts_preserved=True, goal_achieved=False))
        raise
