"""Owned complete native-code CE/execution comparison after prior full stages.

This indexed, unlaunched driver retains every original native source/CE anchor,
three epochs, eight full-horizon attempts and all preceding assessments.
It needs separate complete source preparation and actual idle four-L40
capacity, fixed-horizon stress and fresh readback before any training. Complete
neural records are durable before Adam; controller events contain only timings
and immutable observation receipts. New physical budgets remain launch gates.
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
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/native-code-suffix-aligned-GPU-v3'
DEST = ROOT / 'checkpoints/native-code-suffix-aligned-GPU-v3'
PREP = ROOT / 'reports/native-code-suffix-aligned-preparation-CPU-v3'
CASES = ROOT / 'reports/native-human-code-suffix-actual-capacity-CPU-v2'
DATA = ROOT / 'data/native-human-code-suffix-v3'
REASONING_DATA = ROOT / 'data/public-reasoning-curriculum-v1'
ADMITTED = DATA / 'admission.json'
REASONING_ADMITTED = ROOT / 'reports/public-reasoning-curriculum-v1-readback-admission/complete.json'
PROFILES = ('anchor_CE', 'anchor_execution')
SUITES = ('code', 'broad', 'preferences', 'QA', 'AI2D', 'MBPP', 'words')
MAX_STATE = 1024**3
MAX_EVENT_RAW = 4096 * 11172
MAX_ASSESSMENTS = 4 * 1024**3
RESERVE = 40 * 1024**3
NATIVE_PREFILL = dict(protocol='own-native-complete-text-causal-chunked-prefill-FP32-v1', chunk_tokens=128)
STAGE_OPTIONS = dict(learning_rate=3e-5, gradient_clip=1., group_size=8, KL_weight=.01, anchor_weight=1.,
    cache_absolute_bound=.001, maximum_cache_bytes=8 * 1024**3, gradient_bucket_values=1024**2,
    seed=989711, epochs=3, execution_workers=2, CPU_mechanics_horizon=None)
OPTIMIZER = dict(algorithm='torch.optim.AdamW', betas=[.9, .95], eps=1e-8, weight_decay=.01, foreach=False, fused=False)


def read(path): return json.loads(Path(path).read_text())


def make_optimizer(model, options, settings):
    """The actual stage and its recovery protocol require actor-only AdamW."""
    import torch
    if (settings.get('algorithm') != 'torch.optim.AdamW' or settings.get('foreach') is not False
            or settings.get('fused') is not False):
        raise ValueError('The source-bound ordinary actor-only AdamW recipe is required')
    return torch.optim.AdamW([p for _, p in model.named_selected(('actor',))], lr=options['learning_rate'],
        betas=tuple(settings['betas']), eps=settings['eps'], weight_decay=settings['weight_decay'],
        foreach=False, fused=False)


def configure():
    global owner
    import own_training_gpu_lease_v2 as owner
    owner.REPORTS = OUT
    owner.resources.REPORTS = OUT


def initializer(plan, device='cpu'):
    from slm_lab.own_dense_standalone_v1 import load
    spec = plan['initializer']
    model, metadata = load(spec['checkpoint'], expected_checkpoint_sha256=spec['checkpoint_sha256'], expected_sources=spec['sources'])
    if (model.counts()['total_unique'] != 984399690 or metadata['completed_updates'] != 1926
            or metadata['tokenizer_sha256'] != spec['tokenizer_sha256']
            or metadata['provenance'].get('training_profile') != 'anchor_RL'
            or metadata['provenance'].get('whole_production_source_schedule_completed') is not True):
        raise ValueError('Only the predeclared complete own reasoning anchor_RL native initializer is admitted')
    return model.to(device), metadata


def stage_sources(plan):
    from slm_lab.corpus import file_hash
    recipe = plan['training_recipe']
    if file_hash(recipe['path']) != recipe['sha256']:
        raise ValueError('Every original optimizer/source/objective recipe byte must match')
    return dict(plan['native_source_files_sha256'], initializer_sha256=plan['initializer']['checkpoint_sha256'], tokenizer_sha256=plan['tokenizer_sha256'],
        native_admission_sha256=plan['native_admission_sha256'],
        native_manifest_sha256=plan['native_manifest_sha256'],
        full_capacity_plan_sha256=plan['capacity_plan_file_sha256'], training_recipe_sha256=recipe['sha256'])


def attach_native_prefill(actor, *, purpose, settings):
    from slm_lab.native_text_chunked_prefill_v1 import install, PROTOCOL
    if settings != NATIVE_PREFILL or settings.get('protocol') != PROTOCOL:
        raise ValueError('The exact declared complete-prefix native prefill recipe is required')
    return install(actor, chunk_tokens=settings['chunk_tokens'], purpose=purpose)


def gpu_setup(rank, visible):
    import torch
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != visible or os.sched_getscheduler(0) != os.SCHED_OTHER or os.nice(0) != 0):
        raise ValueError('New actual GPUworkers require normal nice0 scheduling and exact visibility')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.85, device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.get_device_name(device) != 'NVIDIA L40': raise ValueError('Only an original physical L40 is admitted')
    return device


def distributed(profile, phase):
    import random
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.native_suffix_capacity_controller_v2 import run_capacity
    from slm_lab.native_suffix_capacity_evidence_v2 import run_stress_cases, require_capacity
    from slm_lab.native_suffix_capacity_readback_v2 import read_capacity
    from slm_lab.native_suffix_training_controller_v2 import run_training
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.native_suffix_batched_history_stage_v5 import NativeSuffixBatchedHistoryStageV5
    from slm_lab.native_suffix_owned_runtime_contract_v3 import history_budget
    from slm_lab.recovered_native_suffix_train_data_v3 import load_recovered_native_suffix_data
    if (int(os.environ['WORLD_SIZE']) != 4 or int(os.environ['RANK']) != int(os.environ['LOCAL_RANK'])
            or profile not in PROFILES or phase not in ['capacity', 'readback', 'train']):
        raise ValueError('Four original actual local replicas and exact declared experiment phase required')
    rank = int(os.environ['RANK'])
    plan = owner.registered(profile + '-' + phase + '-rank-' + str(rank))
    device = gpu_setup(rank, '0,1,2,3')
    dist.init_process_group('gloo', timeout=timedelta(minutes=45))
    gradient_group = dist.new_group(backend='nccl', timeout=timedelta(minutes=45))
    try:
        guard = FileBindings(plan['source_files_sha256'])
        guard.check(full=True)
        data = load_recovered_native_suffix_data(ROOT, purpose='optimization' if phase == 'train' else 'capacity')
        options = plan['stage_options']
        capacity_plan = read(CASES / 'actual-complete-source-capacity-plan.json')
        base, _ = initializer(plan)
        model = OwnDenseActor(base).to(device)
        prefill = attach_native_prefill(model, purpose='GPU-runtime', settings=plan['native_prefill'])
        optimizer = make_optimizer(model, options, plan['optimizer'])
        random.seed(options['seed'] + rank)
        np.random.seed(options['seed'] + rank)
        torch.random.default_generator.manual_seed(options['seed'] + rank)
        torch.cuda.manual_seed(options['seed'] + rank)
        mode = 'train' if phase == 'train' else 'capacity'
        entries = data.complete_schedule(epochs=options['epochs'], width=4, seed=options['seed']) if mode == 'train' else capacity_plan['capacity_entries']
        stage = NativeSuffixBatchedHistoryStageV5(model, optimizer, data, entries, profile, options, stage_sources(plan),
            mode=mode, rank=rank, world=4, control_group=None, gradient_group=gradient_group,
            history_budgets=history_budget(plan['indexed_history'], mode, len(entries)))
        folder = OUT / profile / phase
        folder.mkdir(parents=True, exist_ok=True)
        state_dir = DEST / profile / (mode + '-private')
        bundle = plan['cold_inputs']
        if file_hash(bundle['path']) != bundle['sha256']: raise ValueError('Complete original cold input bytes must remain exact')
        inputs = torch.load(bundle['path'], map_location='cpu', weights_only=True, mmap=True)
        def independent():
            fresh = OwnDenseActor(initializer(plan)[0]).to(device)
            attach_native_prefill(fresh, purpose='GPU-runtime', settings=plan['native_prefill'])
            return fresh
        if phase == 'capacity':
            result = run_capacity(stage, state_dir, capacity_plan=capacity_plan, independent_actor=independent,
                cold_inputs=inputs, maximum_bytes=MAX_STATE, minimum_free_bytes=RESERVE,
                history_directory=folder / 'indexed-attempts')
            horizon = run_stress_cases(stage)
            require_capacity(stage, stage.gather(result), stage.gather(horizon), capacity_plan)
            atomic_json(folder / ('rank-' + str(rank) + '-full-horizon.json'), horizon)
        else:
            proof_paths = [OUT / profile / 'capacity' / ('rank-' + str(r) + '.json') for r in range(4)]
            reports = [read(p) for p in proof_paths]
            horizon_paths = [OUT / profile / 'capacity' / ('rank-' + str(r) + '-full-horizon.json') for r in range(4)]
            horizons = [read(p) for p in horizon_paths]
            if phase == 'readback':
                FileBindings({str(p): file_hash(p) for p in [*proof_paths, *horizon_paths]}).check(full=True)
                result = read_capacity(stage, state_dir, reports=reports, horizon_reports=horizons, capacity_plan=capacity_plan,
                    independent_actor=independent, cold_inputs=inputs)
            else:
                fresh_paths = [OUT / profile / 'readback' / ('rank-' + str(r) + '.json') for r in range(4)]
                readbacks = [read(p) for p in fresh_paths]
                controls = FileBindings({**{p: h for p, h in plan['source_files_sha256'].items() if p.endswith('.py')},
                    **{str(p): file_hash(p) for p in [*proof_paths, *horizon_paths, *fresh_paths]}})
                def preflight():
                    owner.owned(plan)
                    controls.check()
                    prefill.assert_current()
                counters = dict(context=0, targets=0)
                def progress(row, elapsed):
                    counters['context'] += row['complete_original_context_tokens']
                    counters['targets'] += row['original_response_EOS_targets']
                    atomic_json(OUT / 'progress.json', dict(profile=profile, phase='complete original native-code training',
                        actual_updates=stage.completed, planned_updates=11172, actual_last_update_seconds=elapsed,
                        original_context_tokens=counters['context'], original_response_EOS_targets=counters['targets'],
                        complete_prefix_prefill=prefill.last_observation, goal_achieved=False))
                result = run_training(stage, state_dir, folder / 'indexed-complete-training-attempt', folder / 'all11172-controller-events.jsonl.gz',
                    capacity_plan=capacity_plan, reports=reports, horizon_reports=horizons, readbacks=readbacks,
                    checkpoint_every=plan['checkpoint_every'], maximum_state_bytes=MAX_STATE,
                    minimum_free_bytes=RESERVE, maximum_event_raw_bytes=MAX_EVENT_RAW,
                    runtime_sources=plan['additional_runtime_sources'], preflight=preflight, progress=progress)
                from slm_lab.native_suffix_standalone_export_v2 import FULL
                if any(result['original_complete_training_budget'].get(k) != v for k, v in FULL.items()) or stage.completed != 11172:
                    raise ValueError('The full original source population, three epochs and all answer/EOS targets are required')
        guard.check(full=True)
        prefill.assert_current()
        owner.owned(plan)
        atomic_json(folder / ('rank-' + str(rank) + '.json'), result)
    finally: dist.destroy_process_group()


def export_candidate(profile):
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings, atomic_checkpoint
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.own_dense_standalone_v1 import load
    from slm_lab.parallel_repair_durable_state_v1 import read_latest
    from slm_lab.native_suffix_standalone_export_v2 import export, FULL
    from slm_lab.recovered_native_suffix_train_data_v3 import load_recovered_native_suffix_data
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.sched_getscheduler(0) != os.SCHED_OTHER or os.nice(0) != 0:
        raise ValueError('A fresh normal nice0 CUDA-hidden export process is required')
    plan = owner.registered(profile + '-complete-native-export')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    guard = FileBindings(plan['source_files_sha256'])
    proofs = [read(OUT / profile / 'train' / ('rank-' + str(r) + '.json')) for r in range(4)]
    if any(not p['complete'] or any(p['original_complete_training_budget'].get(k) != v for k, v in FULL.items()) or p['actual_complete_updates'] != 11172 for p in proofs):
        raise ValueError('All four exact complete training receipts are required')
    if len({p['actual_pointer_sha256'] for p in proofs}) != 1 or len({p['actual_checkpoint_sha256'] for p in proofs}) != 1:
        raise ValueError('Every replica must have the same final actual shared state')
    state, receipt, pointer = read_latest(DEST / profile / 'train-private', expected_pointer_sha256=proofs[0]['actual_pointer_sha256'])
    if receipt['checkpoint_sha256'] != proofs[0]['actual_checkpoint_sha256']: raise ValueError('Exact final trained tensor bytes required')
    actor = OwnDenseActor(initializer(plan)[0]).eval()
    data = load_recovered_native_suffix_data(ROOT, purpose='optimization')
    sources = stage_sources(plan)
    actor.import_export(state['model'], expected_sources=sources)
    native, payload, provenance = export(actor, data, state, proofs[0], sources=sources,
        tokenizer_sha256=plan['tokenizer_sha256'], working_budget_bytes=8 * 1024**3,
        runtime_sources=plan['additional_runtime_sources'])
    artifact_sources = payload['complete_dense']['sources']
    folder = DEST / profile / 'native'
    folder.mkdir(parents=True)
    shutil.copyfile(Path(plan['initializer']['checkpoint']).parent / 'tokenizer.json', folder / 'tokenizer.json')
    checkpoint = folder / 'model.pt'
    atomic_checkpoint(checkpoint, payload, plan['maximum_native_bytes'], RESERVE)
    cold, metadata = load(checkpoint, expected_checkpoint_sha256=file_hash(checkpoint), expected_sources=artifact_sources)
    if (metadata['completed_updates'] != 11172 or tensor_digest(cold.named_parameters()) != tensor_digest(native.named_parameters())
            or {p.data_ptr() for p in cold.parameters()} & {p.data_ptr() for p in native.parameters()}):
        raise ValueError('Every actual folded native value must cold-load independently and exactly')
    inputs = torch.load(plan['cold_inputs']['path'], map_location='cpu', weights_only=True, mmap=True)
    arithmetic = {}
    with torch.no_grad(), torch.autocast('cpu', enabled=False):
        for name in ['text', 'photo']:
            wanted, actual = native(**inputs[name])['logits'], cold(**inputs[name])['logits']
            if not torch.equal(wanted, actual) or not bool(torch.isfinite(actual).all()):
                raise ValueError('All complete4096text/448photo native outputs must cold-load exactly')
            hot = actor(**inputs[name])['logits']
            arithmetic[name] = dict(actual_hot_to_folded_max_absolute_logit_difference=float((hot - actual).abs().max()),
                actual_hot_to_folded_argmax_agreement=float((hot.argmax(-1) == actual.argmax(-1)).float().mean()),
                actual_folded_to_cold_bit_exact=True)
            del wanted, actual, hot
        wanted, actual = native.word_logits(inputs['word']['word_pixels']), cold.word_logits(inputs['word']['word_pixels'])
        if actual.dtype != torch.float32 or not torch.equal(wanted, actual) or not bool(torch.isfinite(actual).all()):
            raise ValueError('All complete64x448FP32word readouts must cold-load exactly')
    if torch.cuda.is_initialized(): raise ValueError('CPU export cannot acquire a GPU context')
    guard.check(full=True)
    owner.owned(plan)
    atomic_json(OUT / profile / 'native-export.json', dict(complete=True, profile=profile, checkpoint=str(checkpoint),
        checkpoint_sha256=file_hash(checkpoint), sources=artifact_sources, tokenizer_sha256=plan['tokenizer_sha256'],
        completed_updates=11172, actual_native_values=984399690,
        every_native_value_and_complete_text_photo_word_cold_exact=True, hot_to_folded_arithmetic_observations=arithmetic,
        hot_policy_generation_or_quality_equivalence_not_assumed=True, provenance=provenance,
        final_inference_requires_only_this_native_and_tokenizer=True, candidate_promoted=False, goal_achieved=False))


def load_candidate(plan, device='cpu'):
    from slm_lab.own_dense_standalone_v1 import load
    from slm_lab.dense_foundation_inference_v1 import DenseNativeView
    spec = plan['standalone']
    model, metadata = load(spec['checkpoint'], expected_checkpoint_sha256=spec['checkpoint_sha256'], expected_sources=spec['sources'])
    if plan['endpoint'] == 'initial':
        if metadata['completed_updates'] != 1926: raise ValueError('The exact completed canonical aligned baseline is required')
    elif (metadata['completed_updates'] != 11172 or not metadata['provenance'].get('whole_production_source_schedule_completed')
            or metadata['provenance'].get('training_profile') != plan['endpoint']):
        raise ValueError('Only the complete original trained native candidate may be assessed')
    if model.counts()['total_unique'] != 984399690: raise ValueError('The entire exact native student must be retained')
    model = model.to(device).eval()
    return model, DenseNativeView(model).eval(), metadata


def assessment(endpoint, suite, physical):
    import assess_machine_standalone_full_v2_assessment_recovery_gate_v1 as evaluator
    from slm_lab.visual_native_assessment_binding_v1 import bind
    folder = OUT / endpoint / 'assessment'
    plan = read(folder / 'plan.json')
    def register(subject, gpu):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu) or os.sched_getscheduler(0) != os.SCHED_OTHER or os.nice(0) != 0:
            raise ValueError('The exact owned normal-scheduled single original GPU is required')
        return owner.registered(endpoint + '-' + subject)
    if suite != 'reasoning':
        with bind(evaluator, reports=folder, plan=plan, load_student=load_candidate, register=register) as functions:
            functions[0](suite, physical)
        return
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.diverse_images import atomic_json
    from slm_lab.public_reasoning_reserved_evaluation_v1 import PublicReasoningReserved
    from slm_lab.reasoning_reserved_shards_v1 import evaluate
    register('reserved-reasoning', physical)
    gpu_setup(0, str(physical))
    model, native, _ = load_candidate(plan, 'cuda:0')
    guard = FileBindings(plan['source_files_sha256'])
    population = PublicReasoningReserved(REASONING_DATA, REASONING_ADMITTED)
    with (folder / ('reasoning-shard-' + str(physical) + '-actual-actions.jsonl')).open('x') as stream:
        def record(row):
            stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
            stream.flush()
        evaluate(native, population, folder / ('reasoning-shard-' + str(physical) + '.json'), rank=physical, world=4,
            checkpoint_sha256=plan['standalone']['checkpoint_sha256'], checked=lambda: owner.owned(plan), record=record)
        os.fsync(stream.fileno())
    guard.check(full=True)
    owner.owned(plan)
    atomic_json(folder / ('reasoning-rank-' + str(physical) + '-complete.json'), dict(complete=True, physical_GPU=physical,
        actual_student_values=984399690, all_actual_native_actions_and_whole_reserved_shard_retained=True, model_updates=0))


def parent():
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.joint_visual_runtime_contract_v1 import ready
    from slm_lab.native_suffix_owned_runtime_contract_v3 import require_complete_preparation
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.sched_getscheduler(0) != os.SCHED_OTHER or os.nice(0) != 0
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])
            or OUT.exists() or DEST.exists() or not (PREP / 'ready.json').exists()):
        raise ValueError('Fresh normal nice0 CUDA-hidden owned parent requires complete idle preparation')
    prepared = read(PREP / 'ready.json')
    if not terminal(read(PREP / 'plan.json')['owner']) or not ready(ROOT, **prepared['exact_predecessors']):
        raise ValueError('All exact original canonical/source/preparation owners must finish successfully first')
    require_complete_preparation(ROOT, prepared, options=STAGE_OPTIONS, prefill=NATIVE_PREFILL,
        maximum_state_bytes=MAX_STATE, maximum_event_raw_bytes=MAX_EVENT_RAW,
        maximum_assessment_bytes=MAX_ASSESSMENTS, reserve_bytes=RESERVE,
        indexed_history=prepared['indexed_history'])
    if shutil.disk_usage(ROOT).free < prepared['required_free_bytes_before_launch']:
        raise ValueError('Both complete candidates, all private generations/journals/assessments and40GiBreserve must fit')
    memory = {k: int(v) for k, v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
    if int(Path('/sys/fs/cgroup/memory.max').read_text()) - sum(memory[k] for k in ['anon', 'shmem', 'kernel']) < 112 * 1024**3:
        raise ValueError('Complete working memory and64GiBother-worker headroom are required')
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    guard = FileBindings(prepared['source_files_sha256'])
    guard.check(full=True)
    with owner.LEASE.open('a+') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner.idle_original()
        used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
        if len(used) != 4 or any(int(v) > 128 for v in used): raise ValueError('All four original L40s must actually be idle')
        files = {**prepared['source_files_sha256'], **owner.lease_sources()}
        plan = dict(owner=owner.resources.identity(os.getpid()), waiter=owner.WAITER,
            terminal_waiter_receipt_sha256=owner.terminal_receipt_sha256(), expires_at_unix=time.time() + 7 * 86400,
            source_files_sha256=files, initializer=prepared['initializer'], cold_inputs=prepared['cold_inputs'],
            maximum_native_bytes=prepared['maximum_native_bytes'], cached_baseline=prepared['cached_baseline'],
            cached_own_reference=prepared['cached_own_reference'], pulse_sources=prepared['pulse_sources'],
            tokenizer_sha256=prepared['initializer']['tokenizer_sha256'], suites=list(SUITES), expected_student_values=984399690,
            native_admission_sha256=file_hash(ADMITTED), native_manifest_sha256=file_hash(DATA / 'manifest.json'),
            native_source_files_sha256=prepared['native_source_files_sha256'], native_prefill=dict(NATIVE_PREFILL),
            indexed_history=prepared['indexed_history'], additional_runtime_sources=prepared['additional_runtime_sources'],
            capacity_plan_file_sha256=file_hash(CASES / 'actual-complete-source-capacity-plan.json'),
            stage_options=STAGE_OPTIONS, optimizer=OPTIMIZER, checkpoint_every=64, profiles=list(PROFILES),
            automatic_retry=False, all_original_canonical_scopes_preserved=True, candidate_promoted=False, goal_achieved=False)
        OUT.mkdir()
        DEST.mkdir()
        recipe = OUT / 'training-recipe.json'
        atomic_json(recipe, dict(stage_options=STAGE_OPTIONS, optimizer=OPTIMIZER, checkpoint_every=64,
            full_training_budget=prepared['full_native_budget'], native_prefill=dict(NATIVE_PREFILL), matched_profiles=list(PROFILES),
            indexed_history=prepared['indexed_history'], maximum_state_bytes=MAX_STATE, maximum_event_raw_bytes=MAX_EVENT_RAW,
            additional_runtime_sources=prepared['additional_runtime_sources']))
        plan['training_recipe'] = dict(path=str(recipe), sha256=file_hash(recipe))
        files[str(recipe)] = file_hash(recipe)
        atomic_json(OUT / 'plan.json', plan)
        atomic_json(OUT / 'children.json', [])
        driver = str(Path(__file__).resolve())
        watcher = None
        success = False
        def interrupted(signum, frame): raise KeyboardInterrupt('Owned complete native-code experiment interrupted')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            watcher = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', driver, '--watchdog'], cwd=ROOT,
                stdout=(OUT / 'watchdog.log').open('x'), stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
            def run_assessments(endpoint, standalone, suites):
                from slm_lab.public_reasoning_reserved_evaluation_v1 import PublicReasoningReserved
                from slm_lab.reasoning_reserved_shards_v1 import reconcile
                from slm_lab.visual_native_assessment_binding_v1 import bind
                folder = OUT / endpoint / 'assessment'
                folder.mkdir(parents=True)
                assessment_plan = dict(plan, endpoint=endpoint, standalone=standalone,
                    source_files_sha256={**files, standalone['checkpoint']: standalone['checkpoint_sha256'],
                        str(Path(standalone['checkpoint']).parent / 'tokenizer.json'): plan['tokenizer_sha256']})
                atomic_json(folder / 'plan.json', assessment_plan)
                import assess_machine_standalone_full_v2_assessment_recovery_gate_v1 as evaluator
                for suite in suites:
                    children = [owner.launch(plan, ['-u', driver, '--assess', suite, '--endpoint', endpoint, '--gpu', str(r)],
                        endpoint + '-' + suite + '-' + str(r), str(r)) for r in range(4)]
                    owner.wait_children(plan, children)
                    if suite == 'reasoning':
                        reconcile(PublicReasoningReserved(REASONING_DATA, REASONING_ADMITTED),
                            [folder / ('reasoning-shard-' + str(r) + '.json') for r in range(4)], folder / 'all-reserved-reasoning.json')
                    else:
                        with bind(evaluator, reports=folder, plan=assessment_plan, load_student=load_candidate,
                                register=lambda *args: plan) as functions: functions[1](suite, assessment_plan)
                    if sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()) > MAX_ASSESSMENTS:
                        raise ValueError('Complete candidate assessments exceed their declared bound; preserve all results')
            run_assessments('initial', dict(plan['initializer']), ['reasoning'])
            for profile in PROFILES:
                for phase in ['capacity', 'readback', 'train']:
                    child = owner.launch(plan, ['-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                        driver, '--distributed', phase, '--profile', profile], profile + '-' + phase + '-launcher', '0,1,2,3')
                    owner.wait_children(plan, [child])
                    proofs = [read(OUT / profile / phase / ('rank-' + str(r) + '.json')) for r in range(4)]
                    if any(not p['complete'] for p in proofs): raise ValueError('Every actual phase replica must finish')
                child = owner.launch(plan, ['-u', driver, '--export', '--profile', profile], profile + '-export', '')
                owner.wait_children(plan, [child])
                standalone = read(OUT / profile / 'native-export.json')
                run_assessments(profile, standalone, [*SUITES, 'reasoning'])
                from compare_visual_native_endpoints_v1 import compare
                compare(Path(plan['cached_baseline']['folder']), OUT / profile / 'assessment/standalone',
                    OUT / profile / 'matched-comparisons.json')
            baseline = read(OUT / 'initial/assessment/all-reserved-reasoning.json')
            comparisons = {}
            for profile in PROFILES:
                candidate = read(OUT / profile / 'assessment/all-reserved-reasoning.json')
                gained, lost = [], []
                for a, b in zip(baseline['results'], candidate['results']):
                    if any(a[k] != b[k] for k in ['position', 'source_id', 'source_group', 'task', 'original_prompt_ids', 'original_native_generation_budget']):
                        raise ValueError('Every complete reserved source/input/budget must match both actual native endpoints')
                    if not a['score']['correct'] and b['score']['correct']: gained.append(a['position'])
                    if a['score']['correct'] and not b['score']['correct']: lost.append(a['position'])
                comparisons[profile] = dict(before_correct=baseline['correct'], after_correct=candidate['correct'],
                    full_examples=289, gained_original_positions=gained, lost_original_positions=lost,
                    source_reserved_only_historical_independence_not_certified=True)
            atomic_json(OUT / 'matched-reasoning-comparisons.json', comparisons)
            guard.check(full=True)
            success = True
        except BaseException as exc:
            atomic_json(OUT / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False, goal_achieved=False))
            raise
        finally:
            owner.restore_owned(plan, 'Owned matched native-code experiment ended; only its registered children released')
            if watcher is not None: watcher.wait(timeout=20)
        if success:
            atomic_json(OUT / 'complete.json', dict(complete=True, profiles=list(PROFILES), all44652presentations7010616targets11172updates_per_profile=True,
                all_actual_capacity_fresh_readback_native_export_and_cold_complete=True,
                all_original_full_QA_AI2D_MBPP_human_words_code_broad_suites_each_candidate_complete=True,
                all289reserved_reasoning_cases_at_initial_and_both_candidates_complete=True,
                original_nine_endpoint_pipeline_preserved=True, actual_native_values=984399690,
                independent_neural_score_and_generated_program_readbacks_pending=True, candidate_promoted=False, goal_achieved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--watchdog', action='store_true')
    parser.add_argument('--distributed', choices=['capacity', 'readback', 'train'])
    parser.add_argument('--profile', choices=PROFILES)
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--assess', choices=[*SUITES, 'reasoning'])
    parser.add_argument('--endpoint', choices=['initial', *PROFILES])
    parser.add_argument('--gpu', type=int, choices=range(4))
    args = parser.parse_args()
    configure()
    if args.watchdog: owner.watchdog()
    elif args.distributed and args.profile: distributed(args.profile, args.distributed)
    elif args.export and args.profile: export_candidate(args.profile)
    elif args.assess and args.endpoint and args.gpu is not None: assessment(args.endpoint, args.assess, args.gpu)
    elif args.run: parent()
    else: parser.error('Use explicit --run after preparation, or an owned child mode')
