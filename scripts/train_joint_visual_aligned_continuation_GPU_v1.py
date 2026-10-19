"""Owned full visual continuation after the intact aligned canonical pipeline.

Preparation is a separate idle CPU process. This executable is launched later
with normal scheduling, hidden CUDA in its parent, all original predecessors
terminal and a real exclusive GPUlease. It never reroutes existing workers.
"""
import argparse
from datetime import datetime, timedelta, timezone
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
OUT = ROOT / 'reports/joint-visual-aligned-continuation-GPU-v1'
DEST = ROOT / 'checkpoints/joint-visual-aligned-continuation-GPU-v1'
PREP = ROOT / 'reports/joint-visual-aligned-continuation-preparation-v1'
CASES = ROOT / 'reports/joint-visual-actual-capacity-cases-CPU-v1'
MAX_STATE = 1536 * 1024**2
RESERVE = 40 * 1024**3
MAX_JOURNAL = 256 * 1024**2
SUITES = ('code', 'broad', 'preferences', 'QA', 'AI2D', 'MBPP', 'words')


def read(path):
    return json.loads(Path(path).read_text())


def configure():
    global owner
    import own_training_gpu_lease_v2 as owner
    owner.REPORTS = OUT
    owner.resources.REPORTS = OUT


def stage_sources(plan, mode):
    from slm_lab.corpus import file_hash
    spec = plan['stage_plans'][mode]
    if file_hash(spec['path']) != spec['sha256']:
        raise ValueError('The exact complete canonical visual stage plan must match')
    stage_plan = read(spec['path'])
    return stage_plan, {**stage_plan['sources'], 'run_plan_sha256': spec['sha256']}


def initializer(plan, device='cpu'):
    from slm_lab.own_dense_standalone_v1 import load
    spec = plan['initializer']
    model, metadata = load(spec['checkpoint'], expected_checkpoint_sha256=spec['checkpoint_sha256'],
        expected_sources=spec['sources'])
    if (model.counts()['total_unique'] != 984399690 or metadata['completed_updates'] != 102
            or metadata['tokenizer_sha256'] != spec['tokenizer_sha256']):
        raise ValueError('The complete fixed aligned native student and exact own tokenizer are required')
    return model.to(device), metadata


def gpu_setup(rank, visible):
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != visible or os.sched_getscheduler(0) != os.SCHED_OTHER or os.nice(0) != 0:
        raise ValueError('New GPUworkers require normal scheduling and exact physical visibility')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.85, device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.get_device_name(device) != 'NVIDIA L40':
        raise ValueError('One of the original local physical L40s is required')
    return device


def distributed(mode):
    if int(os.environ['WORLD_SIZE']) != 4 or int(os.environ['LOCAL_RANK']) != int(os.environ['RANK']):
        raise ValueError('Exactly four original local replicas are required')
    rank = int(os.environ['RANK'])
    plan = owner.registered('visual-' + mode + '-rank-' + str(rank))
    import random
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.codenet_checkpoint_v1 import FileBindings, atomic_checkpoint
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.corrected_multimodal_batches_v1 import CorrectedMultimodalBatches
    from slm_lab.joint_visual_controller_v1 import run_capacity, run_training
    from slm_lab.joint_visual_native_export_v1 import export as export_native
    from slm_lab.joint_visual_objective_update_v1 import make_optimizer
    from slm_lab.joint_visual_private_checkpoint_v2 import FixedGuard, _agree
    from slm_lab.joint_visual_source_delivery_v1 import OriginalVisualSources
    from slm_lab.joint_visual_training_stage_v1 import JointVisualStage
    from slm_lab.own_actor_vision_prefix_split_backward_v1 import OwnActorVisionPrefixSplitBackward
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    device = gpu_setup(rank, '0,1,2,3')
    dist.init_process_group('gloo', timeout=timedelta(minutes=45))
    try:
        guard = FileBindings(plan['source_files_sha256'])
        guard.check(full=True)
        base, _ = initializer(plan)
        view = OwnActorVisionPrefixSplitBackward(OwnDenseActor(base)).to(device)
        options = plan['training_options']
        optimizer = make_optimizer(view, learning_rates=options['learning_rates'], betas=options['betas'],
            eps=options['eps'], weight_decay=options['weight_decay'])
        data = CorrectedMultimodalBatches(ROOT, base.config, purpose='optimization' if mode == 'train' else 'capacity')
        provider = OriginalVisualSources(data)
        spec, sources = stage_sources(plan, mode)
        entries = read(CASES / 'actual-complete-source-capacity-plan.json')['capacity_entries'] if mode == 'capacity' else read(
            data.folder / 'complete-corrected-single-epoch-schedule.json')['entries']
        random.seed(options['seed'] + rank)
        np.random.seed(options['seed'] + rank)
        torch.random.default_generator.manual_seed(options['seed'] + rank)
        torch.cuda.manual_seed(options['seed'] + rank)
        stage = JointVisualStage(view, optimizer, provider, plan=spec, sources=sources, entries=entries,
            rank=rank, world=4, device=device, fixed=FixedGuard(view, optimizer))
        folder = OUT / mode
        folder.mkdir(exist_ok=True)
        state_dir = DEST / (mode + '-private')
        if mode == 'capacity':
            bundle = plan['cold_inputs']
            if file_hash(bundle['path']) != bundle['sha256']:
                raise ValueError('Every complete original cold input tensor must retain its bytes')
            inputs = torch.load(bundle['path'], map_location='cpu', weights_only=True, mmap=True)
            proof = run_capacity(stage, state_dir, capacity_plan=read(CASES / 'actual-complete-source-capacity-plan.json'),
                independent_native=lambda: initializer(plan, device)[0], cold_inputs=inputs,
                maximum_bytes=MAX_STATE, minimum_free_bytes=RESERVE)
            guard.check(full=True)
            owner.owned(plan)
            atomic_json(folder / ('rank-' + str(rank) + '.json'), proof)
        else:
            proofs = [read(OUT / 'capacity' / ('rank-' + str(r) + '.json')) for r in range(4)]
            controls = FileBindings({p: h for p, h in plan['source_files_sha256'].items()
                if p.endswith('.py') or p in [s['path'] for s in plan['stage_plans'].values()]})
            def preflight():
                owner.owned(plan)
                controls.check()
            def progress(row, elapsed):
                atomic_json(OUT / 'progress.json', dict(phase='complete original aligned visual continuation',
                    actual_complete_updates=stage.completed, planned_updates=4733, family=row['family'],
                    original_response_EOS_targets=sum(r['response_EOS_targets'] for r in stage.metrics),
                    actual_last_update_seconds=elapsed, optimizer_group_clocks=row['optimizer_group_clocks'], goal_achieved=False))
            result = run_training(stage, state_dir, folder / 'all4733-complete-actual-updates.jsonl', capacity_proofs=proofs,
                capacity_plan_sha256=plan['capacity_plan_semantic_sha256'], every=options['checkpoint_every'],
                maximum_bytes=MAX_STATE, minimum_free_bytes=RESERVE, maximum_journal_bytes=MAX_JOURNAL,
                preflight=preflight, progress=progress)
            if (stage.completed != 4733 or result['actual_complete_source_presentations'] != 302627
                    or result['actual_response_EOS_targets'] != 4669243):
                raise ValueError('The complete original4733step/302627view/4669243target epoch is required')
            error = None
            checkpoint = DEST / 'trained' / 'model.pt'
            if rank == 0:
                try:
                    checkpoint.parent.mkdir()
                    shutil.copyfile(Path(plan['initializer']['checkpoint']).parent / 'tokenizer.json', checkpoint.parent / 'tokenizer.json')
                    payload = export_native(view, sources=sources, tokenizer_sha256=plan['initializer']['tokenizer_sha256'],
                        mode='train', completed=4733, planned=4733, original_epoch_entries=4733)
                    atomic_checkpoint(checkpoint, payload, plan['maximum_native_bytes'], RESERVE)
                except Exception as exc:
                    error = repr(exc)
            _agree(error, None, None, 4)
            # This is a real final standalone load, with no old base needed.
            error = None
            cold = None
            try:
                from slm_lab.own_dense_standalone_v1 import load
                cold, _ = load(checkpoint, expected_checkpoint_sha256=file_hash(checkpoint), expected_sources=sources)
                cold = cold.to(device)
                inputs = torch.load(plan['cold_inputs']['path'], map_location='cpu', weights_only=True, mmap=True)
                if any(not torch.equal(p, dict(cold.named_parameters())[n]) for n, p in base.named_parameters()):
                    raise ValueError('Every actually exported final native value must cold-load exactly')
                view.eval()
                cold.eval()
                batches = {n: {k: v.to(device) for k, v in b.items()} for n, b in inputs.items()}
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    for name in ['text', 'photo']:
                        if not torch.equal(view.native(**batches[name])['logits'], cold(**batches[name])['logits']):
                            raise ValueError('Actual full native text/photo outputs must survive final standalone export exactly')
                    wanted = base.word_logits(batches['word']['word_pixels'])
                    actual = cold.word_logits(batches['word']['word_pixels'])
                    if not torch.equal(wanted, actual) or actual.dtype != torch.float32:
                        raise ValueError('The actual full original FP32 word readout must survive final standalone export exactly')
                cold_proof = dict(complete=True, all_actual_native_parameters_equal=True,
                    actual4096text448photo64x448word_outputs_bit_exact=True, actual_word_readout_dtype='torch.float32',
                    final_standalone_only_loaded_for_this_cold_model=True, no_overlay_or_parameter_copy_after_load=True,
                    actual_native_values=984399690, goal_achieved=False)
            except Exception as exc:
                error = repr(exc)
            _agree(error, None, None, 4)
            del cold
            guard.check(full=True)
            owner.owned(plan)
            atomic_json(folder / ('rank-' + str(rank) + '.json'), dict(complete=True, rank=rank, actual_complete_updates=4733,
                complete_source_presentations=302627, response_EOS_targets=4669243, full_final_native_cold=cold_proof,
                checkpoint_sha256=file_hash(checkpoint), actual_native_values=base.counts()['total_unique'], goal_achieved=False))
            if rank == 0:
                atomic_json(OUT / 'native-export.json', dict(complete=True, checkpoint=str(checkpoint),
                    checkpoint_sha256=file_hash(checkpoint), sources=sources, completed_updates=4733,
                    tokenizer_sha256=plan['initializer']['tokenizer_sha256'], actual_native_values=984399690,
                    every_exported_value_and_original4096text448photo_word_output_cold_load_exact=True,
                    final_inference_requires_only_complete_native_and_tokenizer=True, candidate_promoted=False, goal_achieved=False))
    finally:
        dist.destroy_process_group()


def load_candidate(plan, device='cpu'):
    from slm_lab.own_dense_standalone_v1 import load
    from slm_lab.dense_foundation_inference_v1 import DenseNativeView
    spec = plan['standalone']
    model, metadata = load(spec['checkpoint'], expected_checkpoint_sha256=spec['checkpoint_sha256'], expected_sources=spec['sources'])
    provenance = metadata['provenance']
    if (metadata['completed_updates'] != 4733 or model.counts()['total_unique'] != 984399690
            or metadata['tokenizer_sha256'] != plan['tokenizer_sha256']
            or provenance.get('mode') != 'train' or provenance.get('whole_production_source_epoch_completed') is not True):
        raise ValueError('Only the complete actual visual-native4733update student may enter final evaluation')
    model = model.to(device).eval()
    return model, DenseNativeView(model).eval(), metadata


def assessment(suite, gpu):
    import assess_machine_standalone_full_v2_assessment_recovery_gate_v1 as evaluator
    from slm_lab.visual_native_assessment_binding_v1 import bind
    plan = read(OUT / 'trained-assessment' / 'plan.json')
    def registered(subject, physical):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != str(physical) or os.sched_getscheduler(0) != os.SCHED_OTHER or os.nice(0) != 0:
            raise ValueError('Each new evaluation process needs normal scheduling and its original isolated L40')
        return owner.registered('visual-native-' + subject)
    with bind(evaluator, reports=OUT / 'trained-assessment', plan=plan, load_student=load_candidate, register=registered) as functions:
        functions[0](suite, gpu)


def parent():
    from slm_lab.joint_visual_runtime_contract_v1 import ready
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.sched_getscheduler(0) != os.SCHED_OTHER or os.nice(0) != 0
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])
            or OUT.exists() or DEST.exists() or not (PREP / 'ready.json').exists()):
        raise ValueError('Fresh normal-scheduled CUDA-hidden owned launch requires completed idle preparation')
    prepared = read(PREP / 'ready.json')
    if not terminal(read(PREP / 'plan.json')['owner']) or not ready(ROOT, **prepared['exact_predecessors']):
        raise ValueError('The original canonical/source and actual preparation owners must all be terminal and complete')
    if shutil.disk_usage(ROOT).free < prepared['required_free_bytes_before_launch']:
        raise ValueError('All bounded private generations, full native export, full assessment outputs and40GiBreserve must fit')
    memory = {k: int(v) for k, v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
    if int(Path('/sys/fs/cgroup/memory.max').read_text()) - sum(memory[k] for k in ['anon', 'shmem', 'kernel']) < 112 * 1024**3:
        raise ValueError('The complete new working set and64GiBof other-worker headroom must remain available')
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    guard = FileBindings(prepared['source_files_sha256'])
    guard.check(full=True)
    with owner.LEASE.open('a+') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner.idle_original()
        used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
        if len(used) != 4 or any(int(v) > 128 for v in used):
            raise ValueError('All four original L40s must be idle before this owned continuation')
        import torch
        from slm_lab.corrected_multimodal_batches_v1 import CorrectedMultimodalBatches
        from slm_lab.joint_visual_controller_v1 import require_full_capacity
        from slm_lab.joint_visual_objective_update_v1 import make_optimizer
        from slm_lab.joint_visual_private_checkpoint_v2 import FixedGuard, digest
        from slm_lab.joint_visual_training_stage_v1 import make_plan
        from slm_lab.own_actor_vision_prefix_split_backward_v1 import OwnActorVisionPrefixSplitBackward
        from slm_lab.own_dense_actor_v1 import OwnDenseActor
        from slm_lab.visual_native_assessment_binding_v1 import bind
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        me = owner.resources.identity(os.getpid())
        files = {**prepared['source_files_sha256'], **owner.lease_sources()}
        plan = dict(owner=me, waiter=owner.WAITER, terminal_waiter_receipt_sha256=owner.terminal_receipt_sha256(),
            source_files_sha256=files, expires_at_unix=time.time() + 7 * 86400,
            initializer=prepared['initializer'], cold_inputs=prepared['cold_inputs'], maximum_native_bytes=prepared['maximum_native_bytes'],
            cached_baseline=prepared['cached_baseline'], tokenizer_sha256=prepared['initializer']['tokenizer_sha256'],
            suites=list(SUITES), expected_student_values=984399690, cached_own_reference=prepared['cached_own_reference'],
            pulse_sources=prepared['pulse_sources'], automatic_retry=False, candidate_promoted=False, goal_achieved=False)
        base, _ = initializer(plan)
        if torch.cuda.is_initialized():
            raise ValueError('The owned parent initializes the complete student on CPUonly')
        view = OwnActorVisionPrefixSplitBackward(OwnDenseActor(base))
        options = dict(learning_rates=dict(vision=1e-5, pointing=1e-4, word=1e-4), betas=[.9, .95], eps=1e-5,
            weight_decay=.01, seed=987911, checkpoint_every=128)
        objective = dict(precision='bf16', maximum_examples=4, maximum_padded_tokens=4096,
            matching_weight=.1, temperature=.07, CTC_weight=.25, native_word_CE_weight=.25,
            gradient_clip=1., gradient_bucket_values=1024**2)
        optimizer = make_optimizer(view, learning_rates=options['learning_rates'], betas=options['betas'], eps=options['eps'], weight_decay=options['weight_decay'])
        fixed = FixedGuard(view, optimizer)
        data = CorrectedMultimodalBatches(ROOT, base.config, purpose='capacity')
        source_folder = data.folder
        capacity_plan = read(CASES / 'actual-complete-source-capacity-plan.json')
        entries = read(source_folder / 'complete-corrected-single-epoch-schedule.json')['entries']
        from slm_lab.joint_visual_runtime_contract_v1 import ADMISSION
        bound = dict(base_checkpoint_sha256=plan['initializer']['checkpoint_sha256'], tokenizer_sha256=plan['tokenizer_sha256'],
            source_admission_sha256=file_hash(ROOT / ADMISSION), complete_schedule_sha256=file_hash(source_folder / 'complete-corrected-single-epoch-schedule.json'),
            corrected_index_sha256=file_hash(data.index_path), fixed_values_sha256=fixed.sha256)
        OUT.mkdir()
        DEST.mkdir()
        (OUT / 'stage-plans').mkdir()
        stage_plans = {}
        for mode, selected in [('capacity', capacity_plan['capacity_entries']), ('train', entries)]:
            content = make_plan(view, optimizer, fixed, sources=bound, entries=selected, options=objective, mode=mode)
            path = OUT / 'stage-plans' / (mode + '.json')
            with path.open('xb') as stream:
                stream.write(json.dumps(content, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
                stream.flush()
                os.fsync(stream.fileno())
            if file_hash(path) != digest(content):
                raise ValueError('The canonical numerical stage plan must bind its actual exact file bytes')
            stage_plans[mode] = dict(path=str(path), sha256=file_hash(path))
            files[str(path)] = file_hash(path)
        plan.update(stage_plans=stage_plans, training_options=options, objective_options=objective,
            capacity_plan_semantic_sha256=digest(capacity_plan), complete_visual_source_budget=dict(updates=4733, views=302627,
                context_tokens=17681556, response_EOS_targets=4669243, word_views=91585, whole_CTC_UTF8_bytes=644063),
            old_failed_word_and_prefix_numeric_gates_unwaived=True, all_original_canonical_scopes_preserved=True,
            independent_neural_score_and_generated_program_readbacks_still_required=True)
        guard = FileBindings(files)
        guard.check(full=True)
        atomic_json(OUT / 'plan.json', plan)
        atomic_json(OUT / 'children.json', [])
        del data, base, view, optimizer, fixed
        gc.collect()
        driver = str(Path(__file__).resolve())
        watcher = None
        success = False
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Owned visual continuation interrupted')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            watcher = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', driver, '--watchdog'], cwd=ROOT,
                stdout=(OUT / 'watchdog.log').open('x'), stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
            for mode in ['capacity', 'train']:
                child = owner.launch(plan, ['-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                    driver, '--distributed', mode], 'visual-' + mode + '-launcher', '0,1,2,3')
                owner.wait_children(plan, [child])
                if mode == 'capacity':
                    proofs = [read(OUT / 'capacity' / ('rank-' + str(r) + '.json')) for r in range(4)]
                    receipt = require_full_capacity(proofs, expected_bindings=bound, expected_capacity_plan_sha256=digest(capacity_plan))
                    atomic_json(OUT / 'capacity-complete.json', receipt)
            receipt = read(OUT / 'native-export.json')
            if any(not read(OUT / 'train' / ('rank-' + str(r) + '.json'))['complete'] for r in range(4)):
                raise ValueError('All four actual complete training and final native cold receipts are required')
            assessment_plan = dict(plan, standalone=receipt)
            assessment_plan['source_files_sha256'] = {**files, str(OUT / 'native-export.json'): file_hash(OUT / 'native-export.json'),
                receipt['checkpoint']: receipt['checkpoint_sha256'], str(Path(receipt['checkpoint']).parent / 'tokenizer.json'): receipt['tokenizer_sha256']}
            assessment_folder = OUT / 'trained-assessment'
            assessment_folder.mkdir()
            atomic_json(assessment_folder / 'plan.json', assessment_plan)
            import assess_machine_standalone_full_v2_assessment_recovery_gate_v1 as evaluator
            for suite in SUITES:
                if shutil.disk_usage(ROOT).free < RESERVE + 1024**3:
                    raise ValueError('The complete next assessment suite must fit above the original writer reserve')
                children = [owner.launch(plan, ['-u', driver, '--assess', suite, '--gpu', str(r)],
                    'visual-native-' + suite + '-' + str(r), str(r)) for r in range(4)]
                owner.wait_children(plan, children)
                with bind(evaluator, reports=assessment_folder, plan=assessment_plan, load_student=load_candidate,
                        register=lambda *args: plan) as functions:
                    functions[1](suite, assessment_plan)
                if sum(p.stat().st_size for p in assessment_folder.rglob('*') if p.is_file()) > prepared['maximum_candidate_assessment_bytes']:
                    raise ValueError('The complete preserved candidate assessments exceeded their reserved size; do not truncate or omit results')
            # Baseline results are the exact already completed canonical native
            # endpoint, never a chosen score winner or a smaller re-run.
            from compare_visual_native_endpoints_v1 import compare
            compare(Path(plan['cached_baseline']['folder']), assessment_folder / 'standalone', OUT / 'matched-comparisons.json')
            guard.check(full=True)
            success = True
        except BaseException as exc:
            atomic_json(OUT / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(),
                automatic_retry=False, goal_achieved=False))
            raise
        finally:
            owner.restore_owned(plan, 'Owned complete visual continuation ended; only its registered children released')
            if watcher is not None:
                watcher.wait(timeout=20)
        if success:
            atomic_json(OUT / 'complete.json', dict(complete=True, all4733updates302627views4669243targets_complete=True,
                all4actual_GPUcapacity_recovery_and_final_native_cold_passed=True, native_values=984399690,
                all_full_QA_AI2D_MBPP_human_words_code_broad_candidate_suites_complete=True,
                exact_completed_aligned_native_baseline_reused=True, original_nine_endpoint_pipeline_preserved=True,
                independent_neural_score_and_program_readbacks_pending=True, candidate_promoted=False, goal_achieved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--watchdog', action='store_true')
    parser.add_argument('--distributed', choices=['capacity', 'train'])
    parser.add_argument('--assess', choices=SUITES)
    parser.add_argument('--gpu', type=int, choices=range(4))
    args = parser.parse_args()
    configure()
    if args.watchdog:
        owner.watchdog()
    elif args.distributed:
        distributed(args.distributed)
    elif args.assess:
        assessment(args.assess, args.gpu)
    elif args.run:
        parent()
    else:
        parser.error('Use explicit --run after preparation, or an owned child mode')
