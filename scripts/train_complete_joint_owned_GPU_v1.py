"""Own the full joint experiment after all original experiments complete.

Wait without Torch, GPU contexts or a lease. A fresh funded parent then owns
all four L40s for baseline assessment,47-case capacity/replay, independent
native export/GPUcold, all5362updates, final export/cold and full assessment.
No original process is signalled, reprioritized or adopted as a new child.
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
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RESERVE = 40 * 1024**3


def read(path): return json.loads(Path(path).read_text())


def configure(workspace):
    global OUT, DEST, WORKSPACE, owner
    WORKSPACE = Path(workspace).resolve(); OUT = WORKSPACE / 'reports'; DEST = WORKSPACE / 'checkpoints'
    import own_training_gpu_lease_v2 as owner
    owner.REPORTS = OUT; owner.resources.REPORTS = OUT


def normal():
    if (os.nice(0) != 0 or os.sched_getscheduler(0) != os.SCHED_OTHER
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Only an actual normal nice0 one-thread owned parent/worker is eligible')


def preflight(plan):
    normal(); owner.owned(plan)
    if shutil.disk_usage(WORKSPACE).free < RESERVE:
        raise ValueError('The actual complete joint workspace must preserve the unchanged40GiBwriter reserve')
    stats = {k: int(v) for k, v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
    if int(Path('/sys/fs/cgroup/memory.max').read_text()) - sum(stats[k] for k in ['anon', 'shmem', 'kernel']) < 112 * 1024**3:
        raise ValueError('The complete working memory and original other-worker headroom must remain available')


def gpu(rank, visible):
    import torch
    normal()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != visible:
        raise ValueError('An actual owned worker must retain its exact original physical GPU visibility')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(rank); device = torch.device('cuda', rank)
    torch.cuda.set_per_process_memory_fraction(.85, device)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.get_device_name(device) != 'NVIDIA L40': raise ValueError('Only an original physical L40 is eligible')
    return device


def distributed(mode):
    from slm_lab.complete_joint_GPU_runtime_v1 import build, stage
    from slm_lab.diverse_images import atomic_json
    from torch import distributed as dist
    import torch
    import random
    import numpy as np
    if os.environ.get('WORLD_SIZE') != '4': raise ValueError('Every actual joint stage requires all four original GPUs')
    rank = int(os.environ['RANK']); plan = owner.registered(mode + '-rank-' + str(rank)); preflight(plan)
    torch.set_num_interop_threads(1); device = gpu(rank, '0,1,2,3')
    dist.init_process_group('nccl', timeout=timedelta(minutes=45))
    control = dist.new_group(backend='gloo', timeout=timedelta(minutes=45))
    try:
        parts = build(ROOT, OUT / 'plan.json', mode=mode, device=device)
        runner = stage(parts, OUT, mode=mode, control_group=control, gradient_group=dist.group.WORLD)
        seed = plan['private_seed'] + rank; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        directory = DEST / mode / 'state'; folder = OUT / mode
        def progress(*args):
            preflight(plan)
            if rank == 0:
                value = args[0] if mode == 'train' else dict(actual_completed_capacity_cases=args[0],
                    planned_capacity_cases=47, production_updates=0, goal_achieved=False)
                atomic_json(folder / 'progress.json', value)
        if mode == 'capacity':
            from slm_lab.complete_joint_capacity_run_v2 import run
            proof = run(runner, directory, maximum_bytes=plan['storage_budget']['maximum_actual_checkpoint_bytes'],
                minimum_free_bytes=RESERVE, preflight=lambda: preflight(plan), progress=progress)
        else:
            from slm_lab.complete_joint_training_loop_v1 import run
            capacities = [read(OUT / 'capacity' / ('rank-' + str(r) + '.json')) for r in range(4)]
            colds = [read(OUT / 'capacity' / ('cold-' + str(r) + '.json')) for r in range(4)]
            proof = run(runner, directory, capacity_reports=capacities, cold_reports=colds, capacity_plan=parts['capacity'],
                checkpoint_every=plan['checkpoint_every'], maximum_checkpoint_bytes=plan['storage_budget']['maximum_actual_checkpoint_bytes'],
                minimum_free_bytes=RESERVE, preflight=lambda: preflight(plan), progress=progress)
        parts['initializer_guard'].check(full=True); parts['source'].guard.check(full=True); preflight(plan)
        atomic_json(folder / ('rank-' + str(rank) + '.json'), proof); dist.barrier(group=control)
    finally:
        dist.destroy_process_group()


def export_native(mode):
    import torch
    from slm_lab.complete_joint_GPU_runtime_v1 import build
    from slm_lab.joint_indexed_durable_state_v3 import read_latest
    from slm_lab.joint_selected_native_export_v1 import export
    from slm_lab.diverse_images import atomic_json
    plan = owner.registered(mode + '-CPU-export'); preflight(plan)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU native export requires hidden CUDA')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    parts = build(ROOT, OUT / 'plan.json', mode=mode); directory = DEST / mode / 'state'
    _, _, pointer = read_latest(directory)
    inputs = torch.load(plan['cold_inputs']['path'], map_location='cpu', weights_only=True, mmap=True)
    proof = export(parts['view'], parts['optimizer'], directory, DEST / mode / 'native', plan['initializer']['tokenizer'],
        sources=parts['sources'], entries=parts['entries'], source_files=parts['source_files'], expected_world=4,
        expected_completed=47 if mode == 'capacity' else 5362, expected_pointer_sha256=pointer, mode=mode,
        probes=[('original_complete4096text', inputs['text']), ('original_complete448photo', inputs['photo'])],
        word_pixels=inputs['word']['word_pixels'], working_budget_bytes=16 * 1024**3,
        maximum_checkpoint_bytes=plan['storage_budget']['maximum_complete_native_bytes'], minimum_free_bytes=RESERVE)
    preflight(plan); atomic_json(OUT / mode / 'native-export.json', proof)


def native_cold(mode, physical):
    import torch
    from slm_lab.complete_joint_GPU_runtime_v1 import build
    from slm_lab.joint_indexed_durable_state_v3 import read_latest
    from slm_lab.joint_native_source_v1 import reconstruct
    from slm_lab.joint_native_GPU_cold_v1 import prove
    from slm_lab.own_dense_standalone_v1 import load
    from slm_lab.diverse_images import atomic_json
    plan = owner.registered(mode + '-native-cold-' + str(physical)); preflight(plan)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(physical) or torch.cuda.is_initialized():
        raise ValueError('The independent source must be rebuilt on CPU before acquiring its owned CUDA context')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    parts = build(ROOT, OUT / 'plan.json', mode=mode); directory = DEST / mode / 'state'
    _, _, pointer = read_latest(directory)
    expected, source_proof = reconstruct(parts['view'], parts['optimizer'], directory, sources=parts['sources'], entries=parts['entries'],
        source_files=parts['source_files'], expected_world=4, expected_completed=47 if mode == 'capacity' else 5362,
        expected_pointer_sha256=pointer, mode=mode, working_budget_bytes=16 * 1024**3)
    artifact = read(OUT / mode / 'native-export.json')
    candidate, _ = load(artifact['checkpoint'], expected_checkpoint_sha256=artifact['checkpoint_sha256'], expected_sources=parts['sources'])
    sources = parts['sources']; del parts; gc.collect()
    device = gpu(0, str(physical)); expected.to(device); candidate.to(device)
    raw = torch.load(plan['cold_inputs']['path'], map_location='cpu', weights_only=True, mmap=True)
    inputs = {name: {k: v.to(device) for k, v in raw[name].items() if isinstance(v, torch.Tensor)} for name in ['text', 'photo', 'word']}
    result = prove(expected, candidate, source_proof, checkpoint_sha256=artifact['checkpoint_sha256'], sources=sources,
        inputs=inputs, physical_GPU=physical)
    preflight(plan); atomic_json(OUT / mode / ('cold-' + str(physical) + '.json'), result)


def load_candidate(plan, device='cpu'):
    from slm_lab.own_dense_standalone_v1 import load
    from slm_lab.dense_foundation_inference_v1 import DenseNativeView
    spec = plan['standalone']; model, metadata = load(spec['checkpoint'], expected_checkpoint_sha256=spec['checkpoint_sha256'], expected_sources=spec['sources'])
    if (metadata['completed_updates'] != (0 if plan['endpoint'] == 'initial' else 5362)
            or model.counts()['total_unique'] != 984399690 or metadata['tokenizer_sha256'] != plan['tokenizer_sha256']
            or plan['endpoint'] != 'initial' and not metadata['provenance'].get('completed_original_joint_source_epoch')):
        raise ValueError('Only the exact complete initial or full5362update own native can enter unchanged full assessments')
    model.to(device).eval(); return model, DenseNativeView(model).eval(), metadata


def assessment(endpoint, suite, physical):
    import train_native_code_suffix_aligned_GPU_v4 as original
    previous = original.OUT, original.load_candidate, getattr(original, 'owner', None)
    original.OUT, original.load_candidate, original.owner = OUT, load_candidate, owner
    try:
        original.assessment(endpoint, suite, physical)
    finally:
        original.OUT, original.load_candidate = previous[:2]
        if previous[2] is None: del original.owner
        else: original.owner = previous[2]


def parent(args):
    from slm_lab.complete_joint_owned_recipe_v1 import prepare, SUITES, sha
    from slm_lab.full_experiment_launch_queue_v1 import identity
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.reserved_atomic_metadata_v1 import write_json
    normal()
    workspace = Path(args.workspace).absolute()
    if workspace.exists() or workspace.resolve() != workspace or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'torch' in sys.modules:
        raise ValueError('A fresh CUDA-hidden non-neural observer must precede the new full owned experiment')
    queue = Path(args.queue).resolve(); queue_owner = read(queue / 'plan.json')['owner']
    files = {str(p): sha(p) for p in python_sources(ROOT, [Path(__file__)])}
    files[str(queue / 'plan.json')] = sha(queue / 'plan.json')
    workspace.mkdir(); reports = workspace / 'reports'; reports.mkdir()
    def guard():
        normal()
        if any(sha(p) != h for p, h in files.items()): raise ValueError('Every exact original queue plan and new runtime byte must remain')
    def write(name, value):
        return write_json(reports / name, value, root=workspace, minimum_free_bytes=RESERVE,
            maximum_metadata_bytes=64 * 1024**2, guard=guard)
    write('waiter-plan.json', dict(owner=identity(os.getpid()), source_files_sha256=files, exact_original_queue_owner=queue_owner,
        all_original_experiments_and_scopes_preserved=True, GPU_contexts=0, GPU_leases=0, production_updates=0, automatic_retry=False))
    deadline = time.monotonic() + 35 * 86400
    while True:
        guard()
        recipe = prepare(ROOT, queue=queue, expected_queue_owner=queue_owner, initializer=args.initializer,
            measurement=args.measurement, runtime_sources=files)
        if recipe is not None:
            required = recipe['storage_budget']['required_free_bytes_before_launch']
            if shutil.disk_usage(workspace).free >= required: break
        write('waiting-progress.json', dict(waiting_for='complete exact original experiment queue,own initializer and full physical budget',
            original_queue_owner=queue_owner, no_existing_workers_modified=True, GPU_work_started=False, goal_achieved=False))
        if time.monotonic() > deadline: raise ValueError('The source observation lifetime expired; no original owner is relabelled stopped')
        time.sleep(30)
    configure(workspace)
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.diverse_images import atomic_json
    runtime_guard = FileBindings(recipe['source_files_sha256'])
    with owner.LEASE.open('a+') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB); owner.idle_original()
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
        if len(rows) != 4 or any(r.rsplit(',', 1)[0].strip() != 'NVIDIA L40' or int(r.rsplit(',', 1)[1]) > 128 for r in rows):
            raise ValueError('All four original L40s must be actually idle before new owned work')
        native = DEST / 'initial'; native.mkdir(parents=True)
        for name in ['model.pt', 'tokenizer.json']:shutil.copyfile(Path(recipe['initializer']['checkpoint']).parent / name, native / name)
        initial = dict(recipe['initializer'], checkpoint=str(native / 'model.pt'), tokenizer=str(native / 'tokenizer.json'))
        if sha(native / 'model.pt') != initial['checkpoint_sha256'] or sha(native / 'tokenizer.json') != initial['tokenizer_sha256']:
            raise ValueError('The complete initial native copy must preserve every actual source byte')
        bound = dict(recipe['source_files_sha256']); bound.update(owner.lease_sources())
        bound.update({str(native / name):sha(native / name) for name in ['model.pt', 'tokenizer.json']})
        plan = dict(recipe, owner=identity(os.getpid()), initializer=initial, source_files_sha256=bound,
            waiter=owner.WAITER, terminal_waiter_receipt_sha256=owner.terminal_receipt_sha256(), expires_at_unix=time.time() + 7 * 86400,
            expected_student_values=984399690, tokenizer_sha256=initial['tokenizer_sha256'], automatic_retry=False,
            candidate_promoted=False, goal_achieved=False)
        atomic_json(OUT / 'plan.json', plan); atomic_json(OUT / 'children.json', [])
        for mode in ['capacity', 'train']:(OUT / mode).mkdir(); (DEST / mode).mkdir()
        driver = str(Path(__file__).resolve()); base = ['-u', driver, '--workspace', str(WORKSPACE)]
        watcher = subprocess.Popen([str(ROOT / '.venv/bin/python'), *base, '--watchdog'], cwd=ROOT,
            stdout=(OUT / 'watchdog.log').open('x'), stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
        def interrupted(signum, frame): raise KeyboardInterrupt('Owned joint experiment interrupted')
        signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
        def launch(extra, subject, visible=''):
            preflight(plan); child = owner.launch(plan, [*base, *extra], subject, visible); owner.wait_children(plan, [child])
        def cold(mode):
            children = [owner.launch(plan, [*base, '--cold', mode, '--gpu', str(r)], mode + '-cold-' + str(r), str(r)) for r in range(4)]
            owner.wait_children(plan, children)
        def assess(endpoint, standalone):
            from slm_lab.visual_native_assessment_binding_v1 import bind
            from slm_lab.public_reasoning_reserved_evaluation_v1 import PublicReasoningReserved
            from slm_lab.reasoning_reserved_shards_v1 import reconcile
            import assess_machine_standalone_full_v2_assessment_recovery_gate_v1 as evaluator
            folder = OUT / endpoint / 'assessment'; folder.mkdir(parents=True)
            assessment_plan = dict(plan, endpoint=endpoint, standalone=standalone,
                source_files_sha256={**bound, standalone['checkpoint']:standalone['checkpoint_sha256']})
            atomic_json(folder / 'plan.json', assessment_plan)
            for suite in [*SUITES, 'reasoning']:
                children = [owner.launch(plan, [*base, '--assess', suite, '--endpoint', endpoint, '--gpu', str(r)],
                    endpoint + '-' + suite + '-' + str(r), str(r)) for r in range(4)]
                owner.wait_children(plan, children)
                if suite == 'reasoning':
                    population = PublicReasoningReserved(ROOT / 'data/public-reasoning-curriculum-v1', ROOT / 'reports/public-reasoning-curriculum-v1-readback-admission/complete.json')
                    reconcile(population, [folder / ('reasoning-shard-' + str(r) + '.json') for r in range(4)], folder / 'all-reserved-reasoning.json')
                else:
                    with bind(evaluator, reports=folder, plan=assessment_plan, load_student=load_candidate, register=lambda *a:plan) as functions:
                        functions[1](suite, assessment_plan)
                if sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()) > plan['storage_budget']['maximum_each_complete_assessment_bytes']:
                    raise ValueError('Preserve every original result; full assessment output exceeded its funded physical budget')
        success = False
        try:
            preflight(plan); assess('initial', initial)
            for mode in ['capacity', 'train']:
                child = owner.launch(plan, ['-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                    driver, '--workspace', str(WORKSPACE), '--distributed', mode], mode + '-distributed-launcher', '0,1,2,3')
                owner.wait_children(plan, [child])
                launch(['--export', mode], mode + '-export'); cold(mode)
            final = read(OUT / 'train/native-export.json'); assess('joint', final)
            from compare_visual_native_endpoints_v1 import compare
            compare(OUT / 'initial/assessment/standalone', OUT / 'joint/assessment/standalone', OUT / 'matched-comparisons.json')
            before, after = [read(OUT / endpoint / 'assessment/all-reserved-reasoning.json') for endpoint in ['initial', 'joint']]
            keys = ['position', 'source_id', 'source_group', 'task', 'original_prompt_ids', 'original_native_generation_budget']
            if (len(before['results']) != 289 or len(after['results']) != 289
                    or any(any(a[k] != b[k] for k in keys) for a, b in zip(before['results'], after['results']))):
                raise ValueError('All289original reserved reasoning sources and complete native budgets must match both endpoints')
            atomic_json(OUT / 'matched-reasoning.json', dict(before_correct=before['correct'], after_correct=after['correct'],
                original_cases=289, source_reserved_only_historical_independence_not_certified=True, candidate_promoted=False))
            runtime_guard.check(full=True); preflight(plan); success = True
        except BaseException as exc:
            atomic_json(OUT / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False, goal_achieved=False)); raise
        finally:
            owner.restore_owned(plan, 'Owned full joint experiment ended; only its registered children released')
            watcher.wait(timeout=20)
        if success:
            atomic_json(OUT / 'complete.json', dict(complete=True, all5362original_joint_updates_and_full_native_export_complete=True,
                all47actual_four_L40_capacity_cases_and_private_replay_and_native_GPUcold_complete=True,
                both_full_original_QA_AI2D_MBPP_human_words_code_broad_and_reserved_reasoning_assessments_complete=True,
                actual_native_values=984399690, original_nine_endpoint_pipeline_preserved=True,
                independent_neural_score_and_program_readback_and_fresh_own_output_human_judgments_pending=True,
                candidate_promoted=False, goal_achieved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('/tmp/slm-complete-joint-owned-GPU-v1'))
    parser.add_argument('--queue', type=Path, default=Path('/tmp/slm-full-owned-experiment-launch-queue-v2'))
    parser.add_argument('--initializer', type=Path, default=Path('/tmp/slm-completed-joint-initializer-CPU-v2'))
    parser.add_argument('--measurement', type=Path, default=Path('/tmp/slm-full-joint-selected-state-size-CPU-v1'))
    parser.add_argument('--run', action='store_true'); parser.add_argument('--watchdog', action='store_true')
    parser.add_argument('--distributed', choices=['capacity', 'train']); parser.add_argument('--export', choices=['capacity', 'train'])
    parser.add_argument('--cold', choices=['capacity', 'train']); parser.add_argument('--gpu', type=int, choices=range(4))
    parser.add_argument('--assess', choices=['code', 'broad', 'preferences', 'QA', 'AI2D', 'MBPP', 'words', 'reasoning'])
    parser.add_argument('--endpoint', choices=['initial', 'joint']); args = parser.parse_args()
    if args.run: parent(args)
    else:
        configure(args.workspace)
        if args.watchdog: owner.watchdog()
        elif args.distributed: distributed(args.distributed)
        elif args.export: export_native(args.export)
        elif args.cold and args.gpu is not None: native_cold(args.cold, args.gpu)
        elif args.assess and args.endpoint and args.gpu is not None: assessment(args.endpoint, args.assess, args.gpu)
        else: parser.error('Use explicit --run or an actual owned child mode')
