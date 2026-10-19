"""Build the full joint GPU factory on CPU from the completed own initializer.

This is an actual984Mnative/999Mresearch source/graph check, not GPU capacity,
an owned production experiment, a neural update or a quality assessment.
The full original preceding experiment queue remains independently required.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RESERVE = 40 * 1024**3


def main(source_folder, output):
    from slm_lab.complete_joint_owned_recipe_v1 import read, sha, FORMAT, FULL, OPTIONS, OPTIMIZER, ACTOR
    from slm_lab.completed_joint_initializer_admission_v1 import completed_source
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.local_source_closure_v2 import python_sources
    source_folder, output = Path(source_folder).resolve(), Path(output).absolute()
    if (output.exists() or output.resolve() != output or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])
            or shutil.disk_usage(output.parent).free < RESERVE + 64 * 1024**2):
        raise ValueError('A fresh funded one-thread CUDA-hidden CPU factory diagnostic is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0)); allowed = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {20 if 20 in allowed else max(allowed)})
    original = read(source_folder / 'plan.json'); expected = original['owner']
    files = {str(p): sha(p) for p in python_sources(ROOT, [Path(__file__)])}
    files[str(source_folder / 'plan.json')] = sha(source_folder / 'plan.json')
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]); output.mkdir()
    states = {}
    def publication_guard():
        for path, digest in files.items():
            stat = Path(path).stat(); current = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if path not in states:
                if sha(path) != digest: raise ValueError('An actual original CPU factory source changed')
                states[path] = current
            elif states[path] != current or stat.st_size <= 8 * 1024**2 and sha(path) != digest:
                raise ValueError('Every original CPU factory source byte and identity must remain unchanged')
    def write(name, value):
        from slm_lab.reserved_atomic_metadata_v1 import write_json
        return write_json(output / name, value, root=output, minimum_free_bytes=RESERVE, maximum_metadata_bytes=32 * 1024**2,
            guard=publication_guard)
    write('owner-plan.json', dict(owner=owner, exact_original_initializer_owner=expected, source_files_sha256=files,
        scope='full original source and graph construction on CPU only', GPU_contexts=0, actual_optimizer_updates=0,
        preceding_full_experiment_queue_and_owned_GPU_capacity_training_still_required=True, goal_achieved=False))
    try:
        deadline = time.monotonic() + 6 * 3600
        while not terminal(expected):
            if read(source_folder / 'plan.json')['owner'] != expected: raise ValueError('Preserve the exact original initializer owner')
            write('progress.json', dict(phase='waiting for exact full-size CPU initializer owner', GPU_contexts=0))
            if time.monotonic() > deadline: raise ValueError('The source observation lifetime expired;the original owner is not relabelled stopped')
            time.sleep(15)
        admitted = completed_source(source_folder)
        if admitted is None: raise ValueError('The exact terminal successful original CPU initializer is required')
        files.update(admitted['source_files_sha256'])
        write('initializer-source-admission.json', admitted)
        from slm_lab.codenet_checkpoint_v1 import FileBindings, capture_rng
        from slm_lab.complete_joint_GPU_runtime_v1 import build
        from slm_lab.parallel_function_repair_stage_v1 import value_digest
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
        capacity = ROOT / 'reports/complete-joint-actual-capacity-cases-CPU-v3/actual-complete-joint-capacity-plan.json'
        files[str(capacity)] = sha(capacity)
        guard = FileBindings(files)
        stats = {k: int(v) for k, v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
        if int(Path('/sys/fs/cgroup/memory.max').read_text()) - sum(stats[k] for k in ['anon', 'shmem', 'kernel']) < 112 * 1024**3:
            raise ValueError('The actual full CPU model and unchanged other-worker memory headroom must be funded')
        plan = dict(format=FORMAT, initializer=admitted['initializer'], full_original_budget=FULL, options=OPTIONS,
            optimizer=OPTIMIZER, actor_settings=ACTOR, expected_native_values=984399690, expected_research_values=999706442,
            capacity_plan=str(capacity), capacity_plan_sha256=sha(capacity), source_files_sha256=files,
            explicit_full_size_CPU_graph_diagnostic=True, GPU_training_admitted=False, preceding_full_experiments_still_required=True)
        write('factory-input-plan.json', plan); before = value_digest(capture_rng('cpu'))
        write('progress.json', dict(phase='constructing actual complete own984Mnative999Mresearch joint factory on CPU', GPU_contexts=0))
        parts = build(ROOT, output / 'factory-input-plan.json', mode='capacity', device='cpu')
        if (parts['view'].actor.counts()['actual_unique'] != 999706442 or parts['optimizer'].state
                or len(parts['entries']) != 47 or len(parts['source']) != 5362 or torch.cuda.is_initialized()
                or before != value_digest(capture_rng('cpu'))):
            raise ValueError('Every actual complete original source,own full graph,empty Adam and private CPU RNG must remain')
        parts['fixed_guard'].check(parts['view'], parts['optimizer'], full=True); parts['reference'].assert_bound(full=True)
        guard.check(full=True); parts['source'].guard.check(full=True)
        write('complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
            actual_counts=parts['view'].actor.counts(), actual_selected_parameter_leaves=len(parts['view'].named_selected()),
            actual_selected_values=sum(p.numel() for _, p in parts['view'].named_selected()), actual_full_original_joint_steps=len(parts['source']),
            actual_complete_capacity_entries=len(parts['entries']), actual_joint_source_bindings=parts['sources'],
            full_native_initializer_loaded_and_graph_and_source_identity_reconciled=True, shared_own_text_reference_unchanged=True,
            exact_original_CPU_RNG_unchanged=True, actual_optimizer_updates=0, GPU_contexts=0,
            owned_GPU_capacity_training_and_all_original_assessments_still_required=True, goal_achieved=False))
        print(json.dumps(dict(complete=True, actual_counts=parts['view'].actor.counts(), actual_optimizer_updates=0)), flush=True)
    except BaseException as exc:
        write('failure.json', dict(owner=owner, error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False, goal_achieved=False)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initializer', type=Path, default=Path('/tmp/slm-completed-joint-initializer-CPU-v3'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/slm-complete-joint-GPU-factory-CPU-v2'))
    args = parser.parse_args(); main(args.initializer, args.output)
