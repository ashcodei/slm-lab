"""Non-neural handoff from exact preparers to serial owned experiment drivers.

Each original driver retains its full source/capacity/training/evaluation gates
and exclusive GPU lease. This observer takes no GPU lease or CUDA context,
never changes an original worker and never retries a failed experiment.
Completing this experiment queue does not certify the requested final model.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .exact_waiting_process_retirement_v1 import terminal

STAGES = (
    dict(name='visual', preparation='joint-visual-aligned-continuation-preparation-v1',
        driver='train_joint_visual_aligned_continuation_GPU_v2.py', output='joint-visual-aligned-continuation-GPU-v2',
        count_key='native_values', scope=['all4733updates302627views4669243targets_complete',
            'all_full_QA_AI2D_MBPP_human_words_code_broad_candidate_suites_complete']),
    dict(name='mixed_SFT', preparation='mixed-coding-SFT-aligned-preparation-v1',
        driver='train_mixed_coding_SFT_aligned_GPU_v1.py', output='mixed-coding-SFT-aligned-GPU-v1',
        count_key='actual_native_values', scope=['all84084presentations21316734targets5256updates_per_profile',
            'all_original_full_QA_AI2D_MBPP_human_words_code_broad_suites_each_candidate_complete']),
    dict(name='reasoning', preparation='batched-reasoning-aligned-preparation-v2',
        driver='train_batched_reasoning_aligned_GPU_v2.py', output='batched-reasoning-aligned-GPU-v2',
        count_key='actual_native_values', scope=['all7704presentations57240targets1926updates_per_profile',
            'all_original_full_QA_AI2D_MBPP_human_words_code_broad_suites_each_candidate_complete']),
    dict(name='native_code', preparation='native-code-suffix-aligned-preparation-CPU-v4',
        driver='train_native_code_suffix_aligned_GPU_v4.py', output='native-code-suffix-aligned-GPU-v4',
        count_key='actual_native_values', scope=['all44652presentations7010616targets11172updates_per_profile',
            'all_original_full_QA_AI2D_MBPP_human_words_code_broad_suites_each_candidate_complete',
            'all289reserved_reasoning_cases_at_initial_and_both_candidates_complete']),
)


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())


def identity(pid):
    fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, ticks=fields[19])


def context():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 0
            or os.sched_getscheduler(0) != os.SCHED_OTHER or 'torch' in sys.modules
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Only an actual normal nice0 one-thread CUDA-hidden non-neural queue may hand off a new owned driver')


def preparation(root, spec, captured):
    folder = Path(root) / 'reports' / spec['preparation']
    plan_path = folder / 'plan.json'
    if sha(plan_path) != captured['plan_sha256'] or read(plan_path)['owner'] != captured['owner']:
        raise ValueError('The exact original source preparer and complete captured plan cannot change')
    if not terminal(captured['owner']):
        return None
    if ((folder / 'failure.json').exists() or not (folder / 'complete.json').is_file()
            or not (folder / 'ready.json').is_file()):
        raise ValueError('An exact original preparer stopped without successful complete evidence')
    done, ready = read(folder / 'complete.json'), read(folder / 'ready.json')
    if (done.get('complete') is not True or done.get('ready_sha256') != sha(folder / 'ready.json')
            or ready.get('complete') is not True or ready.get('GPU_work_started') is not False
            or ready.get('production_updates') != 0 or type(ready.get('required_free_bytes_before_launch')) is not int
            or ready['required_free_bytes_before_launch'] <= 40 * 1024**3 or not ready.get('source_files_sha256')):
        raise ValueError('Only a complete original source-only recipe with all declared physical allocations is eligible')
    return ready


def host_resources(root, required_bytes):
    """Read resource state without opening a lease or constructing a model."""
    free = shutil.disk_usage(root).free
    if free < required_bytes:
        return dict(ready=False, waiting_for='complete physical storage funding', actual_free_bytes=free, required_free_bytes=required_bytes)
    stats = {k: int(v) for k, v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
    limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    if limit == 'max': raise ValueError('The original explicit bounded host memory contract is required')
    available = int(limit) - sum(stats[k] for k in ['anon', 'shmem', 'kernel'])
    if available < 112 * 1024**3:
        return dict(ready=False, waiting_for='complete host working memory and other-worker headroom', actual_available_bytes=available)
    lease = Path(root) / 'reports/code-residual-production-v4.gpu.lock'
    if lease.exists():
        stat = lease.stat(); expected = os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino
        for row in Path('/proc/locks').read_text().splitlines():
            fields = row.split()
            if len(fields) >= 8 and fields[1:4] == ['FLOCK', 'ADVISORY', 'WRITE']:
                major, minor, inode = fields[5].split(':')
                if (int(major, 16), int(minor, 16), int(inode)) == expected:
                    return dict(ready=False, waiting_for='the existing exclusive GPU lease', actual_lease_owner_pid=int(fields[4]))
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
    if len(rows) != 4: raise ValueError('The four original physical GPUs must remain present')
    parsed = [row.rsplit(',', 1) for row in rows]
    if any(name.strip() != 'NVIDIA L40' for name, _ in parsed): raise ValueError('Only the original four L40s may be used')
    used = [int(value) for _, value in parsed]
    return dict(ready=all(value <= 128 for value in used), waiting_for='idle original four L40s', actual_GPU_memory_used_MiB=used)


def completion(root, spec, launched):
    folder = Path(root) / 'reports' / spec['output']
    required = [folder / name for name in ['plan.json', 'complete.json', 'children.json', 'restored.json']]
    if any(not path.is_file() or path.is_symlink() for path in required) or (folder / 'failure.json').exists():
        raise ValueError('A launched owned experiment must retain all complete original outputs and successful restoration')
    plan, done, children, restored = [read(path) for path in required]
    if (any(plan['owner'].get(k) != launched[k] for k in ['pid', 'ticks']) or not terminal(launched)
            or any(not terminal(child) for child in children) or done.get('complete') is not True
            or done.get(spec['count_key']) != 984399690 or done.get('original_nine_endpoint_pipeline_preserved') is not True
            or any(done.get(key) is not True for key in spec['scope'])
            or restored.get('all_registered_point_actors_terminal') is not True):
        raise ValueError('Every full declared training/evaluation scope and exact terminal owned actor must remain')
    return dict(stage=spec['name'], owner=launched, complete=True,
        original_complete_receipts_sha256={str(path): sha(path) for path in required}, final_model_capability_not_certified=True)


def run(root, directory, captures, *, guard, publish, resources=host_resources, pause=time.sleep,
        now=time.monotonic, maximum_wait_seconds=35 * 86400, interval_seconds=30):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    context(); guard()
    if set(captures) != {row['name'] for row in STAGES}:
        raise ValueError('Every original full visual, mixed, reasoning and native-code experiment must remain queued')
    deadline = now() + maximum_wait_seconds
    results = []
    for spec in STAGES:
        folder = root / 'reports' / spec['output']; destination = root / 'checkpoints' / spec['output']
        if folder.exists() or destination.exists():
            raise ValueError('Preserve any existing experiment owner/output; this queue cannot adopt or restart it')
        while True:
            context(); guard()
            prepared = preparation(root, spec, captures[spec['name']])
            state = dict(ready=False, waiting_for='exact complete original source preparation')
            if prepared is not None:
                state = resources(root, prepared['required_free_bytes_before_launch'])
            if state.get('ready') is True:
                break
            publish('progress.json', dict(stage=spec['name'], launched_experiments=len(results),
                waiting=state, current_driver_launched=False, goal_achieved=False))
            if now() >= deadline:
                raise ValueError('The source/resource observation lifetime expired; no predecessor is relabelled stopped')
            pause(interval_seconds)
        # The actual owned child repeats all semantic source, storage, memory,
        # exclusive lease, full-size capacity and full assessment checks.
        for path, digest in prepared['source_files_sha256'].items():
            if sha(path) != digest: raise ValueError('A complete prepared original source changed before handoff')
        context(); guard()
        publish(spec['name'] + '-launch-intent.json', dict(stage=spec, original_ready_sha256=sha(root / 'reports' / spec['preparation'] / 'ready.json'),
            inherited_actual_nice=0, inherited_actual_scheduler=os.SCHED_OTHER, automatic_retry=False, goal_achieved=False))
        path = directory / (spec['name'] + '-owned-driver.log')
        with path.open('xb') as log:
            child = subprocess.Popen([str(root / '.venv/bin/python'), '-u', str(root / 'scripts' / spec['driver']), '--run'],
                cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
            actual = identity(child.pid)
            publish(spec['name'] + '-launched.json', dict(stage=spec['name'], actual_owned_driver=actual,
                source_preparation_owner=captures[spec['name']]['owner'], automatic_retry=False, goal_achieved=False))
            while child.poll() is None:
                guard()
                publish('progress.json', dict(stage=spec['name'], launched_experiments=len(results) + 1,
                    actual_owned_driver=actual, actual_same_driver_live=not terminal(actual),
                    driver_keeps_its_own_exclusive_lease_and_watchdog=True, goal_achieved=False))
                pause(interval_seconds)
            log.flush(); os.fsync(log.fileno())
        if child.returncode != 0:
            raise ValueError('An actual owned driver failed; preserve its exact outputs without retrying or launching later stages')
        result = completion(root, spec, actual)
        results.append(result); publish(spec['name'] + '-complete.json', result)
    return dict(complete=True, all_original_experiments=results, automatic_retry=False,
        all_original_training_and_evaluation_populations_retained=True,
        final_shared_model_integration_and_verified_high_end_capability_still_required=True, goal_achieved=False)
