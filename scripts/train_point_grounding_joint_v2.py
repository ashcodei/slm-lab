"""Explicit replacement for the failed deterministic-CUDA point capacity run.

Retain the unchanged v1 collective kernel, owned worker/watchdog lifecycle,
source epoch and evaluations. A versioned vision wrapper changes only the
learned position-grid backward to a deterministic CPU VJP. Wait for the
currently owned coding evaluation; preserve every v1 failure artifact.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import train_point_grounding_joint_v1 as original
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/point-grounding-joint-v2'
OUTPUT = ROOT / 'checkpoints/point-grounding-joint-v2'
QUEUE = ROOT / 'reports/point-grounding-joint-v2-queue.json'
HYPOTHESIS = REPORTS / 'hypothesis.json'
PREDECESSOR_REPORTS = ROOT / 'reports/own-code-delta-public-v1'
PREDECESSOR_QUEUE = ROOT / 'reports/own-code-delta-public-v1-queue.json'
OLD_REPORTS = ROOT / 'reports/point-grounding-joint-v1'
OLD_QUEUE = ROOT / 'reports/point-grounding-joint-v1-queue.json'

# Function globals stay in the imported v1 module. Reuse its exact lifecycle
# with explicit new artifact destinations, never altering any source file.
original.REPORTS = REPORTS
original.OUTPUT = OUTPUT
original.QUEUE = QUEUE
original.HYPOTHESIS = HYPOTHESIS
original.resources.REPORTS = REPORTS
_native_load_head = original.load_head


def load_head(plan, device, phase='baseline'):
    from slm_lab.deterministic_vision_resize_v1 import DeterministicVisionEncoder
    head, metadata = _native_load_head(plan, device, phase)
    head.base.vision = DeterministicVisionEncoder(head.base.vision)
    return head, metadata


original.load_head = load_head


def parent(files, predecessor):
    if (REPORTS / 'plan.json').exists() or OUTPUT.exists():
        raise ValueError('Preserve prior v2 run; no implicit restart')
    if original.resources.same(predecessor):
        raise ValueError('Existing exact coding-evaluation owner must actually end before acquisition')
    prior = json.loads((PREDECESSOR_REPORTS / 'plan.json').read_text())
    if any(prior['owner'][key] != predecessor[key] for key in ['pid', 'ticks']):
        raise ValueError('Exact original coding-evaluation owner binding required')
    if any(original.resources.same(actor) for actor in json.loads((PREDECESSOR_REPORTS / 'children.json').read_text())):
        raise ValueError('All registered original coding-evaluation workers must actually end')
    terminal = PREDECESSOR_REPORTS / 'comparison.json'
    if not terminal.exists():
        terminal = PREDECESSOR_REPORTS / 'failure.json'
    for path in [PREDECESSOR_REPORTS / 'plan.json', PREDECESSOR_REPORTS / 'restored.json', terminal]:
        files[str(path)] = file_hash(path)
    guard = FileBindings(files)
    hypothesis = json.loads(HYPOTHESIS.read_text())
    options = dict(hypothesis['proposed_fixed_training_options'], image_size=448, maximum_whole_query_tokens=34,
                   position_gradient='native_forward_CPU_bicubic_VJP_v1')
    with (ROOT / 'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        original.idle_original()
        memory = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, check=True).stdout.splitlines()
        if len(memory) != 4 or any(int(value.strip()) > 128 for value in memory):
            raise ValueError('All four replacement-experiment GPUs must actually be unused')
        guard.check(full=True)
        probe = json.loads((ROOT / 'reports/point-grounding-full-CPU-v2/complete.json').read_text())
        plan = dict(created_at=datetime.now(timezone.utc).isoformat(), owner=original.resources.identity(os.getpid()),
            waiter=original.WAITER, predecessor=predecessor, predecessor_terminal_artifact=str(terminal),
            source_files_sha256=files, base_checkpoint_sha256=file_hash(original.BASE),
            tokenizer_sha256=file_hash(original.TOKEN), frozen_language_critic_sha256=probe['frozen_language_critic_SHA_after'],
            training_options=options, gradient_reduction='fixed_parameter_order_worker_average_v1',
            maximum_seconds=original.MAXIMUM_SECONDS, expires_at_unix=time.time() + original.MAXIMUM_SECONDS,
            explicit_replacement_of_failed_v1_capacity=True, native_forward_unchanged=True,
            deterministic_algorithms_remain_enabled=True, automatic_retry=False, candidate_promoted=False, goal_achieved=False)
        OUTPUT.mkdir()
        shutil.copy2(original.TOKEN, OUTPUT / 'tokenizer.json')
        atomic_json(REPORTS / 'plan.json', plan)
        atomic_json(REPORTS / 'children.json', [])
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Owned replacement point controller interrupted')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        driver = str(Path(__file__).resolve())
        watcher = subprocess.Popen(['.venv/bin/python', '-u', driver, '--watchdog'], cwd=ROOT,
            stdout=(REPORTS / 'watchdog.log').open('xb'), stderr=subprocess.STDOUT, start_new_session=True)
        try:
            original.idle_original()
            os.kill(original.WAITER['pid'], signal.SIGSTOP)
            deadline = time.monotonic() + 5
            while (original.resources.same(original.WAITER) or {}).get('state') not in ['T', 't']:
                if time.monotonic() > deadline:
                    raise ValueError('Original idle waiter did not enter its owned hold')
                time.sleep(.05)
            if (Path('/proc') / str(original.WAITER['pid']) / 'task' / str(original.WAITER['pid']) / 'children').read_text().strip():
                raise ValueError('Original waiter began work before hold; never launch replacement workers')
            train_args = ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=4', driver]
            child = original.launch(plan, [*train_args, '--worker', 'capacity'], 'capacity', '0,1,2,3')
            original.wait_children(plan, [child])
            for phase in ['baseline', 'candidate']:
                if phase == 'candidate':
                    child = original.launch(plan, [*train_args, '--worker', 'train'], 'training', '0,1,2,3')
                    original.wait_children(plan, [child])
                children = [original.launch(plan, ['-u', driver, '--assess', phase, '--suite', str(i)],
                                            phase + '-' + str(i), str(i)) for i in range(4)]
                original.wait_children(plan, children)
                if any(not (REPORTS / phase / f'suite-{i}-complete.json').exists() for i in range(4)):
                    raise ValueError('Every actual native point/image assessment must complete')
                atomic_json(REPORTS / (phase + '-complete.json'), dict(completed_at=datetime.now(timezone.utc).isoformat(),
                    phase=phase, all_four_suites_complete=True, goal_achieved=False))
            from slm_lab.point_grounding_readback_v1 import reconcile
            reconcile(ROOT, REPORTS, OUTPUT, original.DATA)
            guard.check(full=True)
            atomic_json(REPORTS / 'status.json', dict(status='replacement original attention experiment complete; inspect matched results',
                candidate_promoted=False, goal_achieved=False))
        except BaseException as problem:
            atomic_json(REPORTS / 'failure.json', dict(error=str(problem), automatic_retry=False, goal_achieved=False))
            raise
        finally:
            original.restore_owned(plan, 'Owned replacement point experiment ended; original waiter restored')
            watcher.wait(timeout=20)


def queue():
    if QUEUE.exists() or (REPORTS / 'plan.json').exists() or OUTPUT.exists():
        raise ValueError('Preserve prior v2 queue or run; no implicit duplicate')
    old = json.loads((OLD_REPORTS / 'plan.json').read_text())
    if original.resources.same(old['owner']) or any(original.resources.same(actor)
            for actor in json.loads((OLD_REPORTS / 'children.json').read_text())):
        raise ValueError('Original failed point owner and every registered worker must actually end')
    failure = json.loads((OLD_REPORTS / 'failure.json').read_text())
    log = (OLD_REPORTS / 'capacity.log').read_text()
    if ('Every owned point child' not in failure['error'] or 'upsample_bicubic2d_backward_out_cuda' not in log
            or (OLD_REPORTS / 'training-complete.json').exists()):
        raise ValueError('This replacement addresses only the recorded original deterministic bicubic capacity failure')
    files = dict(json.loads(OLD_QUEUE.read_text())['source_files_sha256'])
    extra = ['scripts/train_point_grounding_joint_v2.py', 'src/slm_lab/deterministic_vision_resize_v1.py',
        'tests/test_deterministic_vision_resize_v1.py', 'tests/test_point_grounding_driver_v2.py',
        'reports/deterministic-vision-resize-v1-tests.log', 'reports/point-grounding-driver-v2-tests.log',
        'reports/point-grounding-joint-v1/failure.json', 'reports/point-grounding-joint-v1/capacity.log',
        'reports/point-grounding-joint-v1/restored.json', 'reports/point-grounding-joint-v1/plan.json',
        'reports/point-grounding-joint-v1/children.json', 'reports/point-grounding-joint-v1-queue.json',
        'reports/own-code-delta-public-v1-queue.json', 'reports/point-grounding-joint-v2/hypothesis.json']
    for name in extra:
        files[str(ROOT / name)] = file_hash(ROOT / name)
    for name, expected in [('deterministic-vision-resize-v1-tests.log', '6 passed'), ('point-grounding-driver-v2-tests.log', '2 passed')]:
        if expected not in (ROOT / 'reports' / name).read_text():
            raise ValueError('Actual bridge/complete-vision/four-worker recovery and replacement-owner checks required')
    guard = FileBindings(files)
    guard.check(full=True)
    predecessor = json.loads(PREDECESSOR_QUEUE.read_text())['owner']
    atomic_json(QUEUE, dict(queued_at=datetime.now(timezone.utc).isoformat(), queue_owner=original.resources.identity(os.getpid()),
        exact_predecessor=predecessor, source_files_sha256=files, actual_model_updates=0,
        pending_not_started=True, automatic_retry=False, explicit_replacement_of_failed_v1_capacity=True, goal_achieved=False))
    while original.resources.same(predecessor):
        guard.check()
        time.sleep(5)
    while True:
        guard.check()
        if not original.resources.same(original.WAITER):
            raise ValueError('Original waiter ended; inspect a new resource window')
        try:
            original.idle_original()
            break
        except ValueError:
            time.sleep(5)
    parent(files, predecessor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['capacity', 'train'])
    parser.add_argument('--watchdog', action='store_true')
    parser.add_argument('--assess', choices=['baseline', 'candidate'])
    parser.add_argument('--suite', type=int, choices=[0, 1, 2, 3])
    args = parser.parse_args()
    if args.watchdog:
        original.watchdog()
    elif args.worker:
        original.worker(args.worker)
    elif args.assess:
        if args.suite is None:
            raise ValueError('Exact isolated assessment suite required')
        original.assess(args.assess, args.suite)
    else:
        queue()
