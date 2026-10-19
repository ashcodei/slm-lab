"""Complete unchanged native evaluations after a terminal observer API failure.

No training, checkpoint edits or failed-output overwrites. A new exclusive
runtime loads the two complete own checkpoints and produces every original
matched code/broad population under the corrected transparent observer.
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
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_pretraining_assessment_recovery_v1 import (
    PUBLIC, FAILED, REPORT, QUEUE, EPOCH, completed_training_after_observer_failure)
import own_training_gpu_lease_v2 as owner

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / REPORT
SOURCE = ROOT / PUBLIC
MODELS = ROOT / 'checkpoints/public-pretraining-v1'


def read(path):
    return json.loads(Path(path).read_text())


def configure():
    owner.REPORTS = OUT
    owner.resources.REPORTS = OUT


def assess(arm, gpu, suite):
    plan = owner.registered('assess-' + arm + '-' + suite + '-' + str(gpu))
    import torch
    from slm_lab.public_pretraining_native_v1 import load
    from slm_lab.dense_native_view_v2 import CompleteDenseNativeView
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    from slm_lab.native_greedy_observer_v2 import observe
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.85, 0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.get_device_name(0) != 'NVIDIA L40':
        raise ValueError('Only the original four L40 devices are admitted')
    checkpoint = MODELS / arm / 'model.pt'
    model, metadata = load(checkpoint, SOURCE / 'plan.json', arm,
        expected_checkpoint_sha256=plan['inputs'][arm]['checkpoint_sha256'],
        expected_plan_sha256=plan['original_neural_plan_sha256'], device='cuda:0')
    view = CompleteDenseNativeView(model).eval()
    destination = OUT / arm
    destination.mkdir(exist_ok=True)
    old_threads = torch.set_num_threads
    torch.set_num_threads = lambda count: None

    def evaluate(module, name, data, **kwargs):
        original = module.load_checkpoint
        module.load_checkpoint = lambda *args, **kw: (view, metadata)
        try:
            module.evaluate(checkpoint, data, destination / name, device='cuda:0', **kwargs)
        finally:
            module.load_checkpoint = original

    try:
        with (destination / (suite + '-actual-native-actions-' + str(gpu) + '.jsonl')).open('x') as actions:
            def record(event):
                actions.write(json.dumps(event) + '\n')
                actions.flush()
            with inference_cache(), observe(view, record):
                if suite == 'code':
                    from slm_lab.native_code_shards_v1 import evaluate_shard
                    for frame in ['completion', 'instruction']:
                        evaluate_shard(view, metadata, checkpoint, destination, rank=gpu, world=4, frame=frame)
                elif gpu == 0:
                    from slm_lab import knowledge_evaluation
                    evaluate(knowledge_evaluation, 'knowledge.json', Path(plan['pulse_sources']['knowledge']),
                        max_examples=56, max_new_tokens=32)
                elif gpu == 1:
                    from slm_lab import math_evaluation
                    evaluate(math_evaluation, 'math.json', ROOT / 'data/math-reasoning-v2/test.jsonl',
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
            os.fsync(actions.fileno())
        owner.owned(plan)
        FileBindings(plan['source_files_sha256']).check(full=True)
    finally:
        torch.set_num_threads = old_threads


def check_population(folder):
    counts = dict(knowledge=read(folder / 'knowledge.json')['overall']['questions'],
        math=read(folder / 'math.json')['evaluated_examples'],
        OCR=read(folder / 'ocr.json')['overall']['examples'],
        objects=read(folder / 'objects.json')['overall']['examples'],
        photos=read(folder / 'vision.json')['images'])
    if counts != dict(knowledge=56, math=16, OCR=20, objects=15, photos=32):
        raise ValueError('Every complete original native development population is required: ' + repr(counts))
    return counts


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists() or (ROOT / QUEUE).exists():
        raise ValueError('Fresh CUDA-hidden evaluation-only recovery with fresh outputs required')
    if not completed_training_after_observer_failure(ROOT):
        raise ValueError('Original parent and all registered GPU children must be terminal')
    tests = ROOT / 'reports/native-greedy-observer-v2-tests.log'
    if '7 passed' not in tests.read_text():
        raise ValueError('Actual text/positional-image/keyword-image transparency controls required')
    gate_tests = ROOT / 'reports/public-pretraining-assessment-recovery-v1-tests.log'
    if '3 passed' not in gate_tests.read_text():
        raise ValueError('Terminal failure, complete epoch and unchanged full-population controls required')
    files = dict(owner.lease_sources())
    paths = [SOURCE / name for name in ['plan.json', 'training-complete.json', 'native-assessment-inputs.json']]
    paths += [ROOT / FAILED / name for name in ['plan.json', 'children.json', 'failure.json', 'restored.json', 'assess-initial-broad-2.log']]
    paths += [ROOT / EPOCH, tests, gate_tests, *python_sources(ROOT, [Path(__file__),
        ROOT / 'tests/test_native_greedy_observer_v2.py', ROOT / 'tests/test_public_pretraining_assessment_recovery_v1.py'])]
    original = read(SOURCE / 'plan.json')
    paths += [Path(p) for p in original['pulse_sources'].values()]
    paths += [ROOT / 'data/math-reasoning-v2/test.jsonl', ROOT / 'data/math-reasoning-v2/manifest.json']
    files.update({str(p): file_hash(p) for p in paths})
    plan = dict(owner=owner.resources.identity(os.getpid()), waiter=owner.WAITER,
        terminal_waiter_receipt_sha256=owner.terminal_receipt_sha256(), source_files_sha256=files,
        expires_at_unix=time.time() + 14 * 86400, original_neural_plan_sha256=file_hash(SOURCE / 'plan.json'),
        inputs=read(SOURCE / 'native-assessment-inputs.json'), pulse_sources=original['pulse_sources'],
        original_complete_epoch_readback_sha256=file_hash(ROOT / EPOCH),
        original_failed_runtime=str(ROOT / FAILED), failure_evidence_preserved=True,
        model_updates=0, original_full_populations=dict(code=48, knowledge=56, math=16, OCR=20, objects=15, photos=32),
        new_external_weights_or_teacher_API=False, candidate_promoted=False, goal_achieved=False)
    with owner.LEASE.open('a+') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner.idle_original()
        used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
        if len(used) != 4 or any(int(x) > 128 for x in used):
            raise ValueError('All original four devices must have no active model')
        if shutil.disk_usage(ROOT).free < 26 * 1024**3:
            raise ValueError('Original disk reserve required')
        FileBindings(files).check(full=True)
        OUT.mkdir()
        atomic_json(OUT / 'plan.json', plan)
        atomic_json(OUT / 'children.json', [])
        atomic_json(ROOT / QUEUE, dict(owner=plan['owner'], source_files_sha256=files,
            status='evaluation-only recovery of complete4447update public model', GPU_claimed=True,
            model_updates=0, goal_achieved=False))
        watcher = None
        success = False
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Evaluation-only recovery interrupted')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            owner.hold_original(plan)
            driver = str(Path(__file__).resolve())
            watcher = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', driver, '--watchdog'], cwd=ROOT,
                stdout=(OUT / 'watchdog.log').open('x'), stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
            # Regenerate both endpoints with exactly the same corrected wrapper.
            for arm in ['initial', 'trained']:
                for suite in ['code', 'broad']:
                    atomic_json(OUT / 'progress.json', dict(phase='native evaluations', endpoint=arm, suite=suite,
                        complete_training_updates=4447, model_updates=0, goal_achieved=False))
                    children = [owner.launch(plan, ['-u', driver, '--assess', arm, '--suite', suite,
                        '--gpu', str(gpu)], 'assess-' + arm + '-' + suite + '-' + str(gpu), str(gpu)) for gpu in range(4)]
                    owner.wait_children(plan, children)
                from slm_lab.native_code_shards_v1 import reconcile
                for frame in ['completion', 'instruction']:
                    reconcile([OUT / arm / ('code-' + frame + '-shard-' + str(rank) + '.json') for rank in range(4)],
                        OUT / arm / ('code-' + frame + '.json'))
                check_population(OUT / arm)
            FileBindings(files).check(full=True)
            success = True
        except BaseException as error:
            atomic_json(OUT / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
                automatic_retry=False, original_failed_evidence_preserved=True, goal_achieved=False))
            raise
        finally:
            owner.restore_owned(plan, 'Evaluation-only recovery ended; every registered GPU child terminal')
            if watcher is not None:
                watcher.wait(timeout=20)
        if success:
            atomic_json(OUT / 'complete.json', dict(complete=True, at=datetime.now(timezone.utc).isoformat(),
                all_original_native_populations_complete=True, original_complete_training_updates=4447,
                original_complete_epoch_readback_sha256=file_hash(ROOT / EPOCH),
                original_failed_runtime_not_relabelled_successful=True, checkpoint_bytes_unchanged=True,
                model_updates=0, original_complete_human_RL_and_final_benchmarks_still_required=True,
                candidate_promoted=False, goal_achieved=False))


if __name__ == '__main__':
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watchdog', action='store_true')
    parser.add_argument('--assess', choices=['initial', 'trained'])
    parser.add_argument('--suite', choices=['code', 'broad'])
    parser.add_argument('--gpu', type=int, choices=range(4))
    args = parser.parse_args()
    if args.watchdog:
        owner.watchdog()
    elif args.assess:
        assess(args.assess, args.gpu, args.suite)
    else:
        try:
            parent()
        except BaseException as error:
            atomic_json(ROOT / (REPORT + '-parent-failure.json'), dict(error=repr(error),
                traceback=traceback.format_exc(), automatic_retry=False, goal_achieved=False))
            raise
