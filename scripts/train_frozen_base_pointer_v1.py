"""Queue whole-source head-only attention learning after the owned coding run."""
import argparse
from datetime import datetime, timezone, timedelta
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time

import train_point_grounding_joint_v1 as owner
from slm_lab.codenet_checkpoint_v1 import FileBindings, atomic_checkpoint
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/frozen-base-pointer-v1'
OUTPUT = ROOT / 'checkpoints/frozen-base-pointer-v1'
QUEUE = ROOT / 'reports/frozen-base-pointer-v1-queue.json'
DATA = ROOT / 'data/localized-pointer-candidate-v1'
BASE = ROOT / 'checkpoints/extended-curriculum-v1/joint/model.pt'
TOKEN = BASE.parent / 'tokenizer.json'
PREVIOUS = ROOT / 'reports/own-code-generation-repair-v1'
PREVIOUS_QUEUE = ROOT / 'reports/own-code-generation-repair-v1-queue.json'
MAX_STATE = 16 * 1024**2
RESERVE = 26 * 1024**3
owner.REPORTS = REPORTS; owner.resources.REPORTS = REPORTS


def load_head(plan, device, phase='baseline'):
    import torch
    if phase == 'candidate':
        from slm_lab.frozen_base_pointer_inference_v1 import load
        receipt = json.loads((REPORTS / 'training-complete.json').read_text())
        return load(OUTPUT / 'head.pt', BASE, TOKEN, REPORTS / 'plan.json',
            expected_head_sha256=receipt['head_sha256'], expected_base_sha256=plan['base_checkpoint_sha256'],
            expected_tokenizer_sha256=plan['tokenizer_sha256'], expected_plan_sha256=file_hash(REPORTS / 'plan.json'),
            expected_data_admission_sha256=plan['data_admission_sha256'], device=device)
    from slm_lab.training import load_checkpoint
    from slm_lab.frozen_base_pointer_stage_v1 import make_head
    torch.manual_seed(plan['training_options']['seed'])
    base, metadata = load_checkpoint(BASE, 'cpu', training_state=False)
    head = make_head(base, rank=plan['head_rank']).to(device)
    return head, dict(metadata, actual_inventory=head.trainable_inventory())


def worker(phase):
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.frozen_base_pointer_stage_v1 import FrozenBasePointerStage, make_optimizer, projections
    from slm_lab.frozen_base_pointer_inference_v1 import export_payload
    from slm_lab.point_grounding_data_v1 import PointerData
    from slm_lab.point_grounding_evaluation_v1 import state_digest
    from slm_lab.point_grounding_stage_v1 import collective_check
    from train_own_code_generation_repair_v1 import optimizer_digest
    # The imported code controller has process-local owner destinations; put
    # the shared lifecycle module back on this actual head experiment.
    owner.REPORTS = REPORTS; owner.resources.REPORTS = REPORTS
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3' or int(os.environ['WORLD_SIZE']) != 4:
        raise ValueError('All four explicitly owned original head-experiment GPUs required')
    plan = owner.registered(phase + '-rank-' + str(rank))
    guard = FileBindings({**plan['source_files_sha256'], str(REPORTS / 'plan.json'): file_hash(REPORTS / 'plan.json')})
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    device = torch.device('cuda', local); torch.cuda.set_device(device)
    if torch.cuda.get_device_name(device) != 'NVIDIA L40' or not torch.cuda.is_bf16_supported():
        raise ValueError('Actual original BF16 L40 population required')
    torch.cuda.set_per_process_memory_fraction(.8, device)
    dist.init_process_group('nccl', timeout=timedelta(seconds=240))
    try:
        head, metadata = load_head(plan, device)
        inventory = head.trainable_inventory()
        if inventory['total_unique_parameters'] != 706799104 or inventory['actual_trainable_parameters'] != 786432:
            raise ValueError('Actual complete own backbone and exactly786432trainable head parameters required')
        initial = state_digest(head); digests = [None] * 4; dist.all_gather_object(digests, initial)
        if len(set(digests)) != 1:
            raise ValueError('All four actual initial own-base/head states must match')
        versions = [p._version for p in head.base.parameters()]
        data = PointerData(DATA, 'train', 'optimization'); options = plan['training_options']
        random.seed(options['seed'] + rank); np.random.seed(options['seed'] + rank); torch.manual_seed(options['seed'] + rank)
        sources = dict(base_checkpoint_sha256=plan['base_checkpoint_sha256'], tokenizer_sha256=plan['tokenizer_sha256'],
            run_plan_sha256=file_hash(REPORTS / 'plan.json'), data_admission_sha256=plan['data_admission_sha256'])
        optimizer = make_optimizer(head, options)
        kernel = FrozenBasePointerStage(head, optimizer, data, options, sources, device,
                                         lambda: torch.autocast('cuda', dtype=torch.bfloat16))
        if kernel.frozen_base_digest != plan['frozen_base_digest']:
            raise ValueError('Every actual original backbone tensor must match the full CPU proof')
        if phase == 'capacity':
            kernel.update(); kernel.validate_Adam()
            disposable = OUTPUT / 'capacity-disposable-resume.pt'
            kernel.save(disposable, MAX_STATE, RESERVE)
            expected = kernel.update()
            expected_head = state_digest(head); expected_Adam = optimizer_digest(optimizer)
            kernel.resume(torch.load(disposable, map_location='cpu', weights_only=True))
            again = kernel.update()
            if (again != expected or state_digest(head) != expected_head or optimizer_digest(optimizer) != expected_Adam
                    or [p._version for p in head.base.parameters()] != versions):
                raise ValueError('Actual CUDA head recovery must match every next metric/head/Adam value and frozen version')
            optimizer.zero_grad(set_to_none=True)
            longest = max(range(len(data)), key=lambda i: len(data.rows[i]['query_input_ids']))
            batch = data.batch([longest, longest], device)
            original_length = batch['input_ids'].shape[1]
            if original_length < 34:
                content = data.rows[longest]['query_input_ids'][1:-1]
                batch['input_ids'] = torch.tensor([[1, *(content * 34)[:32], 2]] * 2, device=device)
                batch['attention_mask'] = torch.ones_like(batch['input_ids'], dtype=torch.bool)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = head(**batch)['loss']
            loss.backward()
            if (not bool(torch.isfinite(loss)) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in kernel.parameters)
                    or any(p.grad is not None for p in head.base.parameters()) or state_digest(head.base) != kernel.frozen_base_digest):
                raise ValueError('Full34token/448px head backward must keep the entire original backbone unchanged')
            proof = dict(rank=rank, actual_GPU=torch.cuda.get_device_name(device), actual_inventory=inventory,
                complete_Adam_states=len(optimizer.state), disposable_logical_updates=2, disposable_actual_step_calls=3,
                actual_CUDA_next_update_head_Adam_and_frozen_versions_exact=True,
                actual_longest_original_query_tokens=original_length, capacity_query_tokens=int(batch['input_ids'].shape[1]),
                maximum_query_is_synthetic_repetition=original_length < 34, actual_pixel_shape=list(batch['pixel_values'].shape),
                capacity_loss=float(loss.detach()), peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                frozen_base_digest=kernel.frozen_base_digest, initial_head_state_digest=initial, production_updates=0)
            proofs = [None] * 4; dist.all_gather_object(proofs, proof)
            if rank == 0:
                atomic_json(REPORTS / 'capacity.json', dict(passed=True, proofs=proofs, actual_four_L40s=True,
                    initial_head_state_digest=initial, temporary_probe_weights_discarded=True, goal_achieved=False))
                disposable.unlink()
            dist.barrier(); return
        capacity = json.loads((REPORTS / 'capacity.json').read_text())
        baseline = json.loads((REPORTS / 'baseline/grounding/plan.json').read_text())
        if (not (REPORTS / 'baseline-complete.json').exists() or not capacity['passed']
                or initial != capacity['initial_head_state_digest'] or initial != baseline['actual_model_state_sha256']):
            raise ValueError('Actual exact initial-head capacity and complete matching baseline required')
        started = time.monotonic()
        for step in range(kernel.steps):
            owner.owned(plan); collective_check(guard.check); record = kernel.update()
            if rank == 0:
                with (REPORTS / 'training-metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, separators=(',', ':')) + '\n')
                if step < 2 or (step + 1) % 8 == 0:
                    atomic_json(REPORTS / 'status.json', dict(status='training only own query/patch head; whole backbone frozen',
                        actual_updates=step + 1, planned_updates=kernel.steps, latest=record, goal_achieved=False))
                    print('frozen_base_pointer_update', step + 1, record['query_mean_nll'], flush=True)
            if step + 1 in [1, 16] or (step + 1) % 128 == 0 or step + 1 == kernel.steps:
                collective_check(lambda: guard.check(full=True))
                if (collective_check(lambda: state_digest(head.base)) != kernel.frozen_base_digest
                        or [p._version for p in head.base.parameters()] != versions):
                    raise ValueError('Every original backbone byte and parameter version must remain unchanged')
                kernel.save(OUTPUT / 'resume.pt', MAX_STATE, RESERVE)
                if rank == 0:
                    atomic_json(REPORTS / 'checkpoint-receipt.json', dict(actual_steps=step + 1,
                        sha256=file_hash(OUTPUT / 'resume.pt'), bytes=(OUTPUT / 'resume.pt').stat().st_size,
                        complete_head_Adam_four_worker_RNG_and_source_cursors_with_immutable_base_reference=True, goal_achieved=False))
        after = collective_check(lambda: state_digest(head.base))
        if after != kernel.frozen_base_digest:
            raise ValueError('Entire original final backbone must remain byte-identical')
        if rank == 0:
            payload = export_payload(head, options, sources, kernel.completed, kernel.frozen_base_digest)
            atomic_checkpoint(OUTPUT / 'head.pt', payload, 8 * 1024**2, RESERVE)
            atomic_json(REPORTS / 'training-complete.json', dict(actual_steps=kernel.completed,
                head_sha256=file_hash(OUTPUT / 'head.pt'), resume_sha256=file_hash(OUTPUT / 'resume.pt'),
                unique_original_queries=36354, actual_exposures=36384, rounding_repeats=30,
                frozen_base_digest_before=kernel.frozen_base_digest, frozen_base_digest_after=after,
                all_original_backbone_parameter_versions_unchanged=True, elapsed_seconds=time.monotonic() - started,
                candidate_promoted=False, goal_achieved=False))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def assess(phase, suite):
    plan = owner.registered(phase + '-suite-' + str(suite))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(suite):
        raise ValueError('Exact isolated owned physical assessment GPU required')
    guard = FileBindings({**plan['source_files_sha256'], str(REPORTS / 'plan.json'): file_hash(REPORTS / 'plan.json')})
    images = FileBindings(plan['native_image_files_sha256'])
    import torch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.8, 0)
    folder = REPORTS / phase; folder.mkdir(exist_ok=True)
    if suite == 0:
        from slm_lab.point_grounding_data_v1 import PointerData
        from slm_lab.point_grounding_evaluation_v1 import evaluate
        head, _ = load_head(plan, 'cuda:0', phase)
        data = PointerData(DATA, 'validation', 'evaluation')
        prior = json.loads((DATA / 'training-position-prior.json').read_text())
        with torch.autocast('cuda', dtype=torch.bfloat16):
            evaluate(head, data, prior, folder / 'grounding', batch_size=4)
    else:
        def evaluate(module, name, data, **kwargs):
            native_loader = module.load_checkpoint
            def frozen_loader(path, device, training_state=False):
                if Path(path) != BASE or training_state:
                    raise ValueError('Only the exact immutable own native image view may use this loader')
                head, metadata = load_head(plan, device, phase)
                return head.base, dict(metadata, training_options=dict(image_size=448),
                                       all_original_backbone_parameters_frozen=True)
            module.load_checkpoint = frozen_loader
            try:
                module.evaluate(BASE, data, folder / name, device='cuda:0', **kwargs)
            finally:
                module.load_checkpoint = native_loader
        if suite == 1:
            from slm_lab import ocr_evaluation
            evaluate(ocr_evaluation, 'ocr.json', ROOT / 'data/alignment-curriculum-v1/images.valid.jsonl',
                     max_examples=256, max_new_tokens=128, image_size=448)
        elif suite == 2:
            from slm_lab import object_evaluation
            evaluate(object_evaluation, 'objects.json', ROOT / 'data/object-qa-v1/valid.jsonl', max_examples=180, max_new_tokens=32)
        else:
            from slm_lab import vision_evaluation
            evaluate(vision_evaluation, 'vision.json', ROOT / 'data/alignment-curriculum-v1/contrastive.valid.jsonl',
                     max_examples=128, image_size=448)
    owner.owned(plan); guard.check(full=True); images.check(full=True)
    atomic_json(folder / f'suite-{suite}-complete.json', dict(suite=suite, phase=phase,
        immutable_base_checkpoint_sha256=plan['base_checkpoint_sha256'], run_plan_sha256=file_hash(REPORTS / 'plan.json'), goal_achieved=False))


def predecessor_artifacts(folder):
    success = folder / 'independent-comparison.json'
    failure = folder / 'failure.json'
    if success.exists():
        return [success]
    if failure.exists():
        return [failure]
    raise ValueError('Exact coding owner ended without complete actual terminal evidence; inspect its reports')


def parent(files, predecessor, hypothesis):
    if (REPORTS / 'plan.json').exists() or OUTPUT.exists():
        raise ValueError('Preserve previous head-only run; no implicit restart')
    if owner.resources.same(predecessor):
        raise ValueError('Exact original coding owner must actually end before head GPU acquisition')
    previous = json.loads((PREVIOUS / 'plan.json').read_text())
    if any(previous['owner'][k] != predecessor[k] for k in ['pid', 'ticks']):
        raise ValueError('Actual original coding owner binding required')
    if any(owner.resources.same(a) for a in json.loads((PREVIOUS / 'children.json').read_text())):
        raise ValueError('Every registered original coding worker must actually end')
    for path in [PREVIOUS / 'plan.json', PREVIOUS / 'restored.json', *predecessor_artifacts(PREVIOUS)]:
        files[str(path)] = file_hash(path)
    guard = FileBindings(files)
    with (ROOT / 'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB); owner.idle_original()
        memory = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout.splitlines()
        if len(memory) != 4 or any(int(v.strip()) > 128 for v in memory):
            raise ValueError('All four head-experiment GPUs must actually be unused')
        guard.check(full=True)
        plan = dict(hypothesis, created_at=datetime.now(timezone.utc).isoformat(), owner=owner.resources.identity(os.getpid()),
            waiter=owner.WAITER, predecessor=predecessor, source_files_sha256=files,
            maximum_seconds=3 * 3600, expires_at_unix=time.time() + 3 * 3600,
            automatic_retry=False, candidate_promoted=False, goal_achieved=False)
        OUTPUT.mkdir(); shutil.copy2(TOKEN, OUTPUT / 'tokenizer.json')
        atomic_json(REPORTS / 'plan.json', plan); atomic_json(REPORTS / 'children.json', [])
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Owned frozen-base head controller interrupted')
        signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
        driver = str(Path(__file__).resolve())
        watcher = subprocess.Popen(['.venv/bin/python', '-u', driver, '--watchdog'], cwd=ROOT,
            stdout=(REPORTS / 'watchdog.log').open('xb'), stderr=subprocess.STDOUT, start_new_session=True)
        def four(phase):
            child = owner.launch(plan, ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=4', driver, '--worker', phase], phase, '0,1,2,3')
            owner.wait_children(plan, [child])
        try:
            owner.idle_original(); os.kill(owner.WAITER['pid'], signal.SIGSTOP)
            deadline = time.monotonic() + 5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in ['T', 't']:
                if time.monotonic() > deadline:
                    raise ValueError('Original idle waiter did not enter the owned hold')
                time.sleep(.05)
            if (Path('/proc') / str(owner.WAITER['pid']) / 'task' / str(owner.WAITER['pid']) / 'children').read_text().strip():
                raise ValueError('Original waiter began work; never launch new head workers')
            four('capacity')
            for phase in ['baseline', 'candidate']:
                if phase == 'candidate':
                    four('train')
                children = [owner.launch(plan, ['-u', driver, '--assess', phase, '--suite', str(i)], phase + '-' + str(i), str(i)) for i in range(4)]
                owner.wait_children(plan, children)
                if any(not (REPORTS / phase / f'suite-{i}-complete.json').exists() for i in range(4)):
                    raise ValueError('Every complete actual original attention/native image assessment required')
                atomic_json(REPORTS / (phase + '-complete.json'), dict(all_four_suites_complete=True, goal_achieved=False))
            from slm_lab.frozen_base_pointer_readback_v1 import reconcile
            reconcile(ROOT, REPORTS, OUTPUT, DATA); guard.check(full=True)
            atomic_json(REPORTS / 'status.json', dict(status='frozen-base head experiment complete; inspect matched controls and image retention',
                                                    candidate_promoted=False, goal_achieved=False))
        except BaseException as problem:
            atomic_json(REPORTS / 'failure.json', dict(error=str(problem), automatic_retry=False, goal_achieved=False)); raise
        finally:
            owner.restore_owned(plan, 'Owned frozen-base head experiment ended; original waiter restored')
            watcher.wait(timeout=20)


def queue():
    if QUEUE.exists() or (REPORTS / 'plan.json').exists() or OUTPUT.exists():
        raise ValueError('Preserve original head queue/run; no implicit duplicate')
    hypothesis = json.loads((REPORTS / 'hypothesis.json').read_text())
    files = dict(hypothesis['source_files_sha256'])
    files[str(REPORTS / 'hypothesis.json')] = file_hash(REPORTS / 'hypothesis.json')
    guard = FileBindings(files); guard.check(full=True)
    predecessor = json.loads(PREVIOUS_QUEUE.read_text())['queue_owner']
    atomic_json(QUEUE, dict(queued_at=datetime.now(timezone.utc).isoformat(), queue_owner=owner.resources.identity(os.getpid()),
        exact_predecessor=predecessor, source_files_sha256=files, actual_new_head_updates=0,
        pending_not_started=True, automatic_retry=False, goal_achieved=False))
    while owner.resources.same(predecessor):
        guard.check(); time.sleep(5)
    while True:
        guard.check()
        if not owner.resources.same(owner.WAITER):
            raise ValueError('Original waiter ended; inspect a fresh resource window')
        try:
            owner.idle_original(); break
        except ValueError:
            time.sleep(5)
    parent(files, predecessor, hypothesis)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['capacity', 'train']); parser.add_argument('--watchdog', action='store_true')
    parser.add_argument('--assess', choices=['baseline', 'candidate']); parser.add_argument('--suite', type=int, choices=[0, 1, 2, 3])
    args = parser.parse_args()
    if args.watchdog:
        owner.watchdog()
    elif args.worker:
        worker(args.worker)
    elif args.assess:
        if args.suite is None:
            raise ValueError('Exact isolated native evaluation suite required')
        assess(args.assess, args.suite)
    else:
        queue()
