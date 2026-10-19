"""Queue a fixed own sub1B generation/repair experiment after owned vision work."""
import argparse
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
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
REPORTS = ROOT / 'reports/own-code-generation-repair-v1'
OUTPUT = ROOT / 'checkpoints/own-code-generation-repair-v1'
QUEUE = ROOT / 'reports/own-code-generation-repair-v1-queue.json'
INITIAL = ROOT / 'checkpoints/own-code-delta-v1/model.pt'
BASE = ROOT / 'checkpoints/extended-curriculum-v1/joint/model.pt'
TOKEN = BASE.parent / 'tokenizer.json'
INITIAL_PLAN = ROOT / 'reports/own-code-delta-v1/plan.json'
PREVIOUS = ROOT / 'reports/point-grounding-joint-v2'
PREVIOUS_QUEUE = ROOT / 'reports/point-grounding-joint-v2-queue.json'
FOLDERS = dict(generation=ROOT / 'data/APPS-verified-generation-v1', repair=ROOT / 'data/APPS-execution-repairs-v1',
               variety=ROOT / 'data/codenet-durable-pack-v2')
MAX_STATE = 5 * 1024**3
RESERVE = 26 * 1024**3
owner.REPORTS = REPORTS
owner.resources.REPORTS = REPORTS


def optimizer_digest(optimizer):
    import torch
    digest = hashlib.sha256()
    for i, state in enumerate(optimizer.state.values()):
        for key in ['step', 'exp_avg', 'exp_avg_sq']:
            tensor = state[key].detach().cpu().contiguous()
            digest.update(json.dumps([i, key, list(tensor.shape), str(tensor.dtype)], separators=(',', ':')).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def streams(config):
    from slm_lab.own_code_training_stage_v1 import PreparedCodeStream
    return {name: PreparedCodeStream(name, folder, TOKEN, config) for name, folder in FOLDERS.items()}


def load_model(plan, phase, device):
    if phase == 'baseline':
        from slm_lab.own_code_delta_inference_v1 import load
        return load(INITIAL, BASE, TOKEN, INITIAL_PLAN, expected_checkpoint_sha256=plan['initial_component_checkpoint_sha256'],
            expected_base_sha256=plan['base_checkpoint_sha256'], expected_target_sha256=plan['own_code_target_sha256'],
            expected_plan_sha256=plan['initial_plan_sha256'], device=device)
    from slm_lab.own_code_training_inference_v1 import load
    receipt = json.loads((REPORTS / 'training-complete.json').read_text())
    return load(OUTPUT / 'model.pt', BASE, TOKEN, INITIAL_PLAN, REPORTS / 'plan.json',
        expected_checkpoint_sha256=receipt['trained_components_sha256'], expected_base_sha256=plan['base_checkpoint_sha256'],
        expected_target_sha256=plan['own_code_target_sha256'], expected_initial_plan_sha256=plan['initial_plan_sha256'],
        expected_training_plan_sha256=file_hash(REPORTS / 'plan.json'), device=device)


def worker(phase):
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.point_grounding_evaluation_v1 import state_digest
    from slm_lab.own_code_training_stage_v1 import OwnCodeTrainingStage, make_optimizer, collective_check
    from slm_lab.own_code_training_inference_v1 import export_payload
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3' or int(os.environ['WORLD_SIZE']) != 4:
        raise ValueError('All four explicitly owned original GPUs required')
    plan = owner.registered(phase + '-rank-' + str(rank))
    guard = FileBindings({**plan['source_files_sha256'], str(REPORTS / 'plan.json'): file_hash(REPORTS / 'plan.json')})
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    device = torch.device('cuda', local); torch.cuda.set_device(device)
    if torch.cuda.get_device_name(device) != 'NVIDIA L40' or not torch.cuda.is_bf16_supported():
        raise ValueError('Actual original BF16 L40 population required')
    torch.cuda.set_per_process_memory_fraction(.85, device)
    dist.init_process_group('nccl', timeout=timedelta(seconds=240))
    try:
        model, metadata = load_model(plan, 'baseline', device)
        if model.counts() != plan['actual_model_counts'] or model.counts()['total_unique'] != 985664512:
            raise ValueError('All actual original/general/code parameters must fit the sealed sub1B inventory')
        model.code.gradient_checkpointing = True
        frozen_before = state_digest(model.base)
        data = streams(model.config)
        random.seed(plan['training_options']['seed'] + rank)
        np.random.seed(plan['training_options']['seed'] + rank)
        torch.manual_seed(plan['training_options']['seed'] + rank)
        options = plan['training_options']; optimizer = make_optimizer(model, options)
        sources = dict(run_plan_sha256=file_hash(REPORTS / 'plan.json'), base_checkpoint_sha256=plan['base_checkpoint_sha256'],
                       initial_component_checkpoint_sha256=plan['initial_component_checkpoint_sha256'],
                       data_admissions_sha256=plan['data_admissions_sha256'])
        kernel = OwnCodeTrainingStage(model, optimizer, data, options, sources, device,
                                      lambda: torch.autocast('cuda', dtype=torch.bfloat16))
        if phase == 'capacity':
            kernel.update(); kernel.validate_Adam(); optimizer.zero_grad(set_to_none=True)
            disposable = OUTPUT / 'capacity-disposable-resume.pt'
            kernel.save(disposable, MAX_STATE, RESERVE + MAX_STATE)
            uninterrupted = kernel.update()
            expected_parts = state_digest(model.parts)
            expected_Adam = optimizer_digest(optimizer)
            kernel.resume(torch.load(disposable, map_location='cpu', weights_only=True, mmap=True))
            recovered = kernel.update()
            if (recovered != uninterrupted or state_digest(model.parts) != expected_parts
                    or optimizer_digest(optimizer) != expected_Adam):
                raise ValueError('Actual four-L40 resumed next update/components/all Adam states must match exactly')
            kernel.validate_Adam()
            optimizer.zero_grad(set_to_none=True)
            longest = max(((int(stream.pack.index[stream.pack.positions[p]]['length']), name, p)
                           for name, stream in data.items() for p in range(len(stream))), key=lambda item: item[0])
            length, name, position = longest
            batch = data[name].encoder.batch([dict(packed_code_index=position)], 'response', device)
            if length != 4096 or batch['input_ids'].shape != (1, 4096):
                raise ValueError('Actual whole longest original4096token code source required')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model.code(**batch)['loss']
            if not bool(torch.isfinite(loss)):
                raise ValueError('Finite actual full-context own-code loss required')
            loss.backward()
            if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.code_parameters())
                    or any(p.grad is not None for p in model.base.parameters()) or state_digest(model.base) != frozen_before):
                raise ValueError('Actual full-context gradients must preserve the entire original general model')
            torch.cuda.synchronize(device)
            proof = dict(rank=rank, actual_GPU=torch.cuda.get_device_name(device), actual_model_counts=model.counts(),
                complete_Adam_states=len(optimizer.state), disposable_logical_updates=2, disposable_actual_step_calls=3,
                actual_CUDA_resumed_next_metrics_components_and_all_Adam_states_exact=True,
                actual_CUDA_RNG_bytes_validated_before_live_mutation=True, longest_original_stream=name,
                longest_original_position=position, longest_original_context_tokens=length,
                longest_original_response_EOS_targets=int((batch['labels'][:, 1:] != -100).sum()),
                full_original_context_backward_loss=float(loss.detach()), peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                frozen_base_digest=frozen_before, production_updates=0, production_model_weights_exported=False,
                temporary_disposable_snapshot_deleted_after_success=True)
            proofs = [None] * 4; dist.all_gather_object(proofs, proof)
            if rank == 0:
                atomic_json(REPORTS / 'capacity.json', dict(passed=True, actual_four_L40s=True, proofs=proofs, goal_achieved=False))
                disposable.unlink()
            dist.barrier()
            return
        if not (REPORTS / 'capacity.json').exists() or not (REPORTS / 'baseline-complete.json').exists():
            raise ValueError('Actual capacity and all fixed baseline suites required before production training')
        started = time.monotonic()
        for step in range(kernel.steps):
            owner.owned(plan); collective_check(guard.check)
            record = kernel.update(); torch.cuda.synchronize(device)
            if rank == 0:
                with (REPORTS / 'training-metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, separators=(',', ':')) + '\n')
                if step < 2 or (step + 1) % 8 == 0:
                    atomic_json(REPORTS / 'status.json', dict(status='own conditional-code generation/repair training',
                        actual_updates=step + 1, planned_updates=kernel.steps, latest=record, goal_achieved=False))
                    print('own_code_generation_repair_update', step + 1, record['summed_objective'], flush=True)
            if step + 1 in [1, 16] or (step + 1) % 128 == 0 or step + 1 == kernel.steps:
                collective_check(lambda: guard.check(full=True))
                if collective_check(lambda: state_digest(model.base)) != frozen_before:
                    raise ValueError('Every original general/vision/critic byte must remain identical')
                kernel.save(OUTPUT / 'resume.pt', MAX_STATE, RESERVE + MAX_STATE)
                if rank == 0:
                    atomic_json(REPORTS / 'checkpoint-receipt.json', dict(steps=step + 1,
                        sha256=file_hash(OUTPUT / 'resume.pt'), bytes=(OUTPUT / 'resume.pt').stat().st_size,
                        all_trainable_components_Adam_four_worker_RNG_and_source_cursors=True,
                        frozen_general_is_exact_immutable_reference=True, goal_achieved=False))
        after = collective_check(lambda: state_digest(model.base))
        if after != frozen_before:
            raise ValueError('Complete frozen original reference digest must match after training')
        if rank == 0:
            initial = torch.load(INITIAL, map_location='cpu', weights_only=True, mmap=True)
            payload = export_payload(model, initial, plan, file_hash(REPORTS / 'plan.json'), kernel.completed)
            atomic_checkpoint(OUTPUT / 'model.pt', payload, 2 * 1024**3, RESERVE)
            atomic_json(REPORTS / 'training-complete.json', dict(actual_updates=kernel.completed,
                trained_components_sha256=file_hash(OUTPUT / 'model.pt'), resumable_checkpoint_sha256=file_hash(OUTPUT / 'resume.pt'),
                frozen_base_digest_before=frozen_before, frozen_base_digest_after=after,
                exposure_budget=plan['predeclared_exposure_budget'], elapsed_seconds=time.monotonic() - started,
                candidate_promoted=False, goal_achieved=False))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def assess(phase, suite, public=False):
    plan = owner.registered(phase + ('-public-' if public else '-suite-') + str(suite))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(suite):
        raise ValueError('Exact isolated owned physical evaluation GPU required')
    guard = FileBindings({**plan['source_files_sha256'], str(REPORTS / 'plan.json'): file_hash(REPORTS / 'plan.json')})
    import torch
    torch.set_num_threads(1); torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.8, 0)
    checkpoint = INITIAL if phase == 'baseline' else OUTPUT / 'model.pt'
    folder = REPORTS / phase; folder.mkdir(exist_ok=True)
    def evaluate(module, name, *args, route='code', **kwargs):
        native_loader = module.load_checkpoint
        def own_loader(path, device, training_state=False):
            if Path(path) != checkpoint or training_state:
                raise ValueError('Only exact owned conditional-code inference may use the explicit loader')
            model, metadata = load_model(plan, phase, device)
            return model.route(route), dict(metadata, explicit_route=route, actual_full_model_counts=model.counts(),
                training_options=dict(image_size=448), declared_evaluation_image_size=448)
        module.load_checkpoint = own_loader
        try:
            owner.owned(plan)
            module.evaluate(checkpoint, *args, folder / name, device='cuda:0', **kwargs)
        finally:
            module.load_checkpoint = native_loader
        guard.check(full=True)
    if public:
        from slm_lab import public_code_benchmark
        evaluate(public_code_benchmark, f'public-{suite}.json',
                 ROOT / 'reports/own-code-delta-public-v1/inputs' / f'shard-{suite}' / 'questions.jsonl', max_new_tokens=512)
    elif suite == 0:
        from slm_lab import code_quality
        for mode in ['completion', 'instruction']:
            evaluate(code_quality, 'code-' + mode + '.json', max_new_tokens=512, prompt_mode=mode)
    elif suite == 1:
        from slm_lab import cpp_evaluation
        evaluate(cpp_evaluation, 'C++.json', max_new_tokens=512)
    elif suite == 2:
        from slm_lab import knowledge_evaluation
        evaluate(knowledge_evaluation, 'knowledge.json', ROOT / 'data/knowledge-eval-v1/questions.jsonl',
                 route='general', max_examples=1000, max_new_tokens=32)
    else:
        from slm_lab import ocr_evaluation, object_evaluation, vision_evaluation
        evaluate(ocr_evaluation, 'ocr.json', ROOT / 'data/alignment-curriculum-v1/images.valid.jsonl',
                 route='general', max_examples=256, max_new_tokens=128, image_size=448)
        evaluate(object_evaluation, 'objects.json', ROOT / 'data/object-qa-v1/valid.jsonl',
                 route='general', max_examples=180, max_new_tokens=32)
        evaluate(vision_evaluation, 'vision.json', ROOT / 'data/alignment-curriculum-v1/contrastive.valid.jsonl',
                 route='general', max_examples=128, image_size=448)
    atomic_json(folder / (('public-' if public else 'suite-') + str(suite) + '-complete.json'),
        dict(checkpoint_sha256=file_hash(checkpoint), suite=suite, explicit_general_route=suite in [2, 3] and not public,
             source_runtime_and_loader_bound_to_plan=True, goal_achieved=False))


def predecessor_artifacts(folder):
    completed = [folder / 'complete-neural-source-readback.json', folder / 'matched-attention-and-native-comparison.json']
    if all(path.exists() for path in completed):
        return completed
    failure = folder / 'failure.json'
    if failure.exists():
        return [failure]
    raise ValueError('Predecessor ended without complete actual success/failure evidence; inspect its original artifacts')


def parent(files, provenance, predecessor, hypothesis):
    if (REPORTS / 'plan.json').exists() or OUTPUT.exists():
        raise ValueError('Preserve previous code run; no implicit restart')
    if owner.resources.same(predecessor):
        raise ValueError('Exact original vision owner must actually end before GPU acquisition')
    previous = json.loads((PREVIOUS / 'plan.json').read_text())
    if any(previous['owner'][k] != predecessor[k] for k in ['pid', 'ticks']):
        raise ValueError('Exact original vision owner binding required')
    if any(owner.resources.same(a) for a in json.loads((PREVIOUS / 'children.json').read_text())):
        raise ValueError('Every registered original vision worker must actually end')
    for path in [PREVIOUS / 'plan.json', PREVIOUS / 'restored.json', *predecessor_artifacts(PREVIOUS)]:
        files[str(path)] = file_hash(path)
    full_guard = FileBindings(provenance); guard = FileBindings(files)
    with (ROOT / 'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB); owner.idle_original()
        memory = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout.splitlines()
        if len(memory) != 4 or any(int(v.strip()) > 128 for v in memory):
            raise ValueError('All four GPUs must actually be unused')
        guard.check(full=True); full_guard.check(full=True)
        plan = dict(hypothesis, created_at=datetime.now(timezone.utc).isoformat(), owner=owner.resources.identity(os.getpid()),
            waiter=owner.WAITER, predecessor=predecessor, source_files_sha256=files, provenance_files_sha256=provenance,
            expires_at_unix=time.time() + 4 * 3600, maximum_seconds=4 * 3600, automatic_retry=False,
            candidate_promoted=False, goal_achieved=False)
        OUTPUT.mkdir(); shutil.copy2(TOKEN, OUTPUT / 'tokenizer.json')
        atomic_json(REPORTS / 'plan.json', plan); atomic_json(REPORTS / 'children.json', [])
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Owned code generation/repair controller interrupted')
        signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
        driver = str(Path(__file__).resolve())
        watcher = subprocess.Popen(['.venv/bin/python', '-u', driver, '--watchdog'], cwd=ROOT,
            stdout=(REPORTS / 'watchdog.log').open('xb'), stderr=subprocess.STDOUT, start_new_session=True)
        def four(phase):
            child = owner.launch(plan, ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=4', driver, '--worker', phase],
                                 phase, '0,1,2,3'); owner.wait_children(plan, [child])
        def suites(phase, public=False):
            children = [owner.launch(plan, ['-u', driver, '--assess', phase, '--suite', str(i), *(['--public'] if public else [])],
                         phase + ('-public-' if public else '-suite-') + str(i), str(i)) for i in range(4)]
            owner.wait_children(plan, children)
        try:
            owner.idle_original(); os.kill(owner.WAITER['pid'], signal.SIGSTOP)
            deadline = time.monotonic() + 5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in ['T', 't']:
                if time.monotonic() > deadline:
                    raise ValueError('Original sleeping waiter did not enter its owned hold')
                time.sleep(.05)
            if (Path('/proc') / str(owner.WAITER['pid']) / 'task' / str(owner.WAITER['pid']) / 'children').read_text().strip():
                raise ValueError('Original waiter began work; do not launch new workers')
            four('capacity'); suites('baseline')
            atomic_json(REPORTS / 'baseline-complete.json', dict(actual_suites=4,
                original_complete_public_baseline='reports/own-code-delta-public-v1/comparison.json', goal_achieved=False))
            full_guard.check(full=True); four('train')
            suites('candidate', public=True); suites('candidate')
            from slm_lab.model import ModelConfig
            from slm_lab.own_code_training_readback_v1 import reconcile
            reconcile(ROOT, REPORTS, OUTPUT, streams(ModelConfig(**json.loads((ROOT / 'configs/model.json').read_text()))))
            guard.check(full=True); full_guard.check(full=True)
            atomic_json(REPORTS / 'status.json', dict(status='fixed own generation/repair experiment complete; inspect measured results',
                candidate_promoted=False, goal_achieved=False))
        except BaseException as problem:
            atomic_json(REPORTS / 'failure.json', dict(error=str(problem), automatic_retry=False, goal_achieved=False)); raise
        finally:
            owner.restore_owned(plan, 'Owned code generation/repair experiment ended; original waiter restored')
            watcher.wait(timeout=20)


def queue():
    if QUEUE.exists() or OUTPUT.exists() or (REPORTS / 'plan.json').exists():
        raise ValueError('Preserve prior code queue/run; no implicit duplicate')
    hypothesis = json.loads((REPORTS / 'hypothesis.json').read_text())
    files = dict(hypothesis['source_files_sha256']); provenance = dict(hypothesis['provenance_files_sha256'])
    files[str(REPORTS / 'hypothesis.json')] = file_hash(REPORTS / 'hypothesis.json')
    guard = FileBindings(files); FileBindings(provenance).check(full=True)
    predecessor = json.loads(PREVIOUS_QUEUE.read_text())['queue_owner']
    atomic_json(QUEUE, dict(queued_at=datetime.now(timezone.utc).isoformat(), queue_owner=owner.resources.identity(os.getpid()),
        exact_predecessor=predecessor, source_files_sha256=files, hypothesis_sha256=file_hash(REPORTS / 'hypothesis.json'),
        actual_code_updates=0, pending_not_started=True, automatic_retry=False, goal_achieved=False))
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
    parent(files, provenance, predecessor, hypothesis)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['capacity', 'train']); parser.add_argument('--watchdog', action='store_true')
    parser.add_argument('--assess', choices=['baseline', 'candidate']); parser.add_argument('--suite', type=int, choices=[0, 1, 2, 3])
    parser.add_argument('--public', action='store_true'); args = parser.parse_args()
    if args.watchdog:
        owner.watchdog()
    elif args.worker:
        worker(args.worker)
    elif args.assess:
        if args.suite is None:
            raise ValueError('Exact isolated native evaluation suite required')
        assess(args.assess, args.suite, args.public)
    else:
        queue()
