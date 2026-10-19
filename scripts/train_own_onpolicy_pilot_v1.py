"""Owned matched128update own-code CE versus actual execution-DPO experiment."""
import argparse
from datetime import datetime, timezone
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
REPORTS = ROOT / 'reports/own-onpolicy-pilot-v1'
OUTPUT = ROOT / 'checkpoints/own-onpolicy-pilot-v1'
MANIFEST = ROOT / 'checkpoints/own-assistant-candidate-v1/manifest.json'
HYPOTHESIS = ROOT / 'reports/own-onpolicy-execution-v1/next-training-hypothesis.json'
PAIRS = ROOT / 'data/own-onpolicy-execution-v1/pairs.jsonl'
BASELINE = ROOT / 'reports/own-code-generation-repair-v1/candidate'
MAX_STATE = 5 * 1024**3
RESERVE = 26 * 1024**3
owner.REPORTS = REPORTS; owner.resources.REPORTS = REPORTS


def optimizer_digest(optimizer):
    import torch
    digest = hashlib.sha256()
    for index, state in enumerate(optimizer.state.values()):
        for key in ['step', 'exp_avg', 'exp_avg_sq']:
            tensor = state[key].detach().cpu().contiguous()
            digest.update(json.dumps([index, key, list(tensor.shape), str(tensor.dtype)], separators=(',', ':')).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def previous_ended():
    for name in ['own-code-generation-repair-v1', 'frozen-base-pointer-v1', 'own-view-contrastive-v1',
                 'own-code-conditioning-v1', 'own-onpolicy-code-64-v1']:
        folder = ROOT / 'reports' / name
        plan = json.loads((folder / 'plan.json').read_text())
        actors = json.loads((folder / 'children.json').read_text())
        if owner.resources.same(plan['owner']) or any(owner.resources.same(actor) for actor in actors):
            raise ValueError('Every exact preceding owner and registered actor must actually end before resource queries')
        if not (folder / 'restored.json').exists():
            raise ValueError('Every preceding owned experiment must restore its original waiter')


def registered(subject):
    plan = json.loads((REPORTS / 'plan.json').read_text()); owner.owned(plan)
    me = owner.resources.identity(os.getpid())
    while True:
        owner.owned(plan)
        actors = json.loads((REPORTS / 'children.json').read_text())
        if any(a['pid'] == me['pid'] and a['ticks'] == me['ticks'] and a['subject'] == subject for a in actors):
            return plan
        time.sleep(.05)


def streams(config, tokenizer):
    from slm_lab.own_code_training_stage_v1 import PreparedCodeStream
    return {name: PreparedCodeStream(name, ROOT / folder, tokenizer, config) for name, folder in
        [('generation', 'data/APPS-verified-generation-v1'), ('variety', 'data/codenet-durable-pack-v2')]}


def budget(data, pairs, options):
    from slm_lab.codenet_stage_v1 import EpochOrder
    from slm_lab.own_code_training_stage_v1 import LanguageCycleOrder
    total = options['steps']; result = {}
    for index, (name, stream) in enumerate(data.items()):
        order = LanguageCycleOrder(stream.language_positions, options['seed'] + 1009) if name == 'variety' else EpochOrder(len(stream), options['seed'])
        positions = [order.at(i) for i in range(total)]
        rows = [stream.pack.index[stream.pack.positions[p]] for p in positions]
        result[name] = dict(population=len(stream), exposures=total, unique_original_positions=len(set(positions)),
            complete_context_tokens=sum(int(r['length']) for r in rows),
            response_EOS_targets=sum(int(r['length']) - int(r['target_start']) for r in rows), full_source_pass=False)
        if name == 'variety':
            from collections import Counter
            language = {p: lang for lang, values in stream.language_positions.items() for p in values}
            result[name]['original_language_exposures'] = dict(Counter(language[p] for p in positions))
    order = EpochOrder(len(pairs), options['seed'] + 2018); selected = [pairs[order.at(i)] for i in range(total)]
    result['pairs'] = dict(population=len(pairs), exposures=total, unique_original_pairs=len({r['id'] for r in selected}),
        chosen_human_response_EOS_targets=sum(sum(v != -100 for v in r['chosen'][1]) for r in selected),
        rejected_actual_own_response_EOS_targets=sum(sum(v != -100 for v in r['rejected'][1]) for r in selected),
        complete_passes=total // len(pairs), next_pass_remainder=total % len(pairs))
    return result


def worker(phase, shard, arm):
    subject = ('capacity-' + str(shard)) if phase == 'capacity' else 'train-' + arm
    plan = registered(subject)
    physical = shard if phase == 'capacity' else plan['arms'][arm]['physical_gpu']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(physical):
        raise ValueError('Exact isolated original L40 required before CUDA initialization')
    guard = FileBindings({**plan['source_files_sha256'], str(REPORTS / 'plan.json'): file_hash(REPORTS / 'plan.json')})
    import numpy as np
    import torch
    from slm_lab.own_assistant_v1 import load_manifest
    from slm_lab.own_onpolicy_stage_v1 import OwnOnPolicyStage
    from slm_lab.own_onpolicy_inference_v1 import options, export_payload
    from slm_lab.own_code_training_stage_v1 import make_optimizer
    from slm_lab.execution_preference_v1 import reference_scores
    from slm_lab.point_grounding_evaluation_v1 import state_digest
    from slm_lab.point_grounding_data_v1 import PointerData
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.85, 0)
    if torch.cuda.get_device_name(0) != 'NVIDIA L40':
        raise ValueError('Actual original L40 required')
    combined, metadata = load_manifest(MANIFEST, expected_manifest_sha256=plan['initial_manifest_sha256'], device='cuda:0')
    model = combined.language
    for p in combined.grounding.parameters():
        p.requires_grad_(False)
    if model.counts() != plan['actual_model_counts'] or combined.counts()['total_unique'] != 986450944:
        raise ValueError('Actual complete own model and unchanged reserved head must match')
    data = streams(model.config, plan['tokenizer']); pairs = list(map(json.loads, PAIRS.read_text().splitlines()))
    if budget(data, pairs, plan['training_options']) != plan['predeclared_exposure_budget']:
        raise ValueError('Every original whole source draw/target/language/pair exposure must match')
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    model.eval()
    initial_parts = state_digest(model.parts)
    if initial_parts != plan['initial_parts_digest']:
        raise ValueError('Both fresh arms/probes must start from the actual same complete815update own policy')
    if phase == 'capacity':
        reference = reference_scores(model.code, pairs, policy_sha256=plan['initial_manifest_sha256'], device='cuda:0', amp=amp)
        reference_path = REPORTS / f'reference-{shard}.json'; atomic_json(reference_path, reference)
    else:
        reference_path = REPORTS / 'reference.json'; reference = json.loads(reference_path.read_text())
        capacity = json.loads((REPORTS / 'capacity.json').read_text())
        if file_hash(reference_path) != capacity['reference_scores_sha256']:
            raise ValueError('Identical frozen own reference measured on all4originalGPUs required')
    reference_sha = file_hash(reference_path)
    ref_guard = FileBindings({str(reference_path): reference_sha})
    frozen = dict(base=state_digest(model.base), query=state_digest(combined.grounding.query_projection),
                  patch=state_digest(combined.grounding.patch_projection))
    versions = [(id(p), p._version, p.data_ptr()) for p in combined.grounding.parameters()]
    photo = PointerData(ROOT / 'data/localized-pointer-candidate-v1', 'train', 'audit')
    original = json.loads((ROOT / 'reports/frozen-base-pointer-full-CPU-v1/complete.json').read_text())
    batch = photo.batch([original['original_source_position']], 'cuda:0')
    def retained():
        with torch.no_grad(), amp():
            generated = model.base.generate(batch['input_ids'], batch['pixel_values'], max_new_tokens=2, eos_token_id=2)
            logits = combined.locate(batch['input_ids'], batch['pixel_values'], attention_mask=batch['attention_mask'], valid_cells=batch['valid_cells'])['logits']
        return generated.cpu(), logits.cpu()
    expected_retained = retained()
    selected_arm = 'execution_DPO' if phase == 'capacity' else arm
    settings = options(plan, selected_arm)
    random.seed(settings['seed']); np.random.seed(settings['seed']); torch.manual_seed(settings['seed'])
    model.code.gradient_checkpointing = True
    optimizer = make_optimizer(model, settings)
    sources = dict(run_plan_sha256=file_hash(REPORTS / 'plan.json'), initial_policy_manifest_sha256=plan['initial_manifest_sha256'],
        reference_scores_sha256=reference_sha, data_admissions_sha256=plan['data_admissions_sha256'],
        initial_code_checkpoint_sha256=plan['initial_code_checkpoint_sha256'], base_checkpoint_sha256=plan['base_checkpoint_sha256'])
    kernel = OwnOnPolicyStage(model, optimizer, data, pairs, reference, settings, sources, 'cuda:0', amp)
    folder = OUTPUT / ('capacity-' + str(shard) if phase == 'capacity' else arm); folder.mkdir()
    started = time.monotonic()
    if phase == 'capacity':
        kernel.update(); kernel.save(folder / 'disposable.pt', MAX_STATE, RESERVE + MAX_STATE)
        expected = kernel.update(); parts_after = state_digest(model.parts); Adam_after = optimizer_digest(optimizer)
        kernel.resume(torch.load(folder / 'disposable.pt', map_location='cpu', weights_only=True, mmap=True))
        actual = kernel.update()
        if expected != actual or parts_after != state_digest(model.parts) or Adam_after != optimizer_digest(optimizer):
            raise ValueError('Actual CUDA exact next-update/parts/everyAdam-state recovery required')
        kernel.validate_Adam()
        stream = data['variety']; position = max(range(len(stream)), key=lambda p: int(stream.pack.index[stream.pack.positions[p]]['length']))
        maximum = stream.encoder.batch([dict(packed_code_index=position)], 'response', 'cuda:0')
        if maximum['input_ids'].shape != (1, 4096):
            raise ValueError('Complete actual original maximum4096token CodeNet source required')
        with amp():
            loss = model.code(**maximum)['loss']
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.code_parameters()):
            raise ValueError('Every own code gradient must be finite at actual4096token maximum')
        optimizer.zero_grad(set_to_none=True)
        (folder / 'disposable.pt').unlink(); folder.rmdir()
    else:
        for step in range(kernel.steps):
            owner.owned(plan); guard.check(); ref_guard.check()
            record = kernel.update()
            with (REPORTS / (arm + '-metrics.jsonl')).open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            if step < 2 or (step + 1) % 16 == 0:
                atomic_json(REPORTS / (arm + '-status.json'), dict(actual_updates=step+1, planned_updates=kernel.steps,
                    latest=record, goal_achieved=False)); print('own_onpolicy_update', arm, step+1, record['summed_objective'], flush=True)
            if step + 1 in [1, 16, kernel.steps]:
                guard.check(full=True); ref_guard.check(full=True)
                kernel.save(folder / 'resume.pt', MAX_STATE, RESERVE + MAX_STATE)
        payload = export_payload(model, plan, file_hash(REPORTS / 'plan.json'), arm, reference_sha, kernel.completed)
        atomic_checkpoint(folder / 'model.pt', payload, 2 * 1024**3, RESERVE)
    after = dict(base=state_digest(model.base), query=state_digest(combined.grounding.query_projection),
                 patch=state_digest(combined.grounding.patch_projection))
    actual_retained = retained()
    if (after != frozen or versions != [(id(p), p._version, p.data_ptr()) for p in combined.grounding.parameters()]
            or any(not torch.equal(a, b) for a, b in zip(expected_retained, actual_retained))
            or any(p.grad is not None for p in combined.grounding.parameters())):
        raise ValueError('Every original general/vision/critic/head weight/version and native route probe must stay exact')
    guard.check(full=True); ref_guard.check(full=True); owner.owned(plan)
    common = dict(actual_model_counts=combined.counts(), frozen_states_before=frozen, frozen_states_after=after,
        all_original_general_vision_critic_head_bytes_versions_and_native_route_probes_exact=True,
        complete_Adam_states=len(optimizer.state), reference_scores_sha256=reference_sha,
        initial_code_parts_digest=initial_parts, trained_code_parts_digest=state_digest(model.parts),
        native_general_probe_tokens=actual_retained[0].tolist(), native_grounding_probe_logits=actual_retained[1].tolist(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
        elapsed_seconds=time.monotonic()-started, candidate_promoted=False, goal_achieved=False)
    if phase == 'capacity':
        atomic_json(REPORTS / f'capacity-{shard}.json', dict(**common, actual_GPU=torch.cuda.get_device_name(0),
            actual_CUDA_next_update_all_metrics_parts_and_Adam_exact=True, maximum_original_context=4096,
            disposable_logical_updates=2, actual_optimizer_calls=3, production_updates=0))
    else:
        model.eval()
        with torch.no_grad(), amp():
            common['native_trained_code_probe_tokens'] = model.code.generate(batch['input_ids'], batch['pixel_values'], max_new_tokens=2, eos_token_id=2).cpu().tolist()
        atomic_json(REPORTS / (arm + '-training-complete.json'), dict(**common, actual_updates=kernel.completed,
            trained_components_sha256=file_hash(folder / 'model.pt'), resume_sha256=file_hash(folder / 'resume.pt'),
            predeclared_exposure_budget=plan['predeclared_exposure_budget']))


def assess(arm, suite, shard):
    plan = registered('assess-' + arm + '-' + suite + '-' + str(shard))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(shard):
        raise ValueError('Exact isolated native assessment GPU required')
    guard = FileBindings({**plan['source_files_sha256'], str(REPORTS / 'plan.json'): file_hash(REPORTS / 'plan.json')})
    import torch
    from slm_lab.own_onpolicy_inference_v1 import load
    torch.set_num_threads(1); torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.7, 0)
    checkpoint = OUTPUT / arm / 'model.pt'; destination = REPORTS / arm; destination.mkdir(exist_ok=True)
    receipt = json.loads((REPORTS / (arm + '-training-complete.json')).read_text())
    if suite == 'retention':
        from slm_lab.own_onpolicy_inference_v1 import attach_original_head
        from slm_lab.point_grounding_data_v1 import PointerData
        from slm_lab.point_grounding_evaluation_v1 import state_digest
        model, metadata = load(checkpoint, REPORTS / 'plan.json', arm, REPORTS / 'reference.json',
            expected_checkpoint_sha256=receipt['trained_components_sha256'], expected_plan_sha256=file_hash(REPORTS / 'plan.json'),
            expected_reference_sha256=receipt['reference_scores_sha256'], device='cuda:0')
        combined, head_metadata = attach_original_head(model, plan)
        for p in combined.grounding.parameters():
            p.requires_grad_(False)
        data = PointerData(ROOT / 'data/localized-pointer-candidate-v1', 'train', 'audit')
        original = json.loads((ROOT / 'reports/frozen-base-pointer-full-CPU-v1/complete.json').read_text())
        batch = data.batch([original['original_source_position']], 'cuda:0')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            generated = combined.generate(batch['input_ids'], batch['pixel_values'], route='general', max_new_tokens=2, eos_token_id=2).cpu().tolist()
            code = combined.generate(batch['input_ids'], batch['pixel_values'], route='code', max_new_tokens=2, eos_token_id=2).cpu().tolist()
            grounding = combined.locate(batch['input_ids'], batch['pixel_values'], attention_mask=batch['attention_mask'], valid_cells=batch['valid_cells'])['logits'].cpu().tolist()
        if (combined.counts()['total_unique'] != 986450944 or generated != receipt['native_general_probe_tokens']
                or code != receipt['native_trained_code_probe_tokens'] or grounding != receipt['native_grounding_probe_logits']
                or state_digest(model.parts) != receipt['trained_code_parts_digest']
                or state_digest(model.base) != receipt['frozen_states_after']['base']):
            raise ValueError('Actual trained full one-backbone reload must reproduce all3routes and component bytes')
        guard.check(full=True); owner.owned(plan)
        atomic_json(destination / 'trained-shared-reload.json', dict(actual_counts=combined.counts(),
            all_three_actual_trained_GPU_routes_exact_after_reload=True, original_backbones_loaded=1,
            full_code_source_or_external_policy_loaded=False, goal_achieved=False))
        return
    def evaluate(module, name, *args, **kwargs):
        original_loader = module.load_checkpoint
        def loader(path, device, training_state=False):
            if Path(path) != checkpoint or training_state:
                raise ValueError('Only exact completed own execution-feedback arm may use this native loader')
            model, metadata = load(checkpoint, REPORTS / 'plan.json', arm, REPORTS / 'reference.json',
                expected_checkpoint_sha256=receipt['trained_components_sha256'], expected_plan_sha256=file_hash(REPORTS / 'plan.json'),
                expected_reference_sha256=receipt['reference_scores_sha256'], device=device)
            return model.code, dict(metadata, explicit_route='code', training_options=dict(image_size=448))
        module.load_checkpoint = loader
        try:
            owner.owned(plan); module.evaluate(checkpoint, *args, destination / name, device='cuda:0', **kwargs)
        finally:
            module.load_checkpoint = original_loader
        guard.check(full=True)
    if suite == 'public':
        from slm_lab import public_code_benchmark as module
        evaluate(module, f'public-{shard}.json', ROOT / 'reports/own-code-delta-public-v1/inputs' / f'shard-{shard}' / 'questions.jsonl', max_new_tokens=512)
    elif suite == 'code':
        from slm_lab import code_quality as module
        for mode in ['completion', 'instruction']:
            evaluate(module, 'code-' + mode + '.json', max_new_tokens=512, prompt_mode=mode)
    else:
        from slm_lab import cpp_evaluation as module
        evaluate(module, 'C++.json', max_new_tokens=512)
    owner.owned(plan)


def parent():
    if REPORTS.exists() or OUTPUT.exists():
        raise ValueError('Preserve every previous experiment attempt; no implicit retry')
    previous_ended(); owner.idle_original()
    from tokenizers import Tokenizer
    from slm_lab.model import ModelConfig
    from slm_lab.own_onpolicy_stage_v1 import validate_reference
    hypothesis = json.loads(HYPOTHESIS.read_text()); spec = json.loads(MANIFEST.read_text())
    old = json.loads((ROOT / 'reports/own-code-generation-repair-v1/plan.json').read_text())
    files = dict(old['source_files_sha256']); provenance = dict(old['provenance_files_sha256'])
    names = ['scripts/train_own_onpolicy_pilot_v1.py', 'src/slm_lab/own_onpolicy_stage_v1.py', 'src/slm_lab/own_onpolicy_inference_v1.py',
        'src/slm_lab/own_onpolicy_readback_v1.py',
        'src/slm_lab/execution_preference_v1.py', 'src/slm_lab/own_onpolicy_execution_v1.py',
        'tests/test_own_onpolicy_stage_v1.py', 'tests/test_own_onpolicy_inference_v1.py',
        'reports/own-onpolicy-kernel-and-loader-v1-tests.log', 'tests/test_own_onpolicy_pilot_driver_v1.py',
        'reports/own-onpolicy-pilot-driver-v1-tests.log', 'src/slm_lab/own_assistant_v1.py',
        'src/slm_lab/frozen_base_pointer_inference_v1.py', 'src/slm_lab/frozen_base_pointer_stage_v1.py',
        'reports/own-conditioning-and-onpolicy-feedback-v1-review.json', 'reports/own-assistant-candidate-v1/validation-complete.json',
        'data/own-onpolicy-execution-v1/admission.json', 'data/own-onpolicy-execution-v1/pairs.jsonl',
        'reports/own-onpolicy-execution-v1/complete.json', 'reports/own-code-generation-repair-v1/independent-comparison.json']
    for name in names:
        files[str(ROOT / name)] = file_hash(ROOT / name)
    for path in [MANIFEST, HYPOTHESIS, *BASELINE.glob('*.json')]:
        files[str(path)] = file_hash(path)
    for key in ['base_checkpoint', 'tokenizer', 'code_checkpoint', 'code_initial_plan', 'code_training_plan', 'grounding_head', 'grounding_plan']:
        path = (MANIFEST.parent / spec[key]).resolve(); files[str(path)] = spec[key + '_sha256']
    if ('4 passed' not in (ROOT / 'reports/own-onpolicy-kernel-and-loader-v1-tests.log').read_text()
            or '2 passed' not in (ROOT / 'reports/own-onpolicy-pilot-driver-v1-tests.log').read_text()):
        raise ValueError('Actual full-gradient/recovery/loader and active-owner rejection checks required')
    settings = hypothesis['predeclared_training']
    training = dict(seed=settings['seed'], steps=settings['updates_per_arm'], world_size=1, microbatch_per_rank=1,
        accumulation=1, global_examples_per_stream=1, learning_rate=settings['learning_rate'], betas=settings['betas'],
        weight_decay=settings['weight_decay'], gradient_norm_clip=settings['gradient_clip'], warmup_fraction=settings['warmup_fraction'],
        cosine_floor_fraction=settings['cosine_floor_fraction'], coefficients=dict(generation=1., variety=.25), DPO_beta=.1)
    tokenizer = (MANIFEST.parent / spec['tokenizer']).resolve()
    config = ModelConfig(**json.loads((ROOT / 'data/APPS-verified-generation-v1/model-config.json').read_text()))
    data = streams(config, tokenizer); pairs = list(map(json.loads, PAIRS.read_text().splitlines()))
    admission = json.loads((ROOT / 'data/own-onpolicy-execution-v1/admission.json').read_text())
    if (file_hash(ROOT / 'data/own-onpolicy-execution-v1/admission.json') != hypothesis['data_admission_sha256']
            or file_hash(PAIRS) != admission['pairs_sha256'] or len(pairs) != 28 or not admission['admitted_for_complete_program_machine_DPO']):
        raise ValueError('All28actual independently admitted own response/human program pairs required')
    guard = FileBindings(files); complete_guard = FileBindings(provenance)
    with (ROOT / 'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB); previous_ended(); owner.idle_original()
        memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
        if len(memory) != 4 or any(int(value.strip()) > 128 for value in memory):
            raise ValueError('All four original L40s must be unused')
        if shutil.disk_usage(ROOT).free < RESERVE + 12 * MAX_STATE:
            raise ValueError('Keep persistent reserve and all four disposable/two production atomic states')
        guard.check(full=True); complete_guard.check(full=True); REPORTS.mkdir(); OUTPUT.mkdir()
        proof = json.loads((ROOT / 'reports/own-assistant-candidate-v1/validation-complete.json').read_text())
        # Use the actual original strict count-key names retained by its plan.
        counts = old['actual_model_counts']
        plan = dict(created_at=datetime.now(timezone.utc).isoformat(), owner=owner.resources.identity(os.getpid()), waiter=owner.WAITER,
            initial_manifest=str(MANIFEST), initial_manifest_sha256=file_hash(MANIFEST),
            initial_code_checkpoint_sha256=spec['code_checkpoint_sha256'], initial_parts_digest=proof['route_parity_proof']['trained_code_parts_digest'],
            base_checkpoint=str((MANIFEST.parent / spec['base_checkpoint']).resolve()), base_checkpoint_sha256=spec['base_checkpoint_sha256'],
            tokenizer=str(tokenizer), tokenizer_sha256=spec['tokenizer_sha256'], actual_model_counts=counts,
            training_options=training, arms=dict(CE_control=dict(DPO_alpha=0., physical_gpu=0), execution_DPO=dict(DPO_alpha=.1, physical_gpu=1)),
            data_admissions_sha256=dict(generation=file_hash(ROOT / 'data/APPS-verified-generation-v1/admission.json'),
                variety=file_hash(ROOT / 'data/codenet-durable-pack-v2/admission.json'), pairs=file_hash(ROOT / 'data/own-onpolicy-execution-v1/admission.json')),
            predeclared_exposure_budget=budget(data, pairs, training), source_files_sha256=files, provenance_files_sha256=provenance,
            baseline='Exact completed815update own component checkpoint/native reports; unchanged original runtime/prompts/precision/budget bound',
            hypothesis_sha256=file_hash(HYPOTHESIS), maximum_seconds=4*3600, expires_at_unix=time.time()+4*3600,
            machine_execution_not_genuine_human_RLHF=True, automatic_retry=False, candidate_promoted=False, goal_achieved=False)
        atomic_json(REPORTS / 'plan.json', plan); atomic_json(REPORTS / 'children.json', [])
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Owned matched execution-feedback experiment interrupted')
        signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
        driver = str(Path(__file__).resolve())
        watcher = subprocess.Popen(['.venv/bin/python', '-u', driver, '--watchdog'], cwd=ROOT,
            stdout=(REPORTS / 'watchdog.log').open('xb'), stderr=subprocess.STDOUT, start_new_session=True)
        def run(tasks):
            children = [owner.launch(plan, ['-u', driver, *args], subject, str(gpu)) for subject, gpu, args in tasks]
            owner.wait_children(plan, children)
        try:
            owner.idle_original(); os.kill(owner.WAITER['pid'], signal.SIGSTOP); until = time.monotonic()+5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in {'T', 't'}:
                if time.monotonic() > until:
                    raise ValueError('Original sleeping childless waiter did not enter owned hold')
                time.sleep(.05)
            if (Path('/proc') / str(owner.WAITER['pid']) / 'task' / str(owner.WAITER['pid']) / 'children').read_text().strip():
                raise ValueError('Original waiter began work; never preempt it')
            run([('capacity-'+str(i), i, ['--worker', 'capacity', '--shard', str(i)]) for i in range(4)])
            proofs = [json.loads((REPORTS / f'capacity-{i}.json').read_text()) for i in range(4)]
            hashes = {p['reference_scores_sha256'] for p in proofs}
            if len(hashes) != 1 or not all(p['actual_CUDA_next_update_all_metrics_parts_and_Adam_exact'] for p in proofs):
                raise ValueError('All four actual CUDA probes and identical frozen own reference scores required')
            shutil.copyfile(REPORTS / 'reference-0.json', REPORTS / 'reference.json')
            atomic_json(REPORTS / 'capacity.json', dict(passed=True, all_four_original_L40s=True,
                reference_scores_sha256=next(iter(hashes)), proofs=proofs, goal_achieved=False))
            complete_guard.check(full=True); guard.check(full=True)
            run([('train-'+arm, plan['arms'][arm]['physical_gpu'], ['--worker', 'train', '--arm', arm]) for arm in plan['arms']])
            run([('assess-'+arm+'-retention-'+str(i), i, ['--assess', 'retention', '--arm', arm, '--shard', str(i)]) for i, arm in enumerate(plan['arms'])])
            for arm in plan['arms']:
                run([('assess-'+arm+'-public-'+str(i), i, ['--assess', 'public', '--arm', arm, '--shard', str(i)]) for i in range(4)])
            run([('assess-'+arm+'-'+suite+'-'+str(gpu), gpu, ['--assess', suite, '--arm', arm, '--shard', str(gpu)])
                for arm, suite, gpu in [('CE_control','code',0), ('CE_control','cpp',1), ('execution_DPO','code',2), ('execution_DPO','cpp',3)]])
            from slm_lab.own_onpolicy_readback_v1 import reconcile
            reconcile(ROOT, REPORTS, OUTPUT, data, pairs)
            guard.check(full=True); complete_guard.check(full=True)
            atomic_json(REPORTS / 'complete.json', dict(status='Both fixed own arms and complete native comparisons finished; inspect measured capability', goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS / 'failure.json', dict(error=str(error), automatic_retry=False, goal_achieved=False)); raise
        finally:
            owner.restore_owned(plan, 'Own matched execution-feedback experiment ended; original waiter restored')
            watcher.wait(timeout=20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watchdog', action='store_true'); parser.add_argument('--worker', choices=['capacity', 'train'])
    parser.add_argument('--assess', choices=['public', 'code', 'cpp', 'retention']); parser.add_argument('--shard', type=int, choices=range(4), default=0)
    parser.add_argument('--arm', choices=['CE_control', 'execution_DPO'], default='execution_DPO')
    args = parser.parse_args()
    if args.watchdog:
        owner.watchdog()
    elif args.worker:
        worker(args.worker, args.shard, args.arm)
    elif args.assess:
        assess(args.arm, args.assess, args.shard)
    else:
        parent()
