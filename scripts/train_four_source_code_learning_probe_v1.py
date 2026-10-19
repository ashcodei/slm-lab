"""Bounded own857M memorization diagnostic, never a general coding-capability claim."""
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
from tokenizers import Tokenizer
import torch

from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_checkpoint_v1 import make_training_state, FORMAT
from slm_lab.code_residual_pilot_v1 import learning_rate, PROFILES
from slm_lab.code_source_generation_diagnostic_v1 import select_controls, reference_agreement
from slm_lab.codenet_checkpoint_v1 import FileBindings, atomic_checkpoint
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.recurrent_ablation import state_digest
from slm_lab.training import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = ROOT / 'reports/four-source-code-learning-probe-v1'
    output = ROOT / 'checkpoints/four-source-code-learning-probe-v1'
    if reports.exists() or output.exists():
        raise ValueError('Inspect earlier partial or complete experiments; no implicit restart')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == __import__('os').getpid():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b'slm_lab.training' in args or any(a.endswith(b'scripts/train_extended_curriculum.py') for a in args):
            raise ValueError('Original main writer must remain terminal before allocating this separate probe')
    if shutil.disk_usage(ROOT).free < 18 * 1024**3 + 64 * 1024**2:
        raise ValueError('One complete2GiB final artifact and64MiB reports above16GiB reserve required')
    device = torch.device('cuda:1')
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(device)[0] < 20 * 1024**3:
        raise ValueError('Initially20GiB free on a BF16-capable diagnostic training device required')
    torch.cuda.set_per_process_memory_fraction(.35, device)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    pilot = ROOT / 'reports/code-residual-lr-pilot-v1'
    prior = json.loads((pilot / 'results.json').read_text())
    prior_plan = json.loads((pilot / 'plan.json').read_text())
    selection_path = pilot / 'selected-original-rows.json'
    if file_hash(selection_path) != prior_plan['selection_sha256']:
        raise ValueError('Exact old train/validation source roles required')
    selected = select_controls(json.loads(selection_path.read_text()), budget=1024)
    train = [r for r in selected if r['role'] == 'train']
    validation = [r for r in selected if r['role'] == 'validation']
    pack_root = ROOT / 'data/codenet-durable-pack-v2'
    packs = {role: CompleteCodePack(pack_root, role,
        purpose='training' if role == 'train' else 'audit') for role in ['train', 'validation']}
    initial_path = Path(prior['best_complete_checkpoint'])
    source = Path(prior_plan['source_checkpoint'])
    token = initial_path.parent / 'tokenizer.json'
    names = ['src/slm_lab/code_residual_branch_v1.py', 'src/slm_lab/code_residual_checkpoint_v1.py',
        'src/slm_lab/code_residual_pilot_v1.py', 'src/slm_lab/code_source_generation_diagnostic_v1.py',
        'src/slm_lab/codenet_checkpoint_v1.py', 'src/slm_lab/codenet_token_pack_v1.py',
        'src/slm_lab/model.py', 'src/slm_lab/training.py', 'src/slm_lab/recurrent_ablation.py']
    paths = [Path(__file__), source, initial_path, token, pilot / 'results.json', pilot / 'plan.json',
        selection_path, pack_root / 'manifest.json', pack_root / 'admission.json',
        ROOT / 'reports/code-residual-source-generation-v1/results.json',
        ROOT / 'reports/code-residual-inference-precision-v1/results.json'] + [ROOT / n for n in names]
    bound = {str(p): file_hash(p) for p in paths}
    bound.update({str(pack_root / name): h for name, h in packs['train'].manifest['files_sha256'].items()})
    if (bound[str(initial_path)] != prior['best_complete_checkpoint_sha256']
            or bound[str(source)] != prior_plan['source_checkpoint_sha256']
            or bound[str(pilot / 'plan.json')] != prior['run_plan_sha256']):
        raise ValueError('Exact own completed pilot and frozen original base required')
    sequences = {}
    for row in selected:
        ids, labels = packs[row['role']].sequence(row['local_position'])
        if (hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest() != row['full_input_ids_sha256']
                or len(ids) != row['context_tokens'] or sum(y != -100 for y in labels) != row['target_tokens']):
            raise ValueError('Every whole original prompt/response/EOS/mask must match')
        sequences[row['id']] = (ids, labels)
    guard = FileBindings(bound)
    reports.mkdir()
    output.mkdir()
    shutil.copy2(token, output / 'tokenizer.json')

    def status(message, **fields):
        atomic_json(reports / 'status.json', dict(status=message,
            updated_at=datetime.now(timezone.utc).isoformat(), candidate_promoted=False,
            goal_achieved=False, **fields))

    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), source_files_sha256=bound,
        starting_policy_checkpoint=str(initial_path), starting_policy_sha256=bound[str(initial_path)],
        original_base_checkpoint=str(source), original_base_checkpoint_sha256=bound[str(source)],
        source_rows=selected, actual_training_sources=4, actual_validation_sources=4,
        repeated_training_problem_groups=len({r['group_id'] for r in train}),
        original_train_validation_groups_disjoint=True, training_updates=256,
        complete_training_programme_exposures=1024,
        complete_original_response_target_exposures=256 * sum(r['target_tokens'] for r in train),
        objective='unchanged target-token-average supervised cross entropy over complete original programs and EOS',
        profile=PROFILES[1], fresh_AdamW=True, old_Adam_or_rng_history_resumed=False,
        weight_decay=.01, gradient_clip=1., foreach=False, seed=810071,
        source_order='C++,Python,D,Java every update; four original sources repeated256 times',
        checkpoint_policy='One final complete residual/Adam/RNG artifact only; no automatic early stop or model selection',
        generation_budget=1024, evaluation_updates=[32, 128, 256],
        reference_reproduction_is_not_functional_correctness=True,
        validation_or_benchmark_answers_used_for_updates=False, existing_benchmark_scores_modified=False,
        original_main_writer_terminal=True, original_future_stage25GiB_preflight_unchanged=True,
        minimum_remaining_home_bytes=16 * 1024**3, maximum_final_checkpoint_bytes=2 * 1024**3,
        maximum_report_bytes=64 * 1024**2, maximum_GPU_memory_fraction=.35,
        full_source_or_final_base_integration=False, candidate_promoted=False, goal_achieved=False,
        scope='Diagnostic saturation on four already declared original training programs over the own frozen706M base. Tests whether complete known programs can be learned/terminated. Even perfect reference reproduction here is memorization, not broad code or independent generalization. All original full-source/replay/multimodal/RLHF and capability gates remain.')
    atomic_json(reports / 'plan.json', plan)
    plan_sha = file_hash(reports / 'plan.json')
    status('loading exact original base and already trained residuals with fresh optimizer')
    random.seed(810071)
    np.random.seed(810072)
    torch.manual_seed(810073)
    torch.cuda.manual_seed_all(810074)
    base, metadata = load_checkpoint(source, 'cpu', training_state=False)
    initial = torch.load(initial_path, map_location='cpu', weights_only=True, mmap=True)
    if (initial['format'] != FORMAT or initial.get('stage') != 'code_residual_lr_pilot_v1'
            or initial['steps'] != initial['planned_steps'] or initial['steps'] != 128
            or initial['base_checkpoint_sha256'] != bound[str(source)]
            or initial['tokenizer_sha256'] != bound[str(token)]
            or metadata['tokenizer_sha256'] != bound[str(token)]):
        raise ValueError('Exact completed128-update own residual warm start required')
    model = RoutedCodeResidual(base, width=2048)
    model.load_adapter_state(initial['residuals'])
    if (state_digest(model.residuals.state_dict()) != state_digest(initial['residuals'])
            or state_digest(model.base.state_dict()) != initial['frozen_backbone_state_sha256']
            or model.parameter_counts() != initial['model_counts']):
        raise ValueError('Every original initial base/residual tensor and parameter count must match')
    del initial
    gc.collect()
    model.to(device)
    model.code.gradient_checkpointing = True
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=3e-5, weight_decay=.01, foreach=False)
    tokenizer = Tokenizer.from_file(str(token))
    base_before = state_digest(model.base.state_dict())
    initial_residual = state_digest(model.residuals.state_dict())
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    metrics = []
    generations = []
    observations = []

    @torch.no_grad()
    def assess(rows):
        model.eval()
        values = []
        for row in rows:
            ids, labels = sequences[row['id']]
            with amp():
                loss = model.code(torch.tensor([ids], device=device),
                    labels=torch.tensor([labels], device=device))['loss']
            values.append(dict(id=row['id'], role=row['role'], language=row['language'],
                targets=row['target_tokens'], mean_NLL=float(loss)))
        return dict(token_mean_NLL=sum(r['targets'] * r['mean_NLL'] for r in values) / sum(r['targets'] for r in values),
            rows=values)

    @torch.no_grad()
    def generate(rows, step):
        model.eval()
        values = []
        for row in rows:
            guard.check()
            ids, labels = sequences[row['id']]
            start = next(i for i, y in enumerate(labels) if y != -100)
            prompt = torch.tensor([ids[:start]], device=device)
            with amp():
                produced = model.code.generate(prompt, max_new_tokens=1024,
                    eos_token_id=2, use_cache=True)[0, start:].tolist()
            raw = tokenizer.decode(produced[:-1] if produced and produced[-1] == 2 else produced,
                skip_special_tokens=False)
            record = dict(step=step, id=row['id'], role=row['role'], language=row['language'],
                generation_budget=1024, generated_token_ids=produced,
                raw_program_text=raw, raw_program_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                **reference_agreement(produced, ids[start:], 2))
            values.append(record)
            generations.append(record)
            atomic_json(reports / 'generations.json', dict(records=generations))
            print('four_source_generation', step, row['role'], row['id'], record['exact_complete_reference_token_match'], flush=True)
        return dict(sources=len(values), exact_reference_matches=sum(r['exact_complete_reference_token_match'] for r in values),
            first_reference_token_matches=sum(r['first_reference_token_match'] for r in values),
            terminal_EOS=sum(r['terminal_EOS'] for r in values))

    initial_losses = dict(train=assess(train), validation=assess(validation))
    atomic_json(reports / 'initial-source-NLL.json', initial_losses)
    # Frozen1024-token initial outputs for all eight exact controls already exist
    # in the bound source-generation report; no benchmark outcomes enter updates.
    total_targets = sum(r['target_tokens'] for r in train)
    for step in range(256):
        guard.check()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(PROFILES[1], step)
        optimizer.param_groups[0]['lr'] = lr
        weighted_loss = 0.
        started = time.monotonic()
        for row in train:
            ids, labels = sequences[row['id']]
            with amp():
                loss = model.code(torch.tensor([ids], device=device),
                    labels=torch.tensor([labels], device=device))['loss']
            if not bool(torch.isfinite(loss)):
                raise ValueError('Finite full original-program loss required')
            (loss * row['target_tokens'] / total_targets).backward()
            weighted_loss += float(loss.detach()) * row['target_tokens'] / total_targets
        if any(p.grad is not None for p in model.base.parameters()):
            raise ValueError('Original general/vision/critic backbone must stay frozen')
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.adapter_parameters()):
            raise ValueError('All48 residual gradients must remain present and finite')
        norm = torch.nn.utils.clip_grad_norm_(model.adapter_parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        metrics.append(dict(step=step + 1, target_tokens=total_targets, mean_NLL_before_update=weighted_loss,
            learning_rate=lr, gradient_norm_before_clipping=float(norm), seconds=time.monotonic()-started))
        if step < 2 or (step + 1) % 8 == 0:
            status('training bounded original four-program memorization diagnostic',
                completed_steps=step + 1, planned_steps=256, latest=metrics[-1])
            print('four_source_update', json.dumps(metrics[-1]), flush=True)
        if step + 1 in [32, 128, 256]:
            observation = dict(step=step + 1, train_NLL=assess(train), validation_NLL=assess(validation),
                train_generation=generate(train, step + 1))
            if step + 1 == 256:
                observation['validation_generation'] = generate(validation, step + 1)
            observations.append(observation)
            atomic_json(reports / 'observations.json', dict(observations=observations))
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    if state_digest(model.base.state_dict()) != base_before:
        raise ValueError('Every original frozen backbone tensor must remain identical')
    guard.check(full=True)
    state = make_training_state(model, optimizer, base_checkpoint_sha256=bound[str(source)],
        tokenizer_sha256=bound[str(token)], run_plan_sha256=plan_sha, completed_steps=256,
        planned_steps=256, device=device, metrics=metrics)
    state.update(stage='code_residual_four_source_learning_probe_v1',
        starting_policy_checkpoint_sha256=bound[str(initial_path)],
        source_manifest_sha256=bound[str(pack_root / 'manifest.json')],
        fresh_optimizer=True, training_sources=4, repeated_training_exposures=1024,
        memorization_diagnostic_only=True)
    checkpoint = output / 'final.pt'
    atomic_checkpoint(checkpoint, state, maximum_bytes=2 * 1024**3, minimum_free_bytes=16 * 1024**3)
    result = dict(completed_at=datetime.now(timezone.utc).isoformat(), source_files_sha256=bound,
        plan_sha256=plan_sha, parameter_counts=model.parameter_counts(), actual_updates=256,
        complete_original_training_programme_exposures=1024, original_training_sources=4,
        response_target_exposures=256 * total_targets, initial_source_losses=initial_losses,
        observations=observations, initial_residual_state_sha256=initial_residual,
        final_residual_state_sha256=state_digest(model.residuals.state_dict()),
        original_base_state_sha256=base_before, final_base_state_sha256=state_digest(model.base.state_dict()),
        checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
        complete_residual_Adam_RNG_saved=True, generation_report_sha256=file_hash(reports / 'generations.json'),
        validation_or_test_rows_used_for_updates=False, full_source_or_final_base_integration=False,
        functional_correctness_assessed=False, candidate_promoted=False, goal_achieved=False,
        scope='Four-original-program memorization diagnostic only. Even4/4 exact reference reproduction demonstrates learning mechanics and termination on trained examples, not independent coding, broad knowledge, multimodal ability or frontier parity. Original queued curriculum and all full-source/capability gates remain.')
    atomic_json(reports / 'results.json', result)
    status('bounded source-learning probe complete; memorization only; no promotion',
        actual_updates=256, observations=observations, checkpoint=str(checkpoint))
    print('four_source_code_learning_probe_complete', flush=True)


if __name__ == '__main__':
    main()
