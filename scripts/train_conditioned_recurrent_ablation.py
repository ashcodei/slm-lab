"""Train input-conditioned normalized recurrence and controls on own public code.

This is a bounded single-seed experiment, isolated from the live main model.
It never promotes a policy or rewrites a bound training module.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import tempfile
import time

import torch
from tokenizers import Tokenizer

from slm_lab.corpus import PackedCorpus, file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.model import ModelConfig
from slm_lab.conditioned_recurrent_model import ConditionedRecurrentLM
from slm_lab.recurrent_ablation import (
    assess_completion, budgets, frozen_digest, language_parameters, state_digest, window_order,
)

ROOT = Path(__file__).resolve().parents[1]
STEPS = 1024
LENGTH = 512
SEED = 82731


def main():
    torch.set_num_threads(1); torch.manual_seed(SEED)
    reports = ROOT / 'reports/conditioned-recurrent-code-ablation-v1'
    if reports.exists(): raise ValueError('Preserve the original full-language recurrence experiment')
    reports.mkdir()
    base = ROOT / 'checkpoints/extended-curriculum-v1/foundation-warmup/model.pt'
    data = ROOT / 'data/systems-variety-code-continuation-v1/new-packed'
    admission_file = ROOT / 'reports/systems-variety-training-admission-readiness-v1.json'
    files = [ROOT / name for name in capture()['files_sha256']] + [Path(__file__),
        ROOT / 'src/slm_lab/recurrent_model.py', ROOT / 'src/slm_lab/recurrent_ablation.py',
        ROOT / 'src/slm_lab/conditioned_recurrent_model.py']
    runtime = capture(files); base_sha = file_hash(base)
    admission = json.loads(admission_file.read_text()); admission_sha = file_hash(admission_file)
    if (admission['passed'] is not True or admission['training_admitted'] is not True
            or admission['new_train_tokens'] != 52569340
            or admission['every_complete_original_text_and_EOS_reencoded_and_matched'] is not True):
        raise ValueError('Require the complete original source/token admission')
    data_sha = file_hash(data / 'corpus.json')
    packed = json.loads((data / 'corpus.json').read_text())
    for split in packed['splits'].values():
        for shard in split['shards']:
            if file_hash(data / shard['path']) != shard['sha256']:
                raise ValueError('Original packed source changed')
    original = torch.load(base, map_location='cpu', mmap=True, weights_only=True)
    if original['stage'] != 'pretrain' or original['steps'] != 4785:
        raise ValueError('Require our own completed scratch warmup')
    token_sha = file_hash(base.parent / 'tokenizer.json')
    if original['tokenizer_sha256'] != token_sha or token_sha != admission['tokenizer_sha256']:
        raise ValueError('Own model/source tokenizer mismatch')
    tokenizer = Tokenizer.from_file(str(base.parent / 'tokenizer.json'))
    model = ConditionedRecurrentLM(ModelConfig(**original['config']), recurrences=1)
    model.load_owned_base(original['model'])
    if model.parameter_counts()['total'] >= 1_000_000_000:
        raise ValueError('Total model exceeds the own parameter budget')
    device = 'cuda:3'; torch.cuda.set_per_process_memory_fraction(.5, 3); model.to(device)
    model.gradient_checkpointing = True
    train = PackedCorpus(data, token_sha, LENGTH)
    valid = PackedCorpus(data, token_sha, LENGTH, split='valid')
    allocation = budgets(model, STEPS, 4, LENGTH)
    order = window_order(len(train), allocation['equal_layer_control_steps'], SEED)
    valid_order = window_order(len(valid), 64, SEED + 1)
    frozen_before = frozen_digest(model)
    calibration_indices = order[:8]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        calibration = model.calibrate_owned_norms(
            torch.tensor([train[index]['tokens']], dtype=torch.long, device=device) for index in calibration_indices)
    initial_new = {name: value.detach().cpu().clone() for name, value in model.new_state().items()}
    initial_policy_digest = state_digest(model.state_dict())
    calibrated_frozen = frozen_digest(model)
    if calibrated_frozen != frozen_before: raise ValueError('Calibration changed frozen own tensors')
    checkpoint_bytes = sum(value.numel() * value.element_size() for value in model.state_dict().values()) + 1024**2
    if shutil.disk_usage('/tmp').free < 3 * checkpoint_bytes + 4 * 1024**3:
        raise ValueError('Preserve four GiB temporary-space reserve for all three isolated checkpoints')
    temp = Path(tempfile.mkdtemp(prefix='slm-conditioned-recurrent-code-v1-', dir='/tmp'))
    plan = {
        'base_checkpoint_sha256': base_sha, 'base_checkpoint_updates': 4785,
        'own_total_parameters': model.parameter_counts()['total'], 'seed': SEED,
        'context_tokens': LENGTH, 'generation_budget': 256,
        'training_windows': order, 'validation_windows': valid_order,
        'source_manifest_sha256': data_sha, 'tokenizer_sha256': token_sha,
        'admission_sha256': admission_sha, 'source_runtime': runtime,
        'device': device, 'per_process_memory_fraction': .5,
        'budgets': allocation, 'learning_rate': 2e-5, 'new_adapter_norm_learning_rate': .0001,
        'initial_recurrence_parameters_sha256': state_digest(initial_new),
        'initial_conditioned_policy_state_sha256': initial_policy_digest,
        'training_only_calibration_indices': calibration_indices, 'training_only_norm_calibration': calibration,
        'optimizer': 'AdamW, weight_decay0.01, clip_norm1, constant learning rates',
        'checkpoint_directory': str(temp), 'automatic_policy_promotion': False,
        'limitations': ['Bounded single-seed continuation from our own warmup; not full pretraining from random weights.',
            'Only new systems source is sampled here; complete old-code rehearsal remains in the later main continuation.',
            'Matched decoder-layer/token executions are a compute proxy; adapter, normalization, head, optimizer and wall time do not match exactly.',
            'Frozen vision tensors do not prove retained multimodal behavior; this experiment cannot promote a policy.',
            'Local completion checks use256 output tokens, unlike the512-token main assessment.'],
    }
    atomic_json(reports / 'plan.json', plan)
    atomic_json(ROOT / 'reports/conditioned-recurrent-code-checkpoint-location-v1.json',
                {'directory': str(temp), 'reports': str(reports), 'goal_achieved': False})

    def status(message, **fields):
        atomic_json(reports / 'status.json', {'status': message,
            'updated_at': datetime.now(timezone.utc).isoformat(), 'goal_achieved': False, **fields})

    def guard():
        if (capture(files) != runtime or file_hash(data / 'corpus.json') != data_sha
                or file_hash(admission_file) != admission_sha):
            raise ValueError('Bound source/runtime/data changed during the experiment')

    def ids(corpus, index):
        return torch.tensor([corpus[index]['tokens']], dtype=torch.long, device=device)

    @torch.no_grad()
    def loss(depth):
        model.eval(); total = 0.; count = 0
        for index in valid_order:
            sequence = ids(valid, index)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                value = model(sequence, labels=sequence, recurrences=depth)['loss']
            total += float(value) * (LENGTH - 1); count += LENGTH - 1
        return {'nll': total / count, 'target_tokens': count, 'validation_windows': len(valid_order)}

    def generated(name, depth):
        guard(); model.recurrences = depth
        status('assessing actual generated code from the isolated recurrence experiment', branch=name, depth=depth)
        def notify(language, task, count):
            status('assessing actual generated code from the isolated recurrence experiment', branch=name,
                   depth=depth, completed_tasks=count)
            print(json.dumps({'branch': name, 'coding_tasks_complete': count, 'language': language, 'task': task}), flush=True)
        return assess_completion(model, tokenizer, device, 256, notify)

    def snapshot(name, completed, depth, elapsed, metrics):
        guard()
        if frozen_digest(model) != frozen_before: raise ValueError('Own vision/critic tensors changed')
        status('assessing held original code windows', branch=name, completed_steps=completed)
        held = {str(value): loss(value) for value in ((1,) if depth == 1 else (1, 2, 4))}
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        digest = state_digest(state)
        output = temp / (name + '.pt')
        if shutil.disk_usage(temp).free < checkpoint_bytes + 4 * 1024**3:
            raise ValueError('Checkpoint output would exhaust the temporary-space reserve')
        torch.save({'architecture': 'own-input-conditioned-normalized-recurrence-v1', 'config': asdict(model.config),
                    'model': state, 'tokenizer_sha256': token_sha, 'base_checkpoint_sha256': base_sha,
                    'steps': completed, 'recurrences': depth, 'core_start': model.core_start,
                    'core_end': model.core_end, 'max_recurrences': model.max_recurrences,
                    'model_state_sha256': digest, 'source_runtime': runtime}, output)
        del state; gc.collect()
        code = generated(name, depth)
        result = {'branch': name, 'completed_steps': completed, 'depth': depth,
            'actual_training_targets': completed * (LENGTH - 1), 'training_elapsed_seconds': elapsed,
            'held_losses': held, 'coding': code, 'checkpoint': str(output),
            'checkpoint_sha256': file_hash(output), 'model_state_sha256': digest,
            'frozen_vision_and_critic_sha256': frozen_before,
            'adapter_state_half_norm': float(model.recurrence_adapter.weight[:, :model.config.hidden_size].detach().norm()),
            'adapter_input_half_norm': float(model.recurrence_adapter.weight[:, model.config.hidden_size:].detach().norm()),
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(3),
            'training_metrics': metrics, 'goal_achieved': False}
        atomic_json(reports / (name + '.json'), result)
        print(json.dumps({'branch': name, 'held_losses': held,
                          'syntax': code['syntax_passes'], 'functional': code['functional_passes'],
                          'efficiency': code['efficiency_passes']}), flush=True)
        return result

    try:
        guard(); status('assessing calibrated initial conditioned policy')
        before_loss = loss(1); before_four = loss(4)
        if not all(torch.isfinite(torch.tensor(v['nll'])) for v in (before_loss, before_four)):
            raise ValueError('Conditioned initial losses are nonfinite')
        before_code = generated('calibrated-initial-one-pass', 1)
        atomic_json(reports / 'base.json', {'held_loss': before_loss, 'initial_four_pass_loss': before_four,
            'coding': before_code, 'source_owned_checkpoint_sha256': base_sha,
            'initial_conditioned_policy_state_sha256': initial_policy_digest, 'goal_achieved': False})
        outcomes = {}
        for branch, depth, steps in [('ordinary', 1, allocation['equal_layer_control_steps']), ('recurrent', 4, STEPS)]:
            model.load_owned_base(original['model'])
            transfer = model.load_state_dict(initial_new, strict=False)
            if transfer.unexpected_keys or set(transfer.missing_keys) != set(original['model']):
                raise ValueError('Calibrated initial recurrence tensor transfer differs')
            model.recurrences = depth
            parameters = language_parameters(model, train_gates=depth > 1)
            new_names = set(model.new_state())
            groups = [{'params': [p for name, p in model.named_parameters() if p.requires_grad and name not in new_names], 'lr': 2e-5},
                      {'params': [p for name, p in model.named_parameters() if p.requires_grad and name in new_names], 'lr': .0001}]
            optimizer = torch.optim.AdamW(groups, weight_decay=.01, foreach=False)
            metrics = []; elapsed = 0.; torch.cuda.reset_peak_memory_stats(3)
            for step in range(steps):
                model.train(); sequence = ids(train, order[step]); start = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    value = model(sequence, labels=sequence, recurrences=depth)['loss']
                value.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
                if not torch.isfinite(value): raise ValueError('Nonfinite full-language training loss')
                optimizer.step(); torch.cuda.synchronize(3)
                duration = time.monotonic() - start; elapsed += duration
                metrics.append({'step': step + 1, 'loss': float(value.detach()), 'grad_norm': float(norm),
                                'training_seconds': duration, 'actual_target_tokens': LENGTH - 1})
                if (step + 1) % 16 == 0:
                    guard(); status('training the isolated full-language recurrence comparison',
                                    branch=branch, depth=depth, completed_steps=step + 1, planned_steps=steps)
                    print(json.dumps({'branch': branch, **metrics[-1]}), flush=True)
                if branch == 'ordinary' and step + 1 == STEPS:
                    outcomes['ordinary-equal-targets'] = snapshot('ordinary-equal-targets', step + 1, depth, elapsed, list(metrics))
            name = 'ordinary-equal-layers' if depth == 1 else 'recurrent-equal-targets'
            optimizer.zero_grad(set_to_none=True); del optimizer
            gc.collect(); torch.cuda.empty_cache()
            outcomes[name] = snapshot(name, steps, depth, elapsed, metrics)
        guard()
        if file_hash(base) != base_sha: raise ValueError('Original own checkpoint changed')
        atomic_json(reports / 'comparison.json', {'plan': plan, 'base_loss': before_loss,
            'base_coding': before_code, 'initial_four_pass_loss': before_four, 'outcomes': outcomes,
            'automatic_policy_promotion': False, 'goal_achieved': False,
            'scope': 'Actual own-model training/generated behavior from one bounded seed; infer no broad capability from prediction alone.'})
        status('full-language recurrence comparison completed; broad capability unverified')
    except Exception as error:
        status('requires inspection; full-language recurrence comparison stopped', error=str(error)); raise


if __name__ == '__main__': main()
