"""Bounded gate-learning feasibility check on our own weights and new public code.

Only three recurrence gates learn here. This is not the matched full-model
training experiment or a functional coding benchmark, and cannot promote a model.
"""
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import torch

from slm_lab.corpus import file_hash, PackedCorpus
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.model import ModelConfig
from slm_lab.recurrent_model import RecurrentMultimodalLM

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1); torch.manual_seed(7321)
    reports = ROOT / 'reports/recurrent-owned-code-probe-v1'
    if reports.exists(): raise ValueError('Preserve the original recurrence feasibility experiment')
    reports.mkdir()
    base = ROOT / 'checkpoints/extended-curriculum-v1/foundation-warmup/model.pt'
    data = ROOT / 'data/systems-variety-code-continuation-v1/new-packed'
    files = [ROOT / name for name in capture()['files_sha256']] + [Path(__file__), ROOT / 'src/slm_lab/recurrent_model.py']
    runtime = capture(files); base_sha = file_hash(base); data_sha = file_hash(data / 'corpus.json')
    readiness = json.loads((ROOT / 'reports/systems-variety-training-admission-readiness-v1.json').read_text())
    if readiness['passed'] is not True or readiness['new_train_tokens'] != 52569340:
        raise ValueError('Require the complete verified original systems token admission')
    metadata = torch.load(base, map_location='cpu', weights_only=True, mmap=True)
    if metadata['steps'] != 4785 or metadata['stage'] != 'pretrain': raise ValueError('Use the exact completed own scratch foundation warmup')
    if (metadata['tokenizer_sha256'] != readiness['tokenizer_sha256']
            or file_hash(base.parent / 'tokenizer.json') != readiness['tokenizer_sha256']):
        raise ValueError('Own checkpoint and admitted systems tokenizers differ')
    packed_manifest = json.loads((data / 'corpus.json').read_text())
    for split in packed_manifest['splits'].values():
        for shard in split['shards']:
            if file_hash(data / shard['path']) != shard['sha256']:
                raise ValueError('Original systems probe token shard changed')
    config = ModelConfig(**metadata['config']); model = RecurrentMultimodalLM(config, recurrences=4)
    model.load_owned_base(metadata['model']); del metadata; gc.collect()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    model.recurrence_gates.requires_grad_(True)
    if model.parameter_counts()['total'] >= 1_000_000_000: raise ValueError('Own recurrent model exceeds the total parameter budget')
    device = 'cuda:3'; torch.cuda.set_per_process_memory_fraction(.2, 3); model.to(device)
    train = PackedCorpus(data, readiness['tokenizer_sha256'], 512)
    valid = PackedCorpus(data, readiness['tokenizer_sha256'], 512, split='valid')
    indices = lambda count, size: [i * (size - 1) // (count - 1) for i in range(count)]
    train_indices = indices(32, len(train)); valid_indices = indices(16, len(valid))
    def sequence(corpus, index): return torch.tensor([corpus[index]['tokens']], dtype=torch.long, device=device)
    def guard():
        if capture(files) != runtime or file_hash(data / 'corpus.json') != data_sha:
            raise ValueError('Own recurrence implementation or admitted token data changed')
    def status(state, **fields):
        atomic_json(reports / 'status.json', {'status': state, 'updated_at': datetime.now(timezone.utc).isoformat(),
                                             'goal_achieved': False, **fields})
    plan = {'base_checkpoint_sha256': base_sha, 'base_checkpoint_updates': 4785,
        'own_total_parameters': model.parameter_counts()['total'], 'new_trainable_gate_parameters': 3,
        'input_data_sha256': data_sha, 'tokenizer_sha256': readiness['tokenizer_sha256'],
        'train_windows': train_indices, 'validation_windows': valid_indices, 'context_tokens': 512,
        'steps': 64, 'learning_rate': .003, 'device': device, 'source_runtime': runtime,
        'decoder_layer_executions': {str(depth): model.execution_layers(depth) for depth in (1, 2, 4)},
        'scope': 'Own completed warmup weights, frozen backbone, only three new recurrence gates; original packed public systems code.',
        'goal_achieved': False, 'automatic_policy_promotion': False,
        'limitations': ['Tiny feasibility sample; not full-model recurrent training or a matched compute-budget comparison.',
            'Prediction loss is not executable-code success, algorithmic efficiency or broad knowledge/image capability.',
            'This base precedes the latest10000 main checkpoint; its scores are a separate architecture experiment.']}
    atomic_json(reports / 'plan.json', plan)
    def evaluate(depth):
        model.eval(); total = 0; targets = 0
        with torch.no_grad():
            for index in valid_indices:
                ids = sequence(valid, index)
                with torch.autocast('cuda', dtype=torch.bfloat16): loss = model(ids, labels=ids, recurrences=depth)['loss']
                total += float(loss) * (ids.shape[1] - 1); targets += ids.shape[1] - 1
        return {'nll': total / targets, 'target_tokens': targets, 'original_validation_windows': len(valid_indices)}
    try:
        guard(); status('assessing own base and zero-gate shared-depth equivalence')
        before = evaluate(1); zero = evaluate(4)
        if abs(before['nll'] - zero['nll']) > 1e-6: raise ValueError('Zero recurrence gates did not preserve the own base loss')
        optimizer = torch.optim.AdamW([model.recurrence_gates], lr=.003, weight_decay=0)
        metrics = []; model.train(); model.gradient_checkpointing = True
        for step in range(64):
            ids = sequence(train, train_indices[step % len(train_indices)]); started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16): loss = model(ids, labels=ids, recurrences=4)['loss']
            loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(model.recurrence_gates.grad).all():
                raise ValueError('Own recurrence gate gradients/loss are nonfinite')
            torch.nn.utils.clip_grad_norm_([model.recurrence_gates], 1); optimizer.step()
            metrics.append({'step': step + 1, 'loss': float(loss), 'actual_target_tokens': ids.shape[1] - 1,
                            'seconds': time.monotonic() - started})
            if (step + 1) % 8 == 0:
                guard(); status('learning three recurrence gates on original public systems code', completed_steps=step + 1)
                print(json.dumps(metrics[-1]), flush=True)
        guard(); status('assessing held source windows after recurrence gate learning')
        after = {str(depth): evaluate(depth) for depth in (1, 2, 4)}
        if after['1'] != before: raise ValueError('Frozen own one-pass backbone changed during gate-only training')
        if file_hash(base) != base_sha: raise ValueError('Own warmup source checkpoint changed')
        result = {'plan': plan, 'before_one_pass': before, 'before_zero_gate_four_passes': zero,
            'after_by_recurrences': after, 'learned_raw_gates': model.recurrence_gates.detach().cpu().tolist(),
            'actual_training_target_tokens': sum(item['actual_target_tokens'] for item in metrics), 'metrics': metrics,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(3), 'goal_achieved': False,
            'automatic_policy_promotion': False, 'functional_coding_gain_measured': False}
        atomic_json(reports / 'result.json', result)
        status('own recurrence gate feasibility experiment completed; functional benefit unverified')
    except Exception as error:
        status('requires inspection; own recurrence experiment stopped', error=str(error)); raise


if __name__ == '__main__': main()
