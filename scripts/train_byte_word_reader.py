"""Train and evaluate the own scratch word recognizer on existing public crops."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader, Dataset

from slm_lab.byte_word_reader import ByteWordReader, evaluate, labels, load_rows, loss, pixels
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.model import ModelConfig, TinyMultimodalLM

ROOT = Path(__file__).resolve().parents[1]


class Words(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        return pixels(row), row['response']


def main():
    torch.set_num_threads(1); torch.manual_seed(42); random.seed(42)
    reports = ROOT / 'reports/byte-word-reader-v1'; checkpoints = ROOT / 'checkpoints/byte-word-reader-v1'
    if reports.exists() or checkpoints.exists(): raise ValueError('Existing byte-word learning needs an inspected resume')
    device = 'cuda:3'; torch.cuda.set_per_process_memory_fraction(.2, 3)
    data = ROOT / 'data/byte-word-reader-v1'
    rows, manifest = load_rows(data)
    model = ByteWordReader().to(device)
    added = sum(parameter.numel() for parameter in model.parameters())
    with torch.device('meta'): policy = TinyMultimodalLM(ModelConfig.load(ROOT / 'configs/model.json'))
    combined = policy.parameter_counts()['total'] + added; del policy
    if combined >= 1_000_000_000: raise ValueError('Hypothetical own policy plus recognizer exceeds requested parameter limit')
    files = [Path(__file__), ROOT / 'src/slm_lab/byte_word_reader.py', ROOT / 'src/slm_lab/ocr_evaluation.py',
             ROOT / 'src/slm_lab/rendered_text.py', ROOT / 'src/slm_lab/model.py', ROOT / 'src/slm_lab/corpus.py',
             ROOT / 'src/slm_lab/diverse_images.py']
    loaded = capture(files); reports.mkdir(); checkpoints.mkdir()
    epochs = 8; batch_size = 64; steps = epochs * math.ceil(len(rows['train']) / batch_size)
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'random_initialization_seed': 42,
            'data_manifest_sha256': file_hash(data / 'manifest.json'), 'source_files_sha256': manifest['source_files_sha256'],
            'train_examples': len(rows['train']), 'validation_examples_available': len(rows['valid']),
            'epochs': epochs, 'batch_size': batch_size, 'steps': steps, 'learning_rate': .001,
            'trainable_parameters': added, 'hypothetical_own_policy_plus_component_parameters': combined,
            'device': device, 'convolution_precision': 'bf16 AMP', 'recurrent_and_CTC_precision': 'fp32',
            'external_model_weights_used': False, 'source_provenance': loaded, 'goal_achieved': False,
            'scope': 'Scratch convolutional/recurrent byte-CTC word component; not yet integrated into native photo understanding.',
            'limitations': manifest['limitations']}
    atomic_json(reports / 'plan.json', plan)

    def status(message, **values):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
                                            'goal_achieved': False, **values})
    def verify():
        if capture(files) != loaded or file_hash(data / 'manifest.json') != plan['data_manifest_sha256']:
            raise ValueError('Loaded byte-word implementation/runtime/input manifest changed')
    def save(name, step, optimizer):
        value = {'model': {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
                 'optimizer': optimizer.state_dict(), 'step': step, 'plan': plan}
        target = checkpoints / name; temporary = target.with_suffix('.tmp')
        torch.save(value, temporary); temporary.replace(target)

    try:
        # Temporary actual-source CUDA updates verify memory/gradient behavior,
        # then restore the original initial tensors before the real experiment.
        initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        probe_optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        torch.cuda.reset_peak_memory_stats(3)
        sample = rows['train'][:8]
        inputs = torch.stack([pixels(row) for row in sample]).to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16): logits = model(inputs)
        probe_loss = loss(logits, [row['response'] for row in sample]); probe_loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); probe_optimizer.step()
        if not torch.isfinite(probe_loss): raise ValueError('Nonfinite actual-source CUDA byte-word loss')
        atomic_json(reports / 'cuda-readiness.json', {'actual_word_crops': len(sample), 'loss': float(probe_loss.detach()),
                    'gradient_norm': float(norm), 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(3),
                    'temporary_learning_discarded': True, 'goal_achieved': False})
        model.load_state_dict(initial)
        if any(not torch.equal(value.detach().cpu(), initial[key]) for key, value in model.state_dict().items()):
            raise ValueError('Temporary CUDA updates were not restored to exact scratch initialization')
        del initial, inputs, logits, probe_optimizer, probe_loss
        model.zero_grad(set_to_none=True)
        status('assessing initial scratch byte-word recognizer')
        baseline = evaluate(model, rows['valid'], reports / 'before.json', device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        generator = torch.Generator().manual_seed(42)
        loader = DataLoader(Words(rows['train']), batch_size=batch_size, shuffle=True, generator=generator,
                            num_workers=1, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        started = time.monotonic(); step = 0; actual_examples = 0; metrics = []
        for epoch in range(epochs):
            verify(); model.train()
            for values, texts in loader:
                values = values.to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True)
                rate = .001 * min(1., (step + 1) / 250) * (.1 + .9 * (1 + math.cos(math.pi * step / steps)) / 2)
                for group in optimizer.param_groups: group['lr'] = rate
                with torch.autocast('cuda', dtype=torch.bfloat16): logits = model(values)
                value = loss(logits, list(texts))
                if not torch.isfinite(value): raise ValueError('Nonfinite scratch byte-word loss')
                value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); optimizer.step()
                step += 1; actual_examples += len(texts)
                if step == 1 or step % 50 == 0:
                    metric = {'step': step, 'epoch': epoch + 1, 'loss': float(value.detach()),
                              'learning_rate': rate, 'actual_examples': actual_examples,
                              'elapsed_seconds': time.monotonic() - started}
                    metrics.append(metric); print(json.dumps(metric), flush=True)
                    status('learning own scratch byte-word recognition', **metric, planned_steps=steps)
                if step % 500 == 0: save('latest.pt', step, optimizer)
            save('latest.pt', step, optimizer)
        if step != steps or actual_examples != epochs * len(rows['train']): raise ValueError('Declared whole-epoch word exposures differ')
        verify()
        save('completed-training-state.pt', step, optimizer)
        target = checkpoints / 'model.pt'; temporary = target.with_suffix('.tmp')
        torch.save({'model': {key: value.detach().cpu() for key, value in model.state_dict().items()}, 'plan': plan, 'step': step}, temporary)
        temporary.replace(target)
        status('assessing completed scratch byte-word recognizer')
        after = evaluate(model, rows['valid'], reports / 'after.json', device)
        verify()
        for path, digest in manifest['source_files_sha256'].items():
            if file_hash(path) != digest: raise ValueError('Original word source changed during training')
        atomic_json(reports / 'comparison.json', {'before': baseline['overall'], 'after': after['overall'],
                    'by_family_before': baseline['by_family'], 'by_family_after': after['by_family'],
                    'model_sha256': file_hash(target), 'actual_training_examples': actual_examples,
                    'hypothetical_own_policy_plus_component_parameters': combined, 'production_promoted': False,
                    'scope': plan['scope'], 'limitations': plan['limitations'], 'goal_achieved': False})
        atomic_json(reports / 'training.json', {'plan': plan, 'metrics': metrics, 'steps': step,
                    'actual_examples': actual_examples, 'goal_achieved': False})
        status('scratch byte-word component trained and assessed; native full-photo reading unverified')
    except Exception as error:
        status('byte-word experiment failed; inspect actual live/artifact evidence before resuming', error=str(error)); raise


if __name__ == '__main__': main()
