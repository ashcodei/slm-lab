"""Queue scratch whole-photo word localization after the verified word-reader run."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import gc
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader, Dataset

from slm_lab.byte_word_reader import ByteWordReader
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.masked_visual_pretraining import process_identity
from slm_lab.model import ModelConfig, TinyMultimodalLM
from slm_lab.word_localizer import WordLocalizer, assignments, encode_targets, load_rows, loss, pixels
from slm_lab.word_localization_evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]
READER_DONE = 'scratch byte-word component trained and assessed; native full-photo reading unverified'


class Photos(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        return pixels(row), encode_targets(row['targets'])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--reader-pid', type=int, required=True)
    parser.add_argument('--reader-start-ticks', required=True); args = parser.parse_args()
    reports = ROOT / 'reports/word-localizer-v1'; checkpoints = ROOT / 'checkpoints/word-localizer-v1'
    if reports.exists() or checkpoints.exists(): raise ValueError('Existing detector work requires an inspected resume')
    if process_identity(args.reader_pid) != args.reader_start_ticks or b'scripts/train_byte_word_reader.py' not in Path(f'/proc/{args.reader_pid}/cmdline').read_bytes().split(b'\0'):
        raise ValueError('Expected word-reader predecessor is not verified live')
    torch.set_num_threads(1); torch.manual_seed(42); random.seed(42)
    files = [Path(__file__)] + [ROOT / 'src/slm_lab' / (name + '.py') for name in
             ['word_localizer', 'word_localization_evaluation', 'text_localization_data', 'text_localization_metrics',
              'byte_word_reader', 'masked_visual_pretraining', 'model', 'training', 'corpus', 'diverse_images', 'evaluation_provenance', 'vqa_pairs', 'public_sources']]
    runtime = capture(files); data = ROOT / 'data/text-localization-v1'; manifest_sha = file_hash(data / 'manifest.json')
    reports.mkdir(); checkpoints.mkdir()
    reader_plan = ROOT / 'reports/byte-word-reader-v1/plan.json'; reader_plan_sha = file_hash(reader_plan)
    readiness_path = ROOT / 'reports/word-localizer-preparation-readiness-v1.json'
    readiness = json.loads(readiness_path.read_text())
    if readiness['data_manifest_sha256'] != manifest_sha or readiness['implementation_sha256'] != file_hash(ROOT / 'src/slm_lab/word_localizer.py'):
        raise ValueError('Detector preparation/readiness report does not match current data/implementation')
    atomic_json(reports / 'queued-plan.json', {'created_at': datetime.now(timezone.utc).isoformat(),
                'reader_pid': args.reader_pid, 'reader_start_ticks': args.reader_start_ticks,
                'train_full_photos': 5000, 'validation_full_photos': 400, 'epochs': 10, 'batch_size': 4,
                'steps': 12500, 'planned_actual_photo_exposures': 50000,
                'own_detector_parameters': readiness['own_detector_parameters'],
                'hypothetical_policy_reader_detector_parameters': readiness['hypothetical_policy_reader_detector_parameters'],
                'data_manifest_sha256': manifest_sha, 'readiness_report_sha256': file_hash(readiness_path),
                'source_provenance': runtime, 'device': 'cuda:3', 'training_started': False,
                'external_weights_used': False, 'automatic_policy_promotion': False, 'goal_achieved': False})

    def status(message, **values):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(), 'goal_achieved': False, **values})
    def verify():
        if capture(files) != runtime or file_hash(data / 'manifest.json') != manifest_sha or file_hash(reader_plan) != reader_plan_sha:
            raise ValueError('Loaded detector implementation/data/predecessor plan changed')
    try:
        status('waiting for verified word-reader training and final assessment', reader_pid=args.reader_pid, reader_start_ticks=args.reader_start_ticks)
        while process_identity(args.reader_pid) == args.reader_start_ticks:
            verify(); time.sleep(15)
        verify()
        predecessor = json.loads((ROOT / 'reports/byte-word-reader-v1/status.json').read_text())
        if predecessor['status'] != READER_DONE or not (ROOT / 'reports/byte-word-reader-v1/comparison.json').is_file():
            raise ValueError('Reader ended without its declared complete training/assessment; inspect before continuing')
        rows, manifest = load_rows(data)
        representation = {}
        for split in ['train', 'valid']:
            counts = Counter()
            for row in rows[split]:
                _, actual = assignments(row['targets']); counts.update(actual)
                if len(actual) == 0: raise ValueError('Missing actual word-assignment audit')
            representation[split] = dict(counts)
        # Every actual full-photo pixel/file hash is checked before learning,
        # and again when the photo is loaded in each pass or evaluation.
        for split in ['train', 'valid']:
            for index, row in enumerate(rows[split]):
                pixels(row)
                if index % 500 == 0: print('Detector photo-byte admission', split, index + 1, len(rows[split]), flush=True)
        device = 'cuda:3'; torch.cuda.set_per_process_memory_fraction(.2, 3)
        model = WordLocalizer().to(device); added = sum(parameter.numel() for parameter in model.parameters())
        with torch.device('meta'):
            policy = TinyMultimodalLM(ModelConfig.load(ROOT / 'configs/model.json')); reader = ByteWordReader()
        combined = policy.parameter_counts()['total'] + sum(parameter.numel() for parameter in reader.parameters()) + added
        del policy, reader
        if combined >= 1_000_000_000: raise ValueError('Own policy plus both proposed image-reading components exceeds requested limit')
        epochs = 10; batch_size = 4; steps = epochs * math.ceil(len(rows['train']) / batch_size)
        plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'train_full_photos': len(rows['train']), 'validation_full_photos': len(rows['valid']),
                'epochs': epochs, 'batch_size': batch_size, 'steps': steps, 'planned_actual_photo_exposures': epochs * len(rows['train']),
                'representation': representation, 'random_seed': 42, 'trainable_parameters': added, 'hypothetical_policy_reader_detector_parameters': combined,
                'external_weights_used': False, 'pretrained_tokenizer_used': False, 'device': device, 'data_manifest_sha256': manifest_sha,
                'source_provenance': runtime, 'reader_plan_sha256': reader_plan_sha,
                'learning_rate': .0003, 'weight_decay': .0001, 'warmup_steps': 250, 'gradient_clip': 5.,
                'objective': 'Center Gaussian focal plus SmoothL1 cell-offset/log-size loss; unreadable regions ignored for negative centers.',
                'reference': 'https://arxiv.org/abs/1904.07850',
                'scope': 'Own standalone full-photo axis-aligned word detector; neither native multimodal policy integration nor reliable reading established.',
                'limitations': manifest['limitations'], 'automatic_policy_promotion': False, 'goal_achieved': False}
        atomic_json(reports / 'plan.json', plan)
        status('verifying scratch full-photo detector CUDA learning')
        initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        probe_optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
        sample_loader = DataLoader(Photos(rows['train'][:batch_size]), batch_size=batch_size, num_workers=0)
        inputs, target = next(iter(sample_loader)); inputs = inputs.to(device)
        target = {stride: {key: value.to(device) for key, value in level.items()} for stride, level in target.items()}
        torch.cuda.reset_peak_memory_stats(3)
        with torch.autocast('cuda', dtype=torch.bfloat16): output = model(inputs)
        value, _ = loss(output, target); value.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); probe_optimizer.step()
        if not torch.isfinite(value): raise ValueError('Nonfinite full-photo detector readiness loss')
        atomic_json(reports / 'cuda-readiness.json', {'full_training_photos': len(inputs), 'loss': float(value.detach()),
                    'gradient_norm': float(norm), 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(3),
                    'temporary_learning_discarded': True, 'goal_achieved': False})
        model.load_state_dict(initial)
        if any(not torch.equal(value.detach().cpu(), initial[key]) for key, value in model.state_dict().items()): raise ValueError('Detector readiness failed to restore exact scratch initialization')
        del initial, inputs, target, output, value, probe_optimizer, sample_loader
        model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        status('assessing scratch detector on all validation full photos')
        before = evaluate(model, rows['valid'], reports / 'before.json', device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
        loader = DataLoader(Photos(rows['train']), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(42),
                            num_workers=1, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        started = time.monotonic(); step = 0; actual_photos = 0; metrics = []
        def save(name):
            path = checkpoints / name; temporary = path.with_suffix('.tmp')
            torch.save({'model': {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
                        'optimizer': optimizer.state_dict(), 'step': step, 'actual_photos': actual_photos, 'plan': plan}, temporary)
            temporary.replace(path)
        for epoch in range(epochs):
            verify(); model.train()
            for inputs, target in loader:
                inputs = inputs.to(device, non_blocking=True)
                target = {stride: {key: value.to(device, non_blocking=True) for key, value in level.items()} for stride, level in target.items()}
                rate = .0003 * min(1., (step + 1) / 250) * (.1 + .9 * (1 + math.cos(math.pi * step / steps)) / 2)
                for group in optimizer.param_groups: group['lr'] = rate
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16): output = model(inputs)
                value, parts = loss(output, target)
                if not torch.isfinite(value): raise ValueError('Nonfinite scratch full-photo word-localization loss')
                value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); optimizer.step()
                step += 1; actual_photos += len(inputs)
                if step == 1 or step % 50 == 0:
                    metric = {'step': step, 'epoch': epoch + 1, 'loss': float(value.detach()), 'center_focal': float(parts['center_focal']),
                              'box_geometry': float(parts['box_geometry']), 'learning_rate': rate, 'actual_photos': actual_photos,
                              'elapsed_seconds': time.monotonic() - started}
                    metrics.append(metric); print(json.dumps(metric), flush=True)
                    status('learning own scratch full-photo word localization', **metric, planned_steps=steps)
                if step % 500 == 0: save('latest.pt')
            save('latest.pt')
        if step != steps or actual_photos != epochs * len(rows['train']): raise ValueError('Actual detector photo exposures differ from declared budget')
        verify(); save('completed-training-state.pt')
        final = checkpoints / 'model.pt'; temporary = final.with_suffix('.tmp')
        torch.save({'model': {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}, 'step': step, 'plan': plan}, temporary); temporary.replace(final)
        status('assessing completed full-photo word localization')
        after = evaluate(model, rows['valid'], reports / 'after.json', device); verify()
        for path, digest in manifest['source_files_sha256'].items():
            if file_hash(path) != digest: raise ValueError('Original word-localization source changed during training')
        atomic_json(reports / 'comparison.json', {'before': before['totals'], 'after': after['totals'], 'model_sha256': file_hash(final),
                    'actual_training_photo_exposures': actual_photos, 'case_ids_match': [row['id'] for row in before['results']] == [row['id'] for row in after['results']],
                    'policy_promoted': False, 'hypothetical_policy_reader_detector_parameters': combined, 'scope': plan['scope'], 'goal_achieved': False})
        atomic_json(reports / 'training.json', {'plan': plan, 'metrics': metrics, 'steps': step, 'actual_photos': actual_photos, 'goal_achieved': False})
        status('scratch full-photo word detector trained and assessed; integrated reliable reading unverified')
    except Exception as error:
        status('word localization failed; inspect actual live/artifact evidence before resuming', error=str(error)); raise


if __name__ == '__main__': main()
