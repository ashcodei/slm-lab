"""Refine our completed own reader using all previous words plus native real words."""
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader

from slm_lab.byte_word_reader import ByteWordReader, evaluate as evaluate256, loss
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.full_byte_reading_evaluation import evaluate as evaluate_all
from slm_lab.real_word_refinement import GroupChunks, Words, load

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1); torch.manual_seed(42); random.seed(42)
    reports = ROOT / 'reports/native-word-refinement-v1'; checkpoints = ROOT / 'checkpoints/native-word-refinement-v1'
    if reports.exists() or checkpoints.exists(): raise ValueError('Existing native word refinement requires an inspected resume')
    source = ROOT / 'checkpoints/byte-word-reader-v1/model.pt'; source_sha = file_hash(source)
    original = json.loads((ROOT / 'reports/byte-word-reader-v1/comparison.json').read_text())
    if source_sha != original['model_sha256'] or json.loads((ROOT / 'reports/byte-word-reader-v1/status.json').read_text())['status'] != 'scratch byte-word component trained and assessed; native full-photo reading unverified':
        raise ValueError('Refinement requires the exact completed own scratch reader')
    state = torch.load(source, map_location='cpu', weights_only=True)
    if state['step'] != state['plan']['steps']: raise ValueError('Refinement received a partial word model')
    data = ROOT / 'data/more-native-real-words-v1'; manifest_sha = file_hash(data / 'manifest.json')
    train, valid, manifest = load(data); reports.mkdir(); checkpoints.mkdir()
    previous_all = ROOT / 'reports/full-byte-reading-v1/summary.json'; previous256 = ROOT / 'reports/byte-word-reader-v1/after.json'
    before_all = json.loads(previous_all.read_text()); before256 = json.loads(previous256.read_text())
    if before_all['checkpoint_sha256'] != source_sha or before_all['overall']['examples'] != len(valid): raise ValueError('Unmatched completed-reader full validation baseline')
    files = [Path(__file__)] + [ROOT / 'src/slm_lab' / (name + '.py') for name in
             ['real_word_refinement', 'more_real_words', 'byte_word_reader', 'full_byte_reading_evaluation',
              'ocr_evaluation', 'rendered_text', 'corpus', 'diverse_images', 'evaluation_provenance', 'vqa_pairs']]
    runtime = capture(files); baseline_hashes = {str(path): file_hash(path) for path in [previous_all, previous256]}
    device = 'cuda:2'; torch.cuda.set_per_process_memory_fraction(.2, 2)
    model = ByteWordReader().to(device); model.load_state_dict(state['model']); del state
    epochs = 4; batch_size = 64; steps = epochs * math.ceil(len(train) / batch_size)
    counts = Counter(row['byte_family'] for row in train)
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'own_source_model_sha256': source_sha,
            'data_manifest_sha256': manifest_sha, 'source_provenance': runtime, 'baseline_report_sha256': baseline_hashes,
            'epochs': epochs, 'batch_size': batch_size, 'steps': steps, 'planned_actual_word_exposures': len(train) * epochs,
            'training_rows': len(train), 'training_families': dict(counts), 'validation_rows': len(valid),
            'seed': 42, 'learning_rate': .0002, 'weight_decay': .0001, 'warmup_steps': 200,
            'gradient_clip': 5., 'native_frame_cache_maximum_bytes_per_worker': 192 * 1024**2,
            'sampler': 'All rows exactly once per epoch, shuffled photo/text chunks of at most8 for frame reuse.',
            'device': device, 'trainable_parameters': sum(value.numel() for value in model.parameters()),
            'new_parameters_added': 0, 'external_weights_or_teacher_outputs_used': False,
            'data_sources': 'All previous91585 words plus121591 unused original human native-photo words; no image downloads.',
            'scope': 'Separate continuation of own scratch word recognizer; full-photo and native-policy gains require separate assessment.',
            'automatic_policy_promotion': False, 'goal_achieved': False}
    atomic_json(reports / 'plan.json', plan)
    def status(message, **values):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(), 'goal_achieved': False, **values})
    def verify():
        if capture(files) != runtime or file_hash(data / 'manifest.json') != manifest_sha or file_hash(source) != source_sha:
            raise ValueError('Refinement implementation/runtime/input/base model changed')
        if any(file_hash(path) != digest for path, digest in baseline_hashes.items()): raise ValueError('Original reading baseline changed')
    try:
        status('verifying native/previous real-source CUDA learning and bounded frame reuse')
        initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        sample_rows = train[:4] + [row for row in train if row['refinement_backend'] == 'native'][:4]
        probe_data = Words(sample_rows); probe = DataLoader(probe_data, batch_size=8, num_workers=0)
        started = time.monotonic(); inputs, texts = next(iter(probe)); load_seconds = time.monotonic() - started
        inputs = inputs.to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.0001)
        torch.cuda.reset_peak_memory_stats(2)
        with torch.autocast('cuda', dtype=torch.bfloat16): logits = model(inputs)
        value = loss(logits, list(texts)); value.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); optimizer.step()
        if not torch.isfinite(value): raise ValueError('Nonfinite actual native-word readiness loss')
        atomic_json(reports / 'cuda-readiness.json', {'old_word_crops': 4, 'additional_native_word_crops': 4,
                    'loss': float(value.detach()), 'gradient_norm': float(norm), 'actual_source_load_seconds': load_seconds,
                    'frame_cache_bytes': probe_data.native.bytes, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(2),
                    'temporary_learning_discarded': True, 'goal_achieved': False})
        model.load_state_dict(initial)
        if any(not torch.equal(value.detach().cpu(), initial[name]) for name, value in model.state_dict().items()): raise ValueError('Refinement readiness did not restore every own-source tensor')
        del initial, inputs, logits, value, optimizer, probe, probe_data; model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        atomic_json(reports / 'before-all.json', before_all)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.0001)
        sampler = GroupChunks(train); loader = DataLoader(Words(train), batch_size=batch_size, sampler=sampler,
                            num_workers=1, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        started = time.monotonic(); step = 0; exposures = 0; metrics = []
        def save(name):
            path = checkpoints / name; temporary = path.with_suffix('.tmp')
            torch.save({'model': {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
                        'optimizer': optimizer.state_dict(), 'plan': plan, 'step': step, 'actual_word_exposures': exposures}, temporary)
            temporary.replace(path)
        for epoch in range(epochs):
            verify(); sampler.epoch = epoch; model.train()
            for inputs, texts in loader:
                inputs = inputs.to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True)
                rate = .0002 * min(1., (step + 1) / 200) * (.1 + .9 * (1 + math.cos(math.pi * step / steps)) / 2)
                for group in optimizer.param_groups: group['lr'] = rate
                with torch.autocast('cuda', dtype=torch.bfloat16): logits = model(inputs)
                value = loss(logits, list(texts))
                if not torch.isfinite(value): raise ValueError('Nonfinite native-word refinement loss')
                value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); optimizer.step()
                step += 1; exposures += len(texts)
                if step == 1 or step % 50 == 0:
                    metric = {'step': step, 'epoch': epoch + 1, 'loss': float(value.detach()), 'learning_rate': rate,
                              'actual_word_exposures': exposures, 'elapsed_seconds': time.monotonic() - started}
                    metrics.append(metric); print(json.dumps(metric), flush=True)
                    status('learning more original real words with previous-word rehearsal', **metric, planned_steps=steps)
                if step % 500 == 0: save('latest.pt')
            save('latest.pt')
        if step != steps or exposures != plan['planned_actual_word_exposures']: raise ValueError('Native-word actual training exposures differ from plan')
        verify(); save('completed-training-state.pt'); final = checkpoints / 'model.pt'; temporary = final.with_suffix('.tmp')
        torch.save({'model': {key: value.detach().cpu() for key, value in model.state_dict().items()}, 'plan': plan, 'step': step}, temporary); temporary.replace(final)
        status('assessing all original real/rendered validation crops after refinement')
        after256 = evaluate256(model, valid, reports / 'after256.json', device)
        after_all = evaluate_all(model, valid, reports / 'after-all.json', device=device, batch_size=32)
        key = lambda row: (row['id'], row['group_id'], row['expected'], row['wrong_image_id'], row['stored_image_sha256'])
        if [key(row) for row in before256['results']] != [key(row) for row in after256['results']]: raise ValueError('Refinement changed prior256 case/control identities')
        before_full = json.loads((ROOT / 'reports/full-byte-reading-v1/reading.json').read_text())
        if [key(row) for row in before_full['results']] != [key(row) for row in after_all['results']]: raise ValueError('Refinement changed full3429 case/control identities')
        verify()
        for path, digest in manifest['native_source_photos_sha256'].items():
            if file_hash(path) != digest: raise ValueError('Native source image changed during cached-frame learning')
        current = {row['id']: row for row in after_all['results']}
        losses = [row['id'] for row in before_full['results'] if row['proper_scores']['exact_match'] and not current[row['id']]['proper_scores']['exact_match']]
        atomic_json(reports / 'comparison.json', {'model_sha256': file_hash(final), 'own_source_model_sha256': source_sha,
                    'before_all': before_all['overall'], 'after_all': after_all['overall'], 'before_by_family': before_all['by_family'],
                    'after_by_family': after_all['by_family'], 'before256': before256['overall'], 'after256': after256['overall'],
                    'lost_previously_correct_word_ids': losses, 'actual_word_exposures': exposures,
                    'case_and_control_scope_preserved': True, 'production_promoted': False,
                    'scope': plan['scope'], 'goal_achieved': False})
        atomic_json(reports / 'training.json', {'plan': plan, 'steps': step, 'actual_word_exposures': exposures, 'metrics': metrics, 'goal_achieved': False})
        status('native-word refinement trained and assessed; full-photo/native-policy benefit unverified')
    except Exception as error:
        status('native-word refinement failed; inspect live/artifact evidence before resuming', error=str(error)); raise


if __name__ == '__main__': main()
