"""Frozen interim word-reading checks, including train/validation and image controls."""
import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import tempfile

import torch

from .character_ctc import batch_pixels, collapse, ctc_loss
from .corpus import file_hash
from .ctc_assessment import select
from .diverse_images import atomic_json, jsonl
from .evaluation_provenance import capture, seal, pixels_sha256
from .learned_character_ctc import LearnedCharacterSequenceModel
from .model import ModelConfig, VisionEncoder
from .ocr_evaluation import transcription_scores
from .visual_character_probe import state_digest


@contextmanager
def frozen_checkpoint(path, directory):
    """Keep the opened checkpoint inode even when the trainer replaces latest.pt."""
    with tempfile.TemporaryDirectory(prefix='.sequence-assessment-', dir=directory) as temporary:
        snapshot = Path(temporary) / 'checkpoint.pt'
        os.link(path, snapshot)
        yield snapshot


def summarize(rows):
    if not rows:
        raise ValueError('No sequence diagnostic cases')
    characters = sum(row['reference_characters'] for row in rows)
    return {
        'examples': len(rows),
        'exact_words': sum(row['exact_match'] for row in rows),
        'exact_accuracy': sum(row['exact_match'] for row in rows) / len(rows),
        'character_error_rate': sum(row['character_edits'] for row in rows) / characters,
        'empty_outputs': sum(not row['predicted'] for row in rows),
        'mean_blank_probability': sum(row['blank_probability'] for row in rows) / len(rows),
        'blank_argmax_frame_fraction': sum(row['blank_argmax_frame_fraction'] for row in rows) / len(rows),
        'normalized_ctc_loss': sum(row['proper_ctc_loss'] for row in rows) / len(rows),
        'blank_ctc_loss_gain': sum(row['blank_ctc_loss'] - row['proper_ctc_loss'] for row in rows) / len(rows),
        'wrong_ctc_loss_gain': sum(row['wrong_ctc_loss'] - row['proper_ctc_loss'] for row in rows) / len(rows),
        'blank_original_target_match': sum(row['blank_predicted'] == row['expected'] for row in rows) / len(rows),
        'wrong_original_target_match': sum(row['wrong_predicted'] == row['expected'] for row in rows) / len(rows),
    }


def exposures(plan, completed_steps):
    """Reconstruct the hash-verified trainer's deterministic shuffled exposure order."""
    if not 0 <= completed_steps <= plan['steps']:
        raise ValueError('Checkpoint step lies outside its training plan')
    generator = torch.Generator().manual_seed(42)
    order = []
    for _ in range(plan['passes']):
        order.extend(torch.randperm(plan['train_rows'], generator=generator).tolist())
    if not order:
        raise ValueError('Empty planned training order')
    batch = plan['batch_size'] * plan['accumulation']
    total = plan['steps'] * batch
    order = (order * ((total + len(order) - 1) // len(order)))[:total]
    stop = completed_steps * batch
    encoder_start = min(completed_steps, plan.get('head_warmup_steps', 0)) * batch
    return Counter(order[:stop]), Counter(order[encoder_start:stop])


@torch.no_grad()
def assess(model, rows, device, size):
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    results = []
    for index, row in enumerate(rows):
        alternatives = rows[index + 1:] + rows[:index]
        other = next((candidate for candidate in alternatives
                      if candidate['ctc_family'] == row['ctc_family']
                      and candidate['group_id'] != row['group_id']
                      and candidate['response'] != row['response']), None)
        if other is None:
            raise ValueError('Each family needs distinct parent images and distinct transcripts for controls')
        images = batch_pixels([row, other], size, device)
        inputs = torch.cat([images[:1], torch.zeros_like(images[:1]), images[1:]], 0)
        with amp():
            logits = model(inputs)
        if not torch.isfinite(logits).all():
            raise ValueError('Non-finite sequence diagnostic logits')
        paths = logits.argmax(-1)
        predictions = [collapse(path) for path in paths.tolist()]
        losses = [float(ctc_loss(value[None], [row['response']])) for value in logits]
        results.append({
            'id': row['id'], 'group_id': row['group_id'], 'family': row['ctc_family'],
            'expected': row['response'], 'predicted': predictions[0],
            'blank_predicted': predictions[1], 'wrong_predicted': predictions[2],
            'wrong_image_id': other['id'], 'wrong_image_group_id': other['group_id'],
            'proper_ctc_loss': losses[0], 'blank_ctc_loss': losses[1], 'wrong_ctc_loss': losses[2],
            'blank_probability': float(logits[0].float().softmax(-1)[:, 0].mean()),
            'blank_argmax_frame_fraction': float((paths[0] == 0).float().mean()),
            'input_pixels_sha256': pixels_sha256(inputs[0]),
            'blank_pixels_sha256': pixels_sha256(inputs[1]),
            'wrong_pixels_sha256': pixels_sha256(inputs[2]),
            'training_exposures': row.get('diagnostic_training_exposures'),
            'encoder_training_exposures': row.get('diagnostic_encoder_exposures'),
            **transcription_scores(row['response'], predictions[0]),
        })
        if len(results) % 32 == 0:
            print('assessed sequence cases', len(results), flush=True)
    report = {'overall': summarize(results),
              'by_family': {family: summarize([row for row in results if row['family'] == family])
                            for family in sorted({row['family'] for row in results})},
              'results': results}
    if any(row['training_exposures'] is not None for row in results):
        report['by_exposure'] = {}
        for name, predicate in [('previously_exposed', lambda row: row['training_exposures'] > 0),
                                ('not_yet_exposed', lambda row: row['training_exposures'] == 0),
                                ('encoder_exposed', lambda row: row['encoder_training_exposures'] > 0)]:
            subset = [row for row in results if predicate(row)]
            if subset:
                report['by_exposure'][name] = summarize(subset)
    return report


def evaluate(checkpoint, data, output, device='cuda:2', maximum_per_family=32):
    if maximum_per_family < 2:
        raise ValueError('Need at least two parent images per family')
    torch.set_num_threads(1)
    if device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.15, int(device.split(':')[1]) if ':' in device else 0)
    checkpoint, data, output = map(Path, [checkpoint, data, output])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('Use a fresh diagnostic report path')
    files = [Path(__file__).with_name(name + '.py') for name in
             ['sequence_diagnostics', 'character_ctc', 'learned_character_ctc', 'ctc_assessment',
              'ctc_decoding', 'model', 'corpus', 'diverse_images', 'evaluation_provenance',
              'ocr_evaluation', 'visual_character_probe']]
    provenance = capture(files)
    manifest_digest = file_hash(data / 'manifest.json')
    manifest = json.loads((data / 'manifest.json').read_text())
    selections = {}
    populations = {}
    all_groups = {}
    for split in ['train', 'valid']:
        path = data / (split + '.jsonl')
        if file_hash(path) != manifest['files_sha256'][path.name]:
            raise ValueError('Frozen sequence records changed')
        rows = list(jsonl(path))
        populations[split] = rows
        all_groups[split] = {row['group_id'] for row in rows}
        selections[split] = select(rows, maximum_per_family)
    if all_groups['train'] & all_groups['valid']:
        raise ValueError('Reading parent groups overlap between training and validation')
    with frozen_checkpoint(checkpoint, output.parent) as snapshot:
        checkpoint_digest = file_hash(snapshot)
        state = torch.load(snapshot, map_location='cpu', weights_only=True, mmap=True)
        plan = state['plan']
        if plan.get('readout_implementation_sha256') != file_hash(Path(__file__).with_name('learned_character_ctc.py')):
            raise ValueError('Checkpoint learned readout implementation differs')
        if plan['implementation_sha256'] != file_hash(Path(__file__).with_name('character_ctc.py')):
            raise ValueError('Checkpoint sequence training implementation differs')
        if plan['manifest_sha256'] != manifest_digest:
            raise ValueError('Checkpoint sequence curriculum differs')
        if plan['train_rows'] != len(populations['train']):
            raise ValueError('Checkpoint training row count differs')
        all_exposures, encoder_exposures = exposures(plan, state['steps'])
        for index, row in enumerate(populations['train']):
            row['diagnostic_training_exposures'] = all_exposures[index]
            row['diagnostic_encoder_exposures'] = encoder_exposures[index]
        model = LearnedCharacterSequenceModel(VisionEncoder(ModelConfig(**plan['config'])), plan['frames'])
        model.load_state_dict(state['model'], strict=True)
        model = model.to(device).eval().requires_grad_(False)
        before = state_digest(model)
        report = {
            'checkpoint_sha256': checkpoint_digest, 'checkpoint_requested_path': str(checkpoint.resolve()),
            'checkpoint_steps': state['steps'], 'plan': plan, 'manifest_sha256': manifest_digest,
            'maximum_per_family': maximum_per_family, 'frames': model.frames,
            'image_size': manifest['configuration']['image_size'], 'component_state_sha256': before,
            'training_population_examples': len(populations['train']),
            'training_examples_previously_exposed': len(all_exposures),
            'training_examples_encoder_exposed': len(encoder_exposures),
            'splits': {split: assess(model, rows, device, manifest['configuration']['image_size'])
                       for split, rows in selections.items()},
            'goal_achieved': False,
            'protocol': 'Hash-ordered unique-parent cases per family; fixed greedy CTC decoding; proper, blank and distinct-transcript wrong images. Loss gains average target-length-normalized CTC losses per example.',
            'limitations': [
                'Training cases diagnose fitting; existing validation splits diagnose development performance.',
                'Hash-ordered training samples can include not-yet-exposed rows; exposure counts and separate summaries identify them.',
                'Not independent final OCR, whole-page reading, handwriting or multilingual evaluation.',
                'Component word reading is separate from native language-model transcription and broad visual ability.',
                'The temporary hard link preserves one checkpoint during assessment; it is removed after reporting.',
            ],
        }
        if state_digest(model) != before or file_hash(snapshot) != checkpoint_digest:
            raise ValueError('Frozen sequence state changed during assessment')
    for name in ['train.jsonl', 'valid.jsonl']:
        if file_hash(data / name) != manifest['files_sha256'][name]:
            raise ValueError('Sequence records changed during assessment')
    seal(report, provenance, files=files, data=data / 'manifest.json', digest=manifest_digest,
         device=device, tokenizer_sha256=plan['tokenizer_sha256'])
    atomic_json(output, report)
    print(json.dumps({'checkpoint_steps': report['checkpoint_steps'],
                      'splits': {name: value['overall'] for name, value in report['splits'].items()}}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--maximum-per-family', type=int, default=32)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data, args.output, args.device, args.maximum_per_family)


if __name__ == '__main__':
    main()
