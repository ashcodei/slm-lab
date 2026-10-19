"""Paired own-encoder caption alignment with exactly frozen language weights."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from tokenizers import Tokenizer

from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .evaluation_provenance import capture
from .model import ModelConfig
from .training import Encoder
from .vision_transfer import language_digest

ROOT = Path(__file__).resolve().parents[2]


def caption_subset(source, output):
    """Keep human/source captions and original licenses, preserving held-out parents."""
    source, output = Path(source), Path(output)
    manifest = json.loads((source / 'manifest.json').read_text())
    inputs = {'manifest.json': file_hash(source / 'manifest.json')}
    rows = {}
    for split in ['train', 'valid']:
        name = split + '.jsonl'
        inputs[name] = file_hash(source / name)
        if inputs[name] != manifest['files_sha256'][name]:
            raise ValueError('Visual source records changed')
        rows[split] = [{**row, 'visual_objective': 'caption alignment'} for row in jsonl(source / name)
                       if row.get('task') == 'caption']
        if not rows[split] or any(not Path(row['image']).is_absolute() for row in rows[split]):
            raise ValueError('Caption inputs require frozen absolute image paths')
        if len({row['group_id'] for row in rows[split]}) != len(rows[split]):
            raise ValueError('Repeated caption parent within a split')
    for field in ['group_id', 'model_pixel_sha256', 'model_pixel_sha256_448']:
        if {row[field] for row in rows['train']} & {row[field] for row in rows['valid']}:
            raise ValueError('Held-out caption images overlap training')
    configuration = {'source': str(source.resolve()), 'input_sha256': inputs,
                     'implementation_sha256': file_hash(__file__)}
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'manifest.json'
    if path.exists():
        prepared = json.loads(path.read_text())
        if prepared['configuration'] != configuration:
            raise ValueError('Existing caption subset differs')
        for name, digest in prepared['files_sha256'].items():
            if file_hash(output / name) != digest:
                raise ValueError('Caption subset changed')
        return prepared
    if any(output.iterdir()):
        raise ValueError('Unrecorded caption subset files need inspection')
    for split, items in rows.items():
        (output / (split + '.jsonl')).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in items))
    prepared = {'configuration': configuration,
                'splits': {split: {'examples': len(items), 'by_source': dict(Counter(row['visual_source'] for row in items))}
                           for split, items in rows.items()},
                'files_sha256': {split + '.jsonl': file_hash(output / (split + '.jsonl')) for split in rows},
                'fashion_examples': 0, 'parent_and_exact_pixel_overlap': 0,
                'limitations': ['Source captions can include metadata not visible in the image.',
                                'Existing development splits; no near-duplicate or independent final-benchmark certification.']}
    atomic_json(path, prepared)
    return prepared


def compare_reports(first, second, family):
    """Reject comparisons with different cases, pixels or evaluation protocols."""
    common = ['data_sha256', 'tokenizer_sha256', 'evaluation_provenance',
              'inference_device_type', 'inference_precision', 'image_size']
    if family in {'ocr', 'objects'}:
        common += ['max_new_tokens', 'protocol']
    for key in common:
        if first[key] != second[key]:
            raise ValueError('Paired visual evaluation differs: ' + key)
    case_fields = {'vision': ['id', 'input_pixels_sha256', 'wrong_image_id', 'wrong_pixels_sha256', 'target_tokens'],
                   'ocr': ['id', 'group_id', 'task', 'expected', 'input_pixels_sha256', 'target_tokens'],
                   'objects': ['id', 'group_id', 'task', 'expected', 'input_pixels_sha256',
                               'wrong_image_id', 'wrong_pixels_sha256', 'target_tokens']}[family]
    cases = 'samples' if family == 'vision' else 'results'
    if [[row[key] for key in case_fields] for row in first[cases]] != [[row[key] for key in case_fields] for row in second[cases]]:
        raise ValueError('Paired visual evaluation cases or image controls differ')
    if family == 'vision':
        keys = ['proper_nll', 'blank_nll_gain', 'mismatch_nll_gain', 'caption_retrieval']
        return {key: {'first': first[key], 'second': second[key]} for key in keys}
    return {'first': first['overall'], 'second': second['overall']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical-gpu', type=int, default=2)
    args = parser.parse_args()
    if args.physical_gpu not in range(4):
        parser.error('Choose one of the four existing local GPUs')
    torch.set_num_threads(1)
    reports = ROOT / 'reports/paired-vision-alignment-v1'; reports.mkdir(exist_ok=True)
    checkpoints = ROOT / 'checkpoints/paired-vision-alignment-v1'

    def status(message, **fields):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
                                             'goal_achieved': False, 'production_promoted': False, **fields})
    try:
        if (reports / 'plan.json').exists():
            raise ValueError('Existing experiment requires inspection; automatic restart is disabled')
        preparation = json.loads((checkpoints / 'initial/preparation.json').read_text())
        data = ROOT / 'data/paired-vision-alignment-v1'
        manifest = caption_subset(ROOT / 'data/visual-pretraining-v1', data)
        for split in ['train', 'valid']:
            for row in jsonl(data / (split + '.jsonl')):
                if file_hash(row['image']) != row['stored_image_sha256']:
                    raise ValueError('Caption image bytes changed: ' + row['image'])
        initial = preparation['branches']['original']
        tokenizer = Path(initial['checkpoint']).parent / 'tokenizer.json'
        if file_hash(tokenizer) != preparation['tokenizer_sha256']:
            raise ValueError('Own tokenizer changed')
        config = ModelConfig(**preparation['config'])
        encoder = Encoder(Tokenizer.from_file(str(tokenizer)), config, data, 448)
        targets = 0; maximum = 0
        for row in jsonl(data / 'train.jsonl'):
            ids, labels = encoder.sequence(row, 'response')
            maximum = max(maximum, len(ids)); targets += sum(value != -100 for value in labels[1:])
            encoder.sequence(dict(row, text=row['response']), None)
        if encoder.truncated:
            raise ValueError('Caption alignment would truncate source labels')
        count = manifest['splits']['train']['examples']
        stages = [{'name': 'contrastive', 'stage': 'contrastive', 'steps': math.ceil(3 * count / 32),
                   'batch_size': 32, 'accumulation': 1, 'image_size': 224, 'learning_rate': 5e-5, 'passes': 3},
                  {'name': 'caption', 'stage': 'sft', 'steps': math.ceil(count / 8),
                   'batch_size': 2, 'accumulation': 4, 'image_size': 448, 'learning_rate': 5e-5, 'passes': 1}]
        for stage in stages:
            stage['actual_row_exposures'] = stage['steps'] * stage['batch_size'] * stage['accumulation']
        specs = [('vision', 'slm_lab.vision_evaluation', data / 'valid.jsonl', ['--max-examples', '128', '--image-size', '448']),
                 ('ocr', 'slm_lab.ocr_evaluation', ROOT / 'data/alignment-curriculum-v1/images.valid.jsonl',
                  ['--max-examples', '256', '--max-new-tokens', '128', '--image-size', '448']),
                 ('objects', 'slm_lab.object_evaluation', ROOT / 'data/object-qa-v1/valid.jsonl',
                  ['--max-examples', '180', '--max-new-tokens', '32'])]
        implementations = capture()
        experiment_sources = {str(path.relative_to(ROOT)): file_hash(path) for path in
                              [Path(__file__), ROOT / 'src/slm_lab/vision_transfer.py', ROOT / 'src/slm_lab/objectives.py']}
        plan = {'preparation': preparation, 'caption_manifest_sha256': file_hash(data / 'manifest.json'),
                'data_sha256': file_hash(data / 'train.jsonl'), 'training_images': count,
                'training_target_tokens_per_pass': targets, 'maximum_sequence_tokens': maximum,
                'stages': stages, 'physical_gpu': args.physical_gpu, 'seed': 42,
                'evaluation_data_sha256': {name: file_hash(path) for name, _, path, _ in specs},
                'evaluation_provenance': implementations, 'experiment_sources': experiment_sources,
                'trainable': 'vision', 'parameters': preparation['parameters'], 'goal_achieved': False,
                'production_promoted': False, 'external_model_weights_used': False,
                'limitations': manifest['limitations'] + ['Caption alignment does not directly teach transcription or object QA.',
                               'Same language tensors must remain unchanged at every stage.',
                               'Three visual development assessments cannot establish general capability or high-end parity.']}
        atomic_json(reports / 'plan.json', plan)
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.physical_gpu), 'OMP_NUM_THREADS': '1',
                       'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}

        def check_runtime():
            if capture() != implementations or any(file_hash(ROOT / path) != digest for path, digest in experiment_sources.items()):
                raise ValueError('Experiment implementation/runtime changed')
            if file_hash(data / 'train.jsonl') != plan['data_sha256']:
                raise ValueError('Caption training data changed')

        def frozen_language(path, branch):
            state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            if language_digest(state['model']) != preparation['branches'][branch]['language_state_sha256']:
                raise ValueError('Caption alignment changed frozen language/critic tensors')
            if state['config'] != preparation['config'] or state['tokenizer_sha256'] != preparation['tokenizer_sha256']:
                raise ValueError('Caption alignment model identity changed')

        def assess(path, branch, position):
            for name, module, held_out, extra in specs:
                check_runtime()
                status('evaluating paired native image use', branch=branch, position=position, family=name)
                output = reports / (branch + '-' + position + '-' + name + '.json')
                subprocess.run([sys.executable, '-u', '-m', module, '--checkpoint', str(path), '--data', str(held_out),
                                '--output', str(output), '--device', 'cuda', *extra], cwd=ROOT, env=environment, check=True)
                result = json.loads(output.read_text())
                if (result['checkpoint_sha256'] != file_hash(path) or result['evaluation_provenance'] != implementations
                        or result['data_sha256'] != plan['evaluation_data_sha256'][name] or result.get('truncated_sequences', 0)):
                    raise ValueError('Native visual report identity/truncation differs')

        for branch in ['original', 'masked']:
            source = Path(preparation['branches'][branch]['checkpoint'])
            if file_hash(source) != preparation['branches'][branch]['checkpoint_sha256']:
                raise ValueError('Initial branch checkpoint changed')
            frozen_language(source, branch)
            assess(source, branch, 'before')
            for stage in stages:
                check_runtime()
                if shutil.disk_usage(ROOT).free < 20 * 1024**3:
                    raise ValueError('Need 20 GiB free for independent and main training snapshots')
                directory = checkpoints / branch / stage['name']
                status('training paired caption alignment', branch=branch, stage=stage['name'],
                       steps=stage['steps'], physical_gpu=args.physical_gpu)
                subprocess.run([sys.executable, '-u', '-m', 'slm_lab.training', stage['stage'], '--checkpoint', str(source),
                                '--tokenizer', str(source.parent / 'tokenizer.json'), '--data', str(data / 'train.jsonl'),
                                '--output', str(directory), '--device', 'cuda', '--trainable', 'vision',
                                '--steps', str(stage['steps']), '--batch-size', str(stage['batch_size']),
                                '--gradient-accumulation', str(stage['accumulation']), '--image-size', str(stage['image_size']),
                                '--learning-rate', str(stage['learning_rate']), '--seed', '42', '--threads', '1',
                                '--gradient-checkpointing', '--checkpoint-every', '200', '--keep-checkpoints', '1',
                                '--log-every', '10'], cwd=ROOT, env=environment, check=True)
                trained = json.loads((directory / 'training.json').read_text())
                if (trained['steps'] != stage['steps'] or trained['stage'] != stage['stage']
                        or trained['trainable'] != 'vision' or trained['data_sha256'] != plan['data_sha256']
                        or trained['image_size'] != stage['image_size'] or trained['truncated_sequences']):
                    raise ValueError('Paired alignment stage did not complete its declared budget')
                source = directory / 'model.pt'; frozen_language(source, branch)
                atomic_json(directory / 'language-preservation.json',
                            {'checkpoint_sha256': file_hash(source),
                             'language_state_sha256': preparation['branches'][branch]['language_state_sha256'],
                             'exact_language_and_critic_preserved': True})
                # The child exited successfully and complete weights are verified;
                # discard only its redundant periodic snapshots, retain final optimizer state.
                for snapshot in directory.glob('step-*.pt'):
                    snapshot.unlink()
            assess(source, branch, 'after')
        comparisons = {}
        for family, _, _, _ in specs:
            records = {(branch, position): json.loads((reports / (branch + '-' + position + '-' + family + '.json')).read_text())
                       for branch in ['original', 'masked'] for position in ['before', 'after']}
            comparisons[family] = {'after_original_vs_masked': compare_reports(records['original', 'after'], records['masked', 'after'], family),
                                   'within_branch': {branch: compare_reports(records[branch, 'before'], records[branch, 'after'], family)
                                                     for branch in ['original', 'masked']}}
        check_runtime()
        atomic_json(reports / 'comparison.json', {'results': comparisons, 'goal_achieved': False,
                    'production_promoted': False, 'exact_language_and_critic_preserved': True,
                    'limitations': plan['limitations']})
        status('paired caption alignment and native visual checks completed; broader capability unverified')
    except Exception as error:
        status('paired image experiment stopped; requires inspection', error=str(error))
        raise


if __name__ == '__main__':
    main()
