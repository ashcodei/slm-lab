"""Frozen own-encoder object-crop diagnostics; fitted heads are never deployed."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from .corpus import file_hash
from .diverse_images import atomic_json, jsonl, pixel_hash
from .evaluation_provenance import capture, seal, pixels_sha256
from .model import ModelConfig, TinyMultimodalLM, VisionEncoder
from .visual_character_probe import features, fit_ridge, scores, state_digest

SOURCE_FILES = [Path(__file__).with_name(name + '.py') for name in
                ['object_feature_probe', 'visual_character_probe', 'model', 'corpus',
                 'diverse_images', 'evaluation_provenance']]


def crop_image(row):
    path = Path(row['image'])
    if file_hash(path) != row['stored_image_sha256']:
        raise ValueError('Object source image changed')
    with Image.open(path) as original:
        image = original.convert('RGB')
    box = row['normalized_box']
    if len(box) != 4 or any(not math.isfinite(value) or not 0 <= value <= 1 for value in box):
        raise ValueError('Invalid normalized object box')
    left, top, right, bottom = box
    if left >= right or top >= bottom:
        raise ValueError('Empty object box')
    width, height = image.size
    # Coordinates already follow the stored photo's recorded source rotation.
    return image.crop((math.floor(left * width), math.floor(top * height),
                       math.ceil(right * width), math.ceil(bottom * height)))


def prepare(source, output, image_size=224, minimum_training=20, train_cap=60, valid_cap=10):
    source, output = map(Path, [source, output])
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / 'manifest.json').read_text())
    for name in ['train.jsonl', 'valid.jsonl']:
        if file_hash(source / name) != manifest['files_sha256'][name]:
            raise ValueError('Frozen human-box source changed')
    configuration = {'version': 1, 'source_manifest_sha256': file_hash(source / 'manifest.json'),
                     'source_root': str(source.resolve()), 'image_size': image_size,
                     'minimum_training': minimum_training, 'training_cap_per_class': train_cap,
                     'validation_cap_per_class': valid_cap, 'implementation_sha256': file_hash(__file__)}
    if (output / 'manifest.json').exists():
        existing = json.loads((output / 'manifest.json').read_text())
        if existing['configuration'] != configuration:
            raise ValueError('Use a new object feature diagnostic directory')
        for name, digest in existing['files_sha256'].items():
            if file_hash(output / name) != digest:
                raise ValueError('Frozen object feature cases changed')
        return existing
    populations = {split: [row for row in jsonl(source / (split + '.jsonl')) if row['task'] == 'object_region']
                   for split in ['train', 'valid']}
    train_groups = defaultdict(set)
    for row in populations['train']:
        train_groups[row['response']].add(row['group_id'])
    classes = sorted(label for label, groups in train_groups.items() if len(groups) >= minimum_training)
    if len(classes) < 2:
        raise ValueError('Need multiple training-supported object classes')
    rows_by_split = {}
    for split, cap in [('train', train_cap), ('valid', valid_cap)]:
        selected = []
        for label in classes:
            seen = set()
            candidates = sorted([row for row in populations[split] if row['response'] == label],
                                key=lambda row: hashlib.sha256(row['id'].encode()).digest())
            for row in candidates:
                if row['group_id'] in seen:
                    continue
                seen.add(row['group_id'])
                cropped = crop_image(row)
                hashes = {str(size): pixel_hash(cropped, size) for size in [image_size, image_size * 2]}
                selected.append({**row, 'crop_input_sha256': hashes, 'crop_dimensions': list(cropped.size)})
                if len(seen) >= cap:
                    break
        rows_by_split[split] = selected
        print('selected object crops', split, len(selected), flush=True)
    labels_by_pixels = defaultdict(set)
    for rows in rows_by_split.values():
        for row in rows:
            labels_by_pixels[row['crop_input_sha256'][str(image_size)]].add(row['response'])
    ambiguous = {key for key, labels in labels_by_pixels.items() if len(labels) > 1}
    protected_groups = {row['group_id'] for row in rows_by_split['valid']}
    protected_pixels = {size: {row['crop_input_sha256'][str(size)] for row in rows_by_split['valid']}
                        for size in [image_size, image_size * 2]}
    exclusions = []
    accepted = {}
    for split in ['valid', 'train']:
        rows = []; seen = set()
        for row in rows_by_split[split]:
            key = row['crop_input_sha256'][str(image_size)]
            reason = None
            if key in ambiguous:
                reason = 'identical object crop has conflicting class labels'
            elif split == 'train' and row['group_id'] in protected_groups:
                reason = 'protected validation photo'
            elif split == 'train' and any(row['crop_input_sha256'][str(size)] in protected_pixels[size] for size in protected_pixels):
                reason = 'exact held-out crop pixels'
            elif key in seen:
                reason = 'duplicate crop within split'
            if reason:
                exclusions.append({'id': row['id'], 'split': split, 'reason': reason})
                continue
            rows.append(row); seen.add(key)
        accepted[split] = rows
    counts = Counter(row['response'] for row in accepted['train'])
    classes = [label for label in classes if counts[label] >= minimum_training]
    for split in accepted:
        removed = [row for row in accepted[split] if row['response'] not in classes]
        exclusions.extend({'id': row['id'], 'split': split, 'reason': 'insufficient training support after crop audit'} for row in removed)
        accepted[split] = [{**row, 'label': classes.index(row['response'])} for row in accepted[split] if row['response'] in classes]
    if len(classes) < 2 or len({row['response'] for row in accepted['valid']}) < 2:
        raise ValueError('Need multiple supported validation classes for wrong-image controls')
    if {row['group_id'] for row in accepted['train']} & {row['group_id'] for row in accepted['valid']}:
        raise ValueError('Photo groups overlap after object crop protection')
    for split, rows in accepted.items():
        (output / (split + '.jsonl')).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    (output / 'excluded.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in exclusions))
    result = {'configuration': configuration, 'classes': classes,
              'splits': {split: {'examples': len(rows), 'photo_groups': len({row['group_id'] for row in rows}),
                                'by_class': dict(Counter(row['response'] for row in rows))} for split, rows in accepted.items()},
              'source_region_cases': {split: len(rows) for split, rows in populations.items()},
              'parent_group_overlap': 0, 'exact_crop_overlap': 0, 'excluded': dict(Counter(row['reason'] for row in exclusions)),
              'files_sha256': {name: file_hash(output / name) for name in ['train.jsonl', 'valid.jsonl', 'excluded.jsonl']},
              'limitations': ['Classes are selected by training support; rare/unknown categories are outside this diagnostic.',
                              'Oracle object boxes are supplied: this is cropped classification, not localization or native VQA.',
                              'Existing development photos; near-duplicates and independent final generalization remain unaudited.',
                              'Probe labels are used to fit a diagnostic head only; no encoder/policy updates or deployed head.']}
    atomic_json(output / 'manifest.json', result)
    return result


def inputs(rows, size):
    pixels = []; geometry = []; colors = []; hashes = []
    for row in rows:
        cropped = crop_image(row)
        image = ImageOps.pad(cropped, (size, size), method=Image.Resampling.BICUBIC, color=(128, 128, 128))
        if hashlib.sha256(image.tobytes()).hexdigest() != row['crop_input_sha256'][str(size)]:
            raise ValueError('Object crop transform differs from the frozen cases')
        values = np.asarray(image, dtype=np.float32) / 127.5 - 1
        tensor = torch.from_numpy(values.copy()).permute(2, 0, 1)
        pixels.append(tensor); hashes.append(pixels_sha256(tensor))
        left, top, right, bottom = row['normalized_box']
        width, height = right - left, bottom - top
        geometry.append([width, height, width * height, width / height, left, top,
                         (left + right) / 2, (top + bottom) / 2])
        channels = values.reshape(-1, 3)
        colors.append(np.concatenate([channels.mean(0), channels.std(0)]).tolist())
    return torch.stack(pixels), torch.tensor(geometry), torch.tensor(colors), hashes


def summary(predictions, targets, classes):
    matched = predictions == targets
    supported = sorted(set(targets.tolist()))
    return {'examples': len(targets), 'accuracy': float(matched.double().mean()),
            'macro_accuracy': sum(float(matched[targets == label].double().mean()) for label in supported) / len(supported),
            'supported_classes': len(supported),
            'per_class': {classes[label]: {'examples': int((targets == label).sum()),
                                          'accuracy': float(matched[targets == label].double().mean())} for label in supported}}


def load_vision(path, config=None):
    torch.manual_seed(42)
    if path is None:
        if config is None:
            raise ValueError('Random control requires an already verified matched architecture')
        model = VisionEncoder(config); model.apply(TinyMultimodalLM._initialize)
        nn_position = model.position
        torch.nn.init.normal_(nn_position, std=.02)
    else:
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        current = ModelConfig(**(state['config'] if 'config' in state else state['plan']['config']))
        if config is not None and config != current:
            raise ValueError('Object probe architectures differ')
        config = current; model = VisionEncoder(config)
        model.load_state_dict({name.removeprefix('vision.'): value for name, value in state['model'].items() if name.startswith('vision.')}, strict=True)
    return model.eval().requires_grad_(False), config


@torch.no_grad()
def evaluate(checkpoints, data, output, device='cuda:2', random_control=True):
    torch.set_num_threads(1)
    if device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.15, int(device.split(':')[1]) if ':' in device else 0)
    data, output = map(Path, [data, output]); output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists():
        raise ValueError('Use a fresh object feature report directory')
    provenance = capture(SOURCE_FILES)
    manifest_digest = file_hash(data / 'manifest.json'); manifest = json.loads((data / 'manifest.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        if file_hash(data / name) != digest:
            raise ValueError('Frozen object feature data changed')
    train = list(jsonl(data / 'train.jsonl')); valid = list(jsonl(data / 'valid.jsonl'))
    classes = manifest['classes']; size = manifest['configuration']['image_size']
    train_pixels, train_geometry, train_colors, _ = inputs(train, size)
    valid_pixels, valid_geometry, valid_colors, pixel_hashes = inputs(valid, size)
    labels = torch.tensor([row['label'] for row in train]); targets = torch.tensor([row['label'] for row in valid])
    baselines = {}
    for name, first, second in [('geometry', train_geometry, valid_geometry),
                                ('geometry_and_color', torch.cat([train_geometry, train_colors], 1), torch.cat([valid_geometry, valid_colors], 1))]:
        head = fit_ridge(first, labels, len(classes), 1.)
        baselines[name] = summary(scores(head, second).argmax(1), targets, classes)
    majority = int(torch.bincount(labels).argmax())
    baselines['training_majority'] = summary(torch.full_like(targets, majority), targets, classes)
    entries = list(checkpoints)
    if random_control: entries.append(('random', None))
    config = None; results = []
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    wrong_indices = torch.tensor([next(j for j in list(range(i + 1, len(valid))) + list(range(i))
                                      if targets[j] != targets[i] and valid[j]['group_id'] != valid[i]['group_id'])
                                  for i in range(len(valid))])
    for name, path in entries:
        checkpoint_digest = file_hash(path) if path else None
        model, config = load_vision(path, config); config.image_token_count(size)
        model = model.to(device); before = state_digest(model)
        extracted = []
        for identity, pixels in [('train', train_pixels), ('valid', valid_pixels)]:
            values = []
            for start in range(0, len(pixels), 16):
                with amp(): values.append(features(model, pixels[start:start + 16].to(device)))
            extracted.append(torch.cat(values))
            print('extracted frozen object crops', name, identity, len(pixels), flush=True)
        with amp(): blank = features(model, torch.zeros(1, 3, size, size, device=device))
        if state_digest(model) != before:
            raise ValueError('Frozen encoder weights changed during object feature assessment')
        if path and file_hash(path) != checkpoint_digest:
            raise ValueError('Object feature checkpoint changed during assessment; use a frozen checkpoint')
        del model
        if device.startswith('cuda'): torch.cuda.empty_cache()
        head = fit_ridge(extracted[0], labels, len(classes), 1.)
        predictions = scores(head, extracted[1]).argmax(1)
        blank_predictions = scores(head, blank).argmax(1).repeat(len(valid))
        wrong_predictions = scores(head, extracted[1][wrong_indices]).argmax(1)
        report = {'name': name, 'checkpoint': str(path) if path else None, 'checkpoint_sha256': checkpoint_digest,
                  'vision_state_sha256': before, 'train': summary(scores(head, extracted[0]).argmax(1), labels, classes),
                  'validation': summary(predictions, targets, classes), 'blank': summary(blank_predictions, targets, classes),
                  'wrong': summary(wrong_predictions, targets, classes),
                  'cases': [{'id': row['id'], 'group_id': row['group_id'], 'expected': row['response'],
                             'predicted': classes[int(predictions[i])], 'blank_predicted': classes[int(blank_predictions[i])],
                             'wrong_predicted': classes[int(wrong_predictions[i])], 'wrong_image_id': valid[int(wrong_indices[i])]['id'],
                             'input_pixels_sha256': pixel_hashes[i], 'wrong_pixels_sha256': pixel_hashes[int(wrong_indices[i])]}
                            for i, row in enumerate(valid)]}
        torch.save(head, output / (name + '-diagnostic-head.pt')); atomic_json(output / (name + '.json'), report)
        results.append({key: value for key, value in report.items() if key != 'cases'})
        print('object feature diagnostic', name, report['validation']['accuracy'], flush=True)
    result = {'manifest_sha256': manifest_digest, 'config': asdict(config), 'classes': classes,
              'training_examples': len(train), 'validation_examples': len(valid), 'uniform_chance': 1 / len(classes),
              'regularization': 1., 'feature_protocol': 'Frozen own pre-projection pooled features, adaptive 2x2 grid; CPU float64 ridge with training-only statistics.',
              'baselines': baselines, 'encoders': results, 'goal_achieved': False, 'production_updated': False,
              'limitations': manifest['limitations'] + ['Different classes share visual properties and annotations can be ambiguous.',
                 'A fitted diagnostic head can exploit background/appearance correlations; geometry/color/random controls bound interpretation.',
                 'This does not establish captioning, OCR, native language answers or high-end-model parity.']}
    for name, digest in manifest['files_sha256'].items():
        if file_hash(data / name) != digest: raise ValueError('Object feature cases changed during assessment')
    seal(result, provenance, files=SOURCE_FILES, data=data / 'manifest.json', digest=manifest_digest, device=device)
    atomic_json(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--source', default='data/object-qa-v1'); parser.add_argument('--data', required=True)
    parser.add_argument('--output'); parser.add_argument('--checkpoint', action='append', default=[])
    parser.add_argument('--device', default='cuda:2'); parser.add_argument('--no-random-control', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(args.source, args.data), indent=2))
    else:
        if not args.output or not args.checkpoint: parser.error('Evaluation needs named checkpoints and output')
        entries = [(name, Path(path)) for name, path in (value.split('=', 1) for value in args.checkpoint)]
        evaluate(entries, args.data, args.output, args.device, not args.no_random_control)


if __name__ == '__main__':
    main()
