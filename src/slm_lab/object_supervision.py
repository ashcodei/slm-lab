"""Own image encoder: paired frozen-head versus supervised encoder classification."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .character_ctc import atomic_torch
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .evaluation_provenance import capture
from .model import ModelConfig, VisionEncoder
from .masked_visual_pretraining import state_digest

ROOT = Path(__file__).resolve().parents[2]


class ObjectClassifier(nn.Module):
    def __init__(self, vision, classes):
        super().__init__(); self.vision = vision
        self.classify = nn.Linear(vision.config.vision_hidden_size, classes)

    def features(self, pixels, gradients=False):
        vision = self.vision
        if pixels.shape[-2:] != (vision.config.image_size, vision.config.image_size):
            raise ValueError('Classification requires the recorded base resolution')
        x = vision.patch(pixels).flatten(2).transpose(1, 2) + vision.position
        for block in vision.blocks:
            x = checkpoint(block, x, use_reentrant=False) if gradients and self.training else block(x)
        return vision.norm(x).mean(1)

    def forward(self, pixels, train_vision=False):
        if train_vision:
            features = self.features(pixels, True)
        else:
            with torch.no_grad(): features = self.features(pixels)
        return self.classify(features)


def pixels(row, size, augment=None):
    with Image.open(row['image']) as image:
        image = ImageOps.pad(image.convert('RGB'), (size, size), Image.Resampling.BICUBIC, color=(128, 128, 128))
        if augment is not None:
            if augment.random() < .5: image = ImageOps.mirror(image)
            image = ImageEnhance.Brightness(image).enhance(augment.uniform(.85, 1.15))
            image = ImageEnhance.Color(image).enhance(augment.uniform(.85, 1.15))
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


@torch.no_grad()
def evaluate(model, rows, classes, device):
    model.eval(); results = []; amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    with amp:
        for index, row in enumerate(rows):
            image = pixels(row, model.vision.config.image_size)[None].to(device)
            mismatch = next(rows[(index + offset) % len(rows)] for offset in range(1, len(rows))
                            if rows[(index + offset) % len(rows)]['label'] != row['label']
                            and rows[(index + offset) % len(rows)]['group_id'] != row['group_id'])
            logits = model(image); prediction = int(logits.argmax(-1)); label = int(row['label'])
            blank_logits = model(torch.full_like(image, 128 / 127.5 - 1))
            wrong_logits = model(pixels(mismatch, model.vision.config.image_size)[None].to(device))
            results.append({'id': row['id'], 'group_id': row['group_id'], 'target_index': label,
                            'target_label': classes[label], 'prediction': prediction, 'correct': prediction == label,
                            'negative_log_likelihood': float(F.cross_entropy(logits.float(), torch.tensor([label], device=device))),
                            'blank_target_match': int(blank_logits.argmax(-1)) == label,
                            'wrong_image_target_match': int(wrong_logits.argmax(-1)) == label,
                            'wrong_image_id': mismatch['id'],
                            'model_pixel_sha256': row['model_pixel_sha256']})
    available = sorted({r['target_index'] for r in results})
    macro = sum(sum(r['correct'] for r in results if r['target_index'] == label) /
                sum(r['target_index'] == label for r in results) for label in available) / len(available)
    return {'examples': len(results), 'parent_images': len({r['group_id'] for r in results}),
            'class_choices': len(classes), 'chance_accuracy': 1 / len(classes),
            'accuracy': sum(r['correct'] for r in results) / len(results), 'macro_class_accuracy': macro,
            'blank_original_target_accuracy': sum(r['blank_target_match'] for r in results) / len(results),
            'wrong_image_original_target_accuracy': sum(r['wrong_image_target_match'] for r in results) / len(results),
            'nll': sum(r['negative_log_likelihood'] for r in results) / len(results), 'results': results}


def train_branch(vision_values, config, classes, rows, validation, label, epochs, batch, output, reports, device, source_sha, data_sha):
    torch.manual_seed(42); rng = random.Random(42)
    vision = VisionEncoder(config); vision.load_state_dict(vision_values, strict=True)
    model = ObjectClassifier(vision, len(classes)).to(device)
    train_vision = label == 'encoder'
    for parameter in model.vision.parameters(): parameter.requires_grad_(train_vision)
    for parameter in model.vision.project.parameters(): parameter.requires_grad_(False)
    initial_digest = state_digest(model.vision)
    frozen_project = model.vision.project.weight.detach().cpu().clone()
    before = evaluate(model, validation, classes, device)
    output.mkdir(); optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    steps = math.ceil(len(rows) * epochs / batch); buckets = defaultdict(list)
    for row in rows: buckets[row['label']].append(row)
    if set(buckets) != set(range(len(classes))): raise ValueError('Training class missing')
    started = time.monotonic(); metrics = []
    torch.cuda.reset_peak_memory_stats(device) if device.startswith('cuda') else None
    for step in range(steps):
        model.train(); optimizer.zero_grad(set_to_none=True)
        chosen = [rng.choice(buckets[rng.randrange(len(classes))]) for _ in range(batch)]
        images = torch.stack([pixels(row, config.image_size, rng) for row in chosen]).to(device)
        targets = torch.tensor([row['label'] for row in chosen], device=device)
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
        with amp: logits = model(images, train_vision); loss = F.cross_entropy(logits.float(), targets)
        if not torch.isfinite(loss): raise ValueError('Nonfinite supervised-object loss')
        loss.backward(); norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm): raise ValueError('Nonfinite supervised-object gradients')
        fraction = (step + 1) / steps
        lr = 1e-4 * min(1., (step + 1) / max(1, steps // 10)) * (.1 + .9 * .5 * (1 + math.cos(math.pi * fraction)))
        for group in optimizer.param_groups: group['lr'] = lr
        optimizer.step()
        if (step + 1) % 50 == 0 or step == 0:
            metric = {'branch': label, 'step': step + 1, 'planned_steps': steps, 'loss': float(loss), 'learning_rate': lr}
            metrics.append(metric); print(json.dumps(metric), flush=True)
            atomic_json(reports / 'status.json', {'status': 'training supervised object labels', **metric,
                        'updated_at': datetime.now(timezone.utc).isoformat(), 'goal_achieved': False})
    elapsed = time.monotonic() - started
    after = evaluate(model, validation, classes, device)
    if not torch.equal(model.vision.project.weight.detach().cpu(), frozen_project): raise ValueError('Unused language projector changed')
    if not train_vision and state_digest(model.vision) != initial_digest: raise ValueError('Frozen-control encoder changed')
    if train_vision and state_digest(model.vision) == initial_digest: raise ValueError('Supervised encoder did not change')
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()): raise ValueError('Nonfinite trained object weights')
    state = {'config': asdict(config), 'vision': {name: value.detach().cpu() for name, value in model.vision.state_dict().items()},
             'classifier': {name: value.detach().cpu() for name, value in model.classify.state_dict().items()}, 'classes': classes,
             'steps': steps, 'planned_steps': steps, 'stage': 'supervised_object_classification', 'branch': label,
             'base_checkpoint_sha256': source_sha, 'data_sha256': data_sha, 'temporary_classifier_head': True,
             'optimizer_history_saved': False, 'goal_achieved': False}
    atomic_torch(output / 'model.pt', state)
    atomic_json(reports / (label + '.json'), {'before': before, 'after': after, 'steps': steps,
                'sample_exposures': steps * batch, 'sampling': 'Seed42 uniform class then uniform example, with replacement; paired augmentation stream.',
                'training_elapsed_seconds': elapsed, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None,
                'unchanged_unused_projector': True, 'vision_weights_changed': train_vision,
                'checkpoint_sha256': file_hash(output / 'model.pt'), 'goal_achieved': False,
                'scope': 'Own oracle-crop classifier only; source language/critic tensors never used or updated, and no automatic encoder transfer.'})
    del model, vision, optimizer
    if device.startswith('cuda'): torch.cuda.empty_cache()
    return after


def group_bootstrap(first, second):
    if [(r['id'], r['group_id'], r['target_index'], r['model_pixel_sha256']) for r in first['results']] != [
            (r['id'], r['group_id'], r['target_index'], r['model_pixel_sha256']) for r in second['results']]:
        raise ValueError('Object classification held cases changed')
    groups = defaultdict(list)
    for old, new in zip(first['results'], second['results']): groups[old['group_id']].append(int(new['correct']) - int(old['correct']))
    keys = sorted(groups); rng = random.Random(42); samples = []
    for _ in range(2000):
        values = [v for key in rng.choices(keys, k=len(keys)) for v in groups[key]]
        samples.append(sum(values) / len(values))
    samples.sort()
    return {'encoder_minus_frozen_accuracy': second['accuracy'] - first['accuracy'],
            'paired_parent_bootstrap_95_interval': [samples[49], samples[1949]], 'bootstrap_parent_images': len(keys)}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--data', required=True); parser.add_argument('--base', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--reports', required=True); parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=32); parser.add_argument('--device', default='cuda:2'); args = parser.parse_args()
    torch.set_num_threads(1)
    data, output, reports = map(Path, [args.data, args.output, args.reports]); reports.mkdir(exist_ok=True)
    if output.exists() or (reports / 'plan.json').exists(): raise ValueError('Use a fresh supervised-object experiment; inspect interrupted jobs before resuming')
    if args.epochs < 1 or args.batch_size < 1: raise ValueError('Positive object-learning budgets required')
    files = [ROOT / name for name in capture()['files_sha256']] + [Path(__file__), ROOT / 'src/slm_lab/object_supervision_data.py',
                                                                 ROOT / 'src/slm_lab/character_ctc.py', ROOT / 'src/slm_lab/masked_visual_pretraining.py']
    provenance = capture(files); manifest = json.loads((data / 'manifest.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        if file_hash(data / name) != digest: raise ValueError('Supervised object data changed')
    rows, validation = [list(jsonl(data / (split + '.jsonl'))) for split in ['train', 'valid']]
    for row in rows + validation:
        if file_hash(row['image']) != row['stored_image_sha256']: raise ValueError('Supervised object crop bytes changed')
    source_sha = file_hash(args.base); base = torch.load(args.base, map_location='cpu', weights_only=True, mmap=True)
    config = ModelConfig(**base['config'])
    vision_values = {name.removeprefix('vision.'): value for name, value in base['model'].items() if name.startswith('vision.')}
    output.mkdir(); classes = manifest['classes']; data_sha = file_hash(data / 'manifest.json')
    atomic_json(reports / 'plan.json', {'base_checkpoint_sha256': source_sha, 'base_checkpoint': str(Path(args.base).resolve()),
                'data_manifest_sha256': data_sha, 'training_crops': len(rows), 'held_crops': len(validation), 'classes': classes,
                'epochs_exposure_budget': args.epochs, 'batch': args.batch_size, 'seed': 42, 'learning_rate': 1e-4,
                'paired_branches': ['frozen', 'encoder'], 'loaded_source_runtime': provenance, 'goal_achieved': False,
                'scope': 'Train-only own label head; oracle object crops; no Fashion-MNIST, pretrained weights, teachers or automatic deployment.'})
    frozen = train_branch(vision_values, config, classes, rows, validation, 'frozen', args.epochs, args.batch_size,
                          output / 'frozen', reports, args.device, source_sha, data_sha)
    if capture(files) != provenance: raise ValueError('Loaded supervised-object implementation/runtime changed')
    encoder = train_branch(vision_values, config, classes, rows, validation, 'encoder', args.epochs, args.batch_size,
                           output / 'encoder', reports, args.device, source_sha, data_sha)
    if capture(files) != provenance or file_hash(args.base) != source_sha: raise ValueError('Supervised-object source/base changed')
    if file_hash(data / 'manifest.json') != data_sha: raise ValueError('Supervised object manifest changed')
    for name, digest in manifest['files_sha256'].items():
        if file_hash(data / name) != digest: raise ValueError('Supervised object rows changed during paired training')
    for row in rows + validation:
        if file_hash(row['image']) != row['stored_image_sha256']: raise ValueError('Object crop bytes changed during paired training')
    comparison = {**group_bootstrap(frozen, encoder), 'frozen': {k: v for k, v in frozen.items() if k != 'results'},
                  'encoder': {k: v for k, v in encoder.items() if k != 'results'}, 'goal_achieved': False,
                  'production_promoted': False, 'limitations': manifest['limitations']}
    atomic_json(reports / 'comparison.json', comparison)
    atomic_json(reports / 'status.json', {'status': 'paired supervised-object classification completed; native multimodal capability unverified',
                'goal_achieved': False, 'production_promoted': False})
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == '__main__': main()
