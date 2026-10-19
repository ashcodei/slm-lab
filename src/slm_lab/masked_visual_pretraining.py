"""Own-encoder masked-image reconstruction, with a training-only decoder.

An MAE-style objective adapted to the existing RMSNorm/SwiGLU vision backbone.
No external weights, teacher outputs or caption targets are used.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .character_ctc import atomic_torch, batch_pixels
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .evaluation_provenance import capture, seal, pixels_sha256
from .model import Block, ModelConfig, RMSNorm, TinyMultimodalLM, VisionEncoder

SOURCES = [Path(__file__).with_name(name + '.py') for name in
           ['masked_visual_pretraining', 'visual_pretraining_data', 'model', 'character_ctc',
            'corpus', 'diverse_images', 'evaluation_provenance']]
LOADED_SOURCES = {str(path): file_hash(path) for path in SOURCES}


def unchanged_sources():
    if {str(path): file_hash(path) for path in SOURCES} != LOADED_SOURCES:
        raise ValueError('Loaded visual-pretraining implementation changed; inspect/reload the waiting process')


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(str(tuple(value.shape)).encode()); digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def patchify(pixels, patch_size):
    if pixels.ndim != 4 or pixels.shape[1] != 3 or pixels.shape[2] != pixels.shape[3] or pixels.shape[2] % patch_size:
        raise ValueError('Expected square RGB images divisible by the patch size')
    batch, _, size, _ = pixels.shape
    side = size // patch_size
    return pixels.reshape(batch, 3, side, patch_size, side, patch_size).permute(0, 2, 4, 3, 5, 1).reshape(batch, side * side, patch_size * patch_size * 3)


def masked_loss(predictions, pixels, mask, patch_size):
    target = patchify(pixels.float(), patch_size)
    if predictions.shape != target.shape or mask.shape != target.shape[:2]:
        raise ValueError('Masked reconstruction target shapes differ')
    if not torch.all((mask == 0) | (mask == 1)) or mask.sum() <= 0:
        raise ValueError('Need a nonempty binary reconstruction mask')
    # Per-patch normalization prevents mean-color prediction from dominating.
    target = (target - target.mean(-1, keepdim=True)) / torch.sqrt(target.var(-1, unbiased=False, keepdim=True) + 1e-6)
    error = (predictions.float() - target).square().mean(-1)
    loss = (error * mask).sum() / mask.sum()
    if not torch.isfinite(loss):
        raise ValueError('Non-finite masked reconstruction loss')
    return loss


class MaskedVisualModel(nn.Module):
    def __init__(self, vision, decoder_layers=2):
        super().__init__()
        self.vision = vision
        self.vision.project.requires_grad_(False)
        config = vision.config
        width = min(256, config.vision_hidden_size)
        heads = 4 if width % 4 == 0 else 1
        count = (config.image_size // config.patch_size) ** 2
        self.decoder_embed = nn.Linear(config.vision_hidden_size, width)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.decoder_position = nn.Parameter(torch.zeros(1, count, width))
        self.decoder = nn.ModuleList(Block(width, width * 3, heads, heads, config.norm_eps) for _ in range(decoder_layers))
        self.decoder_norm = RMSNorm(width, config.norm_eps)
        self.predict = nn.Linear(width, config.patch_size ** 2 * 3)
        nn.init.normal_(self.mask_token, std=.02)
        nn.init.normal_(self.decoder_position, std=.02)

    def forward(self, pixels, noise=None, mask_ratio=.75, gradient_checkpointing=False):
        config = self.vision.config
        if not 0 < mask_ratio < 1 or pixels.shape[-2:] != (config.image_size, config.image_size):
            raise ValueError('Masked experiment requires base image resolution and a partial mask')
        x = self.vision.patch(pixels).flatten(2).transpose(1, 2) + self.vision.position
        batch, count, width = x.shape
        if noise is None:
            noise = torch.rand(batch, count, device=x.device)
        if noise.shape != (batch, count) or not torch.isfinite(noise).all():
            raise ValueError('Invalid random-mask ordering')
        shuffled = noise.argsort(1)
        restore = shuffled.argsort(1)
        keep = max(1, min(count - 1, int(count * (1 - mask_ratio))))
        visible = shuffled[:, :keep]
        x = x.gather(1, visible[:, :, None].expand(-1, -1, width))
        # Hidden patch embeddings never enter the encoder. Decoder targets do
        # not participate in constructing predictions.
        for block in self.vision.blocks:
            x = activation_checkpoint(block, x, use_reentrant=False) if gradient_checkpointing and self.training else block(x)
        x = self.decoder_embed(self.vision.norm(x))
        x = torch.cat([x, self.mask_token.expand(batch, count - keep, -1)], 1)
        x = x.gather(1, restore[:, :, None].expand(-1, -1, x.shape[-1])) + self.decoder_position
        for block in self.decoder:
            x = activation_checkpoint(block, x, use_reentrant=False) if gradient_checkpointing and self.training else block(x)
        predictions = self.predict(self.decoder_norm(x)).float()
        mask = torch.ones(batch, count, device=x.device)
        mask[:, :keep] = 0
        mask = mask.gather(1, restore)
        return {'predictions': predictions, 'mask': mask, 'visible_indices': visible,
                'loss': masked_loss(predictions, pixels, mask, config.patch_size)}


def from_base(path):
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    config = ModelConfig(**state['config'])
    vision = VisionEncoder(config)
    vision.load_state_dict({name.removeprefix('vision.'): value for name, value in state['model'].items() if name.startswith('vision.')}, strict=True)
    model = MaskedVisualModel(vision)
    added = sum(value.numel() for name, value in model.named_parameters() if not name.startswith('vision.'))
    with torch.device('meta'):
        deployed = TinyMultimodalLM(config).parameter_counts()['total']
    if deployed >= 1_000_000_000:
        raise ValueError('Deployment exceeds the user parameter limit')
    return model, {'config': asdict(config), 'base_checkpoint': str(Path(path).resolve()),
                   'base_checkpoint_sha256': file_hash(path), 'tokenizer_sha256': state['tokenizer_sha256'],
                   'training_decoder_parameters': added, 'deployed_parameters': deployed,
                   'decoder_deployed': False, 'external_weights_used': False}


def training_pixels(rows, size, device, first_position, seed):
    tensors = []
    for index, row in enumerate(rows):
        path = Path(row['image'])
        if file_hash(path) != row['stored_image_sha256']:
            raise ValueError('Source training image changed')
        rng = random.Random(seed + (first_position + index) * 1009)
        with Image.open(path) as original:
            image = original.convert('RGB')
            width, height = image.size
            for _ in range(10):
                area = width * height * rng.uniform(.2, 1.)
                ratio = math.exp(rng.uniform(math.log(.75), math.log(4 / 3)))
                crop_width, crop_height = round(math.sqrt(area * ratio)), round(math.sqrt(area / ratio))
                if 0 < crop_width <= width and 0 < crop_height <= height:
                    left, top = rng.randrange(width - crop_width + 1), rng.randrange(height - crop_height + 1)
                    image = image.crop((left, top, left + crop_width, top + crop_height))
                    break
            else:
                side = min(width, height)
                left, top = (width - side) // 2, (height - side) // 2
                image = image.crop((left, top, left + side, top + side))
            image = image.resize((size, size), Image.Resampling.BICUBIC)
            if row['visual_source'] == 'everyday_photos' and rng.random() < .5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            values = np.asarray(image, dtype=np.float32) / 127.5 - 1
        tensors.append(torch.from_numpy(values.copy()).permute(2, 0, 1))
    return torch.stack(tensors).to(device)


@torch.no_grad()
def assess(model, rows, output, device, maximum=128):
    provenance = capture(SOURCES)
    model.eval()
    before = state_digest(model)
    families = sorted({row['visual_source'] for row in rows})
    chosen = [row for family in families for row in sorted([row for row in rows if row['visual_source'] == family],
                    key=lambda row: hashlib.sha256(row['id'].encode()).digest())[:max(2, maximum // len(families))]]
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    records = []
    count = (model.vision.config.image_size // model.vision.config.patch_size) ** 2
    for row in chosen:
        pixels = batch_pixels([row], model.vision.config.image_size, device)
        seed = int(hashlib.sha256(('mae-validation-v1:' + row['id']).encode()).hexdigest()[:12], 16)
        noise = torch.rand(1, count, generator=torch.Generator().manual_seed(seed)).to(device)
        with amp():
            result = model(pixels, noise=noise)
        records.append({'id': row['id'], 'group_id': row['group_id'], 'source': row['visual_source'],
                        'masked_mse': float(result['loss']), 'mask_seed': seed,
                        'input_pixels_sha256': pixels_sha256(pixels[0]),
                        'visible_indices': result['visible_indices'][0].tolist()})
    if state_digest(model) != before:
        raise ValueError('Frozen masked visual weights changed during assessment')
    report = {'component_state_sha256': before, 'images': len(records), 'masked_mse': sum(row['masked_mse'] for row in records) / len(records),
              'by_source': {family: {'images': sum(row['source'] == family for row in records),
                              'masked_mse': sum(row['masked_mse'] for row in records if row['source'] == family) / sum(row['source'] == family for row in records)} for family in families},
              'results': records, 'goal_achieved': False,
              'scope': 'Fixed-mask full-image reconstruction diagnostic; not semantic/OCR/native language capability.',
              'limitations': ['Encoder and temporary decoder both train; lower reconstruction error does not isolate improved encoder features.',
                              'Existing development images, not independent final benchmarks.']}
    seal(report, provenance, files=SOURCES, device=device)
    atomic_json(output, report)
    return report


def train(base, data, output, reports, device='cuda:3', passes=10, batch_size=32, learning_rate=1e-4, resume=False):
    unchanged_sources()
    torch.set_num_threads(1); torch.manual_seed(42)
    if device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.25, int(device.split(':')[1]) if ':' in device else 0)
    data, output, reports = map(Path, [data, output, reports])
    output.mkdir(parents=True, exist_ok=True); reports.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data / 'manifest.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        if file_hash(data / name) != digest:
            raise ValueError('Frozen visual reconstruction data changed')
    rows = list(jsonl(data / 'train.jsonl')); valid = list(jsonl(data / 'valid.jsonl'))
    model, metadata = from_base(base)
    model = model.to(device)
    steps = math.ceil(passes * len(rows) / batch_size)
    plan = {**metadata, 'manifest_sha256': file_hash(data / 'manifest.json'), 'training_images': len(rows),
            'passes': passes, 'steps': steps, 'batch_size': batch_size, 'mask_ratio': .75, 'image_size': model.vision.config.image_size,
            'learning_rate': learning_rate, 'seed': 42, 'implementations': LOADED_SOURCES,
            'augmentation': 'Random resized crop: area 0.2..1, aspect 0.75..1.333; ordinary-photo-only horizontal flip. Validation uses full letterboxed images.',
            'trainable_parameters': sum(value.numel() for value in model.parameters() if value.requires_grad),
            'goal_achieved': False, 'scope': 'Own visual reconstruction pretraining only; no automatic encoder/policy promotion.'}
    if (reports / 'plan.json').exists() and json.loads((reports / 'plan.json').read_text()) != plan:
        raise ValueError('Visual reconstruction plan differs')
    if any(output.iterdir()) and not resume:
        raise ValueError('Existing masked pretraining checkpoints need inspected explicit resume')
    atomic_json(reports / 'plan.json', plan)
    optimizer = torch.optim.AdamW((value for value in model.parameters() if value.requires_grad), lr=learning_rate, weight_decay=.05)
    start = 0
    if resume:
        state = torch.load(output / 'latest.pt', map_location=device, weights_only=True)
        if state['plan'] != plan:
            raise ValueError('Masked pretraining resume plan differs')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); start = state['steps']
        torch.set_rng_state(state['torch_rng_state'].cpu())
        if device.startswith('cuda'): torch.cuda.set_rng_state(state['cuda_rng_state'].cpu(), device)
    else:
        assess(model, valid, reports / 'baseline.json', device)
    generator = torch.Generator().manual_seed(42)
    order = []
    for _ in range(passes): order.extend(torch.randperm(len(rows), generator=generator).tolist())
    total = steps * batch_size
    order = (order * ((total + len(order) - 1) // len(order)))[:total]
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    begin = time.monotonic()

    def save(name, completed, include_optimizer=True):
        value = {'model': {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                 'plan': plan, 'steps': completed, 'torch_rng_state': torch.get_rng_state()}
        if include_optimizer: value['optimizer'] = optimizer.state_dict()
        if device.startswith('cuda'): value['cuda_rng_state'] = torch.cuda.get_rng_state(device)
        atomic_torch(output / name, value)

    for step in range(start, steps):
        model.train(); optimizer.zero_grad(set_to_none=True); started = time.monotonic()
        batch = [rows[index] for index in order[step * batch_size:(step + 1) * batch_size]]
        pixels = training_pixels(batch, model.vision.config.image_size, device, step * batch_size, plan['seed'])
        rate = learning_rate * min(1, (step + 1) / max(1, steps // 10)) * (.1 + .9 * (1 + math.cos(math.pi * step / steps)) / 2)
        for group in optimizer.param_groups: group['lr'] = rate
        with amp(): result = model(pixels, gradient_checkpointing=True)
        result['loss'].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == start or (step + 1) % 10 == 0 or step + 1 == steps:
            status = {'status': 'training own visual masked reconstruction', 'step': step + 1, 'planned_steps': steps,
                      'loss': float(result['loss'].detach()), 'gradient_norm': float(norm), 'learning_rate': rate,
                      'step_seconds': time.monotonic() - started, 'elapsed_seconds': time.monotonic() - begin,
                      'image_exposures': (step + 1) * batch_size, 'goal_achieved': False}
            atomic_json(reports / 'status.json', status); print(json.dumps(status), flush=True)
        if (step + 1) % 100 == 0:
            unchanged_sources(); save('latest.pt', step + 1)
    save('model.pt', steps, False); save('latest.pt', steps)
    final = assess(model, valid, reports / 'final.json', device)
    unchanged_sources()
    atomic_json(reports / 'status.json', {'status': 'masked visual pretraining/reconstruction checks completed; semantic/native capability unverified',
                'steps': steps, 'validation_masked_mse': final['masked_mse'], 'goal_achieved': False, 'production_promoted': False})


def process_identity(pid):
    try:
        # The command may contain parentheses; split after the stat comm field.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in {'Z', 'X'}:
            return None
        return fields[19]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True); parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--reports', required=True)
    parser.add_argument('--device', default='cuda:3'); parser.add_argument('--passes', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wait-for-readout-pid', type=int)
    args = parser.parse_args()
    if args.passes < 1 or args.batch_size < 1: parser.error('Positive training budgets required')
    reports = Path(args.reports); reports.mkdir(parents=True, exist_ok=True)
    try:
        if args.wait_for_readout_pid is not None:
            pid = args.wait_for_readout_pid; identity = process_identity(pid)
            if identity is None:
                raise ValueError('Requested readout PID is not live; inspect before submitting the next job')
            command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ')
            if b'-m slm_lab.learned_character_ctc' not in command:
                raise ValueError('Requested preceding PID is not the expected reading experiment')
            atomic_json(reports / 'status.json', {'status': 'waiting for live reading experiment before sharing its GPU slot',
                        'preceding_pid': pid, 'preceding_start_ticks': identity, 'goal_achieved': False})
            while process_identity(pid) == identity:
                time.sleep(15)
            unchanged_sources()
        train(args.checkpoint, args.data, args.output, args.reports, args.device, args.passes, args.batch_size, resume=args.resume)
    except Exception as error:
        atomic_json(reports / 'status.json', {'status': 'requires inspection; masked visual process stopped',
                    'error': str(error), 'goal_achieved': False})
        raise


if __name__ == '__main__':
    main()
