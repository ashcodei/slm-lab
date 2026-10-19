"""Compose own vision research branches while preserving exact language weights."""
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import torch

from .corpus import file_hash
from .diverse_images import atomic_json
from .model import ModelConfig, TinyMultimodalLM


def language_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name.startswith('vision.'):
            continue
        digest.update(name.encode()); digest.update(str(tuple(value.shape)).encode()); digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepare(source, masked, output):
    source, masked, output = map(Path, [source, masked, output])
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'preparation.json').exists():
        raise ValueError('Existing transfer preparation requires explicit inspection; do not overwrite')
    if any(output.iterdir()):
        raise ValueError('Unrecorded transfer files require inspection')
    # Retain the exact snapshot inode while its active trainer rotates snapshots.
    with tempfile.TemporaryDirectory(prefix='.foundation-transfer-', dir=output) as temporary:
        frozen = Path(temporary) / 'source.pt'; os.link(source, frozen)
        original_sha = file_hash(frozen)
        base = torch.load(frozen, map_location='cpu', weights_only=True, mmap=True)
        donor = torch.load(masked, map_location='cpu', weights_only=True, mmap=True)
        config = ModelConfig(**base['config'])
        if donor['plan']['config'] != asdict(config) or donor['steps'] != donor['plan']['steps']:
            raise ValueError('Masked vision donor is incompatible or incomplete')
        tokenizer = source.parent / 'tokenizer.json'
        if file_hash(tokenizer) != base['tokenizer_sha256'] or donor['plan']['tokenizer_sha256'] != base['tokenizer_sha256']:
            raise ValueError('Own branch tokenizer identities differ')
        with torch.device('meta'):
            parameters = TinyMultimodalLM(config).parameter_counts()['total']
        if parameters >= 1_000_000_000:
            raise ValueError('Transfer model exceeds the user parameter limit')
        original_language = language_digest(base['model'])
        donor_vision = {name: value for name, value in donor['model'].items() if name.startswith('vision.')}
        expected = {name: value for name, value in base['model'].items() if name.startswith('vision.')}
        if set(donor_vision) != set(expected) or any(donor_vision[name].shape != value.shape or donor_vision[name].dtype != value.dtype for name, value in expected.items()):
            raise ValueError('Donor vision tensor names/shapes/precision differ')
        payload = {key: value for key, value in base.items() if key not in
                   {'optimizer', 'reference_model', 'torch_rng_state', 'cuda_rng_state'}}
        branches = {}
        for name in ['original', 'masked']:
            root = output / name; root.mkdir()
            weights = dict(base['model'])
            if name == 'masked': weights.update(donor_vision)
            value = {**payload, 'model': weights, 'foundation_source_sha256': original_sha,
                     'vision_source': 'masked reconstruction experiment' if name == 'masked' else 'original own foundation vision'}
            path = root / 'model.pt'; intermediate = path.with_suffix('.tmp')
            torch.save(value, intermediate); intermediate.replace(path)
            shutil.copy2(tokenizer, root / 'tokenizer.json')
            saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            if language_digest(saved['model']) != original_language:
                raise ValueError('Vision transfer changed language/critic weights')
            if any(not torch.equal(saved['model'][key], tensor) for key, tensor in (donor_vision if name == 'masked' else expected).items()):
                raise ValueError('Saved vision weights differ from the selected own donor')
            branches[name] = {'checkpoint': str(path.resolve()), 'checkpoint_sha256': file_hash(path),
                              'language_state_sha256': original_language}
            del saved, value, weights
        report = {'source_snapshot': str(source.resolve()), 'source_snapshot_sha256': original_sha,
                  'foundation_updates': base['steps'], 'masked_checkpoint': str(masked.resolve()),
                  'masked_checkpoint_sha256': file_hash(masked), 'branches': branches,
                  'config': asdict(config), 'parameters': parameters, 'tokenizer_sha256': base['tokenizer_sha256'],
                  'training_decoder_transferred': False, 'external_model_weights_used': False,
                  'goal_achieved': False,
                  'scope': 'Exact own-language-preserving vision branches; no native capability or policy promotion claim.'}
    atomic_json(output / 'preparation.json', report)
    return report
