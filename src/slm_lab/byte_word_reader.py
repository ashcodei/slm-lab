"""Scratch convolutional/recurrent word-crop recognition with UTF-8 byte CTC."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .ocr_evaluation import transcription_scores
from .rendered_text import text_group

ROOT = Path(__file__).resolve().parents[2]
HEIGHT = 32
WIDTH = 512
FRAMES = 128
MAX_BYTES = 64


def labels(text):
    raw = text.encode('utf-8')
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError('Word transcript must contain1..64 UTF-8 bytes')
    return [value + 1 for value in raw]


def collapse(path):
    values = []; previous = None
    for value in path:
        value = int(value)
        if not 0 <= value <= 256: raise ValueError('Invalid blank/byte output class')
        if value and value != previous: values.append(value - 1)
        previous = value
    raw = bytes(values)
    try: return raw.decode('utf-8'), True
    except UnicodeDecodeError: return raw.decode('utf-8', errors='replace'), False


class ByteWordReader(nn.Module):
    def __init__(self):
        super().__init__()
        def block(first, second):
            return [nn.Conv2d(first, second, 3, padding=1, bias=False), nn.GroupNorm(8, second), nn.GELU()]
        self.features = nn.Sequential(
            *block(3, 32), nn.MaxPool2d(2), *block(32, 64), nn.MaxPool2d(2),
            *block(64, 128), *block(128, 128), nn.MaxPool2d((2, 1)),
            *block(128, 256), *block(256, 256), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 256, (2, 1), bias=False), nn.GroupNorm(8, 256), nn.GELU())
        self.sequence = nn.LSTM(256, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.output = nn.Linear(256, 257)

    def forward(self, pixels):
        if pixels.ndim != 4 or pixels.shape[1:] != (3, HEIGHT, WIDTH):
            raise ValueError('Word reader requires fixed32x512 RGB input without transcript-dependent scaling')
        features = self.features(pixels)
        if features.shape[2:] != (1, FRAMES): raise ValueError('Convolutional byte time-axis differs from declared budget')
        sequence = features[:, :, 0, :].transpose(1, 2)
        # Keep recurrent computation in FP32 independently of convolution AMP.
        with torch.autocast(device_type=pixels.device.type, enabled=False):
            values, _ = self.sequence(sequence.float())
            return self.output(values).float()


def loss(logits, texts):
    targets = [labels(text) for text in texts]
    if any(len(row) + sum(a == b for a, b in zip(row, row[1:])) > logits.shape[1] for row in targets):
        raise ValueError('Byte transcript needs more CTC frames; do not silently zero its loss')
    packed = torch.tensor([value for row in targets for value in row], device=logits.device, dtype=torch.long)
    lengths = torch.tensor([len(row) for row in targets], dtype=torch.long)
    inputs = torch.full((len(targets),), logits.shape[1], dtype=torch.long)
    return F.ctc_loss(logits.float().log_softmax(-1).transpose(0, 1), packed, inputs, lengths,
                      blank=0, reduction='mean', zero_infinity=False)


def pixels(row, verify=True):
    path = Path(row['image'])
    if verify and file_hash(path) != row['stored_image_sha256']: raise ValueError('Original word-crop bytes changed')
    with Image.open(path) as image:
        image = image.convert('RGB').resize((WIDTH, HEIGHT), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def prepare(output):
    output = Path(output)
    if output.exists(): raise ValueError('Preserve existing byte-word dataset; inspect before resuming')
    base = ROOT / 'data/character-sequence-v1'; extra = ROOT / 'data/expanded-scene-words-v1'
    sources = [('base_train', base / 'train.jsonl'), ('base_valid', base / 'valid.jsonl'),
               ('extra_train', extra / 'train.jsonl')]
    manifests = {str(root): json.loads((root / 'manifest.json').read_text()) for root in [base, extra]}
    source_hashes = {}
    for name, path in sources:
        digest = file_hash(path)
        if digest != manifests[str(path.parent)]['files_sha256'][path.name]:
            raise ValueError('Previously prepared word source differs from frozen metadata')
        source_hashes[str(path)] = digest
    original_valid = list(jsonl(base / 'valid.jsonl'))
    protected_groups = {row['group_id'] for row in original_valid}
    protected_words = {text_group(row['response']) for row in original_valid if row['ctc_family'] == 'rendered'}
    selected = {'train': [], 'valid': []}; exclusions = {}; identities = set()
    for name, path in sources:
        split = 'valid' if name == 'base_valid' else 'train'
        for index, row in enumerate(jsonl(path)):
            reason = None
            try: labels(row['response'])
            except ValueError: reason = 'transcript exceeds declared UTF-8 byte budget'
            if split == 'train' and row['group_id'] in protected_groups: raise ValueError('Original validation photo entered byte-word training')
            if split == 'train' and text_group(row['response']) in protected_words: reason = 'protected rendered-validation word'
            identity = (row['group_id'], row.get('original_word_id', row['id']))
            if split == 'train' and identity in identities: reason = 'repeated original word annotation'
            if reason:
                exclusions[reason] = exclusions.get(reason, 0) + 1; continue
            if row.get('task') != 'word_ocr': raise ValueError('Byte-word component requires genuine prepared word crops')
            selected[split].append([name, index])
            if split == 'train': identities.add(identity)
    output.mkdir(parents=True)
    manifest = {'source_files_sha256': source_hashes, 'sources': {name: str(path) for name, path in sources},
                'selected': selected, 'excluded': exclusions, 'input_height': HEIGHT, 'input_width': WIDTH,
                'ctc_frames': FRAMES, 'maximum_transcript_bytes': MAX_BYTES,
                'preparation_implementation_sha256': file_hash(__file__), 'goal_achieved': False,
                'limitations': ['Existing public human word crops and own rendered text; this component does not locate text in full photographs.',
                                'Changed architecture, byte alphabet, resolution and additional data/budget are not matched to the prior character experiment.',
                                'Original frozen full-model/global evaluation scope remains required; word-crop success alone is insufficient.']}
    atomic_json(output / 'manifest.json', manifest)
    return manifest


def load_rows(data):
    data = Path(data); manifest = json.loads((data / 'manifest.json').read_text())
    if manifest['preparation_implementation_sha256'] != file_hash(__file__): raise ValueError('Prepared byte-word implementation changed')
    sources = {}
    for name, source in manifest['sources'].items():
        if file_hash(source) != manifest['source_files_sha256'][source]: raise ValueError('Original byte-word source changed')
        sources[name] = list(jsonl(source))
    rows = {}
    for split, selected in manifest['selected'].items():
        rows[split] = []
        for name, index in selected:
            original = sources[name][index]
            row = {**original, 'image': str((Path(manifest['sources'][name]).parent / original['image']).resolve()),
                   'byte_family': original.get('ctc_family', 'additional_real_word')}
            rows[split].append(row)
    return rows, manifest


@torch.no_grad()
def evaluate(model, rows, output, device):
    # Preserve the exact earlier256 development question/photo identities.
    reference = json.loads((ROOT / 'reports/learned-character-sequence-v1/final-reading.json').read_text())
    by_id = {row['id']: row for row in rows}
    selected = [by_id[row['id']] for row in reference['results']]
    model.eval(); results = []
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    for index, row in enumerate(selected):
        other = next(other for other in selected[index + 1:] + selected[:index] if other['byte_family'] == row['byte_family']
                     and other['group_id'] != row['group_id'] and other['response'] != row['response'])
        proper = pixels(row)[None].to(device); wrong = pixels(other)[None].to(device)
        result = {'id': row['id'], 'group_id': row['group_id'], 'family': row['byte_family'], 'expected': row['response'],
                  'wrong_image_id': other['id'], 'stored_image_sha256': row['stored_image_sha256']}
        for name, values in [('proper', proper), ('blank', torch.zeros_like(proper)), ('wrong', wrong)]:
            with amp(): logits = model(values)
            text, valid = collapse(logits.argmax(-1)[0].tolist())
            result[name + '_generated'] = text; result[name + '_valid_utf8'] = valid
            result[name + '_scores'] = transcription_scores(row['response'], text)
        results.append(result)
    def summary(items):
        value = {'examples': len(items)}
        for name in ['proper', 'blank', 'wrong']:
            value[name + '_exact'] = sum(row[name + '_scores']['exact_match'] for row in items)
            value[name + '_character_error'] = sum(row[name + '_scores']['character_edits'] for row in items) / sum(row[name + '_scores']['reference_characters'] for row in items)
            value[name + '_valid_utf8'] = sum(row[name + '_valid_utf8'] for row in items) / len(items)
        return value
    report = {'overall': summary(results), 'by_family': {family: summary([row for row in results if row['family'] == family])
              for family in sorted({row['family'] for row in results})}, 'results': results,
              'previous_case_report_sha256': file_hash(ROOT / 'reports/learned-character-sequence-v1/final-reading.json'),
              'input_height': HEIGHT, 'input_width': WIDTH, 'goal_achieved': False,
              'scope': 'Standalone scratch byte-CTC recognition on earlier identical word-crop development cases; no native full-photo VQA or causal architecture superiority claim.'}
    atomic_json(Path(output), report)
    return report
