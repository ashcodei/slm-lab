"""Bound existing training replay rows to complete text masks and image bytes."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .corpus import file_hash
from .training import Encoder


INDEX_DTYPE = np.dtype([('offset', '<u8'), ('bytes', '<u4'), ('family', '<u2'),
    ('context_tokens', '<u2'), ('target_tokens', '<u2'), ('row_sha256', 'V32'), ('input_sha256', 'V32')])
READING_TASKS = {'word_ocr', 'region_ocr', 'document_ocr', 'document_region_ocr'}
ARTIFACT_FILES = {'index.npy', 'image-files.jsonl', 'records.jsonl', 'review.jsonl'}


def training_role(row):
    for key in ['split', 'source_split']:
        value = row.get(key)
        if value is not None and value != 'train':
            raise ValueError('Explicit held/source role is not replay training data')


def family(row):
    training_role(row)
    if row.get('image'):
        return 'image_reading' if row.get('task') in READING_TASKS else 'generic_visual'
    source = row.get('source', '')
    if 'TheAlgorithms/Python' in source or 'MBPP' in source:
        return 'verified_programs'
    if row.get('language'):
        return 'code:' + row['language']
    if row.get('domain') == 'math_reasoning' or source.startswith('GSM8K'):
        return 'math'
    if source == 'HuggingFaceH4/no_robots':
        return 'human_instruction'
    raise ValueError('Original replay source needs an explicit family qualification')


class ReplayContextDoesNotFit(ValueError):
    def __init__(self, prompt_tokens, response_tokens, limit):
        self.prompt_tokens, self.response_tokens, self.limit = prompt_tokens, response_tokens, limit
        super().__init__('Whole original replay prompt/response exceeds the text window')


class FullReplayEncoder(Encoder):
    def sequence(self, row, response_field=None):
        if response_field != 'response' or 'tokens' in row:
            raise ValueError('Original complete supervised replay response required')
        training_role(row)
        prompt, response = self.tokens(self.prompt(row)), self.tokens(row['response'])
        if len(prompt) + len(response) + 2 > self.config.max_text_length:
            raise ReplayContextDoesNotFit(len(prompt), len(response), self.config.max_text_length)
        ids = [self.bos] + prompt + response + [self.eos]
        labels = [-100] * (len(prompt) + 1) + response + [self.eos]
        if '_replay_input_sha256' in row:
            digest = hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest()
            if (digest != row['_replay_input_sha256'] or len(ids) != row['_replay_context_tokens']
                or len(response) + 1 != row['_replay_target_tokens']):
                raise ValueError('Actual original replay text/mask differs from bound preparation')
        return ids, labels


class ReplayCorpus:
    """Read original JSON bytes by offset, retaining all source metadata/aliases."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / 'manifest.json').read_text())
        if self.manifest['format'] != 'slm-complete-joint-replay-v1':
            raise ValueError('Known complete replay format required')
        if set(self.manifest['artifact_files_sha256']) != ARTIFACT_FILES:
            raise ValueError('Complete replay artifact inventory required')
        for name, digest in self.manifest['artifact_files_sha256'].items():
            if file_hash(self.folder / name) != digest:
                raise ValueError('Bound replay artifacts differ')
        self.source = Path(self.manifest['original_training_source'])
        if file_hash(self.source) != self.manifest['original_training_source_sha256']:
            raise ValueError('Original replay training file changed')
        self.source_state = self._state(self.source)
        self.index = np.load(self.folder / 'index.npy', mmap_mode='r', allow_pickle=False)
        if self.index.dtype != INDEX_DTYPE or self.index.ndim != 1 or len(self.index) != self.manifest['complete_fit_rows']:
            raise ValueError('Replay index schema/population differs')
        self.families = self.manifest['family_names']
        if np.any(self.index['family'] >= len(self.families)):
            raise ValueError('Replay family index differs')
        if (np.any(self.index['bytes'] == 0)
            or np.any(self.index['offset'] + self.index['bytes'] > self.source.stat().st_size)
            or np.any(self.index['offset'][1:] <= self.index['offset'][:-1])
            or np.any(self.index['context_tokens'] < 2) or np.any(self.index['context_tokens'] > 4096)
            or np.any(self.index['target_tokens'] < 1) or np.any(self.index['target_tokens'] >= self.index['context_tokens'])):
            raise ValueError('Replay offsets/complete target spans differ')
        self.positions = {name: np.flatnonzero(self.index['family'] == number) for number, name in enumerate(self.families)}
        if any(len(self.positions[name]) != self.manifest['family_counts'][name] for name in self.families):
            raise ValueError('Actual replay family populations differ')
        self.images = {}
        for image in map(json.loads, (self.folder / 'image-files.jsonl').open()):
            path = Path(image['path'])
            if file_hash(path) != image['sha256']:
                raise ValueError('Original replay image bytes changed')
            self.images[str(path)] = self._state(path)

    @staticmethod
    def _state(path):
        s = path.stat()
        return s.st_ino, s.st_size, s.st_mtime_ns

    def size(self, name):
        return len(self.positions[name])

    def row(self, name, position):
        if type(position) is not int or not 0 <= position < self.size(name):
            raise IndexError('Local replay-family position outside source population')
        if self._state(self.source) != self.source_state:
            raise ValueError('Original replay source changed during use')
        item = self.index[self.positions[name][position]]
        with self.source.open('rb') as source:
            source.seek(int(item['offset']))
            raw = source.read(int(item['bytes']))
        if hashlib.sha256(raw).digest() != item['row_sha256'].tobytes():
            raise ValueError('Exact original replay row bytes differ')
        row = json.loads(raw)
        if family(row) != name:
            raise ValueError('Original replay family/role changed')
        if row.get('image'):
            path = Path(row['image'])
            path = (path if path.is_absolute() else self.source.parent / path).resolve()
            if str(path) not in self.images or self._state(path) != self.images[str(path)]:
                raise ValueError('Original replay image changed during use')
        return {**row, '_replay_input_sha256': item['input_sha256'].tobytes().hex(),
            '_replay_context_tokens': int(item['context_tokens']), '_replay_target_tokens': int(item['target_tokens'])}
