"""Original later exercise/header inputs, with explicit exercise repetitions."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .codenet_replay_v1 import INDEX_DTYPE
from .corpus import file_hash


EXERCISE_LANGUAGES = ('Python', 'JavaScript', 'Go', 'Java', 'Rust', 'C++')
FAMILIES = tuple('publisher_exercises:' + name for name in EXERCISE_LANGUAGES) + (
    'complete_JS_parameter_headers', 'complete_Go_source_headers', 'complete_Ruby_source_headers')
KINDS = ('exercises', 'javascript', 'goruby')
ARTIFACTS = {'index.npy', 'records.jsonl', 'source-admission.json'}


def source_family(row, kind):
    if (row.get('split') != 'train' or row.get('image') or not row.get('id') or not row.get('group_id')
            or row.get('external_model_teacher_used') is not False or not isinstance(row.get('prompt'), str)
            or not isinstance(row.get('response'), str) or not row['response']):
        raise ValueError('Original complete training-only publisher instruction required')
    language = row.get('language')
    if kind == 'exercises':
        if (language not in EXERCISE_LANGUAGES or not row.get('source', '').startswith('exercism/')
                or row.get('publisher_reference') is not True or row.get('task') != 'public_code_exercise'):
            raise ValueError('Original admitted exercise reference and language required')
        return 'publisher_exercises:' + language
    if (row.get('source') != 'CodeSearchNet' or row.get('original_partition') != 'train'
            or not row.get('parent_instruction_id') or not row.get('complete_source_header')):
        raise ValueError('Complete original documented training header required')
    if kind == 'javascript' and language == 'JavaScript' and row.get('task') == 'complete_documented_code_parameters':
        name, tag = FAMILIES[6], 'javascript'
    elif kind == 'goruby' and language in {'Go', 'Ruby'} and row.get('task') == 'complete_documented_code_header':
        name, tag = 'complete_' + language + '_source_headers', language.lower()
    else:
        raise ValueError('Original source kind, language and complete header task differ')
    match = re.fullmatch(r'```' + tag + r'\n(.*)\n```', row['response'], re.S)
    if match is None or hashlib.sha256(match.group(1).encode()).hexdigest() != row.get('source_code_sha256'):
        raise ValueError('Every original source implementation byte must match')
    return name


def exercise_exposure(row, repetition):
    if type(repetition) is not int or not 0 <= repetition < 4:
        raise ValueError('Original queued four-exposure exercise index required')
    return {**row, 'id': row['id'] + '-replay-' + str(repetition),
        'original_exercise_record_id': row['id'], 'exercise_replay_index': repetition,
        'curriculum_source': 'original_exercism_tasks'}


class LaterInstructionReplay:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / 'manifest.json').read_text())
        m = self.manifest
        if m.get('format') != 'slm-complete-later-instruction-replay-v2' or m.get('family_names') != list(FAMILIES):
            raise ValueError('All ordered original later instruction families required')
        if set(m['artifact_files_sha256']) != ARTIFACTS:
            raise ValueError('Complete original instruction index/admission inventory required')
        for name, digest in m['artifact_files_sha256'].items():
            if file_hash(self.folder / name) != digest:
                raise ValueError('Later instruction replay artifacts changed')
        admission = json.loads((self.folder / 'source-admission.json').read_text())
        if admission.get('passed') is not True or admission.get('held_roles_excluded') is not True:
            raise ValueError('Completed unchanged publisher/header admission checks required')
        if [s['kind'] for s in m['sources']] != list(KINDS):
            raise ValueError('All three original source components required')
        self.sources, self.states = {}, {}
        for s in m['sources']:
            path = Path(s['path'])
            if (file_hash(path) != s['sha256'] or file_hash(s['manifest']) != s['manifest_sha256']
                    or admission['source_manifests_sha256'].get(s['kind']) != s['manifest_sha256']
                    or admission['receipts'][s['kind']].get('passed') is not True):
                raise ValueError('Exact original admitted source/manifest required')
            original = json.loads(Path(s['manifest']).read_text())
            if (original['files_sha256']['train.jsonl'] != s['sha256']
                    or original['tokenizer_sha256'] != m['tokenizer_sha256'] or original['train_examples'] != s['rows']):
                raise ValueError('Actual original train file and shared own tokenizer differ')
            self.sources[s['kind']] = s
            self.states[s['kind']] = self._state(path)
        self.index = np.load(self.folder / 'index.npy', mmap_mode='r', allow_pickle=False)
        if (self.index.dtype != INDEX_DTYPE or self.index.ndim != 1 or len(self.index) != m['unique_source_rows']
                or np.any(self.index['family'] >= len(FAMILIES)) or np.any(self.index['bytes'] == 0)
                or np.any(self.index['context_tokens'] < 3) or np.any(self.index['context_tokens'] > 4096)
                or np.any(self.index['target_tokens'] < 1) or np.any(self.index['target_tokens'] >= self.index['context_tokens'])):
            raise ValueError('Complete unique original instruction index/schema/masks required')
        self.families = list(FAMILIES)
        self.positions = {name: np.flatnonzero(self.index['family'] == i) for i, name in enumerate(FAMILIES)}
        self.kinds = {name: 'exercises' if i < 6 else 'javascript' if i == 6 else 'goruby' for i, name in enumerate(FAMILIES)}
        for kind in KINDS:
            items = self.index[np.isin(self.index['family'], [i for i, name in enumerate(FAMILIES) if self.kinds[name] == kind])]
            expected = np.concatenate((np.zeros(1, dtype=np.uint64), np.cumsum(items['bytes'].astype(np.uint64))[:-1]))
            if (len(items) != self.sources[kind]['rows'] or not np.array_equal(items['offset'], expected)
                    or int(items['bytes'].astype(np.uint64).sum()) != Path(self.sources[kind]['path']).stat().st_size):
                raise ValueError('Every original source row must appear exactly once in its contiguous index')
        for name in FAMILIES:
            if len(self.positions[name]) != m['unique_family_counts'][name] or self.size(name) != m['family_counts'][name]:
                raise ValueError('Original unique/repeated family populations differ')
        if sum(self.size(name) for name in FAMILIES) != m['complete_fit_rows']:
            raise ValueError('Four exercise exposures are explicit; never new unique sources')

    @staticmethod
    def _state(path):
        s = Path(path).stat()
        return s.st_ino, s.st_size, s.st_mtime_ns

    def size(self, name):
        return len(self.positions[name]) * (4 if self.kinds[name] == 'exercises' else 1)

    def row(self, name, position):
        if name not in self.positions:
            raise KeyError(name)
        if type(position) is not int or not 0 <= position < self.size(name):
            raise IndexError('Later instruction source position outside its explicit exposure budget')
        kind = self.kinds[name]
        path = Path(self.sources[kind]['path'])
        if self._state(path) != self.states[kind]:
            raise ValueError('Original later instruction source changed during replay')
        repetition, original_position = divmod(position, len(self.positions[name]))
        item = self.index[self.positions[name][original_position]]
        with path.open('rb') as stream:
            stream.seek(int(item['offset']))
            raw = stream.read(int(item['bytes']))
        if hashlib.sha256(raw).digest() != item['row_sha256'].tobytes():
            raise ValueError('Exact original later instruction source bytes differ')
        row = json.loads(raw)
        if source_family(row, kind) != name:
            raise ValueError('Original source role/family changed')
        if kind == 'exercises':
            row = exercise_exposure(row, repetition)
        return {**row, '_replay_input_sha256': item['input_sha256'].tobytes().hex(),
            '_replay_context_tokens': int(item['context_tokens']), '_replay_target_tokens': int(item['target_tokens'])}
