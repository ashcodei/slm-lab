"""Complete original packed-code windows, including every final causal target.

This adds a source reader and explicit next-token encoder. It does not change
the running curriculum, register a production stage, or supply instruction gold.
"""
from bisect import bisect_right
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np

from .corpus import file_hash
from .diverse_images import atomic_json
from .training import Encoder
from .codenet_replay_v1 import training_role


FAMILY = 'original_packed_code'
FORMAT = 'slm-complete-packed-pretraining-replay-v1'
OBJECTIVE = 'causal_pretraining'
INDEX_DTYPE = np.dtype([('offset', '<u8'), ('length', '<u2'), ('input_sha256', 'V32')])
ARTIFACTS = {'train-index.npy', 'valid-index.npy', 'source-admission.json'}


def window_layout(tokens, length):
    if type(tokens) is not int or tokens < 2 or type(length) is not int or not 2 <= length <= 4096:
        raise ValueError('At least two original tokens and a complete context of2..4096 required')
    full, remainder = divmod(tokens - 1, length - 1)
    return dict(tokens=tokens, windows=full + bool(remainder), target_tokens=tokens - 1,
        full_windows=full, tail_target_tokens=remainder,
        last_window_length=remainder + 1 if remainder else length,
        legacy_dropped_target_tokens=remainder)


def _state(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


class PackedStream:
    """Read exact ordered bytes across shard boundaries, with a bounded map cache."""

    def __init__(self, folder, expected_manifest_sha256, tokenizer_sha256):
        self.folder = Path(folder).resolve()
        self.path = self.folder / 'corpus.json'
        if file_hash(self.path) != expected_manifest_sha256:
            raise ValueError('Exact original packed corpus manifest required')
        self.manifest = json.loads(self.path.read_text())
        m = self.manifest
        if m.get('dtype') != '<u2' or m.get('tokenizer_sha256') != tokenizer_sha256 or set(m['splits']) != {'train', 'valid'}:
            raise ValueError('Original uint16 own-tokenizer train/valid streams required')
        self.manifest_state = _state(self.path)
        self.shards, self.ends, self.states = {}, {}, {}
        self.cache = OrderedDict()
        seen = set()
        for split in ('train', 'valid'):
            total, self.shards[split], self.ends[split] = 0, [], []
            for item in m['splits'][split]['shards']:
                name = item['path']
                if not isinstance(name, str) or Path(name).name != name or name in {'', '.', '..'}:
                    raise ValueError('Original packed shard basename required')
                path = self.folder / name
                identity = path.resolve()
                count = item['tokens']
                if (identity in seen or type(count) is not int or count < 1
                        or path.stat().st_size != count * 2 or file_hash(path) != item['sha256']):
                    raise ValueError('Every original packed shard and distinct split path must match')
                seen.add(identity)
                self.states[str(path)] = _state(path)
                self.shards[split].append(path)
                total += count
                self.ends[split].append(total)
            if total != m['splits'][split]['tokens'] or total < 2:
                raise ValueError('Complete original packed token counts required')
        if sum(m['splits'][s]['tokens'] for s in ('train', 'valid')) != m['total_tokens']:
            raise ValueError('Combined packed token total differs')

    def read(self, split, offset, length):
        if split not in self.shards:
            raise ValueError('Explicit original train/valid partition required')
        if (type(offset) is not int or type(length) is not int or length < 1 or offset < 0
                or offset + length > self.ends[split][-1]):
            raise IndexError('Packed window outside the original split')
        if _state(self.path) != self.manifest_state:
            raise ValueError('Original packed manifest changed during replay')
        pieces, remaining, cursor = [], length, offset
        while remaining:
            shard = bisect_right(self.ends[split], cursor)
            path = self.shards[split][shard]
            if _state(path) != self.states[str(path)]:
                raise ValueError('Original packed shard changed during replay')
            key = (split, shard)
            if key not in self.cache:
                self.cache[key] = np.memmap(path, dtype='<u2', mode='r')
            self.cache.move_to_end(key)
            while len(self.cache) > 4:
                self.cache.popitem(last=False)
            start = cursor - (self.ends[split][shard - 1] if shard else 0)
            amount = min(remaining, self.ends[split][shard] - cursor)
            pieces.append(np.asarray(self.cache[key][start:start + amount]).copy())
            cursor += amount
            remaining -= amount
        return np.concatenate(pieces)


def prepare_packed_replay(output, packed_root, admission, tokenizer_path, maximum_length, vocab_size):
    """Create small byte/hash indices; never copy or retokenize the source code."""
    output, tokenizer_path = Path(output), Path(tokenizer_path).resolve()
    if output.exists():
        raise ValueError('Preserve previous packed replay artifacts; no implicit replacement')
    token_sha = file_hash(tokenizer_path)
    if (admission.get('passed') is not True or admission.get('training_admitted') is not True
            or admission.get('tokenizer_sha256') != token_sha
            or type(vocab_size) is not int or not 3 <= vocab_size <= 65536):
        raise ValueError('Completed original source/token admission and uint16 vocabulary required')
    stream = PackedStream(packed_root, admission['packed_manifest_sha256'], token_sha)
    if stream.manifest['splits']['train']['tokens'] != admission['combined_train_tokens']:
        raise ValueError('Source admission and actual complete train stream differ')
    layouts = {split: window_layout(stream.ends[split][-1], maximum_length) for split in ('train', 'valid')}
    output.mkdir(parents=True)
    for split, layout in layouts.items():
        index = np.empty(layout['windows'], dtype=INDEX_DTYPE)
        for position in range(layout['windows']):
            offset = position * (maximum_length - 1)
            length = min(maximum_length, layout['tokens'] - offset)
            ids = stream.read(split, offset, length)
            if int(ids.max()) >= vocab_size:
                raise ValueError('Original packed token exceeds the exact own vocabulary')
            index[position] = (offset, length, np.void(hashlib.sha256(ids.tobytes()).digest()))
        np.save(output / (split + '-index.npy'), index, allow_pickle=False)
    atomic_json(output / 'source-admission.json', admission)
    manifest = dict(format=FORMAT, family_names=[FAMILY], objective=OBJECTIVE,
        response_field=None, maximum_length=maximum_length, vocab_size=vocab_size,
        tokenizer=str(tokenizer_path), tokenizer_sha256=token_sha,
        packed_root=str(stream.folder), packed_manifest_sha256=admission['packed_manifest_sha256'],
        component_order=[item['name'] for item in stream.manifest.get('components', [])],
        splits=layouts, family_counts={FAMILY: layouts['train']['windows']},
        artifact_files_sha256={name: file_hash(output / name) for name in sorted(ARTIFACTS)},
        original_source_bytes_copied=0, artificial_special_tokens_added=0,
        scope='All original packed bytes/partitions/order retained. One-token context overlaps target every '
            'original adjacent pair exactly once, including a final short window. Existing per-document EOS '
            'tokens are unchanged. Older components are rehearsed, not newly collected. Source admission '
            'reconstructs the new systems files and binds all older packed bytes; this is not a new '
            'cross-source deduplication or semantic-correctness certificate.',
        registered_in_production_runner=False, neural_training_started=False, goal_achieved=False)
    atomic_json(output / 'manifest.json', manifest)
    return manifest


class PackedReplayCorpus:
    families = [FAMILY]

    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / 'manifest.json').read_text())
        m = self.manifest
        if (m.get('format') != FORMAT or m.get('family_names') != [FAMILY]
                or m.get('objective') != OBJECTIVE or m.get('response_field', 'missing') is not None
                or set(m['artifact_files_sha256']) != ARTIFACTS):
            raise ValueError('Explicit complete original causal-pretraining replay required')
        for name, digest in m['artifact_files_sha256'].items():
            if file_hash(self.folder / name) != digest:
                raise ValueError('Original packed replay index/admission artifact changed')
        if file_hash(m['tokenizer']) != m['tokenizer_sha256']:
            raise ValueError('Exact own packed-code tokenizer required')
        admission = json.loads((self.folder / 'source-admission.json').read_text())
        if (admission.get('passed') is not True or admission.get('training_admitted') is not True
                or admission.get('packed_manifest_sha256') != m['packed_manifest_sha256']
                or admission.get('tokenizer_sha256') != m['tokenizer_sha256']):
            raise ValueError('Completed matching original packed source admission required')
        self.stream = PackedStream(m['packed_root'], m['packed_manifest_sha256'], m['tokenizer_sha256'])
        if (self.stream.ends['train'][-1] != admission['combined_train_tokens']
                or [item['name'] for item in self.stream.manifest.get('components', [])] != m['component_order']):
            raise ValueError('All original packed components and token populations required')
        self.index, self.index_states = {}, {}
        for split in ('train', 'valid'):
            layout = window_layout(self.stream.ends[split][-1], m['maximum_length'])
            path = self.folder / (split + '-index.npy')
            index = np.load(path, mmap_mode='r', allow_pickle=False)
            starts = np.arange(layout['windows'], dtype=np.uint64) * (m['maximum_length'] - 1)
            lengths = np.minimum(m['maximum_length'], layout['tokens'] - starts)
            if (layout != m['splits'][split] or index.ndim != 1 or index.dtype != INDEX_DTYPE
                    or len(index) != layout['windows'] or not np.array_equal(index['offset'], starts)
                    or not np.array_equal(index['length'], lengths)):
                raise ValueError('Every causal window/tail must cover the original split exactly once')
            self.index[split], self.index_states[split] = index, _state(path)
        if m['family_counts'] != {FAMILY: len(self.index['train'])}:
            raise ValueError('Complete original packed training-family population required')

    def size(self, name, *, split='train'):
        if name != FAMILY or split not in self.index:
            raise KeyError('Unknown original packed source family/partition')
        return len(self.index[split])

    def row(self, name, position, *, split='train'):
        count = self.size(name, split=split)
        if type(position) is not int or not 0 <= position < count:
            raise IndexError('Original packed window position outside its split')
        if _state(self.folder / (split + '-index.npy')) != self.index_states[split]:
            raise ValueError('Original packed replay index changed during use')
        item = self.index[split][position]
        ids = self.stream.read(split, int(item['offset']), int(item['length']))
        digest = hashlib.sha256(ids.tobytes()).digest()
        if digest != item['input_sha256'].tobytes() or int(ids.max()) >= self.manifest['vocab_size']:
            raise ValueError('Exact original packed window bytes/vocabulary differ')
        return dict(tokens=ids.astype(np.int64).tolist(), split=split, source_split=split,
            source=self.manifest['packed_root'], packed_offset=int(item['offset']),
            _replay_objective=OBJECTIVE, _replay_input_sha256=digest.hex(),
            _replay_context_tokens=len(ids), _replay_target_tokens=len(ids) - 1)


class PackedPretrainingEncoder(Encoder):
    def sequence(self, row, response_field=None):
        training_role(row)
        ids = row.get('tokens')
        if (response_field is not None or row.get('_replay_objective') != OBJECTIVE
                or row.get('split') != 'train' or any(key in row for key in ('prompt', 'response', 'text', 'image'))
                or not isinstance(ids, list) or not 2 <= len(ids) <= self.config.max_text_length
                or any(type(token) is not int or not 0 <= token < min(65536, self.config.vocab_size) for token in ids)):
            raise ValueError('Complete training-only packed next-token objective required; no instruction response')
        digest = hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest()
        if (digest != row.get('_replay_input_sha256') or len(ids) != row.get('_replay_context_tokens')
                or len(ids) - 1 != row.get('_replay_target_tokens')):
            raise ValueError('Original packed token input/causal target mask differs')
        return ids.copy(), ids.copy()
