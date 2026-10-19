"""Durable complete programmes and exact token masks with sealed source roles."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .corpus import file_hash
from .diverse_images import atomic_json


ROLES = {'train': 0, 'validation': 1, 'test': 2}
INDEX_DTYPE = np.dtype([('offset', '<u8'), ('length', '<u2'), ('target_start', '<u2'), ('role', 'u1')])
FILES = {'input-ids.bin', 'index.npy', 'original-rows.jsonl.gz', 'tokenizer.json',
         'model-config.json', 'DATASET-LICENSE.txt', 'publisher-README.md'}


class CodePackWriter:
    def __init__(self, folder, maximum_bytes=768 * 1024**2, minimum_free_gib=26):
        self.folder = Path(folder)
        self.maximum_bytes = maximum_bytes
        self.minimum_free_bytes = int(minimum_free_gib * 1024**3)
        self.entries, self.names, self.counts = [], set(), Counter()
        self.tokens = (self.folder / 'input-ids.bin').open('xb')
        self.compressed = (self.folder / 'original-rows.jsonl.gz').open('xb')
        self.rows = gzip.GzipFile(fileobj=self.compressed, mode='wb', mtime=0)

    def guard(self):
        used = self.tokens.tell() + self.compressed.tell() + len(self.entries) * INDEX_DTYPE.itemsize
        if used > self.maximum_bytes:
            raise ValueError('Durable code-pack size budget exceeded')
        if shutil.disk_usage(self.folder).free < self.minimum_free_bytes:
            raise ValueError('Preserve persistent free-space reserve')

    def add(self, row, ids, labels):
        if row['id'] in self.names or row['split'] not in ROLES or row.get('image'):
            raise ValueError('Unique original text identity and sealed role required')
        if not 3 <= len(ids) <= 4096 or len(ids) != len(labels) or ids[0] != 1 or ids[-1] != 2:
            raise ValueError('Complete original token framing differs')
        if any(not isinstance(value, (int, np.integer)) or not 0 <= value < 32768 for value in ids):
            raise ValueError('Token outside the exact own vocabulary')
        start = next((i for i, value in enumerate(labels) if value != -100), len(ids))
        if not 1 <= start < len(ids) - 1 or labels[:start] != [-100] * start or labels[start:] != ids[start:]:
            raise ValueError('Complete contiguous response/EOS mask required')
        raw = np.asarray(ids, dtype='<u2').tobytes()
        position = len(self.entries)
        self.entries.append((self.tokens.tell() // 2, len(ids), start, ROLES[row['split']]))
        self.names.add(row['id'])
        self.tokens.write(raw)
        exported = {**row, 'token_pack_index': position,
                    'complete_input_ids_sha256': hashlib.sha256(raw).hexdigest(), 'target_tokens': len(ids) - start}
        self.rows.write((json.dumps(exported, ensure_ascii=False) + '\n').encode())
        for key, value in [('examples', 1), ('context_tokens', len(ids)), ('target_tokens', len(ids) - start)]:
            self.counts[(row['split'], key)] += value
        if len(self.entries) % 1000 == 0:
            self.guard()

    def close(self):
        self.rows.close()
        self.compressed.close()
        self.tokens.close()

    def finish(self, provenance):
        if not self.entries:
            raise ValueError('A complete code pack must contain original examples')
        self.guard()
        self.close()
        np.save(self.folder / 'index.npy', np.asarray(self.entries, dtype=INDEX_DTYPE), allow_pickle=False)
        files = sorted(FILES)
        total = sum((self.folder / name).stat().st_size for name in files)
        if total > self.maximum_bytes or shutil.disk_usage(self.folder).free < self.minimum_free_bytes:
            raise ValueError('Final durable pack storage budget differs')
        manifest = {'format': 'slm-codenet-complete-token-pack-v1',
            'source_examples': len(self.entries), 'stored_bytes': total,
            'index_dtype': INDEX_DTYPE.descr, 'input_ids_dtype': '<u2',
            'response_mask': 'All labels before target_start are -100; complete remaining original IDs including EOS are labels.',
            'roles': ROLES, 'counts': {role: {key: self.counts[(role, key)] for key in ['examples', 'context_tokens', 'target_tokens']} for role in ROLES},
            'files_sha256': {name: file_hash(self.folder / name) for name in files},
            'provenance': provenance, 'dataset_admitted_to_training': False,
            'new_model_training_started': False, 'goal_achieved': False}
        atomic_json(self.folder / 'manifest.json', manifest)
        return manifest


class CompleteCodePack:
    def __init__(self, folder, role, purpose='audit'):
        self.folder, self.role = Path(folder), role
        if role not in ROLES or purpose not in {'audit', 'training'}:
            raise ValueError('Explicit sealed role and known purpose required')
        manifest_path = self.folder / 'manifest.json'
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest['format'] != 'slm-codenet-complete-token-pack-v1' or self.manifest['roles'] != ROLES:
            raise ValueError('Code-pack format/roles differ')
        if purpose == 'training':
            if role != 'train':
                raise ValueError('Held source roles cannot be used for optimisation')
            admission_path = self.folder / 'admission.json'
            if not admission_path.exists():
                raise ValueError('Candidate pack has not been admitted to training')
            admission = json.loads(admission_path.read_text())
            if admission.get('dataset_manifest_sha256') != file_hash(manifest_path) or admission.get('admitted_to_training') is not True:
                raise ValueError('Exact data admission attestation required')
        if set(self.manifest['files_sha256']) != FILES:
            raise ValueError('Complete durable artifact inventory required')
        for name, digest in self.manifest['files_sha256'].items():
            if file_hash(self.folder / name) != digest:
                raise ValueError('Durable code-pack bytes differ')
        self.index = np.load(self.folder / 'index.npy', mmap_mode='r', allow_pickle=False)
        if self.index.dtype != INDEX_DTYPE or len(self.index) != self.manifest['source_examples']:
            raise ValueError('Code-pack index schema/population differs')
        if self.index.ndim != 1 or np.any(self.index['role'] > 2):
            raise ValueError('Invalid sealed role index')
        lengths = self.index['length'].astype(np.uint64)
        expected_offsets = np.concatenate((np.zeros(1, dtype=np.uint64), np.cumsum(lengths)[:-1]))
        if (not np.array_equal(self.index['offset'], expected_offsets)
                or np.any(lengths < 3) or np.any(lengths > 4096)
                or np.any(self.index['target_start'] < 1) or np.any(self.index['target_start'] >= lengths - 1)):
            raise ValueError('Code-pack offsets/complete response spans differ')
        if (self.folder / 'input-ids.bin').stat().st_size != int(lengths.sum()) * 2:
            raise ValueError('Code-pack token store size differs')
        self.ids = np.memmap(self.folder / 'input-ids.bin', dtype='<u2', mode='r')
        self.positions = np.flatnonzero(self.index['role'] == ROLES[role])
        if len(self.positions) != self.manifest['counts'][role]['examples']:
            raise ValueError('Sealed role population differs')

    def __len__(self):
        return len(self.positions)

    def sequence(self, position):
        if not isinstance(position, (int, np.integer)) or not 0 <= position < len(self):
            raise IndexError('Local sealed-role row index outside population')
        item = self.index[self.positions[position]]
        ids = self.ids[int(item['offset']):int(item['offset']) + int(item['length'])].astype(np.int64).tolist()
        start = int(item['target_start'])
        if ids[0] != 1 or ids[-1] != 2 or any(value >= 32768 for value in ids):
            raise ValueError('Complete packed framing/vocabulary differs')
        return ids, [-100] * start + ids[start:]


def packed_encoder(pack, tokenizer, config):
    """Use the existing batch/padding machinery without legacy text truncation."""
    from .training import Encoder

    class PackedEncoder(Encoder):
        def sequence(self, row, response_field=None):
            if response_field != 'response' or set(row) != {'packed_code_index'}:
                raise ValueError('Packed complete-code response index required')
            return pack.sequence(row['packed_code_index'])

    return PackedEncoder(tokenizer, config, pack.folder)
