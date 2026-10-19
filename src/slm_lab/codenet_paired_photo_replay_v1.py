"""Original finalized human photo questions; no held rows or new image copies."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .codenet_replay_v1 import INDEX_DTYPE
from .corpus import file_hash
from .diverse_images import atomic_json
from .vqa_pair_evaluation import validate_pairs


FAMILY = 'paired_human_photo_questions'
FORMAT = 'slm-complete-paired-human-photo-replay-v1'
ARTIFACTS = {'index.npy', 'records.jsonl', 'image-files.jsonl', 'source-admission.json'}
ORIGINAL_FILES = {'train.jsonl', 'valid.jsonl', 'pairs.train.jsonl', 'pairs.valid.jsonl', 'images.collected.jsonl'}


def training_row(original, data):
    if (original.get('split') not in {None, 'train'} or original.get('source_split') not in {None, 'train'}
            or original.get('source_subset') != 'train2014' or original.get('task') != 'visual_qa'
            or original.get('domain') != 'everyday_photos' or original.get('external_model_teacher_used') is not False
            or not isinstance(original.get('prompt'), str) or not original['prompt']
            or not isinstance(original.get('response'), str) or not original['response']
            or not isinstance(original.get('pair_ids'), list) or not original['pair_ids']
            or not original.get('id', '').startswith('vqa2:train2014:question:')
            or not original.get('group_id', '').startswith('flickr:')
            or type(original.get('exact_human_agreement')) is not int or original['exact_human_agreement'] < 6):
        raise ValueError('Original complete training-only paired human photo question required')
    data = Path(data).resolve()
    path = (data / original['image']).resolve(strict=True)
    if not path.is_relative_to(data) or not path.is_file():
        raise ValueError('Original paired training photo must remain in its finalized source directory')
    for key in ('stored_image_sha256', 'model_pixel_sha256_448'):
        digest = original.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Original stored photo and448px model-input identities required')
    return {**original, 'image': str(path), 'split': 'train'}


def prepare_paired_replay(output, data, admitted_rows, receipt, encoder, tokenizer_sha256):
    output, data = Path(output), Path(data).resolve()
    if output.exists():
        raise ValueError('Retain previous paired replay artifacts; no implicit replacement')
    path = data / 'manifest.json'
    original = json.loads(path.read_text())
    _receipt(receipt, file_hash(path))
    if (set(original['files_sha256']) != ORIGINAL_FILES or original['tokenizer_sha256'] != tokenizer_sha256
            or original.get('external_model_teacher_used') is not False):
        raise ValueError('Complete original finalized paired source and own tokenizer required')
    for name, digest in original['files_sha256'].items():
        if file_hash(data / name) != digest:
            raise ValueError('Original finalized paired source bytes differ')
    expected = {row['id']: row for row in admitted_rows}
    if len(expected) != len(admitted_rows) or not expected:
        raise ValueError('Every original admitted training photo question exactly once required')
    entries, inventory, images, rows = [], [], {}, []
    with (data / 'train.jsonl').open('rb') as stream:
        while True:
            offset, raw = stream.tell(), stream.readline()
            if not raw:
                break
            original_row = json.loads(raw)
            row = training_row(original_row, data)
            if expected.get(row['id']) != row:
                raise ValueError('Exact original source-to-admitted-row transformation required')
            ids, labels = encoder.sequence(row, 'response')
            targets = sum(value != -100 for value in labels[1:])
            if (len(ids) != original_row['context_tokens'] or targets != original_row['target_tokens']
                    or len(ids) > 4096 or not 0 < targets < len(ids)):
                raise ValueError('Every complete original photo-question input/response mask required')
            digest = hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).digest()
            entries.append((offset, len(raw), 0, len(ids), targets,
                np.void(hashlib.sha256(raw).digest()), np.void(digest)))
            inventory.append(dict(position=len(entries) - 1, id=row['id'], group_id=row['group_id'],
                pair_ids=row['pair_ids'], context_tokens=len(ids), target_tokens=targets,
                input_sha256=digest.hex()))
            record = dict(path=row['image'], sha256=row['stored_image_sha256'], model_pixel_sha256_448=row['model_pixel_sha256_448'])
            if row['image'] in images and images[row['image']] != record:
                raise ValueError('One exact original identity for each repeated training photo required')
            images[row['image']] = record
            rows.append(original_row)
    if {row['id'] for row in rows} != set(expected) or len(rows) != len(expected) or len(rows) != original['splits']['train']['questions']:
        raise ValueError('Every original admitted training photo question must be indexed once')
    validate_pairs([json.loads(line) for line in (data / 'pairs.train.jsonl').open()], rows)
    if encoder.truncated:
        raise ValueError('Original paired photo instructions cannot be clipped')
    for record in images.values():
        if file_hash(record['path']) != record['sha256']:
            raise ValueError('Original stored training photo bytes changed')
    output.mkdir(parents=True)
    np.save(output / 'index.npy', np.asarray(entries, dtype=INDEX_DTYPE), allow_pickle=False)
    (output / 'records.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in inventory))
    (output / 'image-files.jsonl').write_text(''.join(json.dumps(images[name]) + '\n' for name in sorted(images)))
    atomic_json(output / 'source-admission.json', receipt)
    manifest = dict(format=FORMAT, family_names=[FAMILY], family_counts={FAMILY: len(rows)},
        complete_fit_rows=len(rows), unique_original_training_images=len(images),
        tokenizer_sha256=tokenizer_sha256, original_source_directory=str(data),
        original_manifest_sha256=file_hash(path), original_configuration_sha256=file_hash(data / 'configuration.json'),
        original_files_sha256=original['files_sha256'],
        artifact_files_sha256={name: file_hash(output / name) for name in sorted(ARTIFACTS)},
        original_source_bytes_copied=0, original_image_bytes_copied=0,
        production_runner_registered=False, new_neural_training_started=False, goal_achieved=False,
        scope='Every original finalized admitted training photo question appears once, with unchanged human answer, '
            'pair membership, source metadata and full response mask. Original admission reconstructs both '
            'training/validation sources and creator/pixel declarations; validation never enters this reader. '
            'This is supervised visual QA, not an added pairwise contrastive objective or learned-grounding result.')
    atomic_json(output / 'manifest.json', manifest)
    return manifest


def _receipt(receipt, manifest_sha256):
    keys = ('passed', 'all_original_question_answer_photo_pair_rows_reconstructed',
        'actual_complete_input_and_response_only_masks_verified',
        'original_creator_declarations_and_stored_pixels_verified', 'validation_rows_not_returned_for_training')
    if (any(receipt.get(key) is not True for key in keys) or receipt.get('manifest_sha256') != manifest_sha256
            or receipt.get('source_photo_alias_and_exact224_448px_train_validation_overlap') != 0):
        raise ValueError('Completed full original paired-photo source reconstruction/admission required')


class PairedPhotoReplay:
    families = [FAMILY]

    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / 'manifest.json').read_text())
        m = self.manifest
        if (m.get('format') != FORMAT or m.get('family_names') != [FAMILY]
                or set(m['artifact_files_sha256']) != ARTIFACTS or set(m['original_files_sha256']) != ORIGINAL_FILES):
            raise ValueError('Complete original paired human photo replay inventory required')
        for name, digest in m['artifact_files_sha256'].items():
            if file_hash(self.folder / name) != digest:
                raise ValueError('Original paired replay index/admission artifact changed')
        self.data = Path(m['original_source_directory']).resolve()
        path = self.data / 'manifest.json'
        if (file_hash(path) != m['original_manifest_sha256']
                or file_hash(self.data / 'configuration.json') != m['original_configuration_sha256']):
            raise ValueError('Exact finalized original paired-photo source/configuration required')
        original = json.loads(path.read_text())
        if (original['files_sha256'] != m['original_files_sha256'] or original['tokenizer_sha256'] != m['tokenizer_sha256']
                or original.get('external_model_teacher_used') is not False):
            raise ValueError('Original paired file/own-tokenizer authorization differs')
        for name, digest in m['original_files_sha256'].items():
            if file_hash(self.data / name) != digest:
                raise ValueError('Finalized original paired source changed')
        _receipt(json.loads((self.folder / 'source-admission.json').read_text()), m['original_manifest_sha256'])
        self.source = self.data / 'train.jsonl'
        self.source_state = self._state(self.source)
        self.index = np.load(self.folder / 'index.npy', mmap_mode='r', allow_pickle=False)
        if self.index.dtype != INDEX_DTYPE or self.index.ndim != 1 or not len(self.index):
            raise ValueError('Known complete original photo-question index schema required')
        self.index_state = self._state(self.folder / 'index.npy')
        index = self.index
        offsets = np.concatenate((np.zeros(1, dtype=np.uint64), np.cumsum(index['bytes'].astype(np.uint64))[:-1]))
        if (not np.array_equal(index['offset'], offsets) or np.any(index['bytes'] == 0) or np.any(index['family'] != 0)
                or int(index['bytes'].astype(np.uint64).sum()) != self.source.stat().st_size
                or np.any(index['context_tokens'] < 3) or np.any(index['context_tokens'] > 4096)
                or np.any(index['target_tokens'] < 1) or np.any(index['target_tokens'] >= index['context_tokens'])
                or len(index) != m['complete_fit_rows'] or len(index) != original['splits']['train']['questions']
                or m['family_counts'] != {FAMILY: len(index)}):
            raise ValueError('Every original training-photo row and complete target span must appear once')
        self.images = {}
        for record in map(json.loads, (self.folder / 'image-files.jsonl').open()):
            path = Path(record['path'])
            if (str(path) in self.images or not path.is_relative_to(self.data)
                    or file_hash(path) != record['sha256']):
                raise ValueError('Original paired image inventory/path/bytes differ')
            self.images[str(path)] = {**record, 'state': self._state(path)}
        if len(self.images) != m['unique_original_training_images'] or len(self.images) != original['splits']['train']['images']:
            raise ValueError('Every original admitted training photo required')

    @staticmethod
    def _state(path):
        state = Path(path).stat()
        return state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns

    def size(self, name):
        if name != FAMILY:
            raise KeyError(name)
        return len(self.index)

    def row(self, name, position):
        if type(position) is not int or not 0 <= position < self.size(name):
            raise IndexError('Original paired-photo position outside training population')
        if self._state(self.source) != self.source_state:
            raise ValueError('Original paired source changed during replay')
        if self._state(self.folder / 'index.npy') != self.index_state:
            raise ValueError('Original paired replay index changed during use')
        item = self.index[position]
        with self.source.open('rb') as stream:
            stream.seek(int(item['offset']))
            raw = stream.read(int(item['bytes']))
        if hashlib.sha256(raw).digest() != item['row_sha256'].tobytes():
            raise ValueError('Exact original paired source row bytes differ')
        row = training_row(json.loads(raw), self.data)
        image = self.images.get(row['image'])
        if (image is None or self._state(row['image']) != image['state']
                or file_hash(row['image']) != row['stored_image_sha256'] or image['sha256'] != row['stored_image_sha256']
                or image['model_pixel_sha256_448'] != row['model_pixel_sha256_448']):
            raise ValueError('Exact original paired photo and448px pixel identity required')
        return {**row, '_replay_input_sha256': item['input_sha256'].tobytes().hex(),
            '_replay_context_tokens': int(item['context_tokens']), '_replay_target_tokens': int(item['target_tokens'])}
