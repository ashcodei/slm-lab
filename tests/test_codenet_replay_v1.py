from types import SimpleNamespace
import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from slm_lab.codenet_replay_v1 import (ARTIFACT_FILES, INDEX_DTYPE, FullReplayEncoder,
    ReplayContextDoesNotFit, ReplayCorpus, family, training_role)
from slm_lab.corpus import file_hash


class Tokenizer:
    def token_to_id(self, text):
        return {'<bos>': 1, '<eos>': 2, '<pad>': 0}[text]

    def get_vocab_size(self):
        return 32768

    def encode(self, text):
        return SimpleNamespace(ids=list(map(ord, text)))


def encoder(tmp_path):
    config = SimpleNamespace(max_text_length=64, image_size=16, vocab_size=32768,
                             image_token_count=lambda _: 4)
    return FullReplayEncoder(Tokenizer(), config, tmp_path)


def test_complete_image_answer_beyond_legacy_half_cap_is_retained_and_padded(tmp_path):
    path = tmp_path / 'image.png'
    Image.new('RGB', (8, 4), (21, 55, 90)).save(path)
    rows = [{'image': str(path), 'split': 'train', 'prompt': 'p', 'response': 'a' * 40},
            {'image': str(path), 'split': 'train', 'prompt': 'p', 'response': 'b'}]
    e = encoder(tmp_path)
    batch = e.batch(rows, 'response', 'cpu')
    assert (batch['labels'][0] != -100).sum().item() == 41
    assert (batch['labels'][1] != -100).sum().item() == 2
    assert batch['pixel_values'].shape == (2, 3, 16, 16)
    assert e.truncated == 0


def test_full_replay_overflow_is_qualified_without_clipping(tmp_path):
    e = encoder(tmp_path)
    with pytest.raises(ReplayContextDoesNotFit) as error:
        e.sequence({'prompt': 'p', 'response': 'a' * 60}, 'response')
    assert error.value.response_tokens == 60 and e.truncated == 0


def test_all_explicit_held_source_roles_are_rejected():
    for key in ['split', 'source_split']:
        for role in ['valid', 'validation', 'test']:
            with pytest.raises(ValueError, match='held/source role'):
                training_role({key: role})


def test_known_source_families_keep_reading_and_verified_programs_distinct():
    assert family({'image': 'x', 'task': 'word_ocr', 'split': 'train'}) == 'image_reading'
    assert family({'image': 'x', 'task': 'caption', 'split': 'train'}) == 'generic_visual'
    assert family({'source': 'GitHub/TheAlgorithms/Python', 'language': 'Python'}) == 'verified_programs'
    assert family({'source': 'CodeSearchNet', 'language': 'Java'}) == 'code:Java'
    assert family({'source': 'GSM8K original human solutions'}) == 'math'
    with pytest.raises(ValueError, match='explicit family'):
        family({'source': 'unknown'})


def replay_fixture(tmp_path):
    """Two original exposures can share an ID and image without losing a row."""
    path = tmp_path / 'image.png'
    Image.new('RGB', (8, 4), (21, 55, 90)).save(path)
    rows = [{'id': 'repeated', 'image': 'image.png', 'task': 'word_ocr', 'split': 'train',
             'prompt': 'p', 'response': value} for value in ['a' * 40, 'b']]
    source = tmp_path / 'original.jsonl'
    raw_rows = [(json.dumps(row) + '\n').encode() for row in rows]
    source.write_bytes(b''.join(raw_rows))
    output = tmp_path / 'replay'
    output.mkdir()
    e = encoder(tmp_path)
    entries, offset = [], 0
    for row, raw in zip(rows, raw_rows):
        ids, labels = e.sequence(row, 'response')
        digest = hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).digest()
        entries.append((offset, len(raw), 0, len(ids), sum(v != -100 for v in labels),
                        np.void(hashlib.sha256(raw).digest()), np.void(digest)))
        offset += len(raw)
    np.save(output / 'index.npy', np.asarray(entries, dtype=INDEX_DTYPE), allow_pickle=False)
    (output / 'image-files.jsonl').write_text(json.dumps({'path': str(path), 'sha256': file_hash(path)}) + '\n')
    for name in ['records.jsonl', 'review.jsonl']:
        (output / name).write_text('')
    manifest = {'format': 'slm-complete-joint-replay-v1', 'original_training_source': str(source),
                'original_training_source_sha256': file_hash(source), 'complete_fit_rows': 2,
                'family_names': ['image_reading'], 'family_counts': {'image_reading': 2},
                'artifact_files_sha256': {name: file_hash(output / name) for name in ARTIFACT_FILES}}
    (output / 'manifest.json').write_text(json.dumps(manifest))
    return output, source, path, e


def test_bound_reader_retains_repeated_original_exposures_and_full_image_targets(tmp_path):
    output, source, path, e = replay_fixture(tmp_path)
    corpus = ReplayCorpus(output)
    assert corpus.size('image_reading') == 2
    rows = [corpus.row('image_reading', n) for n in range(2)]
    assert rows[0]['id'] == rows[1]['id'] == 'repeated'
    assert rows[0]['response'] != rows[1]['response']
    batch = e.batch(rows, 'response', 'cpu')
    assert (batch['labels'] != -100).sum(dim=1).tolist() == [41, 2]
    assert batch['pixel_values'].shape == (2, 3, 16, 16)
    for position in [-1, 2, True]:
        with pytest.raises(IndexError):
            corpus.row('image_reading', position)
    altered = {**rows[0], 'response': 'c' * 40}
    with pytest.raises(ValueError, match='text/mask differs'):
        e.sequence(altered, 'response')


@pytest.mark.parametrize('changed', ['source', 'image', 'index'])
def test_bound_reader_rejects_changed_original_bytes_and_artifacts(tmp_path, changed):
    output, source, image, _ = replay_fixture(tmp_path)
    corpus = ReplayCorpus(output)
    path = {'source': source, 'image': image, 'index': output / 'index.npy'}[changed]
    with path.open('ab') as stream:
        stream.write(b'changed')
    if changed != 'index':
        with pytest.raises(ValueError, match='changed during use'):
            corpus.row('image_reading', 0)
    with pytest.raises(ValueError, match='changed|differ'):
        ReplayCorpus(output)
