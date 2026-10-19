from types import SimpleNamespace
import json

import numpy as np
import pytest

from slm_lab.codenet_complete_context_v1 import CompleteCodeEncoder
from slm_lab.codenet_token_pack_v1 import CodePackWriter, CompleteCodePack, packed_encoder
from slm_lab.corpus import file_hash


class Tokenizer:
    def token_to_id(self, text):
        return {'<bos>': 1, '<eos>': 2, '<pad>': 0}[text]

    def get_vocab_size(self):
        return 32768

    def encode(self, text):
        return SimpleNamespace(ids=list(map(ord, text)))


def fixture_pack(folder):
    for name in ['tokenizer.json', 'model-config.json', 'DATASET-LICENSE.txt', 'publisher-README.md']:
        (folder / name).write_text('{}')
    config = SimpleNamespace(max_text_length=64, image_size=224, vocab_size=32768,
                             image_token_count=lambda _: 0)
    encoder = CompleteCodeEncoder(Tokenizer(), config, folder)
    rows = [{'id': 'long', 'group_id': 'p00001', 'split': 'train', 'prompt': 'p', 'response': 'r' * 40},
            {'id': 'short', 'group_id': 'p00002', 'split': 'train', 'prompt': 'p', 'response': 'x'},
            {'id': 'valid', 'group_id': 'p00003', 'split': 'validation', 'prompt': 'p', 'response': 'v'},
            {'id': 'test', 'group_id': 'p00004', 'split': 'test', 'prompt': 'p', 'response': 't'}]
    writer = CodePackWriter(folder, minimum_free_gib=0)
    for row in rows:
        writer.add(row, *encoder.sequence(row, 'response'))
    writer.finish({'fixture': True})
    return rows, encoder, config


def test_complete_targets_and_fixed_roles_roundtrip_with_real_batch_padding(tmp_path):
    rows, original, config = fixture_pack(tmp_path)
    pack = CompleteCodePack(tmp_path, 'train')
    assert len(pack) == 2
    assert pack.sequence(0) == original.sequence(rows[0], 'response')
    assert pack.sequence(1) == original.sequence(rows[1], 'response')
    assert CompleteCodePack(tmp_path, 'validation').sequence(0) == original.sequence(rows[2], 'response')
    actual = packed_encoder(pack, Tokenizer(), config).batch([{'packed_code_index': 0}, {'packed_code_index': 1}], 'response', 'cpu')
    expected = original.batch(rows[:2], 'response', 'cpu')
    for key in ['input_ids', 'labels', 'attention_mask']:
        assert actual[key].equal(expected[key])
    assert (actual['labels'][0] != -100).sum().item() == 41


def test_unadmitted_data_and_held_roles_cannot_be_requested_for_training(tmp_path):
    fixture_pack(tmp_path)
    with pytest.raises(ValueError, match='not been admitted'):
        CompleteCodePack(tmp_path, 'train', purpose='training')
    with pytest.raises(ValueError, match='Held source roles'):
        CompleteCodePack(tmp_path, 'test', purpose='training')


def test_corrupt_token_bytes_are_rejected(tmp_path):
    fixture_pack(tmp_path)
    with (tmp_path / 'input-ids.bin').open('ab') as output:
        output.write(b'bad')
    with pytest.raises(ValueError, match='bytes differ'):
        CompleteCodePack(tmp_path, 'train')


def test_self_consistent_hashes_cannot_conceal_invalid_offsets(tmp_path):
    fixture_pack(tmp_path)
    path = tmp_path / 'index.npy'
    array = np.load(path, allow_pickle=False)
    array[1]['offset'] += 1
    np.save(path, array, allow_pickle=False)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    manifest['files_sha256']['index.npy'] = file_hash(path)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='offsets'):
        CompleteCodePack(tmp_path, 'train')


def test_noncontiguous_response_mask_and_oversize_storage_are_rejected(tmp_path):
    writer = CodePackWriter(tmp_path, maximum_bytes=1, minimum_free_gib=0)
    row = {'id': 'a', 'split': 'train', 'prompt': 'p', 'response': 'r'}
    try:
        with pytest.raises(ValueError, match='contiguous'):
            writer.add(row, [1, 100, 101, 2], [-100, 100, -100, 2])
        writer.add(row, [1, 100, 101, 2], [-100, -100, 101, 2])
        with pytest.raises(ValueError, match='size budget'):
            writer.guard()
    finally:
        writer.close()
