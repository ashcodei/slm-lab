from contextlib import nullcontext
import copy
import hashlib
import json
import os

import numpy as np
import pytest
import torch

from slm_lab.codenet_packed_replay_v1 import (FAMILY, PackedPretrainingEncoder, PackedReplayCorpus,
    prepare_packed_replay, window_layout)
from slm_lab.codenet_runner_v1 import FixedReductionModel
from slm_lab.codenet_stage_v1 import Objective, freeze_for_code_stage, objective_gradients
from slm_lab.corpus import file_hash
from slm_lab.model import TinyMultimodalLM

from test_codenet_runner_v1 import Tokenizer, small_config


def artifact(tmp_path, total=18, length=5):
    packed = tmp_path / 'packed'
    packed.mkdir()
    token = tmp_path / 'tokenizer.json'
    token.write_text('own tokenizer fixture')
    sources = {'train': np.arange(3, 3 + total, dtype='<u2'), 'valid': np.asarray([71, 72, 73, 74, 75, 76], dtype='<u2')}
    splits = {}
    for split, ids in sources.items():
        rows = []
        for position, part in enumerate(np.array_split(ids, 3)):
            if not len(part):
                continue
            path = packed / (split + '-' + str(position) + '.bin')
            part.tofile(path)
            rows.append(dict(path=path.name, tokens=len(part), sha256=file_hash(path)))
        splits[split] = dict(tokens=len(ids), shards=rows)
    manifest = dict(dtype='<u2', tokenizer_sha256=file_hash(token), splits=splits,
        total_tokens=sum(len(ids) for ids in sources.values()), components=[{'name': 'older'}, {'name': 'systems'}])
    (packed / 'corpus.json').write_text(json.dumps(manifest))
    receipt = dict(passed=True, training_admitted=True, tokenizer_sha256=file_hash(token),
        packed_manifest_sha256=file_hash(packed / 'corpus.json'), combined_train_tokens=total)
    output = tmp_path / 'replay'
    prepare_packed_replay(output, packed, receipt, token, length, 128)
    return PackedReplayCorpus(output), sources


@pytest.mark.parametrize('total', [2, 4, 5, 6, 8, 9, 10, 17, 18])
def test_every_original_adjacent_pair_exactly_once_including_short_final_window(tmp_path, total):
    reader, originals = artifact(tmp_path, total)
    encoder = PackedPretrainingEncoder(Tokenizer(), small_config(), tmp_path, 16)
    observed, targets = [], 0
    for position in range(reader.size(FAMILY)):
        row = reader.row(FAMILY, position)
        ids, labels = encoder.sequence(row, None)
        assert ids == labels == row['tokens']
        assert len(ids) <= 5
        assert row['_replay_target_tokens'] == len(ids) - 1
        observed.extend(zip(ids[:-1], labels[1:]))
        targets += len(ids) - 1
    assert observed == list(zip(originals['train'][:-1], originals['train'][1:]))
    assert targets == total - 1
    layout = window_layout(total, 5)
    assert len(reader.row(FAMILY, reader.size(FAMILY) - 1)['tokens']) == layout['last_window_length']
    assert reader.manifest['original_source_bytes_copied'] == 0
    assert reader.manifest['artificial_special_tokens_added'] == 0
    assert encoder.truncated == 0


def test_exact_padded_causal_loss_and_gradient_update_use_real_shared_objective(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(71)
    reader, originals = artifact(tmp_path)
    encoder = PackedPretrainingEncoder(Tokenizer(), small_config(), tmp_path, 16)
    rows = [reader.row(FAMILY, i) for i in range(reader.size(FAMILY))]
    batch = encoder.batch(rows, None, 'cpu')
    assert int((batch['labels'][:, 1:] != -100).sum()) == len(originals['train']) - 1
    assert batch['input_ids'][-1].tolist() == [19, 20, 0, 0, 0]
    assert batch['labels'][-1].tolist() == [19, 20, -100, -100, -100]
    assert batch['attention_mask'][-1].tolist() == [True, True, False, False, False]
    raw = TinyMultimodalLM(small_config())
    trainable, frozen = freeze_for_code_stage(raw)
    before = copy.deepcopy(raw.state_dict())
    with torch.no_grad():
        logits = raw(**{key: value for key, value in batch.items() if key != 'labels'})['logits']
        expected = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, 128),
            batch['labels'][:, 1:].reshape(-1), ignore_index=-100)
    optimizer = torch.optim.AdamW(trainable, lr=.003)
    metrics = objective_gradients(FixedReductionModel(raw), [Objective(FAMILY, encoder,
        [[row] for row in rows], None, .5)], 'cpu', nullcontext)
    assert metrics['objectives'][FAMILY]['global_target_tokens'] == len(originals['train']) - 1
    assert metrics['objectives'][FAMILY]['token_mean_nll'] == pytest.approx(float(expected), abs=1e-6)
    assert all(parameter.grad is None for parameter in frozen)
    optimizer.step()
    after = raw.state_dict()
    assert any(not torch.equal(value, after[name]) for name, value in before.items() if name.startswith('blocks.'))
    assert all(torch.equal(value, after[name]) for name, value in before.items() if name.startswith(('vision.', 'value_head.')))


def test_validation_windows_remain_visible_to_audit_and_are_rejected_for_training(tmp_path):
    reader, originals = artifact(tmp_path)
    rows = [reader.row(FAMILY, i, split='valid') for i in range(reader.size(FAMILY, split='valid'))]
    assert [token for i, row in enumerate(rows) for token in row['tokens'][int(i > 0):]] == originals['valid'].tolist()
    encoder = PackedPretrainingEncoder(Tokenizer(), small_config(), tmp_path, 16)
    for row in rows:
        with pytest.raises(ValueError, match='held/source'):
            encoder.sequence(row, None)


@pytest.mark.parametrize('fault', ['field', 'response', 'image', 'objective', 'token', 'length', 'targets'])
def test_packed_data_cannot_be_relabelled_as_instruction_or_fabricated_tokens(tmp_path, fault):
    reader, _ = artifact(tmp_path)
    row = reader.row(FAMILY, 0)
    field = None
    if fault == 'field':
        field = 'response'
    elif fault in {'response', 'image'}:
        row[fault] = 'fabricated'
    elif fault == 'objective':
        row['_replay_objective'] = 'response'
    elif fault == 'token':
        row['tokens'][0] += 1
    elif fault == 'length':
        row['tokens'].pop()
    else:
        row['_replay_target_tokens'] += 1
    with pytest.raises(ValueError):
        PackedPretrainingEncoder(Tokenizer(), small_config(), tmp_path, 16).sequence(row, field)


def test_changed_shard_with_restored_size_and_mtime_rejected(tmp_path):
    reader, _ = artifact(tmp_path)
    path = reader.stream.shards['train'][0]
    state = path.stat()
    raw = bytearray(path.read_bytes())
    raw[0] ^= 1
    path.write_bytes(raw)
    os.utime(path, ns=(state.st_atime_ns, state.st_mtime_ns))
    with pytest.raises(ValueError, match='shard changed'):
        reader.row(FAMILY, 0)
    with pytest.raises(ValueError, match='packed shard'):
        PackedReplayCorpus(reader.folder)


def test_missing_tail_or_duplicate_offset_rejected_even_with_updated_index_hash(tmp_path):
    reader, _ = artifact(tmp_path)
    path = reader.folder / 'train-index.npy'
    index = np.load(path, allow_pickle=False)
    index[-1]['offset'] = index[-2]['offset']
    np.save(path, index, allow_pickle=False)
    manifest = reader.manifest
    manifest['artifact_files_sha256'][path.name] = file_hash(path)
    (reader.folder / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Every causal window/tail'):
        PackedReplayCorpus(reader.folder)


def test_window_hash_is_checked_even_if_index_is_corrupted_before_new_reader(tmp_path):
    reader, _ = artifact(tmp_path)
    path = reader.folder / 'train-index.npy'
    index = np.load(path, allow_pickle=False)
    index[0]['input_sha256'] = np.void(hashlib.sha256(b'wrong original window').digest())
    np.save(path, index, allow_pickle=False)
    manifest = reader.manifest
    manifest['artifact_files_sha256'][path.name] = file_hash(path)
    (reader.folder / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='window bytes'):
        PackedReplayCorpus(reader.folder).row(FAMILY, 0)
