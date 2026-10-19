from dataclasses import replace
import json
import os

import numpy as np
from PIL import Image
import pytest

from slm_lab.codenet_paired_photo_replay_v1 import (FAMILY, PairedPhotoReplay, prepare_paired_replay, training_row)
from slm_lab.codenet_runner_v1 import VerifiedReplayEncoder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import pixel_hash

from test_codenet_runner_v1 import Tokenizer, small_config


def artifact(tmp_path):
    data = tmp_path / 'original'
    (data / 'images').mkdir(parents=True)
    encoder = VerifiedReplayEncoder(Tokenizer(), replace(small_config(), max_text_length=128,
        image_size=224, patch_size=16), data, 448)
    all_rows = {}
    for split, subset in [('train', 'train2014'), ('valid', 'val2014')]:
        rows = []
        for side in range(2):
            identity = (0 if split == 'train' else 100) + side
            path = data / 'images' / f'{identity}.png'
            Image.new('RGB', (8, 4), (identity + 10, side * 80, 23)).save(path)
            answer = 'a' * 70 if split == 'train' and side == 0 else chr(ord('b') + identity % 3)
            with Image.open(path) as image:
                pixels = pixel_hash(image, 448)
            row = dict(id=f'vqa2:{subset}:question:{identity}', group_id=f'flickr:{identity}',
                source_subset=subset, task='visual_qa', domain='everyday_photos',
                external_model_teacher_used=False, pair_ids=[split + '-pair'], prompt='p\nAnswer briefly.',
                response=answer, exact_human_agreement=10, image=str(path.relative_to(data)),
                stored_image_sha256=file_hash(path), model_pixel_sha256_448=pixels,
                original_human_annotation={'answers': [{'answer_id': i, 'answer': answer} for i in range(1, 11)]})
            # Tokenization/masks do not depend on the held/training source role.
            ids, labels = encoder.sequence(row, 'response')
            row.update(context_tokens=len(ids), target_tokens=sum(label != -100 for label in labels[1:]))
            rows.append(row)
        all_rows[split] = rows
        (data / (split + '.jsonl')).write_text(''.join(json.dumps(row) + '\n' for row in rows))
        pair = dict(id=split + '-pair', question_ids=[row['id'] for row in rows], canonical_question='p', source_subset=subset)
        (data / ('pairs.' + split + '.jsonl')).write_text(json.dumps(pair) + '\n')
    (data / 'images.collected.jsonl').write_text('original photo journal fixture\n')
    (data / 'configuration.json').write_text('{}\n')
    names = ['train.jsonl', 'valid.jsonl', 'pairs.train.jsonl', 'pairs.valid.jsonl', 'images.collected.jsonl']
    original = dict(tokenizer_sha256='b' * 64, external_model_teacher_used=False,
        files_sha256={name: file_hash(data / name) for name in names},
        splits={split: dict(questions=2, images=2, pairs=1) for split in all_rows})
    (data / 'manifest.json').write_text(json.dumps(original))
    receipt = dict(passed=True, manifest_sha256=file_hash(data / 'manifest.json'),
        all_original_question_answer_photo_pair_rows_reconstructed=True,
        actual_complete_input_and_response_only_masks_verified=True,
        original_creator_declarations_and_stored_pixels_verified=True,
        validation_rows_not_returned_for_training=True, source_photo_alias_and_exact224_448px_train_validation_overlap=0,
        scope='Explicit fixture receipt; not actual original public-source admission.')
    output = tmp_path / 'replay'
    admitted = [training_row(row, data) for row in all_rows['train']]
    prepare_paired_replay(output, data, admitted, receipt, encoder, 'b' * 64)
    return PairedPhotoReplay(output), encoder, all_rows


def test_complete_original_human_responses_metadata_pairs_and448px_batches_are_preserved(tmp_path):
    reader, encoder, originals = artifact(tmp_path)
    assert reader.size(FAMILY) == 2
    for position in range(2):
        row = reader.row(FAMILY, position)
        expected = training_row(originals['train'][position], reader.data)
        assert {key: value for key, value in row.items() if not key.startswith('_replay_')} == expected
        batch = encoder.batch([row], 'response', 'cpu')
        ids, labels = encoder.sequence(expected, 'response')
        assert batch['input_ids'][0].tolist() == ids
        assert batch['labels'][0].tolist() == labels
        assert tuple(batch['pixel_values'].shape) == (1, 3, 448, 448)
        assert int((batch['labels'][:, 1:] != -100).sum()) == expected['target_tokens']
    assert reader.row(FAMILY, 0)['_replay_target_tokens'] == 71 > encoder.config.max_text_length // 2
    assert reader.manifest['original_source_bytes_copied'] == reader.manifest['original_image_bytes_copied'] == 0
    assert encoder.truncated == 0
    for held in originals['valid']:
        with pytest.raises(ValueError, match='training-only'):
            training_row(held, reader.data)


@pytest.mark.parametrize('changed', ['source', 'image', 'index'])
def test_changed_original_bytes_or_index_rejected_even_with_restored_mtime(tmp_path, changed):
    reader, _, _ = artifact(tmp_path)
    if changed == 'source':
        path = reader.source
    elif changed == 'image':
        path = reader.data / 'images/0.png'
    else:
        path = reader.folder / 'index.npy'
    state = path.stat()
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    os.utime(path, ns=(state.st_atime_ns, state.st_mtime_ns))
    with pytest.raises(ValueError):
        reader.row(FAMILY, 0)


def test_missing_or_duplicate_original_source_index_rejected_with_updated_artifact_digest(tmp_path):
    reader, _, _ = artifact(tmp_path)
    path = reader.folder / 'index.npy'
    index = np.load(path, allow_pickle=False)
    index[1]['offset'] = 0
    np.save(path, index, allow_pickle=False)
    manifest = reader.manifest
    manifest['artifact_files_sha256'][path.name] = file_hash(path)
    (reader.folder / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Every original training-photo row'):
        PairedPhotoReplay(reader.folder)


def test_teacher_or_held_role_cannot_enter_original_training_view(tmp_path):
    reader, _, originals = artifact(tmp_path)
    row = dict(originals['train'][0], external_model_teacher_used=True)
    with pytest.raises(ValueError, match='training-only'):
        training_row(row, reader.data)
    with pytest.raises(ValueError, match='training-only'):
        training_row({**originals['train'][0], 'split': 'valid'}, reader.data)
