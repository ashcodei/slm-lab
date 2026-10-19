from copy import deepcopy
import json

import pytest

from slm_lab.corpus import file_hash
from slm_lab.vision_alignment_experiment import caption_subset, compare_reports


def test_caption_subset_retains_provenance_and_rejects_held_out_pixels(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    for split in ['train', 'valid']:
        row = {'id': split, 'group_id': split, 'image': str((tmp_path / (split + '.png')).resolve()),
               'task': 'caption', 'prompt': 'Describe this image.', 'response': 'Public caption.',
               'source_url': 'https://example.org/' + split, 'license': 'CC0', 'visual_source': 'everyday_photos',
               'model_pixel_sha256': split + '-224', 'model_pixel_sha256_448': split + '-448'}
        other = {**row, 'id': split + '-text', 'task': 'region_ocr'}
        (source / (split + '.jsonl')).write_text(json.dumps(row) + '\n' + json.dumps(other) + '\n')
    manifest = {'files_sha256': {split + '.jsonl': file_hash(source / (split + '.jsonl')) for split in ['train', 'valid']}}
    (source / 'manifest.json').write_text(json.dumps(manifest))
    report = caption_subset(source, tmp_path / 'captions')
    retained = json.loads((tmp_path / 'captions/train.jsonl').read_text())
    assert report['splits']['train']['examples'] == 1 and report['fashion_examples'] == 0
    assert retained['license'] == 'CC0' and retained['source_url'].endswith('/train')
    assert caption_subset(source, tmp_path / 'captions') == report
    train = json.loads((source / 'train.jsonl').read_text().splitlines()[0])
    train['model_pixel_sha256_448'] = 'valid-448'
    (source / 'train.jsonl').write_text(json.dumps(train) + '\n')
    manifest['files_sha256']['train.jsonl'] = file_hash(source / 'train.jsonl')
    (source / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='overlap'):
        caption_subset(source, tmp_path / 'changed')


def test_paired_visual_comparison_requires_same_inputs_and_protocol():
    first = {'data_sha256': 'data', 'tokenizer_sha256': 'own', 'evaluation_provenance': {'files': 'same'},
             'inference_device_type': 'cuda', 'inference_precision': 'bf16', 'image_size': 448,
             'samples': [{'id': 'a', 'input_pixels_sha256': 'pixels', 'wrong_image_id': 'b',
                          'wrong_pixels_sha256': 'other', 'target_tokens': 8}],
             'proper_nll': 3., 'blank_nll_gain': 0., 'mismatch_nll_gain': 0.,
             'caption_retrieval': {'image_to_text_accuracy': 0.}}
    second = deepcopy(first); second['proper_nll'] = 2.
    result = compare_reports(first, second, 'vision')
    assert result['proper_nll'] == {'first': 3., 'second': 2.}
    second['samples'][0]['wrong_pixels_sha256'] = 'changed'
    with pytest.raises(ValueError, match='controls differ'):
        compare_reports(first, second, 'vision')
    second = deepcopy(first); second['image_size'] = 224
    with pytest.raises(ValueError, match='image_size'):
        compare_reports(first, second, 'vision')
