from dataclasses import asdict
import json

from PIL import Image, ImageDraw
import torch
import pytest

from slm_lab.corpus import file_hash
from slm_lab.model import ModelConfig, TinyMultimodalLM
from slm_lab.object_feature_probe import crop_image, prepare, evaluate


def test_object_crop_uses_stored_photo_coordinates_and_rejects_changed_source(tmp_path):
    path = tmp_path / 'photo.png'; image = Image.new('RGB', (80, 40), 'blue')
    ImageDraw.Draw(image).rectangle((20, 10, 39, 29), fill='red'); image.save(path)
    row = {'image': str(path), 'stored_image_sha256': file_hash(path), 'normalized_box': [.25, .25, .5, .75]}
    crop = crop_image(row)
    assert crop.size == (20, 20) and crop.tobytes() == bytes((255, 0, 0)) * 400
    image.save(path, compress_level=1)
    with pytest.raises(ValueError, match='changed'):
        crop_image(row)


def test_frozen_probe_covers_labels_controls_and_train_only_heads_without_changing_checkpoint(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    for split, count, offset in [('train', 3, 10), ('valid', 2, 60)]:
        rows = []
        for label, color in [('red', 'red'), ('green', 'green')]:
            for i in range(count):
                path = source / f'{split}-{label}-{i}.png'; image = Image.new('RGB', (16, 16), 'gray')
                draw = ImageDraw.Draw(image); draw.rectangle((4, 4, 11, 11), fill=color)
                draw.point((4, 4), fill=(offset + i, offset + i, offset + i)); image.save(path)
                rows.append({'id': path.name, 'group_id': path.name, 'image': str(path),
                             'stored_image_sha256': file_hash(path), 'normalized_box': [.25, .25, .75, .75],
                             'task': 'object_region', 'response': label})
        (source / (split + '.jsonl')).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (source / 'manifest.json').write_text(json.dumps({'files_sha256': {name: file_hash(source / name) for name in ['train.jsonl', 'valid.jsonl']}}))
    data = tmp_path / 'data'; manifest = prepare(source, data, image_size=16, minimum_training=2, train_cap=3, valid_cap=2)
    assert manifest['parent_group_overlap'] == manifest['exact_crop_overlap'] == 0
    assert manifest['splits']['train']['examples'] == 6 and manifest['splits']['valid']['examples'] == 4
    config = ModelConfig(vocab_size=260, hidden_size=32, intermediate_size=64, num_layers=1,
                         num_heads=4, num_kv_heads=2, vision_hidden_size=16, vision_intermediate_size=32,
                         vision_layers=1, vision_heads=2, image_size=16, patch_size=4, pool_size=2)
    checkpoint = tmp_path / 'model.pt'
    torch.save({'config': asdict(config), 'model': TinyMultimodalLM(config).state_dict()}, checkpoint)
    before = file_hash(checkpoint)
    report = evaluate([('tiny', checkpoint)], data, tmp_path / 'reports', device='cpu', random_control=False)
    assert report['validation_examples'] == 4 and report['training_examples'] == 6
    assert report['classes'] == ['green', 'red'] and report['uniform_chance'] == .5
    assert report['production_updated'] is False and report['goal_achieved'] is False
    assert report['baselines']['geometry_and_color']['accuracy'] == 1
    cases = json.loads((tmp_path / 'reports/tiny.json').read_text())['cases']
    assert all(row['input_pixels_sha256'] != row['wrong_pixels_sha256'] for row in cases)
    assert file_hash(checkpoint) == before
    assert (tmp_path / 'reports/tiny-diagnostic-head.pt').exists()
