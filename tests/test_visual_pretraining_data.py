import json
from pathlib import Path

from PIL import Image
import pytest

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import pixel_hash
from slm_lab.visual_pretraining_data import prepare


def test_full_images_preserve_sources_and_protect_old_heldout_and_previous_training(tmp_path):
    base, commons = tmp_path / 'base', tmp_path / 'commons'
    base.mkdir(); commons.mkdir()

    def row(root, name, color, task='caption', source='everyday_photos'):
        path = root / (name + '.png')
        image = Image.new('RGB', (32, 24), color)
        image.save(path)
        return {'id': name, 'group_id': name, 'image': path.name, 'stored_image_sha256': file_hash(path),
                'model_pixel_sha256': pixel_hash(image), 'model_pixel_sha256_448': pixel_hash(image, 448),
                'curriculum_source': source, 'task': task, 'prompt': 'Describe this image.',
                'response': name, 'license': 'CC-BY-2.0', 'author': 'fixture author'}

    train = [row(base, 'old-train', 'red'), row(base, 'full-scene', 'green', 'region_ocr'),
             row(base, 'clothing', 'purple', 'category', 'fashion'), row(base, 'word-crop', 'cyan', 'word_ocr')]
    valid = [row(base, 'old-valid', 'blue')]
    new_train = [row(commons, 'new-diagram', 'orange', source='commons'),
                 row(commons, 'duplicate-heldout', 'blue', source='commons')]
    new_valid = [row(commons, 'already-trained', 'red', source='commons'),
                 row(commons, 'new-valid', 'yellow', source='commons')]
    # Original Commons journals omit task/source labels; the assembly supplies them.
    for record in new_train + new_valid:
        record.pop('task')
        record.pop('curriculum_source')
    for root, pairs in [(base, [('images.train.jsonl', train), ('images.valid.jsonl', valid)]),
                        (commons, [('train.jsonl', new_train), ('valid.jsonl', new_valid)])]:
        for name, rows in pairs:
            (root / name).write_text(''.join(json.dumps(record) + '\n' for record in rows))
        (root / 'manifest.json').write_text(json.dumps({'images': sum(len(rows) for _, rows in pairs),
                              'files_sha256': {name: file_hash(root / name) for name, _ in pairs}}))
    before = {path: file_hash(path) for root in [base, commons] for path in root.iterdir()}
    output = tmp_path / 'prepared'
    manifest = prepare(base, commons, output)
    assert manifest['splits']['train']['images'] == 3 and manifest['splits']['valid']['images'] == 2
    assert manifest['fashion_training_images'] == 0 and manifest['parent_group_overlap'] == 0
    assert all(value == 0 for value in manifest['exact_input_pixel_overlap'].values())
    assert manifest['excluded']['exact held-out input pixels'] == 1
    assert manifest['excluded']['additional validation image was used in earlier training'] == 1
    assert all(file_hash(path) == digest for path, digest in before.items())
    rows = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
    assert all(Path(record['image']).is_absolute() and record['author'] == 'fixture author' for record in rows)
    assert all(record['task'] in {'caption', 'region_ocr'} for record in rows)
    assert prepare(base, commons, output) == manifest
    (output / 'train.jsonl').write_text('tampered')
    with pytest.raises(ValueError, match='records changed'):
        prepare(base, commons, output)
