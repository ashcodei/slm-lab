import json
from pathlib import Path

import pytest

from slm_lab.corpus import file_hash
from slm_lab.evaluation_provenance import capture
from slm_lab.textvqa_admission import admit


def audited_inputs(root):
    data = root / 'data/textvqa-human-reading-v1'; data.mkdir(parents=True)
    sources = []
    for name in ['textvqa_data', 'textvqa_input_audit']:
        source = root / 'src/slm_lab' / (name + '.py'); source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('# recorded source ' + name); sources.append(source)
    files = {}; counts = {}; images = []
    for split in ['train', 'valid']:
        image = data / (split + '.png'); image.write_bytes(b'original stored photo ' + split.encode()); images.append(image)
        row = {'id': split + '-question', 'group_id': split + '-photo', 'image': str(image),
               'stored_image_sha256': file_hash(image), 'target_tokens': 2}
        path = data / (split + '.jsonl'); path.write_text(json.dumps(row) + '\n'); files[path.name] = file_hash(path)
        counts[split] = {'examples': 1, 'original_image_groups': 1, 'target_tokens': 2}
    manifest = {'implementation_sha256': file_hash(sources[0]), 'files_sha256': files, 'splits': counts,
                'tokenizer_sha256': 'recorded-own-tokenizer'}
    manifest_path = data / 'manifest.json'; manifest_path.write_text(json.dumps(manifest))
    report = {'passed': True, 'manifest_sha256': file_hash(manifest_path),
              'preparation_source_sha256': file_hash(sources[0]), 'audit_source_sha256': file_hash(sources[1]),
              'files_sha256': files, 'tokenizer_sha256': manifest['tokenizer_sha256'],
              'validation_group_overlap': 0, 'actual_model_pixel_sizes_rechecked': [224, 448],
              'implementation_runtime': capture(sources), 'original_source_and_inventory_sha256': {}, 'splits': counts}
    report_path = root / 'reports/textvqa-original-input-audit-v1.json'; report_path.parent.mkdir()
    report_path.write_text(json.dumps(report))
    return data, images


def test_post_audit_actual_photo_mutation_is_caught_before_training(tmp_path):
    _, images = audited_inputs(tmp_path)
    rows, receipt = admit(tmp_path)
    assert len(rows) == 1 and receipt['actual_photo_bytes_rehashed'] == 2
    images[0].write_bytes(b'replaced photo after audit')
    with pytest.raises(ValueError, match='photo bytes changed'):
        admit(tmp_path)


def test_rehashed_changed_question_manifest_cannot_reuse_the_previous_passed_audit(tmp_path):
    data, _ = audited_inputs(tmp_path)
    admit(tmp_path)
    path = data / 'train.jsonl'; row = json.loads(path.read_text()); row['target_tokens'] = 999
    path.write_text(json.dumps(row) + '\n')
    manifest_path = data / 'manifest.json'; manifest = json.loads(manifest_path.read_text())
    manifest['files_sha256']['train.jsonl'] = file_hash(path); manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='source audit differs'):
        admit(tmp_path)
