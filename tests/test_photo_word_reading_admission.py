import json
from pathlib import Path

import pytest
import torch

from scripts import assess_photo_word_reading as assessment
from slm_lab.corpus import file_hash
from slm_lab.evaluation_provenance import capture


def completed(tmp_path, monkeypatch, step=10):
    monkeypatch.setattr(assessment, 'ROOT', tmp_path)
    name = 'own-component'; reports = tmp_path / 'reports' / name; checkpoints = tmp_path / 'checkpoints' / name
    reports.mkdir(parents=True); checkpoints.mkdir(parents=True)
    source = tmp_path / 'implementation.py'; source.write_text('own_source = 1\n')
    plan = {'steps': 10, 'source_provenance': capture([source])}
    (reports / 'status.json').write_text(json.dumps({'status': 'complete'})); (reports / 'plan.json').write_text(json.dumps(plan))
    model = checkpoints / 'model.pt'; torch.save({'model': {'weight': torch.ones(1)}, 'plan': plan, 'step': step}, model)
    (reports / 'comparison.json').write_text(json.dumps({'model_sha256': file_hash(model)}))
    return name, reports, model, source


def test_admission_requires_exact_completed_steps_and_source_state(tmp_path, monkeypatch):
    name, _, _, _ = completed(tmp_path, monkeypatch)
    snapshot, plan, digest = assessment.verified_state(name, 'complete')
    assert snapshot['step'] == plan['steps'] == 10 and len(digest) == 64


def test_partial_snapshot_cannot_enter_whole_photo_composition(tmp_path, monkeypatch):
    name, _, _, _ = completed(tmp_path, monkeypatch, 9)
    with pytest.raises(ValueError, match='did not complete'): assessment.verified_state(name, 'complete')


def test_changed_component_source_or_checkpoint_cannot_enter_whole_photo_composition(tmp_path, monkeypatch):
    name, _, model, source = completed(tmp_path, monkeypatch)
    source.write_text('different_source = 2\n')
    with pytest.raises(ValueError, match='implementation/runtime changed'): assessment.verified_state(name, 'complete')
    source.write_text('own_source = 1\n'); model.write_bytes(b'different checkpoint')
    with pytest.raises(ValueError, match='not the declared completed model'): assessment.verified_state(name, 'complete')


def test_failed_final_component_status_cannot_enter_whole_photo_composition(tmp_path, monkeypatch):
    name, reports, _, _ = completed(tmp_path, monkeypatch)
    (reports / 'status.json').write_text(json.dumps({'status': 'failed'}))
    with pytest.raises(ValueError, match='not the declared completed model'): assessment.verified_state(name, 'complete')


def test_composed_reading_preserves_all_actual_photo_and_counterpart_inputs():
    rows = [{'id': str(index), 'group_id': 'group' + str(index), 'stored_image_sha256': 'file' + str(index),
             'model_pixel_sha256_768': 'pixels' + str(index)} for index in range(2)]
    cases = [{**row, 'input_pixels_sha256_768': row['model_pixel_sha256_768'],
              'wrong_image_id': rows[1 - index]['id'], 'wrong_pixels_sha256_768': rows[1 - index]['model_pixel_sha256_768']}
             for index, row in enumerate(rows)]
    report = {'full_photos': 2, 'results': cases}
    assert len(assessment.matching_localization_cases(rows, report)) == 2
    cases[0]['input_pixels_sha256_768'] = 'different-image'
    with pytest.raises(ValueError, match='bytes/pixels/counterpart'): assessment.matching_localization_cases(rows, report)
    cases[0]['input_pixels_sha256_768'] = rows[0]['model_pixel_sha256_768']; cases[0]['wrong_image_id'] = '0'
    with pytest.raises(ValueError, match='bytes/pixels/counterpart'): assessment.matching_localization_cases(rows, report)


def test_composed_reading_cannot_silently_reduce_full_photo_coverage():
    rows = [{'id': '0'}, {'id': '1'}]
    with pytest.raises(ValueError, match='coverage'): assessment.matching_localization_cases(rows, {'full_photos': 1, 'results': []})
    with pytest.raises(ValueError, match='cases'): assessment.matching_localization_cases(rows, {'full_photos': 2, 'results': [{'id': '0'}, {'id': '0'}]})
