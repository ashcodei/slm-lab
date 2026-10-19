"""Real missing/failed/CPU scopes cannot be upgraded into full data admission."""
import json
from pathlib import Path

import pytest

from slm_lab.native_suffix_train_data_v1 import NativeSuffixTrainData
from slm_lab.recovered_native_suffix_train_data_v3 import load_recovered_native_suffix_data, check_admission

ROOT = Path(__file__).resolve().parents[1]


def test_missing_full_recovered_native_admission_is_explicitly_rejected(tmp_path):
    for purpose in ['optimization', 'capacity', 'CPU-mechanics', 'evaluation', 'test']:
        with pytest.raises(ValueError, match='admission'): load_recovered_native_suffix_data(tmp_path, purpose=purpose)


def test_actual_original_failed_admission_or_recovered_legacy_reader_is_not_native_data_admission():
    original = ROOT / 'reports/native-human-code-suffix-data-admission-CPU-v1'
    failure = json.loads((original / 'failure.json').read_text())
    assert failure['owner'] == dict(pid=167672, ticks='56126055')
    prior = ROOT / 'reports/native-human-code-suffix-data-admission-CPU-v2'
    assert json.loads((prior / 'failure.json').read_text())['owner'] == dict(pid=188617, ticks='57073359')
    assert not (prior / 'complete.json').exists()
    source = json.loads((ROOT / 'reports/full-public-token-suffix-readback-gzip-recovery-CPU-v4/complete.json').read_text())
    plan = json.loads((ROOT / 'reports/full-public-token-suffix-readback-gzip-recovery-CPU-v4/plan.json').read_text())
    assert source['actual_counts']['complete_original_sources'] == 3721
    with pytest.raises(ValueError): check_admission(source, source, source, plan['owner'], admission_sha256='a'*64, manifest_sha256='b'*64)


def test_actual_nine_presentation_CPU_fixture_remains_one_source_and_cannot_admit_recovered_full_source_data():
    controls = ROOT / 'reports/native-suffix-CPU-mechanics-qualification-v1/closed-qualified-controls.json'
    data = NativeSuffixTrainData.CPU_mechanics(ROOT, controls)
    assert len(data) == 9 and data.full_source_data_admitted is False and data.admission_sha256 is None
    assert len({data.information(p)['original_source_id'] for p in data.slots}) == 1
    proof = json.loads(controls.read_text())
    with pytest.raises(ValueError): check_admission(proof, proof, proof, dict(pid=1,ticks='0'), admission_sha256='a'*64, manifest_sha256='b'*64)
