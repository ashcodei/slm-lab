import json
from pathlib import Path
import sys

import pytest

from slm_lab.corpus import file_hash
from slm_lab import public_pretraining_assessment_recovery_v1 as gate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from recover_public_pretraining_assessments_v1 import check_population


def write(root, name, value):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value))
    return p


def fixture(root):
    public = gate.PUBLIC
    failed = gate.FAILED
    write(root, failed + '/plan.json', dict(owner=dict(gate.ORIGINAL, state='R')))
    write(root, failed + '/children.json', [dict(pid=99, ticks='9')])
    write(root, failed + '/restored.json', dict(all_registered_point_actors_terminal=True))
    write(root, failed + '/failure.json', dict(error="ValueError('An exact owned child failed; preserve partial evidence')"))
    p = root / failed / 'assess-initial-broad-2.log'
    p.write_text('TypeError: observe.<locals>.generate() takes 1 positional argument but 2 were given\n')
    plan = write(root, public + '/plan.json', dict(planned_steps=4447, selected_source_budget={'full': 17}))
    write(root, public + '/training-complete.json', dict(actual_updates=4447,
        all_new_admitted_raw_train_windows_once=True, all826verified_train_functions_once=True,
        actual_selected_source_budget={'full': 17}))
    write(root, public + '/native-assessment-inputs.json', {a: dict(checkpoint_sha256=a) for a in ['initial', 'trained']})
    bindings = {str(plan): file_hash(plan)}
    bindings.update({str(root / 'checkpoints/public-pretraining-v1' / a / 'model.pt'): a for a in ['initial', 'trained']})
    write(root, gate.EPOCH, dict(passed=True, actual_complete_updates=4447,
        actual_unique_neural_values=984399690, source_files_sha256=bindings))


def test_completed_epoch_gate_accepts_only_exact_terminal_births_and_preserved_failure(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.setattr(gate, 'terminal', lambda birth: True)
    assert gate.completed_training_after_observer_failure(tmp_path)
    monkeypatch.setattr(gate, 'terminal', lambda birth: birth['pid'] != 99)
    assert not gate.completed_training_after_observer_failure(tmp_path)
    monkeypatch.setattr(gate, 'terminal', lambda birth: True)
    write(tmp_path, gate.PUBLIC + '/training-complete.json', dict(actual_updates=4446))
    with pytest.raises(ValueError):
        gate.completed_training_after_observer_failure(tmp_path)


def test_failed_or_incomplete_recovery_cannot_release_successors(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.setattr(gate, 'terminal', lambda birth: True)
    assert not gate.assessment_ready(tmp_path)
    write(tmp_path, gate.REPORT + '/plan.json', dict(owner=dict(pid=42, ticks='42')))
    write(tmp_path, gate.REPORT + '/children.json', [])
    write(tmp_path, gate.REPORT + '/failure.json', dict(error='actual failure'))
    with pytest.raises(ValueError, match='assessment recovery failed'):
        gate.assessment_ready(tmp_path)


def test_complete_population_reconciliation_rejects_missing_or_smaller_suites(tmp_path):
    for name, value in [('knowledge', dict(overall=dict(questions=56))),
            ('math', dict(evaluated_examples=16)), ('ocr', dict(overall=dict(examples=20))),
            ('objects', dict(overall=dict(examples=15))), ('vision', dict(images=32))]:
        write(tmp_path, name + '.json', value)
    assert check_population(tmp_path) == dict(knowledge=56, math=16, OCR=20, objects=15, photos=32)
    write(tmp_path, 'knowledge.json', dict(overall=dict(questions=55)))
    with pytest.raises(ValueError, match='complete original'):
        check_population(tmp_path)
