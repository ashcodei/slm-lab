import pytest

from slm_lab.exercise_probe import exposure_plan, validate_splits


def row(name, group, language='Python', split='train'):
    return {'id': name, 'group_id': group, 'language': language, 'split': split, 'prompt': 'task', 'response': 'complete source'}


def test_another_language_port_cannot_leak_a_training_exercise_into_validation():
    with pytest.raises(ValueError, match='Cross-language'):
        validate_splits([row('py-one', 'same')], [row('cpp-one', 'same', 'C++', 'valid')])
    validate_splits([row('py-one', 'train-group')], [row('cpp-one', 'held-group', 'C++', 'valid')])


def test_shuffled_training_remainder_is_counted_and_keeps_every_full_module():
    rows = [row(str(i), str(i), 'C++' if i % 2 else 'Python') for i in range(9)]
    plan, records = exposure_plan(rows, list(range(1, 10)))
    assert plan['steps'] == 5 and plan['sample_exposures'] == 40 and plan['remainder_exposures'] == 4
    assert plan['minimum_implementation_exposures'] == 4 and plan['maximum_implementation_exposures'] == 5
    assert plan['planned_target_exposures'] == sum(int(record['id']) + 1 for record in records)
    assert set(record['id'] for record in records) == {r['id'] for r in rows}
    assert plan['exposures_by_language']['C++'] + plan['exposures_by_language']['Python'] == 40


def test_incomplete_target_accounting_cannot_start_training():
    with pytest.raises(ValueError): exposure_plan([row('one', 'one')], [])
    with pytest.raises(ValueError): exposure_plan([row('one', 'one')], [0])
