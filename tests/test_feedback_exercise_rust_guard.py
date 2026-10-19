from copy import deepcopy
from unittest.mock import Mock

import pytest

from slm_lab.exercise_rust_evaluation_v2 import PROTOCOL, measure_response
from slm_lab.feedback_exercise_rust_guard import enforce, validate_coverage
from test_exercise_rust_evaluation import report, success


def fixture():
    before = report(); before['protocol'] = PROTOCOL; before['declared_generation_budget'] = 2048
    after = deepcopy(before); after['checkpoint_sha256'] = 'candidate'
    references = [{'id': row['id'], 'group_id': row['group_id'],
        'original_test_sha256': row['original_test_sha256'], 'original_reference_file_sha256': {'src/lib.rs': 'original'},
        'result': {'inventory': {'cases': ['original::test']}}} for row in before['results']]
    result = {'base_checkpoint_sha256': 'own', 'candidate_checkpoint_sha256': 'candidate',
              'checks': {}, 'failed_checks': [], 'eligible_for_selection': True}
    return before, after, references, result


def test_original_complete_failed_policy_can_be_compared_without_capability_claim():
    before, after, references, result = fixture()
    assert enforce(result, before, after, references)['eligible_for_selection']


def test_rust_tie_cannot_hide_lost_module_and_cannot_override_older_failed_guard():
    before, after, references, result = fixture(); success(before, 0); success(after, 1)
    for value, index in [(before, 0), (after, 1)]:
        value['results'][index]['execution'].update(compilation_passed=True, compilation=[{'returncode': 0, 'limit': None}])
    guarded = enforce(result, before, after, references)
    assert not guarded['eligible_for_selection'] and guarded['rust_module_guard']['previously_correct_modules_lost'] == ['0']
    before, after, references, result = fixture(); result['eligible_for_selection'] = False
    assert not enforce(result, before, after, references)['eligible_for_selection']


def test_original_full_case_inventory_and_compilation_required():
    before, after, references, result = fixture(); success(after, 0)
    with pytest.raises(ValueError): validate_coverage(after, references)
    after['results'][0]['execution'].update(compilation_passed=True, compilation=[{'returncode': 0, 'limit': None}])
    references[0]['result']['inventory']['cases'].append('original::another_test')
    with pytest.raises(ValueError): validate_coverage(after, references)


def test_wrong_policy_and_partial_source_population_rejected():
    before, after, references, result = fixture(); result['candidate_checkpoint_sha256'] = 'wrong'
    with pytest.raises(ValueError): enforce(result, before, after, references)
    with pytest.raises(ValueError): validate_coverage(before, references[:-1])


def test_changed_original_source_test_or_interface_rejected():
    before, after, references, result = fixture(); references[0]['original_test_sha256'] = {'changed.rs': 'different'}
    with pytest.raises(ValueError): validate_coverage(before, references)


def test_generated_original_cargo_change_is_failed_contract_without_running_it():
    raw = 'File: src/lib.rs\n```rust\npub fn f() {}\n```\nFile: Cargo.toml\n```toml\n[package]\nname = "changed"\n```'
    runner = Mock(); value = measure_response(raw, ['src/lib.rs', 'Cargo.toml'], runner,
        {'tests': {'Cargo.toml': b'[package]\nname = "original"\n'}}, {})
    assert value['full_file_delivery'] and not value['compilation_passed']
    assert 'package_contract_error' in value; runner.run.assert_not_called()


def test_actual_backend_failure_propagates_and_is_not_scored_as_model_failure():
    runner = Mock(); runner.run.side_effect = RuntimeError('Compiler integrity changed')
    raw = 'File: src/lib.rs\n```rust\npub fn f() {}\n```'
    with pytest.raises(RuntimeError): measure_response(raw, ['src/lib.rs'], runner, {'tests': {}}, {})
