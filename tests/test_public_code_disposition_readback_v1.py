import copy
import hashlib
import json

import pytest

from slm_lab.held_text_overlap_v1 import normalize
from slm_lab.public_code_disposition_readback_v1 import rebuild, row_sha, verify_observation
from slm_lab.public_code_instruction_screen_v1 import parse_whole, observe
from slm_lab.public_synthetic_code_screen_policy_v1 import body_fingerprint, supervision_scope


class BytesTokenizer:
    def encode(self, value, add_special_tokens):
        assert add_special_tokens is False
        return type('Encoding', (), dict(ids=[v + 4 for v in value.encode()]))()

    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens is False
        return bytes(v - 4 for v in ids).decode()


def fixture():
    row = dict(id='control', input='Return x plus one. Unicode: café\u2028line',
        output='```python\ndef f(x):\n    return x + 1\n```',
        unit_tests=json.dumps(['assert f(2) == 3', 'assert f(-1) == 0']),
        tests_execution_status=json.dumps(['pass', 'pass']), average_test_score=1, domain='generic')
    parsed = parse_whole(row)
    observed = dict(all_cases_observed=True, all_published_tests_passed=True,
        expected_outputs_sent_to_worker=False, whole_original_tests=2, published_tests_are_synthetic=True,
        independent_semantic_correctness_not_certified=True,
        program_sha256=hashlib.sha256(parsed['program'].encode()).hexdigest(),
        code_bytes_unchanged_after_fence_extraction=True, OS_confinement='Landlock and seccomp before interpreter startup',
        execution=dict(status='completed', stdout_sha256='0' * 64), status='all_published_tests_pass',
        passed_tests=2, test_passes=[True, True], actual_value_literals=['3', '0'])
    prior = dict(original_shard_position=0, source_id='control', original_entry_sha256=row_sha(row),
        source_domain='generic', scope=supervision_scope(), screen_pass_candidate=True, protected_overlap=[],
        input_only_group_sha256=hashlib.sha256(normalize(row['input']).encode()).hexdigest(),
        complete_context_tokens=2 + len(('User: ' + row['input'] + '\nAssistant:').encode()) + len(row['output'].encode()),
        response_EOS_targets=len(row['output'].encode()) + 1, complete_original_tests=2,
        program_AST_without_docstrings_sha256=body_fingerprint(parsed['program']),
        execution_reused_from_exact_fresh_pilot_readback=False, actual_whole_execution=observed,
        status='weak_SFT_screen_pass_candidate')
    veto = dict(original_entry_sha256='f' * 64, evidence_path='/control/semantic-veto.json')
    return row, prior, parsed, veto


def test_original_multibyte_whole_response_and_eos_rebuild():
    row, prior, parsed, veto = fixture()
    result, native, actual = rebuild(row, 0, [], BytesTokenizer(), veto, 'f' * 64, prior)
    assert result == prior and actual == parsed
    assert native['response_EOS_targets'] == len(row['output'].encode()) + 1
    assert native['target_start'] + native['response_EOS_targets'] == native['complete_context_tokens']
    assert len(native['native_input_u16le_sha256']) == 64


@pytest.mark.parametrize('change', ['false_pass', 'missing_output', 'wrong_source', 'missing_test', 'special_scope'])
def test_forged_parent_observations_rejected(change):
    row, prior, parsed, veto = fixture()
    if change == 'false_pass':
        prior['actual_whole_execution']['actual_value_literals'][0] = '4'
    elif change == 'missing_output':
        prior['actual_whole_execution']['actual_value_literals'].pop()
    elif change == 'wrong_source':
        row['input'] += ' changed'
    elif change == 'missing_test':
        row['unit_tests'] = json.dumps(['assert f(2) == 3'])
    else:
        prior['scope']['human_preferences'] = True
    with pytest.raises(ValueError):
        rebuild(row, 0, [], BytesTokenizer(), veto, 'f' * 64, prior)


def test_exact_pilot_receipt_required_when_reuse_is_claimed():
    row, prior, parsed, veto = fixture()
    prior['execution_reused_from_exact_fresh_pilot_readback'] = True
    with pytest.raises(ValueError):
        rebuild(row, 0, [], BytesTokenizer(), veto, 'f' * 64, prior)
    pilot = dict(fresh_whole_program_execution=copy.deepcopy(prior['actual_whole_execution']))
    rebuild(row, 0, [], BytesTokenizer(), veto, 'f' * 64, prior, pilot)


def test_passing_known_bad_body_is_vetoed_before_replay():
    row, prior, parsed, veto = fixture()
    prior.pop('actual_whole_execution')
    prior.pop('execution_reused_from_exact_fresh_pilot_readback')
    prior.update(status='known_independent_semantic_veto', screen_pass_candidate=False, veto_evidence=veto['evidence_path'])
    rebuild(row, 0, [], BytesTokenizer(), veto, body_fingerprint(parsed['program']), prior)


def test_fresh_actual_confined_program_and_each_published_assertion():
    _, _, parsed, _ = fixture()
    actual = observe(parsed, seconds=3)
    verify_observation(parsed, actual)
    assert actual['all_published_tests_passed'] is True
    assert actual['actual_value_literals'] == ['3', '0']
    actual['actual_value_literals'] = ['3', '1']
    with pytest.raises(ValueError):
        verify_observation(parsed, actual)


def test_runtime_abstention_cannot_be_marked_test_pass():
    _, prior, parsed, _ = fixture()
    actual = prior['actual_whole_execution']
    actual.update(status='runtime_resource_or_confinement_abstention', all_cases_observed=False,
        all_published_tests_passed=False, execution=dict(status='timeout', stdout_sha256='0' * 64))
    verify_observation(parsed, actual)
    actual['all_published_tests_passed'] = True
    with pytest.raises(ValueError):
        verify_observation(parsed, actual)
