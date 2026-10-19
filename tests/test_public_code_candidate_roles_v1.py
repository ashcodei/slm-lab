import copy
import hashlib

import pytest

from slm_lab.public_code_candidate_roles_v1 import assign_roles, verify_role_separation


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def row(position, question, body):
    return dict(original_shard_position=position, input_only_group_sha256=digest(question),
                program_AST_without_docstrings_sha256=digest(body))


def test_transitive_question_body_chain_stays_together_and_order_does_not_matter():
    # The endpoints share neither key; their middle member still connects them.
    records = [row(7, 'a', 'x'), row(19, 'a', 'y'), row(101, 'b', 'y')]
    roles = assign_roles(records)
    assert roles == assign_roles(reversed(records))
    assert len({r['source_role'] for r in roles.values()}) == 1
    assert {r['declared_duplicate_component_size'] for r in roles.values()} == {3}
    summary = verify_role_separation(records, roles)
    assert summary['declared_duplicate_components'] == 1
    assert summary['declared_input_and_AST_cross_role_intersections'] == 0


def test_all_original_positions_reserved_before_outcomes_and_no_failed_replay_reroll():
    records = [row(i, 'question' + str(i), 'body' + str(i)) for i in range(500)]
    roles = assign_roles(records)
    summary = verify_role_separation(records, roles)
    assert set(roles) == set(range(500))
    assert 0 < summary['reserved_candidates'] < 500
    observed = copy.deepcopy(records)
    for record in observed:
        record['fresh_execution_status'] = 'runtime_abstention' if record['original_shard_position'] % 2 else 'pass'
    assert assign_roles(observed) == roles
    assert all(r['optimization_forbidden'] == (r['source_role'] == 'reserved') for r in roles.values())


@pytest.mark.parametrize('corruption', ['omit', 'change_role', 'change_component', 'change_optimization'])
def test_tampered_roles_are_rejected(corruption):
    records = [row(1, 'q', 'x'), row(2, 'q', 'y'), row(3, 'r', 'z')]
    roles = assign_roles(records)
    if corruption == 'omit':
        del roles[1]
    elif corruption == 'change_role':
        roles[1]['source_role'] = 'reserved' if roles[1]['source_role'] == 'train' else 'train'
    elif corruption == 'change_component':
        roles[1]['declared_duplicate_component_sha256'] = digest('forged')
    else:
        roles[1]['optimization_forbidden'] = not roles[1]['optimization_forbidden']
    with pytest.raises(ValueError):
        verify_role_separation(records, roles)


@pytest.mark.parametrize('records', [[row(1, 'q', 'b'), row(1, 'r', 'c')],
    [dict(row(1, 'q', 'b'), input_only_group_sha256='missing')], [row(-1, 'q', 'b')]])
def test_missing_identity_or_duplicates_rejected(records):
    with pytest.raises(ValueError):
        assign_roles(records)
