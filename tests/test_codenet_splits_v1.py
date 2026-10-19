import pytest

from slm_lab.codenet_splits_v1 import component_role, partition


def row(problem='p00001', submission='s000000001', digest='a' * 64, group=None):
    return dict(archive_member=f'Project_CodeNet/data/{problem}/Python/{submission}.py',
                problem_id=problem, language='Python', proposed_problem_group=group or problem,
                split=None, context_qualification='complete_original_pair_fits',
                complete_input_ids_sha256=digest, target_tokens=12, context_tokens=30)


def test_aliases_and_copies_keep_fixed_component_role_and_deterministic_representative():
    records = [row(submission='s000000002'), row(), row('p00002', group='p00001')]
    mapping = {'p00001': 'p00001', 'p00002': 'p00001'}
    first, counts = partition(records, mapping)
    assert partition(list(reversed(records)), mapping) == (first, counts)
    assert {r['split'] for r in first} == {component_role('p00001')}
    assert sum(not r['duplicate_complete_example'] for r in first) == 1
    assert all(r['complete_example_representative'] == records[1]['archive_member'] for r in first)
    assert sum(c['unique_target_tokens'] for c in counts.values()) == 12


def test_full_pair_copies_cannot_cross_components_even_when_role_matches():
    with pytest.raises(ValueError, match='crosses problem components'):
        partition([row(), row('p00002')], {'p00001': 'p00001', 'p00002': 'p00002'})


def test_mask_start_is_part_of_duplicate_identity_and_review_rows_keep_roles():
    a, b, c = row(), row(submission='s000000002'), row(submission='s000000003')
    b['target_tokens'] = 13
    c['context_qualification'] = 'original_media_review_needed'
    for key in ['target_tokens', 'context_tokens', 'complete_input_ids_sha256']:
        c.pop(key)
    result, _ = partition([a, b, c], {'p00001': 'p00001'})
    assert [r['duplicate_complete_example'] for r in result] == [False, False, None]
    assert all(not r['dataset_admitted_to_training'] for r in result)


def test_inconsistent_identity_or_past_assignment_is_rejected():
    a = row()
    a['split'] = 'train'
    with pytest.raises(ValueError, match='Unassigned'):
        partition([a], {'p00001': 'p00001'})
    a['split'] = None
    a['problem_id'] = 'p00002'
    with pytest.raises(ValueError, match='identity'):
        partition([a], {'p00001': 'p00001', 'p00002': 'p00002'})
