import random

import pytest

from slm_lab.codenet_candidate_selection_v1 import select


def row(number, user, language='Python', status='Accepted', size='100'):
    return {'submission_id': f's{number:09d}', 'problem_id': 'p00001',
            'user_id': user, 'language': language, 'status': status, 'code_size': size}


def test_source_order_does_not_change_distinct_author_and_submission_choices():
    rows = [row(i, 'u' + str(i // 3)) for i in range(1, 100)]
    expected, counts = select(rows, 'p00001')
    random.Random(19).shuffle(rows)
    actual, actual_counts = select(iter(rows), 'p00001')
    assert actual == expected and actual_counts == counts
    assert len(actual) == 4
    assert len({r['metadata']['user_id'] for r in actual}) == 4
    for chosen in actual:
        original = [r for r in rows if r['user_id'] == chosen['metadata']['user_id']]
        assert chosen['metadata']['submission_id'] == min(r['submission_id'] for r in original)


def test_languages_keep_separate_budgets_and_rejections_are_not_positive_targets():
    rows = [row(i, f'u{i}', language) for i in range(1, 12) for language in ['C++', 'Python', 'Rust']]
    rows += [row(99, 'rejected', status='Wrong Answer'), row(100, 'large', size='50000'), row(101, 'empty', size='0')]
    chosen, counts = select(rows, 'p00001', per_language=2)
    assert len(chosen) == 6
    assert {r['metadata']['language'] for r in chosen} == {'C++', 'Python', 'Rust'}
    assert counts == {'rows': 36, 'accepted': 35, 'accepted_within_declared_byte_bound': 33}
    assert not {'rejected', 'large', 'empty'} & {r['metadata']['user_id'] for r in chosen}


def test_metadata_identity_mismatch_stops_selection():
    with pytest.raises(ValueError, match='problem identity'):
        select([row(1, 'u1')], 'p00002')
