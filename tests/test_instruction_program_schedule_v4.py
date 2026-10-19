from collections import Counter
from copy import deepcopy

import pytest

from slm_lab.instruction_program_schedule_v4 import LANGUAGES, schedule


def pools(rows):
    result = {name: [object()] for name in [*LANGUAGES, 'human_instruction']}
    result['verified_python'] = rows
    return result


def test_distinct_algorithms_in_one_file_are_all_seen_before_name_variants():
    rows = [{'group_id': 'same-publisher-file', 'original_program_sha256': program,
             'variant_index': variant}
            for program in ['algorithm-a', 'algorithm-b'] for variant in [2, 1, 0]]
    original = deepcopy(rows)
    selected = [rows[update[2]['index']] for update in schedule(pools(rows), 6, 17)]
    assert {row['original_program_sha256'] for row in selected[:2]} == {'algorithm-a', 'algorithm-b'}
    assert [row['variant_index'] for row in selected] == [0, 0, 1, 1, 2, 2]
    assert rows == original  # File-level train/validation identities remain intact.


def test_language_allocation_and_repeated_schedule_are_preserved():
    source = pools([{'original_program_sha256': 'program', 'variant_index': 0}])
    allocation = schedule(source, 1500, 17)
    counts = Counter(pick['pool'] for update in allocation for pick in update)
    assert all(counts[name] == 500 for name in LANGUAGES)
    assert counts['verified_python'] == counts['human_instruction'] == 1500
    assert allocation == schedule(source, 1500, 17)


@pytest.mark.parametrize('rows', [
    [{'group_id': 'file', 'variant_index': 0}],
    [{'original_program_sha256': 'program', 'variant_index': 1}],
    [{'original_program_sha256': 'program', 'variant_index': 0}] * 2,
])
def test_missing_algorithm_identity_original_or_duplicate_variant_is_rejected(rows):
    with pytest.raises(ValueError):
        schedule(pools(rows), 6, 17)
