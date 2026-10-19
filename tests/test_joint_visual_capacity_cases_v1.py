"""Whole-source selection controls; no neural or full-size admission claims."""
import copy

import numpy as np
import pytest

from slm_lab.joint_visual_capacity_cases_v1 import choose, FAMILIES, DTYPE, digest


def sources():
    names = list(FAMILIES)
    rows, entries, words = [], [], []
    for number, name in enumerate(names):
        count = 129 if name == 'words' else 64
        for position in range(count):
            row = np.zeros((), dtype=DTYPE)
            row['family'], row['original_local_position'] = number, position
            length = 35 if name == 'words' and position < 64 else 21
            if name == 'images:own_context_photo_reading' and position == 3:
                length = 2952
            row['context_tokens'], row['response_EOS_targets'] = length, 4
            rows.append(row)
            if name == 'words':
                words.append(dict(original_local_position=position, source_role='train',
                    whole_word_UTF8_bytes=50 if position == 77 else 5,
                    minimum_repeated_byte_CTC_steps=56 if position == 77 else 5,
                    whole_word_UTF8_sha256='a' * 64, complete_context_tokens=length,
                    response_EOS_targets=4, byte_family='real_word'))
        for begin in range(0, count, 64):
            positions = list(range(begin, min(begin + 64, count)))
            selected = rows[-count:][begin:begin + 64]
            entries.append(dict(step=len(entries) + 1, family=name, original_local_positions=positions,
                complete_context_tokens=sum(int(r['context_tokens']) for r in selected),
                response_EOS_targets=4 * len(positions),
                proposed_rank_source_counts=[len(positions[r::4]) for r in range(4)]))
    return names, np.asarray(rows, dtype=DTYPE), entries, words


def test_all_nine_families_and_actual_worst_shapes_preserve_complete_original_entries():
    names, index, entries, words = sources()
    original = copy.deepcopy(entries)
    plan = choose(names, index, entries, words)
    assert entries == original
    assert plan['actual_original_epoch_source_views'] == 641
    assert plan['maximum_original_native_context'] == 2952
    assert plan['maximum_original_photo_negative_set'] == 64
    assert plan['maximum_original_word_UTF8_bytes'] == 50
    assert plan['longest_original_word_local_position'] == 77
    assert plan['actual_full_size_GPU_capacity_passed'] is False and plan['training_admitted'] is False
    assert {e['family'] for e in plan['capacity_entries']} == set(FAMILIES)
    assert [FAMILIES[e['family']] for e in plan['capacity_entries'][:3]] == ['native', 'photo', 'word']
    assert plan['capacity_entries'][-1]['original_local_positions'] == [128]
    assert plan['capacity_entries'][-1]['proposed_rank_source_counts'] == [1, 0, 0, 0]
    for case, capacity in zip(plan['capacity_cases'], plan['capacity_entries']):
        original = entries[case['original_complete_epoch_step'] - 1]
        assert case['original_complete_entry_sha256'] == digest(original)
        assert {k: v for k, v in original.items() if k != 'step'} == {k: v for k, v in capacity.items() if k != 'step'}


@pytest.mark.parametrize('case', ['duplicate_occurrence', 'missing_tail', 'wrong_budget', 'wrong_rank_counts',
    'held_word', 'partial_word', 'unknown_family', 'wrong_word_mask'])
def test_partial_or_changed_original_populations_and_targets_reject(case):
    names, index, entries, words = sources()
    if case == 'duplicate_occurrence':
        entries[-1]['original_local_positions'] = [77]
    elif case == 'missing_tail':
        entries.pop()
    elif case == 'wrong_budget':
        entries[0]['response_EOS_targets'] -= 1
    elif case == 'wrong_rank_counts':
        entries[-1]['proposed_rank_source_counts'] = [0, 1, 0, 0]
    elif case == 'held_word':
        words[77]['source_role'] = 'validation'
    elif case == 'partial_word':
        words.pop()
    elif case == 'unknown_family':
        index['family'][0] = 999
    elif case == 'wrong_word_mask':
        words[77]['response_EOS_targets'] -= 1
    with pytest.raises(ValueError):
        choose(names, index, entries, words)
