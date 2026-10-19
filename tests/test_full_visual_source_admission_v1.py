import copy

import pytest

from slm_lab.full_visual_source_admission_v1 import FAMILIES, validate


def evidence():
    # Structural gate fixture only. No claim that these synthetic records
    # are observations of actual production data or grant training admission.
    budgets = {name: dict(original_views=count, complete_context_tokens=1, response_EOS_targets=1)
        for name, count in FAMILIES.items()}
    budgets['words'].update(complete_context_tokens=17681548, response_EOS_targets=4669235)
    index = dict(complete=True, training_admitted=False, all_original_train_views=302627,
        actual_context_tokens=17681556, actual_response_EOS_targets=4669243,
        actual_original_negative_targets_retained=80, actual_whole_CTC_UTF8_byte_targets=644063,
        actual_family_budgets=budgets)
    reader = dict(complete=True, training_admitted=False, actual_all_original_entries=302627,
        actual_current_positive_native_entries_changed=4, actual_context_tokens=17681556,
        actual_response_EOS_targets=4669243, actual_whole_word_UTF8_bytes=644063, actual_epoch_updates=4733,
        actual_distinct_stored_image_paths=195121, actual_family_views=FAMILIES,
        actual_all_complete_raw_source_ranges_hashed=True,
        actual_all91585packed_word_sequences_and_complete_CTC_bytes_rechecked=True,
        actual_all_original_occurrences_once=True)
    pixels = dict(complete=True, passed=True, actual_counts=dict(original_physical_positions=195121,
        pixel_integrity_passes=195121), actual_physical_states_rechecked=195121,
        actual_physical_files_changed_after_observation=0)
    pixel_reader = dict(complete=True, passed=True, training_admitted=False, source_role='train',
        actual_original_physical_paths=195121, actual_original_native_views=302627,
        actual_original_whole_word_views=91585, actual_fresh_source_states_rechecked=195121,
        actual_end_file_state_differences=0, actual_family_views=FAMILIES,
        all_original_protocol_expectations_independently_rebuilt=True,
        actual_counts=dict(original_physical_positions=195121, fresh_integrity_passes=195121, original_word_views=91585))
    batches = dict(complete=True, passed=True, training_admitted=False, source_role='train',
        actual_complete_global_steps=4733, actual_full_source_views=302627,
        actual_totals=dict(actual_rank_calls=18932, actual_native_rank_batches=18929, actual_empty_ranks=3,
            original_source_views=302627, whole_word_views=91585, whole_word_UTF8_bytes=644063,
            actual_whole_photo_matching_pairs=78255, maximum_native_context=2952),
        all302627whole_original_current_views_once_and_all4rank_deliveries_rechecked=True,
        all_native_tensors_bitexact_to_established_encoders=True,
        all4complete_original_preference_negatives_preserved=True, actual_family_budgets=copy.deepcopy(budgets))
    corrections = dict(training_admitted=True, admitted_role='train', exact_original_prepared_views=4,
        positive_response_EOS_targets=20, complete_original_negative_targets=80)
    words = dict(complete=True, passed=True, actual_counts=dict(words=91585, UTF8_bytes=644063,
        context_tokens=1810266, response_EOS_targets=436491), all_whole_native_responses_decode_to_original_transcripts=True)
    return [index, reader, pixels, pixel_reader, batches, corrections, words]


def test_structural_complete_scope_is_accepted_without_creating_training_admission():
    before = evidence()
    snapshot = copy.deepcopy(before)
    assert validate(*before) is None and before == snapshot
    assert before[0]['training_admitted'] is False and before[4]['training_admitted'] is False


@pytest.mark.parametrize('position,path,value', [
    (3, ['actual_original_physical_paths'], 195120),
    (4, ['actual_complete_global_steps'], 12),
    (4, ['actual_totals', 'whole_word_UTF8_bytes'], 644062),
    (4, ['actual_totals', 'actual_empty_ranks'], 0),
    (4, ['actual_family_budgets', 'replay:image_reading', 'response_EOS_targets'], 0),
    (5, ['complete_original_negative_targets'], 79),
    (1, ['actual_all91585packed_word_sequences_and_complete_CTC_bytes_rechecked'], False),
    (3, ['source_role'], 'valid'),
])
def test_partial_or_held_or_budget_changed_proofs_cannot_admit_original_full_scope(position, path, value):
    records = evidence()
    target = records[position]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate(*records)
