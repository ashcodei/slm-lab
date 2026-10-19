"""Require full actual source populations before admitting visual data only.

This is not GPU/runtime/model admission. Passing small representative tensor
controls, changing weights or supplying a generic passed flag cannot replace
any complete source population or original budget below.
"""
FAMILIES = {'images:additional_photo_captions': 78255, 'images:additional_real_words': 23847,
    'images:additional_commons': 987, 'images:original_visual_reasoning': 40000,
    'images:human_photo_reading': 5858, 'images:own_context_photo_reading': 5858,
    'replay:generic_visual': 21024, 'replay:image_reading': 35213, 'words': 91585}


def require_counts(record, expected, context):
    if not isinstance(record, dict) or any(type(record.get(k)) is not int or record[k] != v for k, v in expected.items()):
        raise ValueError('The complete original ' + context + ' population/budget is required')


def validate(index, index_reader, pixels, pixel_reader, batches, corrections, words):
    if any(not value.get('complete') for value in [index, index_reader, pixels, pixel_reader, batches, words]):
        raise ValueError('Every original source and independent reader must be complete')
    if any(value.get('passed') is not True for value in [pixels, pixel_reader, batches, words]):
        raise ValueError('Every full physical, batch and whole-word observation must actually pass')
    if (index.get('training_admitted') is not False or index_reader.get('training_admitted') is not False
            or pixel_reader.get('training_admitted') is not False or batches.get('training_admitted') is not False
            or pixel_reader.get('source_role') != 'train' or batches.get('source_role') != 'train'):
        raise ValueError('Exact original unadmitted train-only data evidence required')
    require_counts(index, dict(all_original_train_views=302627, actual_context_tokens=17681556,
        actual_response_EOS_targets=4669243, actual_original_negative_targets_retained=80,
        actual_whole_CTC_UTF8_byte_targets=644063), 'corrected index')
    require_counts(index_reader, dict(actual_all_original_entries=302627, actual_current_positive_native_entries_changed=4,
        actual_context_tokens=17681556, actual_response_EOS_targets=4669243,
        actual_whole_word_UTF8_bytes=644063, actual_epoch_updates=4733,
        actual_distinct_stored_image_paths=195121), 'index readback')
    if (index_reader.get('actual_family_views') != FAMILIES
            or index_reader.get('actual_all_complete_raw_source_ranges_hashed') is not True
            or index_reader.get('actual_all91585packed_word_sequences_and_complete_CTC_bytes_rechecked') is not True
            or index_reader.get('actual_all_original_occurrences_once') is not True):
        raise ValueError('Every original source byte range, packed word and family occurrence must be independently read')
    require_counts(pixels.get('actual_counts'), dict(original_physical_positions=195121, pixel_integrity_passes=195121), 'physical pixel')
    require_counts(pixels, dict(actual_physical_states_rechecked=195121, actual_physical_files_changed_after_observation=0), 'physical end state')
    require_counts(pixel_reader, dict(actual_original_physical_paths=195121, actual_original_native_views=302627,
        actual_original_whole_word_views=91585, actual_fresh_source_states_rechecked=195121,
        actual_end_file_state_differences=0), 'fresh physical readback')
    require_counts(pixel_reader.get('actual_counts'), dict(original_physical_positions=195121,
        fresh_integrity_passes=195121, original_word_views=91585), 'fresh physical disposition')
    if (pixel_reader.get('actual_family_views') != FAMILIES
            or pixel_reader.get('all_original_protocol_expectations_independently_rebuilt') is not True):
        raise ValueError('Fresh pixel readback must reconstruct every original family and protocol expectation')
    require_counts(batches, dict(actual_complete_global_steps=4733, actual_full_source_views=302627), 'rank batch')
    require_counts(batches.get('actual_totals'), dict(actual_rank_calls=18932, actual_native_rank_batches=18929,
        actual_empty_ranks=3, original_source_views=302627, whole_word_views=91585, whole_word_UTF8_bytes=644063,
        actual_whole_photo_matching_pairs=78255, maximum_native_context=2952), 'delivered rank batch')
    if (batches.get('all302627whole_original_current_views_once_and_all4rank_deliveries_rechecked') is not True
            or batches.get('all_native_tensors_bitexact_to_established_encoders') is not True
            or batches.get('all4complete_original_preference_negatives_preserved') is not True):
        raise ValueError('Every actual full rank tensor and all four complete original negatives must be retained')
    original_budgets, delivered = index.get('actual_family_budgets', {}), batches.get('actual_family_budgets', {})
    if set(original_budgets) != set(FAMILIES) or set(delivered) != set(FAMILIES):
        raise ValueError('All nine complete original visual families must be retained')
    for name, count in FAMILIES.items():
        require_counts(original_budgets[name], dict(original_views=count), 'original family ' + name)
        for field in ['original_views', 'complete_context_tokens', 'response_EOS_targets']:
            if type(delivered[name].get(field)) is not int or delivered[name][field] != original_budgets[name][field]:
                raise ValueError('Every actual current family context/EOS budget must match the original corrected index')
    if (sum(v['complete_context_tokens'] for v in delivered.values()) != 17681556
            or sum(v['response_EOS_targets'] for v in delivered.values()) != 4669243):
        raise ValueError('Complete current context and true response/EOS budgets are required')
    require_counts(corrections, dict(exact_original_prepared_views=4, positive_response_EOS_targets=20,
        complete_original_negative_targets=80), 'admitted four-correction')
    if corrections.get('training_admitted') is not True or corrections.get('admitted_role') != 'train':
        raise ValueError('Only the separately independently qualified four current source corrections may be used')
    require_counts(words.get('actual_counts'), dict(words=91585, UTF8_bytes=644063,
        context_tokens=1810266, response_EOS_targets=436491), 'whole-word native')
    if words.get('all_whole_native_responses_decode_to_original_transcripts') is not True:
        raise ValueError('Every whole word native response must decode to its complete original UTF8transcript')
