"""Rebuild complete native arrays directly from original public text.

Independent of the pack-row formatter and token writer. This validates native
text/masks and sealed roles, not source execution, semantics or admission.
"""
import hashlib
import struct


def verify(original, joined, role, row, ids, labels, tokenizer):
    view = row['view']
    if view not in ('code_only', 'reasoning'):
        raise ValueError('Declared full source view required')
    source_role = role['source_role']
    if (source_role not in ('train', 'reserved') or role['role_fixed_before_fresh_replay'] is not True
            or role['optimization_forbidden'] is not (source_role == 'reserved')
            or row['split'] != ('train' if source_role == 'train' else 'validation')
            or row['group_id'] != role['declared_duplicate_component_sha256']
            or row['original_fixed_role'] != role or row['publisher_split'] != 'train'
            or original['split'] != 'train' or row['original_source_id'] != original['id']):
        raise ValueError('Original source and immutable train/reserved role must match')
    prompt = joined['views'][view]['prompt']
    response = original['solution'] if view == 'code_only' else original['r1_generation']
    if (row['prompt'] != prompt or row['response'] != response
            or row['original_response_utf8_sha256'] != hashlib.sha256(response.encode()).hexdigest()
            or row['admitted_use'] != 'weak_supervised_coding_only'
            or row['model_generated_examples'] is not True or row['human_preferences'] is not False
            or row['own_model_on_policy'] is not False or row['use_as_RL_reward_oracle_forbidden'] is not True
            or row['external_teacher_weights_loaded_by_this_pipeline'] is not False
            or row['live_teacher_API_used'] is not False):
        raise ValueError('Whole original text and weak public synthetic supervision scope required')
    text = 'User: ' + prompt + '\nAssistant:'
    head = tokenizer.encode(text, add_special_tokens=False).ids
    target = tokenizer.encode(response, add_special_tokens=False).ids
    expected = [1] + head + target + [2]
    boundary = 1 + len(head)
    expected_labels = [-100] * boundary + expected[boundary:]
    if (not target or not 3 <= len(expected) <= 4096
            or any(not 4 <= value < 32768 for value in head + target)
            or tokenizer.decode(head, skip_special_tokens=False) != text
            or tokenizer.decode(target, skip_special_tokens=False) != response
            or ids != expected or labels != expected_labels):
        raise ValueError('Every original prompt/response/EOS token and exact loss mask must match without cropping')
    digest = hashlib.sha256(struct.pack('<' + 'H' * len(expected), *expected)).hexdigest()
    mask_digest = hashlib.sha256(bytes([0] * boundary + [1] * (len(target) + 1))).hexdigest()
    metadata = dict(complete_context_tokens=len(expected), response_EOS_targets=len(target)+1, target_start=boundary,
        native_input_u16le_sha256=digest, native_target_mask_u8_sha256=mask_digest)
    if (row['original_native_metadata'] != metadata or row['complete_input_ids_sha256'] != digest
            or row['target_tokens'] != len(target)+1
            or joined['views'][view]['entire_context_tokens'] != len(expected)
            or joined['views'][view]['prompt_tokens_including_BOS'] != boundary
            or joined['views'][view]['entire_response_tokens_including_EOS'] != len(target)+1):
        raise ValueError('Complete native metadata, physical input bytes and original length arithmetic differ')
    return dict(complete_context_tokens=len(expected), response_EOS_targets=len(target)+1,
        sealed_role=source_role, view=view, original_text_and_native_arrays_exact=True,
        training_admission_granted=False)
