"""Rebuild suffix descriptors from original source bytes without producer code."""
import hashlib
import json
import struct


def checksum(values):
    return hashlib.sha256(struct.pack('<' + 'H' * len(values), *values)).hexdigest()


def rebuild(tokenizer, row, ids, labels, level, observed):
    if (type(level) is not int or level not in [4, 16, 64, 0]
            or row['split'] != 'train' or row['publisher_split'] != 'train'
            or row['external_model_teacher_used'] is not False or row['model_generated_examples'] is not False
            or len(ids) != len(labels) or not 3 <= len(ids) <= 4096 or ids[0] != 1 or ids[-1] != 2
            or any(type(t) is not int or not 0 <= t < 32768 for t in ids)
            or checksum(ids) != row['complete_input_ids_sha256']):
        raise ValueError('Every original human train role, whole native source and EOS boundary must independently reconstruct')
    unmasked = [i for i, t in enumerate(labels) if t != -100]
    if not unmasked or unmasked != list(range(unmasked[0], len(ids))) or not 1 <= unmasked[0] < len(ids) - 1:
        raise ValueError('Every original contiguous complete answer/EOS label must be retained')
    start = unmasked[0]
    if any(labels[i] != ids[i] for i in unmasked):
        raise ValueError('Original native target values cannot change')
    reference = ids[start:-1]
    text = row['response']
    if tokenizer.encode(text, add_special_tokens=False).ids != reference or tokenizer.decode(reference, skip_special_tokens=False) != text:
        raise ValueError('All original human response bytes and native IDs must agree')
    bound = max(0, len(reference) - level) if level else 0
    # Find the largest allowed boundary whose independent decode is an exact
    # original text prefix. No line-boundary or source-rewriting assumption.
    selected = None
    for end in range(bound, -1, -1):
        candidate = tokenizer.decode(reference[:end], skip_special_tokens=False)
        if text[:len(candidate)] == candidate:
            selected = end, candidate
            break
    keep, prefix = selected
    missing = ids[start + keep:]
    action_budget = min(4096 - start - keep, max(512, len(missing) + 64))
    if action_budget < len(missing) or ids[:start + keep] + missing != ids:
        raise ValueError('The complete reference suffix and EOS must fit the actual native budget')
    expected = dict(source_id=row['id'], nominal_missing_non_EOS_tokens=level,
        original_complete_input_sha256=checksum(ids), original_context_tokens=len(ids), original_prompt_tokens=start,
        original_response_EOS_targets=len(unmasked), retained_reference_tokens=keep,
        actual_missing_response_EOS_targets=len(missing), conditional_rollout_prefix_tokens=start + keep,
        full_native_action_budget=action_budget, original_reference_prefix_sha256=checksum(reference[:keep]),
        original_reference_prefix_text_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
        original_missing_suffix_and_EOS_sha256=checksum(missing),
        original_unconditional_labels_sha256=hashlib.sha256(json.dumps(labels, separators=(',', ':')).encode()).hexdigest(),
        original_has_embedded_boundary_tokens=any(t in [0, 1, 2] for t in ids[1:-1]),
        exact_unhinted_original_source=keep == 0, training_context_only=True, unassisted_capability_score=False,
        per_view_full_reference_and_prefix_only_controls_required=True, source_training_admitted=False,
        production_RL_admitted=False, student_generations=0, model_updates=0)
    if observed != expected:
        raise ValueError('The complete independent native view differs from the producer descriptor')
    return dict(metadata=expected, original_prefix_text=prefix, original_reference_prefix_ids=reference[:keep],
        original_missing_suffix_and_EOS_ids=missing)


def eligible(metadata, full_reference, prefix_only):
    rewards = [full_reference.get('reward'), prefix_only.get('reward')]
    if any(type(r) not in [int, float] or not 0 <= r <= 1 for r in rewards):
        return False
    return (rewards[0] == 1 and rewards[1] != 1 and not metadata['original_has_embedded_boundary_tokens']
        and (metadata['nominal_missing_non_EOS_tokens'] == 0 or metadata['retained_reference_tokens'] > 0))
