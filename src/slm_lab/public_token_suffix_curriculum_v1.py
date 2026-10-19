"""Original human code with progressively longer missing native token suffixes.

This proposes training contexts, never an unassisted benchmark or RL admission.
Every original unconditional CE target remains. Full-reference and prefix-only
execution controls must qualify each exact view before a later RL experiment.
"""
import hashlib
import struct

LEVELS = (4, 16, 64, 0)  # 0 means the entire unhinted response, not zero work.


def tokens_sha(ids):
    return hashlib.sha256(struct.pack('<' + 'H' * len(ids), *ids)).hexdigest()


def view(tokenizer, row, ids, labels, missing_tokens):
    if (row.get('split') != 'train' or row.get('publisher_split') != 'train'
            or row.get('external_model_teacher_used') is not False or row.get('model_generated_examples') is not False):
        raise ValueError('Only original public human publisher-train solutions may supply a suffix curriculum')
    if type(missing_tokens) is not int or missing_tokens not in LEVELS:
        raise ValueError('Explicit4/16/64missing-token or0unhinted level required')
    if (len(ids) != len(labels) or not 3 <= len(ids) <= 4096 or ids[0] != 1 or ids[-1] != 2
            or any(type(v) is not int or not 0 <= v < 32768 for v in ids)
            or tokens_sha(ids) != row['complete_input_ids_sha256']):
        raise ValueError('Every original complete native byte and actual EOS boundary must be preserved')
    start = next((i for i, value in enumerate(labels) if value != -100), len(labels))
    if not 1 <= start < len(ids) - 1 or labels[:start] != [-100] * start or labels[start:] != ids[start:]:
        raise ValueError('All original unconditional response and EOS targets must remain contiguous and intact')
    response = row['response']
    original = ids[start:-1]
    if tokenizer.encode(response, add_special_tokens=False).ids != original or tokenizer.decode(original, skip_special_tokens=False) != response:
        raise ValueError('The original complete human source must reconstruct from its original native response IDs')
    keep = max(0, len(original) - missing_tokens) if missing_tokens else 0
    # Byte-level token boundaries can fall inside a Unicode character. Retain
    # only a complete exact original Unicode prefix; never replace a character.
    while keep and not response.startswith(tokenizer.decode(original[:keep], skip_special_tokens=False)):
        keep -= 1
    prefix = tokenizer.decode(original[:keep], skip_special_tokens=False)
    if prefix != response[:len(prefix)]:
        raise ValueError('Supplied code must be an exact original Unicode prefix')
    rollout = ids[:start + keep]
    suffix = ids[start + keep:]
    budget = min(4096 - len(rollout), max(512, len(suffix) + 64))
    if rollout + suffix != ids or budget < len(suffix) or keep and len(suffix) - 1 < missing_tokens:
        raise ValueError('Every original missing token, complete program and genuine EOS must fit without truncation')
    return dict(source_id=row['id'], nominal_missing_non_EOS_tokens=missing_tokens,
        original_complete_input_sha256=tokens_sha(ids), original_context_tokens=len(ids),
        original_prompt_tokens=start, original_response_EOS_targets=len(ids) - start,
        retained_reference_tokens=keep, actual_missing_response_EOS_targets=len(suffix),
        conditional_rollout_prefix_tokens=len(rollout), full_native_action_budget=budget,
        original_reference_prefix_sha256=tokens_sha(original[:keep]),
        original_reference_prefix_text_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
        original_missing_suffix_and_EOS_sha256=tokens_sha(suffix),
        original_unconditional_labels_sha256=hashlib.sha256(__import__('json').dumps(labels, separators=(',', ':')).encode()).hexdigest(),
        original_has_embedded_boundary_tokens=any(v in [0, 1, 2] for v in ids[1:-1]),
        native_rollout_prefix_ids=rollout, public_reference_prefix_ids=original[:keep],
        public_reference_prefix_text=prefix, original_missing_suffix_and_EOS_ids=suffix,
        complete_original_anchor_ids=list(ids), complete_original_anchor_labels=list(labels),
        exact_unhinted_original_source=keep == 0, training_context_only=True,
        unassisted_capability_score=False, per_view_full_reference_and_prefix_only_controls_required=True,
        source_training_admitted=False, production_RL_admitted=False, student_generations=0, model_updates=0)


def descriptor(value):
    omitted = {'native_rollout_prefix_ids', 'public_reference_prefix_ids', 'public_reference_prefix_text',
        'original_missing_suffix_and_EOS_ids', 'complete_original_anchor_ids', 'complete_original_anchor_labels'}
    return {k: v for k, v in value.items() if k not in omitted}


def nontrivial(value, reference, prefix_only):
    if any(type(x.get('reward')) not in [int, float] or not 0 <= x['reward'] <= 1 for x in [reference, prefix_only]):
        raise ValueError('Actual bounded full-reference and prefix-only rewards are required')
    return bool(reference['reward'] == 1 and prefix_only['reward'] < 1
        and not value['original_has_embedded_boundary_tokens']
        and (value['nominal_missing_non_EOS_tokens'] == 0 or value['retained_reference_tokens'] > 0))
