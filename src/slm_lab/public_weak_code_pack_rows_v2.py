"""Whole public synthetic coding rows: weak SFT only, never an RL oracle.

Reuse the two closed actual execution observations, preserving the role fixed
before replay. Passing generated tests establishes agreement with those tests;
it does not independently certify the question, algorithm or test semantics.
"""
import hashlib
import struct

from .held_text_overlap_v1 import normalize
from .public_code_disposition_readback_v1 import row_sha, verify_observation
from .public_code_instruction_screen_v1 import parse_whole
from .public_synthetic_code_screen_policy_v1 import body_fingerprint

REVISION = '8f3ba5bafe4d6e8db46082cf7ae6741bc370604d'
ATTRIBUTION = 'Wasi Uddin Ahmad et al., NVIDIA, OpenCodeInstruct (2025)'


def build(original, prior, observed, role, tokenizer, *, veto_sha256, veto_AST_sha256):
    position = observed['original_shard_position']
    if (type(position) is not int or not 0 <= position < 100000 or observed['source_id'] != original.get('id')
            or observed['original_entry_sha256'] != row_sha(original)
            or prior['original_shard_position'] != position or prior['original_entry_sha256'] != row_sha(original)
            or observed['original_disposition_rebuilt_exactly'] is not True
            or observed['original_candidate'] is not prior['screen_pass_candidate']):
        raise ValueError('Every complete original public source byte and both closed observations must match')
    if not observed['original_candidate']:
        if role is not None or observed['source_role'] != 'excluded':
            raise ValueError('Originally excluded rows cannot acquire a new training or reserved role')
        return None, 'original_exclusion:' + observed['original_screen_status']
    if (role is None or role['original_shard_position'] != position or role['role_fixed_before_fresh_replay'] is not True
            or role['source_role'] not in ['train', 'reserved'] or role['optimization_forbidden'] is not (role['source_role'] == 'reserved')
            or any(observed.get(k) != v for k, v in role.items()) or prior.get('protected_overlap') not in (None, [])
            or observed['independent_semantic_correctness_certified'] is not False
            or observed['use_as_RL_reward_oracle_forbidden'] is not True):
        raise ValueError('Every fixed original role, protected-input exclusion and weak-only scope must remain')
    parsed = parse_whole(original)
    body = body_fingerprint(parsed['program'])
    if row_sha(original) == veto_sha256 or body == veto_AST_sha256:
        return None, 'preserved_independent_semantic_veto'
    question = hashlib.sha256(normalize(original['input']).encode()).hexdigest()
    if (question != observed['input_only_group_sha256'] or body != observed['program_AST_without_docstrings_sha256']
            or question != prior['input_only_group_sha256'] or body != prior['program_AST_without_docstrings_sha256']):
        raise ValueError('Both whole original duplicate keys must reconstruct without changing source bytes')
    for value in [prior['actual_whole_execution'], observed['fresh_whole_program_execution']]: verify_observation(parsed, value)
    fresh = observed['fresh_whole_program_execution']
    if (prior['actual_whole_execution']['all_published_tests_passed'] is not True
            or observed['fresh_complete_pass_weak_SFT_candidate'] is not fresh['all_published_tests_passed']):
        raise ValueError('The actual original and fresh returned-value decisions must agree with their retained labels')
    if not fresh['all_published_tests_passed']:
        return None, 'preserved_fresh_' + fresh['status']
    prompt = 'User: ' + original['input'] + '\nAssistant:'
    head = tokenizer.encode(prompt, add_special_tokens=False).ids
    answer = tokenizer.encode(original['output'], add_special_tokens=False).ids
    ids = [1] + head + answer + [2]
    start = len(head) + 1
    labels = [-100] * start + ids[start:]
    mask = bytes([False] * start + [True] * (len(answer) + 1))
    native = dict(native_input_u16le_sha256=hashlib.sha256(struct.pack('<' + 'H' * len(ids), *ids)).hexdigest(),
        native_target_mask_u8_sha256=hashlib.sha256(mask).hexdigest(), complete_context_tokens=len(ids),
        response_EOS_targets=len(answer) + 1, target_start=start)
    if (not 3 <= len(ids) <= 4096 or any(not 4 <= t < 32768 for t in ids[1:-1])
            or tokenizer.decode(head, skip_special_tokens=False) != prompt
            or tokenizer.decode(answer, skip_special_tokens=False) != original['output']
            or native != observed['complete_native']):
        raise ValueError('All original native question/response/EOS positions must reconstruct exactly without cropping')
    split = 'train' if role['source_role'] == 'train' else 'validation'
    row = dict(id='public-weak-code:' + str(position) + ':' + original['id'], group_id=role['declared_duplicate_component_sha256'],
        split=split, publisher_split='train', original_shard_position=position, original_source_id=original['id'],
        original_fixed_role=role, prompt=original['input'], response=original['output'], original_entry=original,
        source='OpenCodeInstruct public weak synthetic coding SFT', source_revision=REVISION,
        source_url='https://huggingface.co/datasets/nvidia/OpenCodeInstruct/tree/' + REVISION,
        source_file='data/train-00000-of-00050.parquet', original_entry_sha256=row_sha(original),
        attribution=ATTRIBUTION, license='CC-BY-4.0', license_url='https://creativecommons.org/licenses/by/4.0/',
        response_origin='Public precomputed Qwen2.5-Coder-32B-Instruct response',
        test_origin='Public precomputed Qwen2.5-Coder-32B-Instruct tests',
        model_generated_examples=True, external_model_teacher_used=True,
        external_teacher_weights_loaded_by_this_pipeline=False, live_teacher_API_used=False,
        human_preferences=False, own_model_on_policy=False, admitted_use='weak_supervised_coding_only',
        generated_test_agreement_not_independent_semantic_correctness=True, use_as_RL_reward_oracle_forbidden=True,
        declared_input_and_AST_cross_role_separation_only=True, historical_or_arbitrary_semantic_independence_not_certified=True,
        original_native_metadata=native, original_complete_execution=prior['actual_whole_execution'],
        original_fresh_execution=fresh)
    return (row, ids, labels), 'weak_SFT_positive:' + role['source_role']
