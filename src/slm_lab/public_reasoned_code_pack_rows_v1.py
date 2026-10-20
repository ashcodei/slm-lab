"""Lossless native SFT views of independently checked public R1 examples.

Public synthetic teacher text remains weak SFT only. This helper grants no
training admission and never supplies a human-preference or RL-reward API.
"""
import hashlib
import re
import struct

from .held_text_overlap_v1 import normalize
from .public_reasoned_code_metadata_v1 import reported_all_pass
from .public_synthetic_code_screen_policy_v1 import body_fingerprint

REVISION = 'eadf535931451525f3e5621d0f960c240bc62fd9'
PUBLISHER_FILE = 'train/python/train-00000-of-00070.parquet'
HEX = re.compile(r'[0-9a-f]{64}')


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def build(original, question, joined, screened, positive, verified, view, tokenizer):
    position = joined['position']
    if (type(position) is not int or not 0 <= position < 20000 or view not in ('code_only', 'reasoning')
            or original['split'] != 'train' or joined['split'] != 'train' or question['split'] != 'train'
            or not reported_all_pass(original['pass_rate'])
            or original['id'] != joined['id'] or original['id'] != screened['id']
            or original['id'] != positive['id'] or original['id'] != verified['id']
            or original['question_id'] != joined['question_id'] or original['question_id'] != screened['question_id']
            or any(item['original_shard_position'] != position for item in (screened, positive, verified))):
        raise ValueError('One complete original upstream-train source and exact checked position required')
    key = (original['dataset'], int(original['index']))
    if (key != (question['dataset'], question['index']) or key != (joined['dataset'], joined['index'])
            or key != (screened['dataset'], screened['source_index'])
            or screened['status'] != 'screened_metadata_positive_pending_fresh_execution'
            or screened['fresh_execution_pending'] is not True or screened['protected_overlap'] is not False
            or verified['finite_case_positive_verified'] is not True
            or verified['original_source_runtime_qualified'] is not True
            or verified['original_status'] != 'all_complete_original_cases_pass'
            or verified['data_admitted_to_training'] is not False or verified['training_admitted'] is not False
            or not isinstance(positive['verified_original_result_sha256'], str)
            or HEX.fullmatch(positive['verified_original_result_sha256']) is None
            or positive['verified_original_result_sha256'] != verified['original_result_sha256']):
        raise ValueError('Exact independently checked finite-case positive and protected-source exclusion required')
    role = screened['source_role']
    fixed = screened['fixed_pre_execution_role']
    if (role not in ('train', 'reserved') or positive['source_role'] != role or verified['source_role'] != role
            or positive['fixed_pre_execution_role'] != fixed or fixed['source_role'] != role
            or fixed['role_fixed_before_fresh_replay'] is not True
            or fixed['optimization_forbidden'] is not (role == 'reserved')
            or HEX.fullmatch(fixed['declared_duplicate_component_sha256']) is None
            or view not in screened['fitting_views'] or positive['fitting_views'] != screened['fitting_views']):
        raise ValueError('Original duplicate-component role and complete fitting views cannot change')
    code = original['solution']
    if (sha(code) != screened['publisher_solution_sha256']
            or body_fingerprint(code) != screened['program_AST_without_docstrings_sha256']
            or hashlib.sha256(normalize(question['question']).encode()).hexdigest() != screened['input_only_group_sha256']
            or sha(question['question']) != question['question_utf8_sha256']
            or question['question_utf8_sha256'] != joined['question_utf8_sha256']):
        raise ValueError('Entire original code and source question must retain their checked identities')
    starter = question.get('starter_code', '')
    if not isinstance(starter, str):
        raise ValueError('Whole original starter-code string required')
    suffix = '\n\nStarter code:\n' + starter if starter.strip() else ''
    if view == 'code_only':
        prompt = 'Write a Python solution to the following problem. Follow its specified interface. Return only the complete code.\n\n' + question['question'] + suffix
        response = code
    else:
        prompt = 'Solve the following programming problem in Python. Explain your reasoning and provide the complete solution.\n\n' + question['question'] + suffix
        response = original['r1_generation']
        if not code.strip() or code.strip() not in response:
            raise ValueError('Whole reasoning target must retain its tested publisher solution')
    saved = joined['views'][view]
    if (saved['prompt'] != prompt or saved['entire_target_utf8_sha256'] != sha(response)
            or saved['complete_context_fits_4096'] is not True
            or saved['contains_reserved_native_source_token'] is not False):
        raise ValueError('Complete source prompt/response and predeclared native-context fit required')
    for token, expected in [('<pad>', 0), ('<bos>', 1), ('<eos>', 2), ('<unk>', 3)]:
        if tokenizer.token_to_id(token) != expected:
            raise ValueError('Own native special-token identities required')
    text = 'User: ' + prompt + '\nAssistant:'
    head = tokenizer.encode(text, add_special_tokens=False).ids
    target = tokenizer.encode(response, add_special_tokens=False).ids
    ids = [1] + head + target + [2]
    start = len(head) + 1
    if (not target or not 3 <= len(ids) <= 4096 or any(not 4 <= value < 32768 for value in head + target)
            or tokenizer.decode(head, skip_special_tokens=False) != text
            or tokenizer.decode(target, skip_special_tokens=False) != response
            or saved['prompt_tokens_including_BOS'] != start
            or saved['entire_response_tokens_including_EOS'] != len(target) + 1
            or saved['entire_context_tokens'] != len(ids)):
        raise ValueError('Entire native source, response, EOS and mask must reconstruct without cropping')
    labels = [-100] * start + ids[start:]
    native = dict(complete_context_tokens=len(ids), response_EOS_targets=len(target)+1, target_start=start,
        native_input_u16le_sha256=hashlib.sha256(struct.pack('<' + 'H' * len(ids), *ids)).hexdigest(),
        native_target_mask_u8_sha256=hashlib.sha256(bytes([0]*start + [1]*(len(target)+1))).hexdigest())
    row = dict(id='public-reasoned-code:' + str(position) + ':' + view + ':' + original['id'],
        group_id=fixed['declared_duplicate_component_sha256'], split='train' if role == 'train' else 'validation',
        publisher_split='train', original_shard_position=position, original_source_id=original['id'],
        original_question_id=original['question_id'], upstream_dataset=key[0], upstream_index=key[1],
        original_fixed_role=fixed, view=view, prompt=prompt, response=response,
        original_question_utf8_sha256=question['question_utf8_sha256'], original_response_utf8_sha256=sha(response),
        original_tested_code_utf8_sha256=sha(code), verified_original_result_sha256=positive['verified_original_result_sha256'],
        source='NVIDIA OpenCodeReasoning-2 public synthetic SFT', source_revision=REVISION,
        source_file=PUBLISHER_FILE, source_url='https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2/tree/' + REVISION,
        attribution='NVIDIA Corporation, OpenCodeReasoning-2 (2025)', license='CC-BY-4.0',
        upstream_declared_license=original.get('license'),
        response_origin='Public precomputed DeepSeek-R1 response; code-only view uses the entire publisher solution field',
        model_generated_examples=True, external_model_teacher_used=True,
        external_teacher_weights_loaded_by_this_pipeline=False, live_teacher_API_used=False,
        human_preferences=False, own_model_on_policy=False, admitted_use='weak_supervised_coding_only',
        full_published_case_agreement_not_independent_semantic_correctness=True,
        reasoning_text_not_independently_certified_by_code_execution=True,
        historical_or_semantic_independence_not_certified=True, use_as_RL_reward_oracle_forbidden=True,
        original_native_metadata=native)
    return row, ids, labels
