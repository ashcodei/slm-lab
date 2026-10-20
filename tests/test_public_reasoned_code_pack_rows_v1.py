from copy import deepcopy
import hashlib
import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from slm_lab.held_text_overlap_v1 import normalize
from slm_lab.public_reasoned_code_cases_v1 import contract, execute, qualify_reference, score_candidate
from slm_lab.public_reasoned_code_output_readback_v1 import verify
from slm_lab.public_reasoned_code_pack_rows_v1 import build
from slm_lab.public_synthetic_code_screen_policy_v1 import body_fingerprint


@pytest.fixture(scope='module')
def checked_source():
    sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
    code = 'def twice(n):\n    return 2*n\n'
    reasoning = 'Multiplication by two gives twice the integer. λ\n```python\n' + code + '```\n\n'
    question = dict(dataset='apps', split='train', index=7, question='Given integer n, return twice its value.', starter_code='')
    question['question_utf8_sha256'] = sha(question['question'])
    original = dict(id='a'*32, question_id='b'*32, dataset='apps', split='train', index='7',
        solution=code, r1_generation=reasoning, pass_rate='1.0', license='mit')
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator([code, reasoning, question['question']], trainers.BpeTrainer(vocab_size=512,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=['<pad>', '<bos>', '<eos>', '<unk>'], show_progress=False))
    prompts = dict(code_only='Write a Python solution to the following problem. Follow its specified interface. Return only the complete code.\n\n' + question['question'],
        reasoning='Solve the following programming problem in Python. Explain your reasoning and provide the complete solution.\n\n' + question['question'])
    joined = dict(position=0, id=original['id'], question_id=original['question_id'], dataset='apps', split='train', index=7,
        question_utf8_sha256=question['question_utf8_sha256'], views={})
    for view, response in [('code_only', code), ('reasoning', reasoning)]:
        prefix = 1 + len(tokenizer.encode('User: ' + prompts[view] + '\nAssistant:', add_special_tokens=False).ids)
        targets = len(tokenizer.encode(response, add_special_tokens=False).ids) + 1
        joined['views'][view] = dict(prompt=prompts[view], entire_target_utf8_sha256=sha(response),
            prompt_tokens_including_BOS=prefix, entire_response_tokens_including_EOS=targets,
            entire_context_tokens=prefix+targets, contains_reserved_native_source_token=False, complete_context_fits_4096=True)
    role = dict(source_role='train', declared_duplicate_component_sha256='c'*64, declared_duplicate_component_size=1,
        role_fixed_before_fresh_replay=True, optimization_forbidden=False)
    screened = dict(original_shard_position=0, id=original['id'], question_id=original['question_id'], dataset='apps', source_index=7,
        status='screened_metadata_positive_pending_fresh_execution', fresh_execution_pending=True, protected_overlap=False,
        publisher_solution_sha256=sha(code), program_AST_without_docstrings_sha256=body_fingerprint(code),
        input_only_group_sha256=sha(normalize(question['question'])), source_role='train', fixed_pre_execution_role=role,
        fitting_views=['code_only', 'reasoning'])
    cases = contract(json.dumps(dict(inputs=[[3], [5]], outputs=[6, 10], fn_name='twice')).encode())
    reference_observation = execute(code, cases, seconds=5)
    fixed = qualify_reference(cases, reference_observation)
    noop = execute('def twice(*args, **kwargs):\n    return None\n', cases, seconds=5)
    reference = dict(original_reference_observation=reference_observation, fixed_output_convention=fixed,
        no_op_observation=noop, no_op_score=score_candidate(cases, noop, fixed))
    observed = execute(code, cases, seconds=5)
    record = dict(candidate_execution_attempted=True, data_admitted_to_training=False, candidate_observation=observed,
        candidate_score=score_candidate(cases, observed, fixed), status='all_complete_original_cases_pass')
    verified = verify(cases, code, code, reference, record, stdin_comparison='stdin_outside_whitespace_only')
    assert verified['finite_case_positive_verified']
    proof_sha = sha(json.dumps(record, sort_keys=True))
    verified.update(original_shard_position=0, id=original['id'], source_role='train', original_result_sha256=proof_sha,
        original_status='all_complete_original_cases_pass', data_admitted_to_training=False)
    positive = dict(original_shard_position=0, id=original['id'], source_role='train', fixed_pre_execution_role=deepcopy(role),
        fitting_views=['code_only', 'reasoning'], verified_original_result_sha256=proof_sha)
    return original, question, joined, screened, positive, verified, tokenizer


def test_whole_code_and_reasoning_have_exact_native_masks_and_shared_role(checked_source):
    original, question, joined, screened, positive, verified, tokenizer = checked_source
    rows = []
    for view, field in [('code_only', 'solution'), ('reasoning', 'r1_generation')]:
        row, ids, labels = build(original, question, joined, screened, positive, verified, view, tokenizer)
        start = row['original_native_metadata']['target_start']
        assert tokenizer.decode(ids[start:-1], skip_special_tokens=False) == original[field]
        assert labels[:start] == [-100]*start and labels[start:] == ids[start:]
        assert ids[0] == 1 and ids[-1] == labels[-1] == 2 and len(ids) <= 4096
        assert row['use_as_RL_reward_oracle_forbidden'] and not row['human_preferences']
        rows.append(row)
    assert rows[0]['group_id'] == rows[1]['group_id'] and rows[0]['id'] != rows[1]['id']
    assert rows[1]['response'].endswith('\n\n') and 'λ' in rows[1]['response']


def test_reserved_component_remains_evaluation_only_and_cannot_be_relabelled(checked_source):
    original, question, joined, screened, positive, verified, tokenizer = deepcopy(checked_source)
    for item in (screened, positive, verified):
        item['source_role'] = 'reserved'
    for item in (screened, positive):
        item['fixed_pre_execution_role'].update(source_role='reserved', optimization_forbidden=True)
    row, _, _ = build(original, question, joined, screened, positive, verified, 'code_only', tokenizer)
    assert row['split'] == 'validation' and row['original_fixed_role']['optimization_forbidden']
    screened['source_role'] = 'train'
    with pytest.raises(ValueError):
        build(original, question, joined, screened, positive, verified, 'code_only', tokenizer)


def test_corrupt_source_proof_fit_or_native_specials_cannot_be_packed(checked_source):
    for mutation in ('proof', 'response', 'fit', 'special_id'):
        original, question, joined, screened, positive, verified, tokenizer = deepcopy(checked_source)
        if mutation == 'proof':
            verified['finite_case_positive_verified'] = False
        elif mutation == 'response':
            original['solution'] += '#changed complete source\n'
        elif mutation == 'fit':
            joined['views']['code_only']['entire_context_tokens'] -= 1
        else:
            tokenizer = Tokenizer(models.WordLevel({'<unk>':0, '<bos>':1, '<eos>':2, '<pad>':3}, unk_token='<unk>'))
        with pytest.raises(ValueError):
            build(original, question, joined, screened, positive, verified, 'code_only', tokenizer)
