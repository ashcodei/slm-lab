import copy

import pytest

from slm_lab.nq_open_data import canonical, example


def test_original_reference_strings_and_official_development_split_are_preserved():
    original = {'question': 'Which country contains Reykjavík?', 'answer': ['Iceland', 'Republic of Iceland']}
    row = example(original, 'dev', 7)
    assert row['split'] == 'valid' and row['source_split'] == 'dev' and row['source_line'] == 7
    assert row['response'] == original['answer'][0] and row['reference_answers'] == original['answer']
    assert row['source_original_question'] == original and row['license'] == 'CC BY-SA 3.0'


def test_changing_reference_answers_does_not_change_input_prompt_or_question_group():
    original = {'question': 'What is the capital of Iceland?', 'answer': ['Reykjavík']}
    changed = copy.deepcopy(original); changed['answer'] = ['LEAKED TARGET']
    first = example(original, 'train', 1); second = example(changed, 'train', 1)
    assert first['prompt'] == second['prompt'] and first['group_id'] == second['group_id']
    assert 'LEAKED TARGET' not in second['prompt']


def test_question_identity_protects_unicode_case_and_whitespace_aliases():
    assert canonical(' WHO owns Ａ  ship? ') == canonical('who owns a ship?')
    original = {'question': 'WHO owns Ａ  ship?', 'answer': ['original']}
    changed = {'question': 'who owns a ship?', 'answer': ['different']}
    assert example(original, 'train', 1)['group_id'] == example(changed, 'dev', 2)['group_id']


def test_unknown_annotation_schema_is_not_marked_as_human_original_answer():
    with pytest.raises(ValueError): example({'question': 'q', 'answer': [], 'model_answer': 'teacher'}, 'train', 1)
    with pytest.raises(ValueError): example({'question': 'q', 'answer': ['target']}, 'test', 1)
