import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('vqa_source_inspection', Path(__file__).resolve().parents[1] / 'scripts/inspect_vqa_sources.py')
inspection = importlib.util.module_from_spec(spec); spec.loader.exec_module(inspection)


def annotation():
    return {'multiple_choice_answer': 'red', 'answers': [{'answer_id': i + 1, 'answer': 'red' if i < 6 else 'blue'} for i in range(10)]}


def test_agreement_uses_actual_ten_human_votes():
    assert inspection.agreed_answer(annotation()) == ('red', 6)


@pytest.mark.parametrize('change', ['missing', 'repeated', 'wrong_majority'])
def test_incomplete_repeated_or_inconsistent_answer_records_are_rejected(change):
    row = annotation()
    if change == 'missing': row['answers'].pop()
    elif change == 'repeated': row['answers'][1]['answer_id'] = row['answers'][0]['answer_id']
    else: row['multiple_choice_answer'] = 'blue'
    with pytest.raises(ValueError): inspection.agreed_answer(row)


@pytest.mark.parametrize('change', ['different_question', 'same_image', 'same_answer'])
def test_accepted_visual_pair_must_keep_question_and_change_image_and_answer(change):
    first = {'question': {'question': 'What COLOR is this?', 'image_id': 1}, 'answer': 'red'}
    second = {'question': {'question': 'What color is this?', 'image_id': 2}, 'answer': 'blue'}
    assert inspection.is_pair(first, second)
    altered = copy.deepcopy(second)
    if change == 'different_question': altered['question']['question'] = 'How many objects?'
    elif change == 'same_image': altered['question']['image_id'] = 1
    else: altered['answer'] = 'red'
    assert not inspection.is_pair(first, altered)
