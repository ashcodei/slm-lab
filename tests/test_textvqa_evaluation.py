import pytest
from slm_lab.textvqa_evaluation import select


def test_real_reading_assessment_keeps_one_original_question_per_photo():
    rows = [{'id': 'question-' + str(i), 'group_id': 'photo-' + str(i // 2), 'response': str(i)} for i in range(8)]
    selected = select(rows, 8)
    assert len(selected) == 4
    assert len({row['group_id'] for row in selected}) == 4
    assert all(any(other['response'] != row['response'] for other in selected) for row in selected)


def test_repeated_questions_or_uninformative_wrong_images_are_rejected():
    first = {'id': 'a', 'group_id': 'photo-a', 'response': 'same'}
    second = {'id': 'b', 'group_id': 'photo-b', 'response': 'same'}
    with pytest.raises(ValueError): select([first, first], 2)
    with pytest.raises(ValueError): select([first, second], 2)
    with pytest.raises(ValueError): select([first, {**second, 'response': ' Same  '}], 2)
