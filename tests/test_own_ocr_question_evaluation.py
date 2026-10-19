import pytest

from slm_lab.own_ocr_question_evaluation import aggregate, control_words


def test_control_contexts_never_reuse_original_photo_words_for_changed_images():
    proper = [{'text': 'proper photo'}]; wrong = [{'text': 'different photo'}]; blank = []
    result = control_words('one', 'two', {'one': proper, 'two': wrong}, blank)
    assert result == {'proper': proper, 'wrong': wrong, 'blank': []}
    assert result['wrong'] is wrong and result['blank'] is blank
    with pytest.raises(ValueError): control_words('one', 'one', {'one': proper}, blank)


def test_all_declared_modes_and_controls_are_reported_with_actual_answer_scores():
    rows = []
    for index in range(2):
        row = {'group_id': str(index), 'modes': {}}
        for mode in ['plain', 'empty_context', 'own_context']:
            row['modes'][mode] = {control: {'exact': mode == 'own_context' and control == 'proper' and index == 0,
                                          'human_agreement': .5 if index == 0 else 0} for control in ['proper', 'blank', 'wrong']}
        rows.append(row)
    result = aggregate(rows)
    assert result['examples'] == result['original_photos'] == 2
    assert result['own_context']['proper']['correct'] == 1 and result['own_context']['proper']['accuracy'] == .5
    assert result['own_context']['blank']['correct'] == result['own_context']['wrong']['correct'] == 0
    assert result['plain']['proper']['mean_human_agreement'] == .25
