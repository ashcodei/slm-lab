import copy
import pytest

from slm_lab.feedback_reading_guard import enforce


def reports():
    case_fields = ['id', 'group_id', 'expected', 'task', 'prompt_sha256', 'input_pixels_sha256',
                   'wrong_pixels_sha256', 'wrong_image_id', 'target_tokens', 'original_answers_sha256']
    rows = [{**{key: key + str(index) for key in case_fields}, 'proper_exact': index == 0}
            for index in range(128)]
    before = {key: 'same-protocol' for key in ['evaluation_provenance', 'data_sha256', 'manifest_sha256',
              'tokenizer_sha256', 'inference_precision', 'inference_device_type', 'protocol', 'human_score_protocol']}
    before.update(results=rows, overall={'examples': 128, 'correct_image_exact_accuracy': 1 / 128},
                  max_examples=128, max_new_tokens=32, image_size=448, checkpoint_sha256='base-policy')
    after = copy.deepcopy(before); after['checkpoint_sha256'] = 'candidate-policy'
    result = {'eligible_for_selection': True, 'base_checkpoint_sha256': 'base-policy',
              'candidate_checkpoint_sha256': 'candidate-policy', 'checks': {}, 'failed_checks': []}
    return result, before, after


def test_feedback_with_same_total_accuracy_but_lost_reading_answer_is_withheld():
    result, before, after = reports()
    after['results'][0]['proper_exact'] = False; after['results'][1]['proper_exact'] = True
    guarded = enforce(result, before, after)
    assert not guarded['eligible_for_selection']
    assert guarded['failed_checks'] == ['real_photo_reading_retention']
    assert result['eligible_for_selection'] and not result['failed_checks']


def test_passing_reading_guard_does_not_override_existing_feedback_failure():
    result, before, after = reports(); result['eligible_for_selection'] = False
    result['failed_checks'] = ['functional_code_regression']
    guarded = enforce(result, before, after)
    assert not guarded['eligible_for_selection'] and guarded['failed_checks'] == ['functional_code_regression']


@pytest.mark.parametrize('change', ['policy', 'answers', 'photo', 'insufficient_photos'])
def test_unmatched_or_narrowed_real_reading_evidence_cannot_approve_feedback(change):
    result, before, after = reports()
    if change == 'policy': result['candidate_checkpoint_sha256'] = 'another-policy'
    elif change == 'answers': after['results'][0]['original_answers_sha256'] = 'changed-human-answer-list'
    elif change == 'photo': after['results'][0]['input_pixels_sha256'] = 'another-photo'
    else: after['results'][1]['group_id'] = after['results'][0]['group_id']
    with pytest.raises(ValueError): enforce(result, before, after)
