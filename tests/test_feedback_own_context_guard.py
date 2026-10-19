import copy

import pytest

from slm_lab.feedback_own_context_guard import enforce, validate_coverage
from slm_lab.object_evaluation import summarize
from slm_lab.own_context_transfer_evaluation import PROTOCOL


def reports():
    rows = []
    for index in range(128):
        row = {'id': 'q' + str(index), 'group_id': 'photo' + str(index), 'expected': 'yes',
               'task': 'own_context_reading_question', 'prompt_sha256': 'p' + str(index),
               'input_pixels_sha256': 'rgb' + str(index), 'wrong_pixels_sha256': 'other' + str(index),
               'wrong_image_id': 'q' + str((index + 1) % 128), 'target_tokens': 2,
               'original_answers_sha256': 'human' + str(index),
               'own_OCR_input_rgb_sha256_768': 'reader' + str(index), 'wrong_OCR_rgb_sha256_768': 'other-reader' + str(index)}
        for control in ['proper', 'blank', 'wrong']:
            correct = control == 'proper' and index < 2
            row.update({control + '_generated': 'yes' if correct else 'no', control + '_exact': correct,
                        control + '_nll': .5, control + '_prompt_sha256': control + str(index),
                        control + '_target_tokens': 2, control + '_selected_context': []})
        rows.append(row)
    before = {key: 'fixed' for key in ['evaluation_provenance', 'data_sha256', 'manifest_sha256', 'tokenizer_sha256',
                                      'inference_precision', 'inference_device_type', 'human_score_protocol',
                                      'own_context_manifest_sha256', 'detector_sha256', 'reader_sha256']}
    before.update(checkpoint_sha256='base', protocol=PROTOCOL, image_size=448, max_examples=128,
                  max_new_tokens=32, image_size_own_reader=768, context_headroom_tokens=256,
                  maximum_context_words=64, total_inference_parameters=709755680, results=rows, overall=summarize(rows))
    after = copy.deepcopy(before); after['checkpoint_sha256'] = 'candidate'
    result = {'eligible_for_selection': True, 'base_checkpoint_sha256': 'base',
              'candidate_checkpoint_sha256': 'candidate', 'checks': {'plain_reading': {'passed': True}}, 'failed_checks': []}
    return result, before, after


def test_positive_preference_and_same_net_reading_score_cannot_hide_lost_original_answer():
    result, before, after = reports()
    after['results'][0]['proper_generated'] = 'no'; after['results'][0]['proper_exact'] = False
    after['results'][2]['proper_generated'] = 'yes'; after['results'][2]['proper_exact'] = True
    after['overall'] = summarize(after['results'])
    assert before['overall']['correct_image_exact_accuracy'] == after['overall']['correct_image_exact_accuracy']
    guarded = enforce(result, before, after)
    assert not guarded['eligible_for_selection']
    assert guarded['failed_checks'] == ['own_context_reading_retention']
    assert guarded['own_context_reading_guard']['previously_correct_questions_lost'] == ['q0']
    assert result['eligible_for_selection'] and not result['failed_checks']


def test_passing_conditioned_reading_never_overrides_existing_code_or_plain_reading_failure():
    result, before, after = reports()
    result['eligible_for_selection'] = False
    result['failed_checks'] = ['functional_code', 'real_photo_reading_retention']
    guarded = enforce(result, before, after)
    assert not guarded['eligible_for_selection'] and guarded['failed_checks'] == result['failed_checks']
    assert guarded['checks']['plain_reading'] == result['checks']['plain_reading']


@pytest.mark.parametrize('change', ['policy', 'subset', 'repeat_photo', 'budget', 'parameters', 'protocol'])
def test_feedback_cannot_use_different_policy_narrower_cases_or_different_interface(change):
    result, before, after = reports()
    if change == 'policy': result['candidate_checkpoint_sha256'] = 'different feedback branch'
    if change == 'subset':
        after['results'].pop(); after['overall'] = summarize(after['results'])
    if change == 'repeat_photo': after['results'][1]['group_id'] = after['results'][0]['group_id']
    if change == 'budget': after['context_headroom_tokens'] = 32
    if change == 'parameters': after['total_inference_parameters'] = 1_000_000_000
    if change == 'protocol': after['protocol'] = 'old untrained raw-float context'
    with pytest.raises(ValueError): enforce(result, before, after)


def test_full_matched_case_evidence_can_admit_an_otherwise_eligible_feedback_branch():
    result, before, after = reports()
    validate_coverage(before)
    guarded = enforce(result, before, after)
    assert guarded['eligible_for_selection'] and not guarded['failed_checks']
    assert guarded['checks']['own_context_reading_retention']['passed']
