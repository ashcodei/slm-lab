import copy

import pytest

from slm_lab.clevr_evaluation import compare_reports, select


def report():
    return {'evaluation_provenance': {'fixture': True}, 'data_sha256': 'data', 'manifest_sha256': 'manifest',
            'tokenizer_sha256': 'own', 'image_size': 448, 'inference_precision': 'bf16', 'inference_device_type': 'cuda',
            'max_new_tokens': 16, 'max_examples': 256, 'protocol': 'fixture', 'checkpoint_sha256': 'before',
            'overall': {'examples': 2, 'correct_image_exact_accuracy': .5},
            'results': [{'id': name, 'group_id': name, 'task': 'count', 'expected': str(i), 'prompt_sha256': name,
                         'input_pixels_sha256': name, 'wrong_pixels_sha256': 'wrong-' + name, 'wrong_image_id': 'other-' + name,
                         'target_tokens': 2, 'proper_exact': i == 0} for i, name in enumerate(['first', 'second'])]}


def test_aggregate_visual_improvement_cannot_hide_losing_a_correct_question():
    before = report(); after = copy.deepcopy(before)
    after['checkpoint_sha256'] = 'after'; after['results'][0]['proper_exact'] = False; after['results'][1]['proper_exact'] = True
    guard = compare_reports(before, after)
    assert not guard['passed'] and guard['previously_correct_questions_lost'] == ['first']


@pytest.mark.parametrize('changed', ['pixels', 'question', 'budget'])
def test_changed_visual_assessment_cannot_be_compared(changed):
    before = report(); after = copy.deepcopy(before)
    if changed == 'pixels': after['results'][0]['input_pixels_sha256'] = 'changed'
    elif changed == 'question': after['results'][0]['prompt_sha256'] = 'changed'
    else: after['max_new_tokens'] = 8
    with pytest.raises(ValueError): compare_reports(before, after)


def test_visual_sample_does_not_repeatedly_count_one_image_within_reasoning_type():
    rows = [{'id': str(i), 'group_id': str(i // 2), 'reasoning_type': 'count', 'response': str(i % 2)} for i in range(12)]
    chosen = select(rows, 6)
    assert len(chosen) == 6 and len({row['group_id'] for row in chosen}) == 6
    assert {row['response'] for row in chosen} == {'0', '1'}


@pytest.mark.parametrize('changed', ['aggregate', 'missing', 'duplicate'])
def test_visual_report_integrity_is_required_even_when_checkpoint_and_protocol_match(changed):
    before = report(); after = copy.deepcopy(before)
    if changed == 'aggregate': after['overall']['correct_image_exact_accuracy'] = 1.
    elif changed == 'missing': after['results'].pop()
    else: after['results'][1]['id'] = after['results'][0]['id']
    with pytest.raises(ValueError): compare_reports(before, after)
