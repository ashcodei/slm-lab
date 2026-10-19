import copy
import pytest

from slm_lab.clevr_joint_rehearsal import compare_scene_reports, question_subset
from slm_lab.clevr_scene_evaluation import score, summarize

TARGET = ('Objects from left to right:\n1. small red metal cube.\n'
          '2. large blue rubber sphere.\n3. small green metal cube.\n'
          'Shape counts: cubes 2, cylinders 0, spheres 1.')


def report():
    row = {'id': 'scene-a', 'group_id': 'parent-a', 'expected': TARGET,
           'prompt_sha256': 'prompt', 'input_pixels_sha256': 'proper',
           'wrong_pixels_sha256': 'wrong', 'wrong_image_id': 'scene-b', 'target_tokens': 30}
    for control in ['proper', 'blank', 'wrong']:
        row[control + '_generated'] = TARGET
        row[control + '_scores'] = score(TARGET, TARGET)
        row[control + '_nll'] = 0.3
    result = {key: 'unchanged' for key in ['evaluation_provenance', 'data_sha256', 'manifest_sha256',
              'tokenizer_sha256', 'image_size', 'inference_precision', 'inference_device_type',
              'max_new_tokens', 'max_examples', 'protocol']}
    result.update(results=[row], overall=summarize([row]))
    return result


def test_qa_exposures_reproduce_actual_python_seed42_sampler_prefix():
    # Recorded Python shuffle output, rather than another call to the implementation.
    rows = [{'id': index} for index in range(10)]
    assert [row['id'] for row in question_subset(rows, 5)] == [7, 3, 2, 8, 5]
    assert [row['id'] for row in rows] == list(range(10))
    with pytest.raises(ValueError):
        question_subset(rows, 11)


def test_scene_retention_tracks_actual_format_and_content_forgetting():
    before = report(); after = copy.deepcopy(before)
    after['results'][0]['proper_generated'] = 'cyan'
    after['results'][0]['proper_scores'] = score('cyan', TARGET)
    after['overall'] = summarize(after['results'])
    guard = compare_scene_reports(before, after)
    assert not guard['passed']
    assert guard['previously_correct_cases_lost']['valid_format'] == ['scene-a']
    assert guard['previously_correct_cases_lost']['object_count_exact'] == ['scene-a']


@pytest.mark.parametrize('change', ['pixels', 'prompt', 'aggregate', 'fabricated_case_score'])
def test_mismatched_or_fabricated_scene_evidence_cannot_support_rehearsal_comparison(change):
    before = report(); after = copy.deepcopy(before)
    if change == 'pixels': after['results'][0]['wrong_pixels_sha256'] = 'another-image'
    elif change == 'prompt': after['results'][0]['prompt_sha256'] = 'gold-leaked-prompt'
    elif change == 'aggregate': after['overall']['proper_full_scene_exact'] = 0.0
    else: after['results'][0]['proper_scores']['full_scene_exact'] = False
    with pytest.raises(ValueError):
        compare_scene_reports(before, after)
