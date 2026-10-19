from slm_lab.object_evaluation import select, summarize


def test_sampling_preserves_answer_diversity_and_avoids_one_photo_dominating():
    rows = [{'id': str(i), 'group_id': 'many-crops', 'task': 'object_count', 'response': '1'} for i in range(50)]
    rows += [{'id': 'two', 'group_id': 'second-photo', 'task': 'object_count', 'response': '2'},
             {'id': 'three', 'group_id': 'third-photo', 'task': 'object_count', 'response': '3'}]
    chosen = select(rows, 3)
    assert {row['response'] for row in chosen} == {'1', '2', '3'}
    assert len({row['group_id'] for row in chosen}) == 3


def test_intervention_metrics_measure_the_original_target_without_claiming_new_scene_accuracy():
    rows = [{'group_id': 'one', 'expected': 'Cat', 'target_tokens': 2, 'proper_exact': True,
             'blank_exact': False, 'wrong_exact': True, 'proper_nll': 1., 'blank_nll': 3., 'wrong_nll': 2.},
            {'group_id': 'two', 'expected': 'Dog', 'target_tokens': 4, 'proper_exact': False,
             'blank_exact': False, 'wrong_exact': False, 'proper_nll': 4., 'blank_nll': 5., 'wrong_nll': 6.}]
    summary = summarize(rows)
    assert summary['correct_image_exact_accuracy'] == .5
    assert summary['wrong_image_original_target_match'] == .5
    assert summary['answer_mode_baseline'] == .5
    assert summary['blank_nll_gain'] == 8 / 6
    assert summary['mismatch_nll_gain'] == 10 / 6
