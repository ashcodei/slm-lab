import copy

import pytest

from slm_lab.ai2d_exposure_patterns_v1 import KINDS
from slm_lab.ai2d_exposure_strata_v1 import CONDITIONS, COHORTS, cohort_identities, join, summarize


def cases_and_audits():
    cases = [dict(id=str(i), original_question_id='original:' + str(i), image_id=i,
        original_ABC_labels=False, purpose='evaluation_only', admitted_to_training=False, split='test') for i in range(3)]
    texts = [dict(id=r['id'], original_question_id=r['original_question_id'], image_id=r['image_id'], original_ABC_labels=False,
        literal_text_patterns={k: dict(inspected_exact_match_scopes=[], found=False) for k in KINDS}) for r in cases]
    images = [dict(id=r['id'], image_id=r['image_id'], original_ABC_labels=False,
        exact_original_diagram_file_in_inspected_current_pool=False,
        exact_actual_proper_native448_input_in_inspected_current_pool=False) for r in cases]
    texts[0]['literal_text_patterns']['original_question'] = dict(inspected_exact_match_scopes=['current_native'], found=True)
    images[1]['exact_actual_proper_native448_input_in_inspected_current_pool'] = True
    return cases, texts, images


def test_generic_question_and_exact_pixels_are_distinct_without_dropping_any_question():
    cases, texts, images = cases_and_audits(); rows = join(cases, texts, images); groups = cohort_identities(rows)
    assert groups['all_original'] == ['0', '1', '2']
    assert groups['generic_question_in_inspected_text'] == ['0']
    assert groups['proper_native448_in_current_image_pool'] == ['1']
    assert groups['no_stronger_exact_match_in_inspected_text_and_current_image_pools'] == ['0', '2']
    changed = copy.deepcopy(cases)
    for r in changed: r['answer_index'] = 17
    assert join(changed, texts, images) == rows
    with pytest.raises(ValueError): join(cases, texts[:-1], images)
    bad = copy.deepcopy(texts); bad[2]['literal_text_patterns']['whole_native_prompt'] = dict(inspected_exact_match_scopes=['current_native'], found=True)
    with pytest.raises(ValueError, match='stronger'): join(cases, bad, images)


def test_overlapping_source_groups_preserve_primary_actual_verdicts_and_all_controls():
    cases, texts, images = cases_and_audits(); rows = join(cases, texts, images)
    metrics = ['strict_correct', 'exact_duplicate_equivalent_correct', 'answer_parseable',
        'spaced_letter_strict_correct', 'spaced_letter_exact_duplicate_equivalent_correct',
        'direct_letter_strict_correct', 'direct_letter_exact_duplicate_equivalent_correct']
    deliveries = [dict(id=r['id'], image_id=r['image_id'], conditions={c: dict(verdict={m: r['id'] == '0' for m in metrics}) for c in CONDITIONS}) for r in cases]
    result = summarize(rows, deliveries)
    assert tuple(result['cohorts']) == COHORTS and result['cohorts_overlap_and_must_not_be_added_together']
    assert result['cohorts']['all_original']['conditions']['proper_image']['strict_correct'] == 1
    assert result['cohorts']['all_original']['conditions']['gray_image']['strict_accuracy'] == 1 / 3
    assert result['cohorts']['no_stronger_exact_match_in_inspected_text_and_current_image_pools']['conditions']['proper_image']['strict_accuracy'] == .5
    assert result['cohorts']['ordered_options_in_inspected_text']['conditions']['proper_image']['strict_accuracy'] is None
    bad = copy.deepcopy(deliveries); bad[0]['conditions'].pop('wrong_image')
    with pytest.raises(ValueError): summarize(rows, bad)
