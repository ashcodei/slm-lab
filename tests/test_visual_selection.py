import copy

import pytest

from slm_lab.visual_selection import compare


def reports():
    return {
        'ocr': {'checkpoint_sha256': 'base', 'data_sha256': 'ocr', 'image_size': 448,
                'max_new_tokens': 128, 'protocol': 'greedy', 'results': [{'id': 'ocr-1'}],
                'by_task': {'word_ocr': {'examples': 64, 'exact_match_accuracy': .02, 'character_error_rate': 1.0}}},
        'vision': {'checkpoint_sha256': 'base', 'data_sha256': 'photos', 'image_size': 448,
                   'images': 128, 'samples': [{'id': 'photo-1'}], 'proper_nll': 2.0, 'mismatch_nll_gain': .1},
        'knowledge': {'checkpoint_sha256': 'base', 'questions_sha256': 'questions', 'protocol': 'fixed',
                      'predictions': [{'id': 'question-1'}], 'by_domain': {'math': {}},
                      'overall': {'generated_accuracy': .1, 'likelihood_accuracy': .2}},
        **{name: {'checkpoint_sha256': 'base', 'suite_sha256': 'code', 'task_count': 24,
                  'generation_budget': 512, 'decoding': 'greedy', 'prompt_mode': name,
                  'results': [{'language': 'Python', 'task': 'add'}],
                  'syntax_pass_rate': .5, 'functional_pass_at_1': .2, 'efficiency_stress_pass_rate': .1}
           for name in ['code', 'code_instruction']},
    }


def improved():
    result = copy.deepcopy(reports())
    for value in result.values():
        value['checkpoint_sha256'] = 'candidate'
    result['ocr']['by_task']['word_ocr'].update(exact_match_accuracy=.15, character_error_rate=.75)
    return result


def test_visual_gain_is_not_promoted_if_code_regresses():
    assert compare(reports(), improved())['promote_candidate']
    candidate = improved(); candidate['code_instruction']['functional_pass_at_1'] = .1
    outcome = compare(reports(), candidate)
    assert not outcome['promote_candidate']
    assert 'code_instruction_functional_pass_at_1' in outcome['failed_checks']
    assert not outcome['goal_achieved']


def test_incomparable_or_mixed_reports_cannot_select_a_checkpoint():
    candidate = improved(); candidate['ocr']['image_size'] = 224
    with pytest.raises(ValueError, match='protocol'):
        compare(reports(), candidate)
    candidate = improved(); candidate['knowledge']['checkpoint_sha256'] = 'another'
    with pytest.raises(ValueError, match='different checkpoints'):
        compare(reports(), candidate)
    candidate = improved(); candidate['ocr']['results'][0]['id'] = 'different-case'
    with pytest.raises(ValueError, match='evaluation cases'):
        compare(reports(), candidate)


def test_nonfinite_or_no_reading_gain_is_not_promoted():
    assert not compare(reports(), reports())['promote_candidate']
    candidate = improved(); candidate['vision']['proper_nll'] = float('nan')
    with pytest.raises(ValueError, match='Non-finite'):
        compare(reports(), candidate)
