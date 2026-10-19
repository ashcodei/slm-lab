import copy

import pytest

from slm_lab.code_continuation_receipt import verify_targets


def report():
    return {'stage': 'pretrain', 'trainable': 'language', 'world_size': 4, 'steps': 2, 'planned_steps': 2,
            'truncated_sequences': 0, 'gradient_accumulation': 16,
            'metrics': [{'step': i, 'global_target_tokens': 262080} for i in [1, 2]]}


def test_actual_full_window_exposures_are_verified_independently_of_step_completion():
    receipt = verify_targets(report(), 2, 4096)
    assert receipt['actual_target_exposures'] == 524160 and receipt['targets_per_update'] == 262080
    incomplete = report(); incomplete['metrics'][1]['global_target_tokens'] -= 4095
    with pytest.raises(ValueError, match='exposures'): verify_targets(incomplete, 2, 4096)


@pytest.mark.parametrize('change', ['missing_update', 'duplicate_update', 'wrong_world', 'truncated', 'wrong_component'])
def test_complete_status_cannot_hide_incomplete_or_changed_training(change):
    value = copy.deepcopy(report())
    if change == 'missing_update': value['metrics'].pop()
    elif change == 'duplicate_update': value['metrics'][1]['step'] = 1
    elif change == 'wrong_world': value['world_size'] = 1
    elif change == 'truncated': value['truncated_sequences'] = 1
    else: value['trainable'] = 'all'
    with pytest.raises(ValueError): verify_targets(value, 2, 4096)
