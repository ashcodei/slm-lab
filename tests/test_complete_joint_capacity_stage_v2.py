import copy

import pytest

from slm_lab.complete_joint_capacity_stage_v2 import capacity_metric
from slm_lab.complete_joint_source_schedule_v1 import digest
from slm_lab.parallel_function_repair_stage_v1 import value_digest


def control():
    original_entry = dict(step=297, objectives=[dict(task='native_image', positive_EOS=409, negative_EOS=20)])
    local_entry = dict(original_entry, step=17)
    case = dict(original_joint_step=297, original_checkpoint_entry_sha256=value_digest(original_entry))
    metric = dict(step=297, entry_sha256=value_digest(original_entry), normalized_task_losses=[.25],
        gradient_interaction={'actual_all_rank_tasks': [{'rank': r, 'original': 'unchanged'} for r in range(4)]})
    return original_entry, local_entry, case, metric


def test_original_bridge_typed_digest_rebases_only_capacity_cursor_and_preserves_all_raw_records():
    original, entry, case, metric = control(); before = copy.deepcopy(metric)
    assert digest(original) != value_digest(original)
    actual = capacity_metric(metric, case, entry, 'a' * 64)
    assert actual['step'] == 17 and actual['entry_sha256'] == value_digest(entry)
    assert actual['normalized_task_losses'] == metric['normalized_task_losses']
    assert actual['gradient_interaction']['actual_all_rank_tasks'] == metric['gradient_interaction']['actual_all_rank_tasks']
    assert actual['gradient_interaction']['actual_original_joint_step'] == 297
    assert actual['gradient_interaction']['actual_original_checkpoint_entry_sha256'] == value_digest(original)
    assert metric == before


@pytest.mark.parametrize('where', ['workload', 'actual_bridge'])
def test_wrong_JSON_digest_protocol_is_rejected_before_journaling_or_optimizer(where):
    original, entry, case, metric = control()
    if where == 'workload':
        case['original_checkpoint_entry_sha256'] = digest(original)
    else:
        metric['entry_sha256'] = digest(original)
    with pytest.raises(ValueError, match='typed entry digest'):
        capacity_metric(metric, case, entry, 'a' * 64)
