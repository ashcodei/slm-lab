import copy

import pytest

from slm_lab.complete_joint_capacity_plan_v1 import select


def rows():
    # Distinct context and response extremes, a real partial task, a
    # correction and a late required visual batch must all survive.
    values = [
        (['code', 'human'], {'code:context': 4096, 'code:targets': 1}, [], 0),
        (['code', 'visual', 'human'], {'code:context': 100, 'code:targets': 99}, [], 0),
        (['visual', 'human'], {'word:bytes': 50}, ['visual'], 0),
        (['code', 'visual', 'human'], {'code:context': 90}, [], 2),
        (['code', 'visual', 'human'], {'code:context': 100}, [], 0),
        (['human'], {'human:context': 995}, [], 0),
        (['human'], {'human:context': 30}, [], 0),
    ]
    return [dict(step=i, original_entry_sha256=str(i) * 64, components=c, limits=l,
        empty_rank_tasks=e, corrective_pairs=n) for i, (c, l, e, n) in enumerate(values, 1)]


def test_all_distinct_extremes_partial_tasks_corrections_and_original_required_batch_survive_without_mutation():
    original = rows(); before = copy.deepcopy(original)
    chosen = select(original, required_steps={5: ['longest_actual_original_real_word']})
    assert [r['original_joint_step'] for r in chosen] == list(range(1, 8))
    assert original == before
    assert all(r['source_summary'] == original[r['original_joint_step'] - 1] for r in chosen)
    assert 'maximum_code:context' in chosen[0]['reasons']
    assert 'maximum_code:targets' in chosen[1]['reasons']
    assert 'actual_original_empty_rank_tasks' in chosen[2]['reasons']
    assert 'every_original_corrective_pair_batch' in chosen[3]['reasons']
    assert 'longest_actual_original_real_word' in chosen[4]['reasons']
    assert chosen[-1]['capacity_step'] == 7


@pytest.mark.parametrize('change', [
    lambda data: data[1].update(step=1),
    lambda data: data[1]['components'].append('code'),
    lambda data: data[1]['limits'].update({'truncated_unknown': -1}),
    lambda data: data[1].update(corrective_pairs=True),
])
def test_incomplete_or_corrupt_original_summary_is_rejected(change):
    data = rows(); change(data)
    with pytest.raises(ValueError):
        select(data, required_steps={})


def test_required_original_case_outside_the_full_epoch_is_rejected():
    with pytest.raises(ValueError):
        select(rows(), required_steps={8: ['original_tail']})
