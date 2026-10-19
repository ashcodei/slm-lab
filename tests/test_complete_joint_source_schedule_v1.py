import copy

import pytest

from slm_lab.complete_joint_source_schedule_v1 import STREAMS, combine, digest, require_original_scope


def fixture():
    streams = {}
    for name, population, epochs in [('code', 3, 2), ('visual', 5, 1), ('human', 11, 1)]:
        rows = []
        for epoch in range(epochs):
            for start in range(0, population, 2):
                positions = list(range(start, min(start + 2, population)))
                rows.append(dict(source_step=len(rows) + 1, epoch=epoch, family=name, original_positions=positions,
                    complete_context_tokens=4 * len(positions), response_EOS_targets=2 * len(positions),
                    rank_source_counts=[len(positions[r::4]) for r in range(4)], independent_denominators={'targets': 2 * len(positions)}))
        streams[name] = dict(source_role='train', source_identity_sha256='a' * 64, entries=rows)
    return streams


def test_every_complete_stream_reference_original_epoch_and_true_partial_task_tail_survives():
    streams = fixture(); before = copy.deepcopy(streams); joined = combine(streams)
    assert streams == before and joined['actual_joint_updates'] == 6
    assert joined['task_combinations'] == {'code+visual+human': 3, 'code+human': 1, 'human': 2}
    for name in STREAMS:
        refs = [p for row in joined['entries'] for p in row['components'] if p['stream'] == name]
        assert [p['source_step'] for p in refs] == list(range(1, len(streams[name]['entries']) + 1))
        assert [p['original_entry_sha256'] for p in refs] == [digest(row) for row in streams[name]['entries']]
        assert joined['component_streams_sha256'][name] == digest(streams[name])
    assert joined['original_stream_budgets']['visual']['source_presentations'] == 5
    assert joined['original_stream_budgets']['human']['source_presentations'] == 11
    assert joined['original_stream_budgets']['human']['task_empty_rank_calls'] == 13
    assert joined['production_training_admitted'] is False
    with pytest.raises(ValueError, match='entire original'): require_original_scope(joined)


@pytest.mark.parametrize('case', ['missing_stream', 'held', 'empty', 'duplicate', 'repeated_epoch_position',
    'rank_count', 'late_step', 'bad_identity', 'truncated_context'])
def test_incomplete_or_changed_source_scope_is_rejected_before_mutating_any_input(case):
    streams = fixture()
    if case == 'missing_stream': streams.pop('visual')
    elif case == 'held': streams['human']['source_role'] = 'valid'
    elif case == 'empty': streams['visual']['entries'] = []
    elif case == 'duplicate': streams['visual']['entries'][0]['original_positions'] = [0, 0]
    elif case == 'repeated_epoch_position': streams['code']['entries'][1]['original_positions'] = [0]
    elif case == 'rank_count': streams['human']['entries'][-1]['rank_source_counts'] = [1, 1, 1, 1]
    elif case == 'late_step': streams['human']['entries'][-1]['source_step'] -= 1
    elif case == 'bad_identity': streams['code']['source_identity_sha256'] = 'z' * 64
    else: streams['visual']['entries'][-1]['complete_context_tokens'] = 1
    before = copy.deepcopy(streams)
    with pytest.raises(ValueError): combine(streams)
    assert streams == before


def test_full_scope_totals_and_task_combinations_cannot_be_replaced_by_a_smaller_curriculum():
    joint = combine(fixture())
    with pytest.raises(ValueError): require_original_scope(joint)
    # Exact counts alone are a necessary planning check, never admission.
    joint['actual_joint_updates'] = 5362
    joint['task_combinations'] = {'code+visual+human': 4733, 'code+human': 523, 'human': 106}
    for name, values in dict(code=[5256, 84084, 42271575, 21316734], visual=[4733, 302627, 17681556, 4669243],
            human=[5362, 85786, 31147346, 8514104]).items():
        joint['original_stream_budgets'][name].update(zip(['updates', 'source_presentations', 'complete_context_tokens', 'response_EOS_targets'], values))
    require_original_scope(joint)
    assert joint['production_training_admitted'] is False
    joint['original_stream_budgets']['code']['response_EOS_targets'] -= 1
    with pytest.raises(ValueError): require_original_scope(joint)
