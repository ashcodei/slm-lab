"""Streaming accounting against real closed whole-source CPU observations."""
import copy
import gc
import json
import sys

import pytest

from slm_lab.native_history_summary_v1 import Summary, summarize
from slm_lab.parallel_native_suffix_stage_v1 import CE_ONLY, KERNEL
from test_indexed_native_history_v1 import actual_rows


def test_real_complete_native_CE_only_and_empty_tail_counts_match_original_list_oracle():
    rows, _ = actual_rows()
    actual = summarize(iter(rows), group_size=8)
    kernels = [o['kernel'] for r in rows for o in r['actual_rank_observations'] if o['kernel'] is not None]
    native = [k for k in kernels if k['format'] == KERNEL]
    ce = [k for k in kernels if k['format'] == CE_ONLY]
    assert actual['actual_complete_updates'] == 3 and actual['actual_source_presentations'] == 9
    assert actual['actual_execution_eligible_presentations'] == len(native) == 8
    assert actual['actual_complete_CE_only_presentations'] == len(ce) == 1
    assert actual['actual_unconditional_original_CE_targets'] == sum(k['unconditional_original_CE_anchor']['local_original_response_EOS_targets'] for k in kernels) == 612
    assert actual['actual_original_context_tokens'] == sum(r['complete_original_context_tokens'] for r in rows)
    assert actual['actual_single_source_three_empty_rank_steps'] == 1
    assert actual['actual_committed_prefix_trajectory_count'] == sum(len(k['trajectories']) for k in native) == 64
    assert actual['actual_committed_prefix_generated_action_count'] == sum(k['actual_sampled_action_tokens'] for k in native)
    assert actual['actual_genuine_EOS_stops'] == sum(t['actual_stopped_on_EOS'] for k in native for t in k['trajectories'])
    assert actual['actual_reward1_trajectories'] == sum(v == 1. for k in native for v in k['actual_rewards'])
    assert actual['actual_reward_varying_groups'] == sum(len(set(v for v in k['actual_rewards'] if v is not None)) > 1 for k in native)
    assert actual['actual_abstaining_groups'] == sum(k['policy_gradient_abstained_due_to_missing_verdict'] for k in native)
    assert not actual['production_training_admitted']


@pytest.mark.parametrize('fault', ['lost_anchor', 'missing_attempt', 'wrong_rank', 'invented_empty_forward', 'wrong_actions', 'wrong_step'])
def test_invalid_next_record_keeps_entire_previous_prefix_exact(fault):
    rows, _ = actual_rows()
    count = Summary(group_size=8)
    count.append(rows[0]); before = count.result()
    wrong = copy.deepcopy(rows[1])
    if fault == 'lost_anchor': wrong['actual_rank_observations'][0]['kernel']['unconditional_original_CE_anchor']['local_original_response_EOS_targets'] -= 1
    elif fault == 'missing_attempt': wrong['actual_rank_observations'][1]['kernel']['trajectories'].pop()
    elif fault == 'wrong_rank': wrong['actual_rank_observations'][1]['rank'] = True
    elif fault == 'invented_empty_forward':
        wrong = copy.deepcopy(rows[2]); wrong['step'] = 2
        wrong['actual_rank_observations'][3]['kernel'] = copy.deepcopy(wrong['actual_rank_observations'][0]['kernel'])
    elif fault == 'wrong_actions': wrong['actual_rank_observations'][1]['kernel']['actual_sampled_action_tokens'] += 1
    else: wrong['step'] = True
    with pytest.raises(ValueError): count.append(wrong)
    assert count.result() == before
    count.append(rows[1]); count.append(rows[2])
    assert count.result() == summarize(rows, group_size=8)


def test_missing_verdicts_remain_real_abstentions_and_changed_semantics_change_digest():
    rows, _ = actual_rows()
    original = summarize(rows, group_size=8)
    changed = copy.deepcopy(rows)
    kernel = changed[0]['actual_rank_observations'][0]['kernel']
    kernel['actual_rewards'][0] = None
    kernel['policy_gradient_abstained_due_to_missing_verdict'] = True
    actual = summarize(changed, group_size=8)
    assert actual['actual_abstaining_groups'] == original['actual_abstaining_groups'] + 1
    assert actual['actual_mathematical_history_sha256'] != original['actual_mathematical_history_sha256']


def test_long_source_shaped_generator_never_retains_observation_history(tmp_path):
    rows, reference = actual_rows()
    reference_counts = []
    def stream():
        for i in range(48):
            # Repeated closed records with only their transport counter changed;
            # these are not new model actions or source-training updates.
            row = dict(rows[i % 3])
            row['step'] = i + 1
            yield row
            # One generator local, the consumer loop variable, and this
            # temporary getrefcount argument. Retaining any prior row adds
            # another reference and fails this actual plain-dict check.
            reference_counts.append(sys.getrefcount(row))
    actual = summarize(stream(), group_size=8)
    gc.collect()
    assert max(reference_counts) == 3
    assert actual['actual_complete_updates'] == 48 and actual['actual_source_presentations'] == 16 * 9
    assert actual['actual_unconditional_original_CE_targets'] == 16 * 612
    (tmp_path / 'actual-streamed-history-proof.json').write_text(json.dumps(dict(
        actual_closed_source_map_entries=len(reference['sources']), actual_source_shaped_record_count=48,
        actual_maximum_record_reference_count_after_processing=max(reference_counts),
        expected_only_generator_consumer_and_getrefcount_argument=3,
        no_processed_observation_retained_by_summary=True, actual_summary=actual,
        original_closed_CPU_fixture_only=True, source_counter_changes_not_new_neural_actions=True,
        production_updates=0, full_size_GPUcapacity_or_quality_gain=False), indent=2))
