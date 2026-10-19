from collections import Counter

import pytest

from slm_lab.expanded_joint_sources_v1 import CompletePhaseSchedule


class Sources:
    lengths={'code':[9,2,7,3,8,5,4,6,10],'book':[4,6,3],'QA':[8,3,4,5,2]}
    def phases(self):return dict(foundation=['code','book'],instruction_and_vision=['QA'])


def test_complete_phases_keep_all_original_rows_once_and_empty_final_ranks():
    data=Sources();widths=dict(code=4,book=4,QA=4);schedule=CompletePhaseSchedule(data,widths,719)
    observed=Counter();empty_ranks=0;last='foundation'
    for step in range(len(schedule)):
        name,positions=schedule.at(step);phase=schedule.phase_for_step[step]
        if last=='instruction_and_vision':assert phase==last
        last=phase;observed.update((name,p) for p in positions)
        shards=[schedule.rank_positions(step,rank,4)[1] for rank in range(4)]
        assert sorted(p for shard in shards for p in shard)==sorted(positions)
        empty_ranks+=sum(not shard for shard in shards)
    assert observed==Counter((name,p) for name,values in data.lengths.items() for p in range(len(values)))
    assert empty_ranks and schedule.entries==CompletePhaseSchedule(data,widths,719).entries
    with pytest.raises(IndexError):schedule.at(len(schedule))


def test_missing_source_family_width_is_rejected_before_a_smaller_pass_can_be_claimed():
    with pytest.raises(ValueError,match='every complete source'):
        CompletePhaseSchedule(Sources(),dict(code=4,book=4),719)
