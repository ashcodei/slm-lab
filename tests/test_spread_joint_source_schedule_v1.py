import copy
from itertools import combinations

import pytest

from slm_lab.complete_joint_source_schedule_v1 import STREAMS, digest
from slm_lab.spread_joint_source_schedule_v1 import spread
from test_complete_joint_source_schedule_v1 import fixture


def test_same_whole_sources_and_order_end_together_without_modality_free_suffixes():
    streams=fixture(); before=copy.deepcopy(streams); result=spread(streams)
    assert streams==before and result['actual_joint_updates']==6
    for name in STREAMS:
        actual=[part for row in result['entries'] for part in row['components'] if part['stream']==name]
        assert [p['source_step'] for p in actual]==list(range(1,len(streams[name]['entries'])+1))
        assert [p['original_entry_sha256'] for p in actual]==[digest(r) for r in streams[name]['entries']]
        assert result['actual_spaced_gaps'][name]['first_active_step']==1
        assert result['actual_spaced_gaps'][name]['last_active_step']==6
    assert [p['stream'] for p in result['entries'][0]['components']]==list(STREAMS)
    assert [p['stream'] for p in result['entries'][-1]['components']]==list(STREAMS)
    assert result['original_prefix_gaps']['visual']['longest_consecutive_steps_without_task']==3
    assert result['actual_spaced_gaps']['visual']['longest_consecutive_steps_without_task']==2
    assert result['production_training_admitted'] is False and result['model_retention_or_quality_gain_unmeasured']


def test_independent_exhaustive_small_schedule_oracle_proves_minimal_maximum_gap_with_fixed_endpoints():
    for horizon in range(2,9):
        for size in range(2,horizon+1):
            streams=fixture()
            for name,count in [('code',size),('visual',horizon),('human',horizon)]:
                rows=[]
                for i in range(count):
                    rows.append(dict(source_step=i+1,epoch=0,family=name,original_positions=[i],complete_context_tokens=4,
                        response_EOS_targets=2,rank_source_counts=[1,0,0,0]))
                streams[name]['entries']=rows
            result=spread(streams)
            optimum=min(max(b-a-1 for a,b in zip([1,*inside],[*inside,horizon]))
                for inside in combinations(range(2,horizon),size-2))
            assert result['actual_spaced_gaps']['code']['longest_consecutive_steps_without_task']==optimum


def test_single_component_occurrence_is_retained_once_at_the_final_step():
    streams=fixture();streams['visual']['entries']=streams['visual']['entries'][:1]
    result=spread(streams)
    locations=[r['step'] for r in result['entries'] if any(p['stream']=='visual' for p in r['components'])]
    assert locations==[6] and result['actual_spaced_gaps']['visual']['actual_component_steps']==1
