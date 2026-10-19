from types import SimpleNamespace

import pytest

from slm_lab.focused_code_continuation_data_v1 import REPLAY as PREVIOUS_REPLAY
from slm_lab.focused_code_continuation_data_v2 import CODE,EXECUTION,RAW_REPLAY,REPLAY,FocusedContinuationSchedule


def fixture():
    lengths={CODE:[64+i%100 for i in range(193)],EXECUTION:[13+i%10 for i in range(21)],
        **{name:[8+i%200 for i in range(199)] for name in REPLAY}}
    return SimpleNamespace(purpose='optimization',lengths=lengths,targets={name:[n-1 for n in values] for name,values in lengths.items()})


def test_retention_extension_keeps_all_original_replay_and_exactly_once_new_sources_and_all_tail_ranks():
    data=fixture();schedule=FocusedContinuationSchedule(data,width=8)
    assert set(PREVIOUS_REPLAY)<set(REPLAY) and set(RAW_REPLAY)<=set(REPLAY)
    assert set(name for name,_ in schedule.entries)>=set(REPLAY)
    for name in [CODE,EXECUTION]:
        assert sorted(p for family,positions in schedule.entries if family==name for p in positions)==list(range(len(data.lengths[name])))
    for _,positions in schedule.entries:
        assert sorted(p for rank in range(4) for p in positions[rank::4])==sorted(positions)
    assert all(schedule.selected_budget(data)[name]['response_EOS_targets']>0 for name in RAW_REPLAY)


def test_missing_raw_family_or_unadmitted_auxiliary_cannot_silently_change_production_scope():
    data=fixture();del data.lengths[RAW_REPLAY[0]]
    with pytest.raises(ValueError):FocusedContinuationSchedule(data,width=8)
    data=fixture();del data.lengths[EXECUTION]
    with pytest.raises(ValueError):FocusedContinuationSchedule(data,width=8)
    data.purpose='source_preparation';schedule=FocusedContinuationSchedule(data,width=8)
    assert set(name for name,_ in schedule.entries)>=set(REPLAY)
    assert all(name!=EXECUTION for name,_ in schedule.entries)
