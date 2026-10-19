from types import SimpleNamespace

import pytest

from slm_lab.focused_code_continuation_data_v2 import FocusedContinuationSchedule as PreviousSchedule
from slm_lab.focused_code_continuation_data_v3 import CODE,EXECUTION,STATE,REPLAY,FocusedContinuationSchedule


def fixture():
    lengths={CODE:[64+i%100 for i in range(193)],EXECUTION:[13+i%10 for i in range(21)],STATE:[27+i%50 for i in range(45)],
        **{name:[8+i%200 for i in range(199)] for name in REPLAY}}
    return SimpleNamespace(purpose='optimization',lengths=lengths,targets={n:[v-1 for v in vs] for n,vs in lengths.items()})


def test_full_state_epoch_preserves_every_old_code_output_and_all12_replay_positions_and_tails():
    data=fixture();schedule=FocusedContinuationSchedule(data,width=8);old=PreviousSchedule(data,width=8)
    assert [(n,p) for n,p in schedule.entries if n!=STATE]==old.entries
    for name in [CODE,EXECUTION,STATE]:
        assert sorted(p for n,ps in schedule.entries if n==name for p in ps)==list(range(len(data.lengths[name])))
    assert set(REPLAY)<=set(n for n,_ in schedule.entries)
    tails=[ps for n,ps in schedule.entries if n==STATE and len(ps)<8]
    assert len(tails)==1 and sorted(p for rank in range(4) for p in tails[0][rank::4])==sorted(tails[0])
    assert len(schedule)==len(old)+6


def test_state_or_public_output_cannot_silently_be_removed_from_full_scope():
    for name in [STATE,EXECUTION]:
        data=fixture();del data.lengths[name]
        with pytest.raises(ValueError,match='Both admitted'):FocusedContinuationSchedule(data,width=8)
