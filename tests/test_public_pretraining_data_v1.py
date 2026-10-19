from collections import Counter
from types import SimpleNamespace

import pytest

from slm_lab.public_pretraining_data_v1 import RAW, TokenBudgetSchedule


def data():
    return SimpleNamespace(role='train', purpose='optimization', retention_identity_sha256='a'*64,
        lengths={RAW:[4096]*17+[100], 'code':[10,25,50], 'photos':[20,12], 'words':[9]},
        targets={RAW:[4095]*17+[98], 'code':[5,10,30], 'photos':[5,3], 'words':[3]},
        kinds={RAW:'text', 'code':'text', 'photos':'image', 'words':'word'},
        retention_entries=[('code',[0,1,0]), ('photos',[0,1]), ('words',[0]), ('code',[2])])


def test_every_raw_window_once_and_exact_repeated_replay_exposures_survive_compute_bucketing():
    d = data()
    schedule = TokenBudgetSchedule(d, context_budget=16384, maximum_rows=7, image_prefix=64)
    same = TokenBudgetSchedule(d, context_budget=16384, maximum_rows=7, image_prefix=64)
    assert schedule.entries == same.entries and schedule.identity_sha256 == same.identity_sha256
    actual = Counter((name,p) for name,positions in schedule.entries for p in positions)
    expected = Counter((RAW,p) for p in range(len(d.lengths[RAW])))
    expected.update((name,p) for name,positions in d.retention_entries for p in positions)
    assert actual == expected and actual['code',0] == 2
    for name, positions in schedule.entries:
        prefix = 64 if d.kinds[name] in {'image','word'} else 0
        assert len(positions) <= 7
        assert len(positions)*(max(d.lengths[name][p] for p in positions)+prefix) <= 16384
        shards = [positions[rank::4] for rank in range(4)]
        assert Counter(p for shard in shards for p in shard) == Counter(positions)
    totals = schedule.selected_budget(d)
    assert totals[RAW]['response_EOS_targets'] == sum(d.targets[RAW])
    assert totals['code']['presentations'] == 4
    assert any(len(p) < 4 for _,p in schedule.entries)


def test_held_role_and_oversize_whole_examples_are_rejected_instead_of_clipped_or_dropped():
    d = data();d.role = 'valid'
    with pytest.raises(ValueError, match='Admitted train'):
        TokenBudgetSchedule(d)
    d = data()
    with pytest.raises(ValueError, match='no source clipping'):
        TokenBudgetSchedule(d, context_budget=4000)
