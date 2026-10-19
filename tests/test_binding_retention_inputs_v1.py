from copy import deepcopy
import pytest

from slm_lab.binding_retention_inputs_v1 import update_rows


def fixture(monkeypatch):
    import slm_lab.binding_retention_inputs_v1 as module
    monkeypatch.setattr(module, 'complete_encoding', lambda row, encoder: {'target_tokens': row['targets'], 'context_tokens': row['context']})
    names = ['Python', 'JavaScript', 'verified_python', 'human_instruction', 'retention']
    pools = {name: [{'id': name, 'targets': 10, 'context': 30}] for name in names}
    picks = [{'pool': name, 'index': 0, 'id': name, 'update': 1, 'target_tokens': 10, 'context_tokens': 30} for name in names]
    return pools, picks[:4], picks[4]


def test_exact_five_complete_source_rows_are_returned(monkeypatch):
    pools, primary, rehearsal = fixture(monkeypatch)
    assert [r['id'] for r in update_rows(pools, primary, rehearsal, None)] == ['Python', 'JavaScript', 'verified_python', 'human_instruction', 'retention']


@pytest.mark.parametrize('change', ['id', 'target_count', 'context_count', 'index', 'update', 'same_language', 'image'])
def test_foreign_or_incomplete_exposure_cannot_reach_training(monkeypatch, change):
    pools, primary, rehearsal = fixture(monkeypatch)
    if change == 'id': rehearsal['id'] = 'unrelated-source'
    elif change == 'target_count': primary[2]['target_tokens'] = 9
    elif change == 'context_count': primary[2]['context_tokens'] = 29
    elif change == 'index': rehearsal['index'] = -1
    elif change == 'update': rehearsal['update'] = 2
    elif change == 'same_language': primary[1] = deepcopy(primary[0])
    else: pools['retention'][0]['image'] = 'foreign-image.jpg'
    with pytest.raises(ValueError): update_rows(pools, primary, rehearsal, None)
