from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_public_pretraining_sources_CPU_v1 as driver


def test_every_previous_image_word_state_code_context_and_target_is_required_with_full_raw_billion():
    old = {n:dict(presentations=7,complete_context_tokens=50,response_EOS_targets=20,updates=2)
           for n in ['code','state','photo','words','broad']}
    selected = deepcopy(old)
    for row in selected.values():row['updates'] = 1
    selected[driver.RAW] = dict(presentations=300000,complete_context_tokens=1000000001,response_EOS_targets=10**9,updates=4000)
    driver.validate_selected_budget(selected,old)
    for family in old:
        wrong = deepcopy(selected);wrong[family]['response_EOS_targets'] -= 1
        with pytest.raises(ValueError,match='No previous'):driver.validate_selected_budget(wrong,old)
    wrong = deepcopy(selected);wrong[driver.RAW]['response_EOS_targets'] -= 1
    with pytest.raises(ValueError,match='full billion'):driver.validate_selected_budget(wrong,old)


def test_live_exact_source_reader_cannot_be_replaced_by_a_prefix_admission(tmp_path,monkeypatch):
    import json
    reader = tmp_path/'reader';reader.mkdir()
    (reader/'plan.json').write_text(json.dumps(dict(owner=driver.READER_OWNER)))
    monkeypatch.setattr(driver,'READER',reader)
    monkeypatch.setattr(driver,'terminal',lambda owner:False)
    assert not driver.dependencies_ready()
    monkeypatch.setattr(driver,'terminal',lambda owner:True)
    with pytest.raises(ValueError,match='not an admitted prefix'):driver.dependencies_ready()
