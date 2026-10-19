from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_public_pretraining_sources_CPU_v2 as driver


def test_all_original_billion_raw_and_multimodal_retention_and_every_new_function_are_required():
    previous={n:dict(presentations=20,complete_context_tokens=500,response_EOS_targets=200,updates=5)
        for n in ['words','photo','state','code']}
    previous[driver.RAW]=dict(presentations=300000,complete_context_tokens=1000500000,response_EOS_targets=10**9,updates=4000)
    selected=deepcopy(previous)
    selected[driver.FUNCTIONS]=dict(presentations=826,complete_context_tokens=797062,response_EOS_targets=389458,updates=4)
    driver.validate_selected_budget(selected,previous)
    for name in previous:
        bad=deepcopy(selected);bad[name]['response_EOS_targets']-=1
        with pytest.raises(ValueError):driver.validate_selected_budget(bad,previous)
    bad=deepcopy(selected);bad[driver.FUNCTIONS]['presentations']-=1
    with pytest.raises(ValueError,match='every826'):driver.validate_selected_budget(bad,previous)


def test_live_or_incomplete_independent_source_predecessors_cannot_be_substituted(tmp_path,monkeypatch):
    (tmp_path/'plan.json').write_text(json.dumps(dict(owner=dict(pid=123,ticks='456'))))
    monkeypatch.setattr(driver,'terminal',lambda owner:False)
    assert not driver.stage_ready(tmp_path,'dataset_admitted_to_function_CE_training')
    monkeypatch.setattr(driver,'terminal',lambda owner:True)
    with pytest.raises(ValueError,match='must finish'):driver.stage_ready(tmp_path,'dataset_admitted_to_function_CE_training')
    (tmp_path/'complete.json').write_text(json.dumps(dict(complete=True,dataset_admitted_to_function_CE_training=True)))
    assert driver.stage_ready(tmp_path,'dataset_admitted_to_function_CE_training')
