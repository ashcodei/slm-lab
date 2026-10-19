import importlib.util
import json
from pathlib import Path

import pytest


def driver():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('conditional_CPU_trial_unit',root/'scripts/train_conditional_prefix_full_CPU_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_source_rule_requires_all_four_same_condition_rewards_and_uses_only_original_training_order(tmp_path):
    module=driver();module.PREVIOUS=tmp_path
    (tmp_path/'plan.json').write_text(json.dumps(dict(selected_original_train_source_ids=['first','second','third'])))
    report=dict(summary={'3:full_vocabulary_temperature1':dict(source_rewards={
        'first':[0.,0.,0.,0.],'second':[0.,1.,0.,0.],'third':[1.,0.,1.,0.]})})
    assert module.selected_source(report)=='second'
    report['summary']['3:full_vocabulary_temperature1']['source_rewards']['first']=[0.,1.,0.]
    with pytest.raises(ValueError,match='four-sample'):module.selected_source(report)
    report['summary']['3:full_vocabulary_temperature1']['source_rewards']={
        'first':[1.,1.,1.,1.],'second':[0.,0.,0.,0.],'third':[1.,1.,1.,1.]}
    with pytest.raises(ValueError,match='no within-source'):module.selected_source(report)


def test_missing_or_reused_predecessor_PID_is_not_a_live_wait(tmp_path,monkeypatch):
    module=driver()
    assert module.same(dict(pid=999999999,ticks='1')) is None
