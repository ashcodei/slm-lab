import importlib.util
import json
import sys
from pathlib import Path

import pytest


def driver():
    path=Path(__file__).resolve().parents[1]/'scripts/train_dense_human_preference_v1.py'
    sys.path.insert(0,str(path.parent));spec=importlib.util.spec_from_file_location('dense_preference_driver_test',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def test_exact_prior_owner_and_children_block_even_with_stale_completion_and_failure_rejects(monkeypatch,tmp_path):
    d=driver();monkeypatch.setattr(d,'PREVIOUS',tmp_path);prior=dict(pid=10,ticks='100');child=dict(pid=11,ticks='101')
    (tmp_path/'children.json').write_text(json.dumps([child]));(tmp_path/'complete.json').write_text('{}')
    monkeypatch.setattr(d.owner.resources,'same',lambda a:a if a==prior else None);assert d.previous_ended(prior) is False
    monkeypatch.setattr(d.owner.resources,'same',lambda a:a if a==child else None);assert d.previous_ended(prior) is False
    monkeypatch.setattr(d.owner.resources,'same',lambda a:None)
    with pytest.raises(ValueError,match='successfully'):d.previous_ended(prior)
    for name in ('restored.json','independent-comparison.json'):(tmp_path/name).write_text('{}')
    assert d.previous_ended(prior)
    (tmp_path/'failure.json').write_text('{}')
    with pytest.raises(ValueError,match='successfully'):d.previous_ended(prior)


def test_active_original_work_rejects_before_GPU_query_or_any_signal(monkeypatch):
    d=driver()
    def active():raise ValueError('Original active work')
    monkeypatch.setattr(d.owner,'idle_original',active)
    monkeypatch.setattr(d.subprocess,'check_output',lambda *a,**k:pytest.fail('GPU query before original admission'))
    monkeypatch.setattr(d.os,'kill',lambda *a,**k:pytest.fail('Signal to active original work'))
    with pytest.raises(ValueError,match='active'):d.require_idle_devices()


def test_complete_control_and_DPO_exposures_match_sources_with_no_rounded_pairs_and_distinct_reference_use():
    from test_dense_human_preference_stage_v1 import pairs,options
    from slm_lab.dense_preference_evidence_v1 import budget,positions
    from slm_lab.codenet_stage_v1 import EpochOrder
    data=pairs();setting=options(data);order=EpochOrder(len(data),setting['seed'])
    draws=[positions(data,setting,order,s) for s in range(setting['steps'])]
    assert [len(row) for row in draws]==[2,2,1,2,2,1]
    assert sorted(sum(draws[:3],[]))==sorted(sum(draws[3:],[]))==list(range(5))
    dpo=budget(data,setting);ce=budget(data,{**setting,'DPO_weight':0})
    assert dpo['pair_exposures']==ce['pair_exposures']==10 and dpo['rounded_pair_repeats']==ce['rounded_pair_repeats']==0
    assert dpo['actual_policy_context_tokens']==dpo['actual_reference_context_tokens']==110
    assert ce['actual_policy_context_tokens']==50 and ce['actual_reference_context_tokens']==0
    assert ce['actual_policy_rejected_targets']==0 and dpo['actual_policy_rejected_targets']==40
    assert dpo['actual_last_update_pairs']==ce['actual_last_update_pairs']==1


def test_full_original_word_population_and_predeclared_merge_loss_bound():
    import copy
    d=driver();ids=[str(i) for i in range(200)]
    source=dict(results=[dict(id=i) for i in ids],overall=dict(examples=200,conditions=dict(proper=dict(CTC_exact=100))))
    fused=copy.deepcopy(source);fused['overall']['conditions']['proper']['CTC_exact']=98
    assert d.word_merge_quality(source,fused,ids)['passed']
    fused['overall']['conditions']['proper']['CTC_exact']=97
    assert not d.word_merge_quality(source,fused,ids)['passed']
    fused['results'].pop()
    with pytest.raises(ValueError,match='Every original'):d.word_merge_quality(source,fused,ids)
