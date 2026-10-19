import importlib.util
import json
from pathlib import Path

import pytest


def driver():
    import sys
    path=Path(__file__).resolve().parents[1]/'scripts/train_shared_words_v1.py'
    sys.path.insert(0,str(path.parent));spec=importlib.util.spec_from_file_location('shared_word_driver_test',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def test_exact_live_prior_owner_and_workers_block_despite_stale_completion_and_bad_endpoints_reject(monkeypatch,tmp_path):
    d=driver();monkeypatch.setattr(d,'PREVIOUS',tmp_path)
    prior=dict(pid=11,ticks='101');child=dict(pid=12,ticks='102')
    (tmp_path/'children.json').write_text(json.dumps([child]));(tmp_path/'complete.json').write_text('{}')
    monkeypatch.setattr(d.owner.resources,'same',lambda a:a if a==prior else None)
    assert d.previous_ended(prior) is False
    monkeypatch.setattr(d.owner.resources,'same',lambda a:a if a==child else None)
    assert d.previous_ended(prior) is False
    monkeypatch.setattr(d.owner.resources,'same',lambda a:None)
    with pytest.raises(ValueError,match='successfully'):d.previous_ended(prior)
    (tmp_path/'restored.json').write_text('{}');(tmp_path/'independent-comparison.json').write_text('{}')
    assert d.previous_ended(prior)
    (tmp_path/'failure.json').write_text('{}')
    with pytest.raises(ValueError,match='successfully'):d.previous_ended(prior)


def test_active_original_work_rejected_before_GPU_queries_or_signals(monkeypatch):
    d=driver()
    def active():raise ValueError('Original work is active')
    monkeypatch.setattr(d.owner,'idle_original',active)
    monkeypatch.setattr(d.subprocess,'check_output',lambda *a,**k:pytest.fail('No GPU query before original idle admission'))
    monkeypatch.setattr(d.os,'kill',lambda *a,**k:pytest.fail('No signals to active original work'))
    with pytest.raises(ValueError,match='active'):d.require_idle_devices()


def test_complete_multi_epoch_exposure_keeps_last_partial_batch_and_never_rounds_source_repeats():
    from slm_lab.shared_word_evidence_v1 import budget,positions
    from test_shared_word_stage_v1 import options,synthetic_words
    data=synthetic_words();setting=options(data)
    draws=[positions(data,setting,step) for step in range(setting['steps'])]
    assert [len(d) for d in draws]==[2,2,1,2,2,1]
    assert sorted(sum(draws[:3],[]))==sorted(sum(draws[3:],[]))==list(range(5))
    actual=budget(data,setting)
    assert actual['source_population']==5 and actual['source_exposures']==10
    assert actual['unique_original_sources']==5 and actual['complete_passes']==2 and actual['repeated_rounding_exposures']==0
    assert actual['UTF8_byte_targets']==24 and actual['complete_context_tokens']==70 and actual['response_EOS_targets']==40
