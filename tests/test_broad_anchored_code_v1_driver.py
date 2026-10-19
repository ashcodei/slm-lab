import importlib.util
import json
from pathlib import Path
import sys

import pytest


def driver():
    root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('broad_anchored_code_driver_test',root/'scripts/train_broad_anchored_code_v1.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_exact_live_human_owner_or_worker_prevents_acquisition(tmp_path,monkeypatch):
    module=driver(); module.PREVIOUS=tmp_path; identity=dict(pid=101,ticks='201')
    monkeypatch.setattr(module.owner.resources,'same',lambda p:dict(state='R'))
    assert not module.previous_ready(identity)
    (tmp_path/'children.json').write_text(json.dumps([dict(pid=303,ticks='403')]))
    monkeypatch.setattr(module.owner.resources,'same',lambda p:dict(state='R') if p['pid']==303 else None)
    assert not module.previous_ready(identity)


def test_terminal_human_needs_both_complete_epochs_private_states_assessments_and_restoration(tmp_path,monkeypatch):
    module=driver(); module.PREVIOUS=tmp_path; identity=dict(pid=101,ticks='201')
    monkeypatch.setattr(module.owner.resources,'same',lambda p:None)
    with pytest.raises(ValueError,match='Complete human'): module.previous_ready(identity)
    (tmp_path/'plan.json').write_text(json.dumps(dict(training_options=dict(steps=10))))
    (tmp_path/'restored.json').write_text('{}')
    (tmp_path/'complete.json').write_text(json.dumps(dict(all_original_human_train_pairs_once_each_arm=True,
        all_held_human_pairs_and_native_development_assessments_complete=True,actual_source_epoch_updates_per_arm=10)))
    for arm in ['chosen_CE','human_DPO']:
        (tmp_path/(arm+'-training-complete.json')).write_text(json.dumps(dict(actual_updates=10,all_original_human_pairs_once=True)))
        (tmp_path/('train-'+arm+'-resume-distributed-state.json')).write_text(json.dumps(dict(completed=10,all_four_private_states=[dict(rank=r) for r in range(4)])))
    assert module.previous_ready(identity)
    receipt=tmp_path/'train-human_DPO-resume-distributed-state.json'; original=receipt.read_text()
    receipt.write_text(json.dumps(dict(completed=9,all_four_private_states=[dict(rank=r) for r in range(4)])))
    with pytest.raises(ValueError,match='four-replica'): module.previous_ready(identity)
    receipt.write_text(original); (tmp_path/'failure.json').write_text('{}')
    with pytest.raises(ValueError,match='Complete human'): module.previous_ready(identity)


def test_actual_complete_seeded_curriculum_keeps_all_sources_and_extreme_native_action_budget():
    from slm_lab.broad_anchored_execution_stage_v1 import source_order
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    root=Path(__file__).resolve().parents[1]; corpus=ExpandedExecutionCorpus(root)
    order=source_order(corpus,963117)
    assert len(order)==3721 and sorted(order)==list(range(3721))
    assert all(corpus.rows[p]['kind']=='documented' for p in order[:173])
    assert all(corpus.rows[p]['original']['difficulty']=='introductory' for p in order[173:2165])
    assert max(row['full_training_action_budget'] for row in corpus.rows)==3059
    assert sum(row['response_EOS_targets'] for row in corpus.rows)==584218
