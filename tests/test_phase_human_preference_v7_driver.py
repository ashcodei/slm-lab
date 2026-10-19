import importlib.util
import json
from pathlib import Path
import sys

import pytest


def driver():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'scripts'))
    spec = importlib.util.spec_from_file_location('phase_human_preference_driver_test',
        root/'scripts/train_phase_human_preference_v7.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_exact_foundation_or_child_prevents_GPU_acquisition(tmp_path, monkeypatch):
    module = driver()
    module.PREVIOUS = tmp_path
    owner = dict(pid=101, ticks='202')
    monkeypatch.setattr(module.owner.resources, 'same', lambda actor:dict(state='R'))
    assert not module.previous_ready(owner)
    (tmp_path/'children.json').write_text(json.dumps([dict(pid=303,ticks='404')]))
    monkeypatch.setattr(module.owner.resources, 'same', lambda actor:dict(state='R') if actor['pid'] == 303 else None)
    assert not module.previous_ready(owner)


def test_terminal_foundation_needs_full_epoch_assessments_restoration_and_no_failure(tmp_path, monkeypatch):
    module = driver()
    module.PREVIOUS = tmp_path
    owner = dict(pid=101, ticks='202')
    monkeypatch.setattr(module.owner.resources, 'same', lambda actor:None)
    (tmp_path/'children.json').write_text('[]')
    with pytest.raises(ValueError, match='Complete code-first'):
        module.previous_ready(owner)
    for name, value in {
        'restored.json':{}, 'training-complete.json':dict(actual_updates=10,all58488new_code_and_all_admitted_public_execution_targets_once=True,all_admitted_public_state_examples_once=True),
        'complete.json':dict(all58488new_code_and_all_admitted_execution_examples_once=True,all_admitted_public_state_examples_once=True,original_local_code_and_broad_assessments_complete=True),
        'plan.json':dict(planned_steps=10,source_start_is_complete_main=True,complete_new_code_and_execution_source_epoch_once=True,complete_public_state_source_epoch_once=True,complete_source_budget={'new:public_source_state_prediction':{'examples':1}},selected_source_budget={'new:public_source_state_prediction':{'presentations':1}})}.items():
        (tmp_path/name).write_text(json.dumps(value))
    assert module.previous_ready(owner)
    source_plan=json.loads((tmp_path/'plan.json').read_text());source_plan['complete_public_state_source_epoch_once']=False
    (tmp_path/'plan.json').write_text(json.dumps(source_plan))
    with pytest.raises(ValueError,match='Actual full'):module.previous_ready(owner)
    source_plan['complete_public_state_source_epoch_once']=True;(tmp_path/'plan.json').write_text(json.dumps(source_plan))
    (tmp_path/'training-complete.json').write_text(json.dumps(dict(actual_updates=9)))
    with pytest.raises(ValueError, match='Actual full'):
        module.previous_ready(owner)
    (tmp_path/'training-complete.json').write_text(json.dumps(dict(actual_updates=10,all58488new_code_and_all_admitted_public_execution_targets_once=True,all_admitted_public_state_examples_once=True)))
    (tmp_path/'complete.json').write_text(json.dumps(dict(all58488new_code_and_all_admitted_execution_examples_once=False,original_local_code_and_broad_assessments_complete=True)))
    with pytest.raises(ValueError, match='Actual full'):
        module.previous_ready(owner)
    (tmp_path/'failure.json').write_text('{}')
    with pytest.raises(ValueError, match='Complete code-first'):
        module.previous_ready(owner)


def test_actual_human_cohort_width16_order_covers_every_pair_once_including_partial_final_batch():
    import math
    import torch
    from slm_lab.parallel_human_preference_v2 import ParallelHumanPreferenceStage
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from test_own_dense_actor_v1 import actor
    root = Path(__file__).resolve().parents[1]
    data = WholeHumanData(root/'data/whole-human-alignment-v2', 'preferences', 'train', 'optimization')
    model = actor()
    options = dict(seed=959117,epochs=1,steps=math.ceil(len(data)/16),microbatch=1,accumulation=16,
        learning_rate=3e-6,betas=[.9,.95],weight_decay=.01,gradient_clip=1.,DPO_beta=.1,DPO_weight=1,chosen_CE_weight=.1)
    optimizer = torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],
        lr=3e-6,betas=(.9,.95),weight_decay=.01,foreach=False)
    stage = ParallelHumanPreferenceStage(model,optimizer,data,options,{'source_fixture':'a'*64})
    batches = [stage.positions(i) for i in range(options['steps'])]
    assert len(data) == 85786 and len(batches) == 5362
    assert sorted(p for batch in batches for p in batch) == list(range(len(data)))
    assert [len(batches[-1][rank::4]) for rank in range(4)] == [3,3,2,2]
    assert stage.cursor(options['steps'])['actual_pair_draws'] == len(data)
