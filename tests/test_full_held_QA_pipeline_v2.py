import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def driver():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('full_QA_pipeline_unit',root/'scripts/assess_full_held_QA_pipeline_v2.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_eight_exact_own_native_endpoints_are_fixed_before_scores(tmp_path):
    module=driver();values=module.descriptors(tmp_path)
    assert tuple(values)==module.ENDPOINTS and len(values)==8
    assert values['human_chosen_CE']['arm']=='chosen_CE' and values['human_DPO']['arm']=='human_DPO'
    assert values['phase_trained']['checkpoint']==str(tmp_path/'checkpoints/phase-joint-v3/trained/model.pt')
    assert values['code_anchor_execution']['checkpoint']==str(tmp_path/'checkpoints/broad-anchored-code-v4/anchor_execution-actor.pt')
    assert not any('resume' in v['checkpoint'] for v in values.values())


def test_live_predecessor_owner_or_worker_prevents_new_GPU_acquisition(tmp_path,monkeypatch):
    module=driver();module.PREVIOUS=tmp_path;identity=dict(pid=101,ticks='201')
    monkeypatch.setattr(module.owner.resources,'same',lambda p:dict(state='R'))
    assert not module.previous_ready(identity)
    (tmp_path/'children.json').write_text(json.dumps([dict(pid=303,ticks='403')]))
    monkeypatch.setattr(module.owner.resources,'same',lambda p:dict(state='R') if p['pid']==303 else None)
    assert not module.previous_ready(identity)


def test_terminal_code_requires_complete_both_epochs_assessments_private_receipts_and_restoration(tmp_path,monkeypatch):
    module=driver();module.PREVIOUS=tmp_path;identity=dict(pid=101,ticks='201')
    monkeypatch.setattr(module.owner.resources,'same',lambda p:None)
    with pytest.raises(ValueError):module.previous_ready(identity)
    (tmp_path/'plan.json').write_text(json.dumps(dict(original_coding_sources=3721,planned_steps=3721)))
    (tmp_path/'complete.json').write_text(json.dumps(dict(all_3721_original_coding_sources_once_each_arm=True,
        matched_native_code_broad_human_preference_and_held_APPS_assessments_complete=True)))
    (tmp_path/'restored.json').write_text('{}')
    for arm in ['anchor_CE','anchor_execution']:
        (tmp_path/(arm+'-training-complete.json')).write_text(json.dumps(dict(actual_updates=3721,all_3721_original_coding_sources_once=True)))
        (tmp_path/('train-'+arm+'-resume-distributed-state.json')).write_text(json.dumps(dict(completed=3721,all_four_private_states=[dict(rank=r) for r in range(4)])))
    assert module.previous_ready(identity)
    receipt=tmp_path/'train-anchor_execution-resume-distributed-state.json'
    receipt.write_text(json.dumps(dict(completed=3720,all_four_private_states=[dict(rank=r) for r in range(4)])))
    with pytest.raises(ValueError):module.previous_ready(identity)


def test_paired_counts_use_all_original_questions_and_reject_changed_context_or_budget():
    module=driver()
    # Synthetic rows exercise the reconciliation arithmetic only, not source
    # admission or a full neural/benchmark proof.
    rows=[dict(audit_id='unit:'+str(i),benchmark='unit-only',original_native_prompt_ids=[1,7,8],
        generation_budget=32,inspected_exact_match_scopes=[],verdict=dict(correct=False)) for i in range(16961)]
    before=dict(original_population_sha256='a'*64,fixed_generation_budgets={'unit-only':32},
        inference_device_type='cuda',inference_precision='BF16_autocast',results=rows)
    after=copy.deepcopy(before);after['results'][3]['verdict']['correct']=True
    result=module.paired(before,after)['by_original_benchmark']['unit-only']
    assert result['questions']==16961 and result['before_correct']==0 and result['after_correct']==1
    assert result['gained']==['unit:3'] and result['lost']==[]
    after['results'][4]['original_native_prompt_ids']=[1,9,8]
    with pytest.raises(ValueError):module.paired(before,after)
    after=copy.deepcopy(before);after['fixed_generation_budgets']={'unit-only':16}
    with pytest.raises(ValueError):module.paired(before,after)


def test_finalizer_must_be_terminal_complete_and_cold_checked(tmp_path,monkeypatch):
    module=driver();module.FINALIZER=tmp_path;identity=dict(pid=101,ticks='201')
    monkeypatch.setattr(module.owner.resources,'same',lambda p:dict(state='R'))
    assert not module.finalizer_ready(identity)
    monkeypatch.setattr(module.owner.resources,'same',lambda p:None)
    with pytest.raises(ValueError):module.finalizer_ready(identity)
    done=dict(complete=True,completed_updates=3721,actual_counts=dict(total_unique=984399690),
        all_actual_exported_parameters_and_complete_native_probes_cold_load_bit_exact=True,
        real_word_logits_exact_through_fold_and_cold_native=True,old_checkpoint_actor_optimizer_or_reference_not_needed_for_inference=True)
    (tmp_path/'complete.json').write_text(json.dumps(done));assert module.finalizer_ready(identity)
    done['completed_updates']=3720;(tmp_path/'complete.json').write_text(json.dumps(done))
    with pytest.raises(ValueError):module.finalizer_ready(identity)
