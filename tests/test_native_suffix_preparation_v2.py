"""Full-count metadata fixtures test preparation; no neural model is created.

Opaque fixture checkpoint/tokenizer bytes cannot be loaded for training.
Passing these tests establishes source assembly and rejection behavior only.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import train_native_code_suffix_aligned_GPU_v2 as driver
from slm_lab.native_suffix_preparation_v2 import (assemble, completed_stages, full_baseline, merge, read, sha, FULL, PRIOR, SUITES)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    original = dict(source_files_sha256={}, exact_predecessors={}, cached_own_reference=dict(fixture_only=True),
        pulse_sources=dict(fixture_only=True), initializer={})
    cold = root / 'opaque-cold-input-fixture.bin'; cold.write_bytes(b'not a tensor archive')
    original['cold_inputs'] = dict(path=str(cold), sha256=sha(cold))
    for index, (name, scope, assessment) in enumerate(PRIOR):
        folder = root / 'reports' / name
        write(folder / 'plan.json', dict(owner=dict(pid=99999991 + index, ticks='1')))
        write(folder / 'children.json', [])
        write(folder / 'restored.json', dict(all_registered_point_actors_terminal=True))
        write(folder / 'complete.json', dict(complete=True, original_nine_endpoint_pipeline_preserved=True,
            native_values=984399690, actual_native_values=984399690, **{scope: True, assessment: True}))
    checkpoint = root / 'checkpoints/batched-reasoning-aligned-GPU-v2/anchor_RL/native/model.pt'
    checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b'opaque fixture; cannot be loaded as a neural model')
    tokenizer = checkpoint.parent / 'tokenizer.json'; tokenizer.write_bytes(b'opaque tokenizer fixture')
    original['initializer']['tokenizer_sha256'] = sha(tokenizer)
    spec = dict(complete=True, profile='anchor_RL', completed_updates=1926, checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint), tokenizer_sha256=sha(tokenizer), sources={'fixture': 'a'*64},
        actual_native_values=984399690, every_native_value_and_complete_text_photo_word_cold_exact=True,
        provenance=dict(training_profile='anchor_RL', whole_production_source_schedule_completed=True))
    write(root / 'reports/batched-reasoning-aligned-GPU-v2/anchor_RL/native-export.json', spec)
    assessment = root / 'reports/batched-reasoning-aligned-GPU-v2/anchor_RL/assessment'
    baseline = assessment / 'standalone'
    for name, count in [('full-QA.json',16961),('full-AI2D.json',3088),('full-original-MBPP.json',600),('held-preferences.json',4642)]:
        rows = [dict(fixture_source_position=p) for p in range(count)]
        if name == 'full-AI2D.json':
            for row in rows: row['conditions'] = {k: {} for k in ['proper_image','gray_image','wrong_image']}
        write(baseline / name, dict(results=rows))
    write(baseline / 'all-held-words.json', dict(generation_budget=64, complete_held_source_words=True,
        held_source_gradients=False, results=[dict(source_position=p, conditions={k:{} for k in ['proper','blank','wrong']}) for p in range(3429)]))
    for frame in ['completion','instruction']:
        write(baseline / ('code-'+frame+'.json'), dict(task_count=24,results=[dict(fixture_task=p) for p in range(24)]))
    for name in ['knowledge','math','ocr','objects','vision']: write(baseline / (name+'.json'), dict(fixture_only=True))
    for suite in SUITES:
        for rank in range(4):
            write(baseline / f'assessment-{suite}-{rank}-complete.json', dict(physical_GPU=rank,actual_student_values=984399690,
                checkpoint_sha256=spec['checkpoint_sha256'], model_updates=0, all_loaded_values_unchanged=True,
                only_standalone_weights_and_tokenizer_for_neural_inference=True))
    write(assessment / 'all-reserved-reasoning.json', dict(complete=True,examples=289,
        checkpoint_sha256=spec['checkpoint_sha256'],optimization_forbidden=True,
        all_original_whole_prompts_and_full64or256budgets_retained=True,results=[dict(position=p) for p in range(289)]))
    native = root / 'native-pack-fixture.bin'; native.write_bytes(b'opaque source fixture')
    case = root / 'reports/native-human-code-suffix-actual-capacity-CPU-v2/actual-complete-source-capacity-plan.json'
    write(case, dict(geometry_only=False,full_source_data_admitted=True,original_human_sources=3721,original_source_views=14884,
        epochs=3,seed=989711,capacity_entries=[[0,1,2,3]],complete_training_budget=FULL,
        all_12_real_single_source_three_empty_rank_training_tails_retained=True))
    capacity = dict(complete=True,source_selection_only=True,complete_training_budget=FULL,
        models_constructed=0,neural_forwards=0,GPU_contexts=0,GPU_leases=0,optimizer_updates=0,
        full_size_GPUcapacity_passed=False,production_training_admitted=False,
        actual_fixed_horizon_source_positions=[0],predicted_full_size_selected_logical_KV_bytes=1024,
        declared_cache_bound_bytes=driver.STAGE_OPTIONS['maximum_cache_bytes'],
        source_files_sha256={str(native):sha(native),str(case):sha(case)})
    code = root / 'source-fixture.py'; code.write_text('fixture = True\n')
    return original, capacity, {str(code):sha(code)}, checkpoint


def build(root, original, capacity, files):
    stages = completed_stages(root, {})
    return assemble(root, original, capacity, stages, owner=dict(pid=99999999,ticks='1'), source_files=files,driver=driver)


def test_full_scope_source_recipe_uses_fixed_reasoning_endpoint_and_preserves_all_baselines(tmp_path):
    original, capacity, files, checkpoint = fixture(tmp_path)
    result = build(tmp_path, original, capacity, files)
    assert Path(result['initializer']['checkpoint']) == checkpoint
    assert result['initializer']['checkpoint_sha256'] == sha(checkpoint)
    assert Path(result['cached_baseline']['folder']) == tmp_path/'reports/batched-reasoning-aligned-GPU-v2/anchor_RL/assessment/standalone'
    assert result['full_native_budget'] == FULL and result['native_stage_options'] == driver.STAGE_OPTIONS
    assert result['required_free_bytes_before_launch'] == int(228.0625 * 1024**3)
    assert result['production_updates'] == result['models_constructed'] == result['GPU_contexts'] == 0
    assert not result['GPU_work_started'] and result['full_size_GPUcapacity_training_export_and_quality_still_required']
    assert all(sha(p)==h for p,h in result['source_files_sha256'].items())


@pytest.mark.parametrize('case',['missing_image_control','truncated_reasoning','wrong_worker_checkpoint','different_initializer','changed_source','partial_native'])
def test_partial_changed_or_alternative_sources_cannot_produce_a_closed_runtime_recipe(tmp_path,case):
    original,capacity,files,_=fixture(tmp_path)
    base=tmp_path/'reports/batched-reasoning-aligned-GPU-v2/anchor_RL'
    if case=='missing_image_control':
        p=base/'assessment/standalone/full-AI2D.json';x=read(p);del x['results'][8]['conditions']['wrong_image'];write(p,x)
    elif case=='truncated_reasoning':
        p=base/'assessment/all-reserved-reasoning.json';x=read(p);x['results'].pop();write(p,x)
    elif case=='wrong_worker_checkpoint':
        p=base/'assessment/standalone/assessment-MBPP-2-complete.json';x=read(p);x['checkpoint_sha256']='b'*64;write(p,x)
    elif case=='different_initializer':
        p=base/'native-export.json';x=read(p);x['profile']='anchor_CE';write(p,x)
    elif case=='changed_source':
        p=next(iter(files));state=Path(p).stat();Path(p).write_text('fixture = None\n');os.utime(p,ns=(state.st_atime_ns,state.st_mtime_ns))
    else: capacity['complete_training_budget']=dict(FULL,source_presentations=44651)
    with pytest.raises(ValueError):build(tmp_path,original,capacity,files)


def test_future_stage_is_not_called_live_or_complete_and_captured_plan_cannot_change(tmp_path):
    assert completed_stages(tmp_path,{}) is None
    original,capacity,files,_=fixture(tmp_path)
    captured={};assert completed_stages(tmp_path,captured)
    p=tmp_path/'reports'/PRIOR[0][0]/'plan.json';x=read(p);x['owner']['ticks']='2';write(p,x)
    with pytest.raises(ValueError):completed_stages(tmp_path,captured)
    # A genuinely live process remains pending even if completion metadata exists.
    fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    write(p,dict(owner=dict(pid=os.getpid(),ticks=fields[19])))
    assert completed_stages(tmp_path,{}) is None


def test_conflicting_binding_rejected_and_source_preparation_imports_are_non_neural():
    with pytest.raises(ValueError):merge({'/tmp/fixed-native-input':'a'*64},{'/tmp/fixed-native-input':'b'*64})
    code="import runpy,sys;runpy.run_path('scripts/prepare_native_code_suffix_aligned_CPU_v2.py');assert 'torch' not in sys.modules;print('source-only import verified')"
    result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=30)
    assert result.returncode==0 and result.stdout.strip()=='source-only import verified'
