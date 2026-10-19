import random

import pytest

from slm_lab.code_residual_pilot_v1 import PROFILES, learning_rate, select_complete_rows


def population(role='train'):
    return [dict(global_position=i,local_position=i,id=f'{role}:{i}',group_id=f'{role}:g{i//2}',
                 language=['C','Python','Rust'][i%3],role=role,context_tokens=100+i,
                 target_tokens=4+i%31,full_input_ids_sha256='a'*64) for i in range(120)]


def test_whole_row_selection_covers_all_languages_extremes_and_is_order_independent():
    rows=population();rows[30]['target_tokens']=120
    before=random.getstate()
    selected=select_complete_rows(rows,number=32,per_language=4,seed=21,role='train')
    assert len(selected)==len({r['id'] for r in selected})==32
    assert all(sum(r['language']==language for r in selected)>=4 for language in ['C','Python','Rust'])
    assert rows[-1] in selected and rows[30] in selected
    assert selected==select_complete_rows(list(reversed(rows)),number=32,per_language=4,seed=21,role='train')
    assert random.getstate()==before
    assert len({r['group_id'] for r in selected})>=30


@pytest.mark.parametrize('fault',['mixed_role','duplicate_id','duplicate_position','clipped_context','zero_targets','budget'])
def test_bad_roles_incomplete_contexts_and_insufficient_budget_reject(fault):
    rows=population();number=32
    if fault=='mixed_role':rows[0]['role']='validation'
    elif fault=='duplicate_id':rows[0]['id']=rows[1]['id']
    elif fault=='duplicate_position':rows[0]['global_position']=rows[1]['global_position']
    elif fault=='clipped_context':rows[0]['context_tokens']=4097
    elif fault=='zero_targets':rows[0]['target_tokens']=0
    elif fault=='budget':number=6
    with pytest.raises(ValueError):select_complete_rows(rows,number=number,per_language=4,seed=21,role='train')


def test_validation_view_is_separate_and_preserves_original_role():
    train=select_complete_rows(population(),number=32,per_language=4,seed=21,role='train')
    valid=select_complete_rows(population('validation'),number=32,per_language=4,seed=22,role='validation')
    assert {r['group_id'] for r in train}.isdisjoint(r['group_id'] for r in valid)
    assert all(r['role']=='validation' for r in valid)


def test_warmup_is_the_only_schedule_difference_at_equal_peak():
    constant,warmup,small=PROFILES
    assert learning_rate(constant,0)==3e-5
    assert learning_rate(warmup,0)==3e-5/16
    assert learning_rate(warmup,15)==learning_rate(constant,15)==3e-5
    assert learning_rate(warmup,127)==learning_rate(constant,127)==3e-5
    assert learning_rate(small,0)==1e-5/16
    assert learning_rate(small,127)==1e-5
