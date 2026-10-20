import copy
import hashlib

import pytest

from slm_lab.held_text_overlap_v1 import HeldTextOverlap
from slm_lab.whole_reasoning_view_screen_v1 import apply_component_veto, screen


SOLUTION = 'def solve():\n    return 7\n'
HELD = 'A protected original problem about orbital clocks and integer divisibility.'


def view(text, length=6000):
    return dict(prompt='User: Solve this separate public problem.\nAssistant:',
        prompt_tokens_including_BOS=20, entire_response_tokens_including_EOS=length-20,
        entire_context_tokens=length, entire_target_utf8_sha256=hashlib.sha256(text.encode()).hexdigest(),
        contains_reserved_native_source_token=False, complete_context_fits_4096=length <= 4096)


def judge(text, **kwargs):
    return screen(view(text), text, SOLUTION, maximum_context=16384,
        overlap=HeldTextOverlap([dict(audit_id='held-1', question=HELD)]), **kwargs)


def original(position, role, component, observation):
    return dict(original_shard_position=position, source_role=role,
        fixed_pre_execution_role=dict(source_role=role, declared_duplicate_component_sha256=component,
            role_fixed_before_fresh_replay=True, optimization_forbidden=role == 'reserved'),
        reasoning_screen=observation)


def test_held_stem_only_at_end_of_whole_long_response_is_detected():
    text='Public derivation. ' * 700 + HELD + '\n' + SOLUTION
    assert HELD not in text[:4096]
    result=judge(text)
    assert result['component_overlap_veto'] and result['status']=='held_literal_overlap'
    assert set(result['entire_target_literal_overlap']['target_literal'])=={'held-1'}
    assert result['whole_response_sha256']==hashlib.sha256(text.encode()).hexdigest()


def test_cropped_source_wrong_geometry_and_unusable_whole_views_cannot_become_positive():
    text='A complete derivation.\n'+SOLUTION
    original_view=view(text);overlap=HeldTextOverlap([dict(audit_id='held-1',question=HELD)])
    with pytest.raises(ValueError):screen(original_view,text[:-1],SOLUTION,maximum_context=16384,overlap=overlap)
    corrupt=copy.deepcopy(original_view);corrupt['entire_context_tokens']+=1
    with pytest.raises(ValueError):screen(corrupt,text,SOLUTION,maximum_context=16384,overlap=overlap)
    assert judge('Different reasoning without that original code.')['status']=='publisher_solution_not_present_in_entire_reasoning'
    assert judge(text+'\0')['status']=='source_NUL_qualification'
    assert screen(view(text,20000),text,SOLUTION,maximum_context=16384,overlap=overlap)['status']=='whole_context_or_native_source_token_qualification'
    assert judge(text)['status']=='whole_reasoning_metadata_screen_positive'


def test_whole_component_veto_preserves_every_position_role_and_reserved_protection():
    clean=judge('Reasoning.\n'+SOLUTION);bad=judge(HELD+'\n'+SOLUTION)
    rows=[original(1,'train','a',bad),original(2,'train','a',clean),original(3,'reserved','b',clean)]
    result=apply_component_veto(rows)
    assert [r['original_shard_position'] for r in result]==[1,2,3]
    assert [r['source_role'] for r in result]==['train','train','reserved']
    assert result[0]['fitting_screened_views']==result[1]['fitting_screened_views']==[]
    assert result[2]['fitting_screened_views']==['code_only','reasoning'] and result[2]['optimization_forbidden']
    assert all(r['training_admitted'] is False for r in result)
    changed=copy.deepcopy(rows);changed[1]['source_role']='reserved'
    with pytest.raises(ValueError):apply_component_veto(changed)
    changed=copy.deepcopy(rows);changed[2]['fixed_pre_execution_role']['declared_duplicate_component_sha256']='a'
    with pytest.raises(ValueError):apply_component_veto(changed)
