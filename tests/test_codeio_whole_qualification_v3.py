import copy
import hashlib
import json

import pytest

from slm_lab.codeio_whole_qualification_v3 import qualify


def inputs(code, output):
    line=(json.dumps(dict(refcode=code,funcname='main_solution',ios=[dict(input={'x':2},output=output)]))+'\n').encode()
    original=dict(original_position=0,original_row_sha256=hashlib.sha256(line).hexdigest(),
        fixed_role={'source_role':'reserved','optimization_forbidden':True},
        entire_component_quarantined_for_protected_signal=False)
    screen=dict(original_position=0,original_row_sha256=original['original_row_sha256'],fixed_role=original['fixed_role'],
        source_role_unchanged=True,data_admitted_to_training=False,optimization_forbidden=True,
        entire_component_quarantined_after_full_target_screen=True)
    return line,original,screen


def test_actual_whole_cached_case_positive_keeps_reserved_role_and_stronger_full_target_veto():
    line,original,screen=inputs('def main_solution(x):\n    return x+1\n',3)
    result=qualify(line,original,screen)
    assert result['all_original_cases_and_seeds_match']
    assert result['fixed_role']==original['fixed_role'] and result['source_optimization_forbidden']
    assert result['whole_component_quarantined'] and not result['original_index_component_veto']
    assert not result['RL_reward_admitted'] and not result['independent_semantic_correctness_certified']
    changed=copy.deepcopy(screen);changed['fixed_role']['source_role']='train'
    with pytest.raises(ValueError):qualify(line,original,changed)


def test_actual_runtime_mismatch_and_trivial_None_populations_do_not_become_positives():
    for code,expected,status in [
        ('def main_solution(x):\n    return x-1\n',3,'complete_local_context_cached_value_mismatch'),
        ('import numpy\ndef main_solution(x):\n    return x+1\n',3,'reference_runtime_or_resource_population_qualification'),
        ('def main_solution(x):\n    return None\n',None,'no_OP_runtime_or_trivial_population_qualification')]:
        result=qualify(*inputs(code,expected))
        assert result['status']==status and not result['all_original_cases_and_seeds_match']
        assert not result['RL_reward_admitted'] and result['source_optimization_forbidden']
