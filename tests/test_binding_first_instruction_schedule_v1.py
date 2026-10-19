from copy import deepcopy

import pytest

from slm_lab.binding_first_instruction_schedule_v1 import schedule
from slm_lab.instruction_program_schedule_v4 import LANGUAGES, schedule as earlier_schedule


def pools():
    values = {name: [object()] for name in [*LANGUAGES, 'human_instruction']}
    values['verified_python'] = [
        {'id': name + str(variant), 'original_program_sha256': name, 'group_id': 'shared-file',
         'variant_index': variant, **({'binding_variant': True, 'parent_variant_index': 0} if variant >= 4 else {})}
        for name, variants in [('a', [0, 1, 2, 4, 5]), ('b', [0, 1])]
        for variant in variants]
    return values


def test_originals_then_binding_variants_use_actual_indices_without_mutating_rows():
    source = pools(); before = deepcopy(source['verified_python'])
    updates = schedule(source, 7, 17)
    selected = [source['verified_python'][step[2]['index']] for step in updates]
    assert {r['id'] for r in selected[:2]} == {'a0', 'b0'}
    assert {r['id'] for r in selected[2:4]} == {'a4', 'b1'}
    assert selected[4]['id'] == 'a5'
    assert {r['id'] for r in selected} == {r['id'] for r in before}
    assert source['verified_python'] == before
    old = earlier_schedule(source, 7, 17)
    assert [[step[i] for i in [0, 1, 3]] for step in updates] == [[step[i] for i in [0, 1, 3]] for step in old]
    assert updates == schedule(source, 7, 17)


def test_no_binding_rows_retain_the_existing_schedule():
    source = pools(); source['verified_python'] = [r for r in source['verified_python'] if not r.get('binding_variant')]
    assert schedule(source, 15, 17) == earlier_schedule(source, 15, 17)


@pytest.mark.parametrize('change', ['duplicate', 'flag_original', 'invalid_parent', 'string_flag'])
def test_real_duplicate_or_invalid_binding_identity_is_rejected(change):
    source = pools()
    if change == 'duplicate': source['verified_python'].append(dict(source['verified_python'][0]))
    elif change == 'flag_original': source['verified_python'][0].update(binding_variant=True, parent_variant_index=0)
    elif change == 'invalid_parent': source['verified_python'][3]['parent_variant_index'] = 1
    else: source['verified_python'][3]['binding_variant'] = 'true'
    with pytest.raises(ValueError): schedule(source, 7, 17)
