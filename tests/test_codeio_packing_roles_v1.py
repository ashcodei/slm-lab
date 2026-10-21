import copy

import pytest

from slm_lab.codeio_packing_roles_v1 import join


def records(reserved=False, old=False, new=False):
    role = dict(source_role='reserved' if reserved else 'train', component_sha256='a'*64,
                component_size=3, optimization_forbidden=reserved, role_fixed_before_fresh_execution=True)
    original = dict(original_position=7, original_row_sha256='b'*64, fixed_role=role,
                    entire_component_quarantined_for_protected_signal=old, data_admitted_to_training=False)
    screened = dict(original_position=7, original_row_sha256='b'*64, fixed_role=copy.deepcopy(role),
                    entire_component_quarantined_after_full_target_screen=old or new,
                    optimization_forbidden=reserved or old or new,
                    source_role_unchanged=True, data_admitted_to_training=False)
    return original, screened


def test_stronger_code_cached_value_veto_and_reserved_source_roles_survive_native_preparation():
    for reserved, old, new in [(False, False, False), (False, False, True),
                               (False, True, False), (True, False, False), (True, False, True)]:
        original, screened = records(reserved, old, new)
        before = copy.deepcopy((original, screened))
        result = join(original, screened, position=7, row_sha256='b'*64)
        assert result['source_optimization_forbidden'] == (reserved or old or new)
        assert result['whole_component_quarantined'] == (old or new)
        assert result['original_fixed_role'] == original['fixed_role']
        assert not result['data_admitted_to_training']
        assert (original, screened) == before


def test_changed_roles_missing_veto_and_misaligned_complete_sources_reject():
    for key, value in [('original_position', 8), ('original_row_sha256', 'c'*64),
                       ('optimization_forbidden', False),
                       ('entire_component_quarantined_after_full_target_screen', False),
                       ('source_role_unchanged', False), ('data_admitted_to_training', True),
                       ('fixed_role', {'source_role': 'train'})]:
        original, screened = records(False, True, False); screened[key] = value
        with pytest.raises(ValueError):
            join(original, screened, position=7, row_sha256='b'*64)
