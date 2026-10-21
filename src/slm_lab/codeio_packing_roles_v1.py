"""Keep the stronger complete-target exclusions in native candidate packs."""


def join(original, screened, *, position, row_sha256):
    role = original.get('fixed_role', {})
    old = original.get('entire_component_quarantined_for_protected_signal')
    whole = screened.get('entire_component_quarantined_after_full_target_screen')
    if (type(position) is not int or position < 0
            or original.get('original_position') != position
            or screened.get('original_position') != position
            or original.get('original_row_sha256') != row_sha256
            or screened.get('original_row_sha256') != row_sha256
            or not isinstance(row_sha256, str) or len(row_sha256) != 64
            or any(c not in '0123456789abcdef' for c in row_sha256)
            or role.get('source_role') not in {'train', 'reserved'}
            or role.get('optimization_forbidden') is not (role.get('source_role') == 'reserved')
            or role.get('role_fixed_before_fresh_execution') is not True
            or screened.get('fixed_role') != role
            or type(old) is not bool or type(whole) is not bool or old and not whole
            or screened.get('optimization_forbidden') is not (role['optimization_forbidden'] or whole)
            or screened.get('source_role_unchanged') is not True
            or original.get('data_admitted_to_training') is not False
            or screened.get('data_admitted_to_training') is not False):
        raise ValueError('Exact original row, fixed role and stronger whole-component veto required')
    return dict(original_fixed_role=dict(role), original_index_component_veto=old,
                whole_component_quarantined=whole,
                source_optimization_forbidden=role['optimization_forbidden'] or whole,
                whole_target_veto_applied=True, source_role_unchanged=True,
                data_admitted_to_training=False)
