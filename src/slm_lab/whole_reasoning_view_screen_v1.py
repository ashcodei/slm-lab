"""Screen whole longer public responses without changing original source roles.

This adds the same literal held-stem and solution-inclusion checks used for
4K views. It does not certify semantic independence or execute any program.
"""
import hashlib


def screen(view, response, solution, *, maximum_context, overlap):
    if type(maximum_context) is not int or not 4096 <= maximum_context <= 32768:
        raise ValueError('An explicit bounded whole-context study is required')
    if not isinstance(response, str) or not isinstance(solution, str):
        raise ValueError('The complete original publisher response and solution are required')
    digest = hashlib.sha256(response.encode()).hexdigest()
    if digest != view['entire_target_utf8_sha256']:
        raise ValueError('An altered or cropped response cannot inherit original whole-token geometry')
    p = view['prompt_tokens_including_BOS']
    r = view['entire_response_tokens_including_EOS']
    n = view['entire_context_tokens']
    if (any(type(v) is not int or v < 1 for v in (p, r, n)) or n != p + r
            or type(view['contains_reserved_native_source_token']) is not bool
            or not isinstance(view['prompt'], str)):
        raise ValueError('Exact original complete prompt/response token measurements are required')
    clean = r > 1 and not view['contains_reserved_native_source_token']
    if view['complete_context_fits_4096'] is not (clean and n <= 4096):
        raise ValueError('The original 4K geometry must remain consistent')
    result = dict(status='whole_context_or_native_source_token_qualification',
        whole_response_sha256=digest, prompt_tokens=p, response_EOS_tokens=r,
        whole_context_tokens=n, maximum_context=maximum_context,
        original_4K_fit=view['complete_context_fits_4096'],
        component_overlap_veto=False, training_admitted=False)
    if not clean or n > maximum_context:
        return result
    if any('\0' in text for text in (view['prompt'], response, solution)):
        result['status'] = 'source_NUL_qualification'
        return result
    if not solution.strip() or solution.strip() not in response:
        result['status'] = 'publisher_solution_not_present_in_entire_reasoning'
        return result
    matches = dict(prompt_literal=overlap.literal(view['prompt']), target_literal=overlap.literal(response))
    veto = any(matches.values())
    result.update(status='held_literal_overlap' if veto else 'whole_reasoning_metadata_screen_positive',
        entire_target_literal_overlap=matches, component_overlap_veto=veto,
        complete_publisher_solution_occurs_in_response=True,
        publisher_solution_utf8_sha256=hashlib.sha256(solution.encode()).hexdigest())
    return result


def apply_component_veto(records):
    """Keep every original role and position, including quarantined components."""
    positions = set(); owners = {}; bad = set()
    for row in records:
        position = row['original_shard_position']; fixed = row['fixed_pre_execution_role']
        role = row['source_role']; component = fixed['declared_duplicate_component_sha256']
        if (type(position) is not int or position < 0 or position in positions
                or role not in ('train', 'reserved') or fixed['source_role'] != role
                or fixed['role_fixed_before_fresh_replay'] is not True
                or fixed['optimization_forbidden'] is not (role == 'reserved')):
            raise ValueError('Every unique original fixed pre-execution role must remain exact')
        positions.add(position)
        if component in owners and owners[component] != role:
            raise ValueError('A declared original duplicate component cannot cross roles')
        owners[component] = role
        if row['reasoning_screen']['component_overlap_veto']:
            bad.add(component)
    result = []
    for row in records:
        blocked = row['fixed_pre_execution_role']['declared_duplicate_component_sha256'] in bad
        views = [] if blocked else ['code_only']
        if not blocked and row['reasoning_screen']['status'] == 'whole_reasoning_metadata_screen_positive':
            views.append('reasoning')
        result.append(dict(row, status='long_context_overlap_component_quarantined' if blocked
            else 'metadata_positive_pending_original_execution_readback',
            fitting_screened_views=views, component_quarantined=blocked,
            optimization_forbidden=row['source_role'] == 'reserved', training_admitted=False))
    return result
