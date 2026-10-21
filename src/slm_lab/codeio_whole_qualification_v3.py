"""Join unchanged whole-source cached-case observations with the full veto ledger."""
import hashlib

from .codeio_case_segments_v1 import validate_record
from .codeio_published_cases_v1 import contract, strict_loads
from .codeio_whole_cases_v2 import execute


def qualify(line, original, screened):
    position = original['original_position']
    if (hashlib.sha256(line).hexdigest() != original['original_row_sha256']
            or screened.get('original_position') != position
            or screened.get('original_row_sha256') != original['original_row_sha256']
            or screened.get('fixed_role') != original['fixed_role']
            or screened.get('source_role_unchanged') is not True
            or screened.get('data_admitted_to_training') is not False):
        raise ValueError('Every original whole source, fixed role and complete-target veto must align')
    result = dict(original_position=position, original_row_sha256=original['original_row_sha256'],
        fixed_role=original['fixed_role'],
        source_optimization_forbidden=screened['optimization_forbidden'],
        whole_component_quarantined=screened['entire_component_quarantined_after_full_target_screen'],
        original_index_component_veto=original['entire_component_quarantined_for_protected_signal'],
        all_original_cases_and_seeds_match=False, source_only_no_training_admission=True,
        independent_semantic_correctness_certified=False, RL_reward_admitted=False)
    try:
        cases = contract(strict_loads(line))
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        result.update(status='original_complete_schema_qualification', diagnostic=str(error))
    else:
        observations = []
        for seed in [0, 1]:
            observed = execute(cases, seed=seed, seconds=20)
            observations.append(observed)
            if not observed['all_original_cases_observed']:
                break
        whole = len(observations) == 2 and all(o['all_original_cases_observed'] for o in observations)
        agree = whole and all(o['all_original_cached_values_match'] for o in observations)
        result.update(total_original_cases=cases['total_cases'], original_program_sha256=cases['original_program_sha256'],
            original_cases_sha256=cases['original_cases_sha256'], reference_observations=observations)
        if not whole:
            result['status'] = 'reference_runtime_or_resource_population_qualification'
        elif not agree:
            result['status'] = 'complete_local_context_cached_value_mismatch'
        else:
            noop = 'def '+cases['function']+'(**kwargs):\n    return None\n'
            control = execute(cases, seed=0, seconds=20, program=noop)
            result['no_OP_control'] = control
            if not control['all_original_cases_observed'] or control['all_original_cached_values_match']:
                result['status'] = 'no_OP_runtime_or_trivial_population_qualification'
            else:
                result.update(status='all_original_cached_values_match_two_local_seed_contexts', all_original_cases_and_seeds_match=True)
    validate_record(result, position)
    return result
