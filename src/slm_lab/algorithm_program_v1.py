"""One bounded execution path for held algorithm references and generated code."""
from collections import Counter
import hashlib
import tempfile

from .codenet_algorithm_assessment_v1 import check_output
from .standalone_gnu_v1 import compile_gnu_program
from .standalone_numpy_v1 import prepare_numpy
from .standalone_program_v1 import run_bounded

PROFILE = {'C++': 'verified GNU C++17/static,1536MiB address-space',
           'Python': 'isolated CPython3.11 with active verified NumPy,384MiB address-space',
           'functional_wall_seconds': 3, 'stress_wall_seconds': 10, 'runtime_CPU_seconds': 8,
           'maximum_input_bytes': 4 * 1024**2, 'combined_output_bytes': 1024**2,
           'compilation_excluded_from_case_wall_time': True,
           'case_wall_time_includes_process_startup': True}


def assess_program(code, language, cases, tools):
    if language not in {'C++', 'Python'} or not cases or not isinstance(code, bytes):
        raise ValueError('Known standalone language, complete programme bytes and fixed cases required')
    result = {'language': language, 'source_sha256': hashlib.sha256(code).hexdigest(),
              'profile': PROFILE, 'cases': []}
    with tempfile.TemporaryDirectory(prefix='slm-held-algorithm-v1-') as work:
        if language == 'C++':
            constraints, compiled = compile_gnu_program(code, work, tools)
            result['compilation'] = {'status': compiled['status'], 'seconds': compiled['seconds'],
                'stderr_tail': compiled['stderr'][-1500:].decode(errors='replace')}
            if constraints is None:
                result['qualification'] = 'compiler_or_environment_review_needed'
                return result
        else:
            constraints = prepare_numpy(code, work)
        for case in cases:
            observed = run_bounded(constraints, case['input'].encode(),
                seconds=PROFILE['stress_wall_seconds'] if case['kind'] == 'stress' else PROFILE['functional_wall_seconds'],
                output_bytes=PROFILE['combined_output_bytes'], python=language == 'Python')
            actual = observed['stdout'].decode(errors='replace')
            passed = observed['status'] == 'completed' and check_output(
                case['problem_id'], case['input'], case['expected_output'], actual)
            result['cases'].append({'case_id': case['id'], 'kind': case['kind'], 'status': observed['status'],
                'seconds_including_startup': observed['seconds'], 'oracle_passed': passed,
                'stdout_sha256': hashlib.sha256(observed['stdout']).hexdigest(),
                'stdout_bytes': len(observed['stdout']), 'stdout_prefix': actual[:256] if not passed else None,
                'stderr_tail': observed['stderr'][-500:].decode(errors='replace'),
                'input_bytes': observed['input_bytes'], 'input_bytes_sent': observed['input_bytes_sent']})
    counts = Counter(case['kind'] for case in result['cases'])
    result['case_counts'] = dict(counts)
    result['passed_counts'] = {kind: sum(case['oracle_passed'] for case in result['cases'] if case['kind'] == kind) for kind in counts}
    if any(case['status'] != 'completed' for case in result['cases']):
        result['qualification'] = 'runtime_or_resource_profile_review_needed'
    elif all(case['oracle_passed'] for case in result['cases']):
        result['qualification'] = 'all_selected_cases_agree_under_fixed_profile'
    else:
        result['qualification'] = 'source_case_or_checker_review_needed'
    return result
