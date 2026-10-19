"""Ordinary stdlib Python modules in the existing Landlock/seccomp runtime.

Gold values remain outside the candidate-readable filesystem/process. The
parent compares every raw returned value using exact Python equality. This
does not use the APPS comparator's optional-list compatibility rule, inject
publisher globals, rewrite generated code, or claim asymptotic efficiency.
"""
import hashlib
import json
import math
from pathlib import Path
import tempfile

from .corpus import file_hash
from .standalone_program_v1 import python_policy, run_bounded

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / 'scripts/python_function_observations_v1.py'
FORMAT = 'own-ordinary-stdlib-Python-module-function-cases-v1'


def execute(program, function, cases, *, seconds=4):
    if (not isinstance(program, str) or not 0 < len(program.encode()) <= 50000 or '\0' in program
            or not isinstance(function, str) or not function.isidentifier() or function.startswith('__')
            or not isinstance(cases, list) or not 1 <= len(cases) <= 1000
            or any(not isinstance(r, dict) or set(r) != {'args', 'expected'} or not isinstance(r['args'], list) for r in cases)
            or type(seconds) not in [int, float] or not math.isfinite(seconds) or not 0 < seconds <= 30):
        raise ValueError('Complete bounded program, explicit function and every JSON-compatible finite case required')
    # Roundtrip explicitly checks the domain instead of silently changing
    # tuples/non-string dictionary keys or sending extra fields to a worker.
    encoded = json.dumps(cases, allow_nan=False)
    if json.loads(encoded) != cases: raise ValueError('All original argument/gold values must survive JSON unchanged')
    if len(encoded.encode()) > 4 * 1024**2: raise ValueError('Complete finite-case payload exceeds declared audit budget')
    base = dict(format=FORMAT, program_sha256=hashlib.sha256(program.encode()).hexdigest(), function=function,
        total_cases=len(cases), passed_cases=0, functional_passed=False, all_cases_observed=False,
        syntax_valid=False, original_cases_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
        source_bytes_not_repaired=True, expected_outputs_not_sent_to_confined_worker=True,
        normal_import_style_module_name=True, standard_library_only=True,
        OS_sandbox='Existing pre-exec Landlock filesystem and seccomp runtime', empirical_or_asymptotic_efficiency_not_assessed=True)
    try: compile(program, '<candidate-module>', 'exec')
    except (SyntaxError, ValueError, RecursionError) as error: return dict(base, status='syntax_error', diagnostic=str(error)[:300])
    base['syntax_valid'] = True
    with tempfile.TemporaryDirectory(prefix='slm-ordinary-Python-function-v1-') as temporary:
        work = Path(temporary); (work / 'program.py').write_text(program); (work / 'harness.py').write_bytes(HARNESS.read_bytes())
        settings = dict(program=str(work / 'program.py'), function=function, inputs=[r['args'] for r in cases])
        (work / 'settings.json').write_text(json.dumps(settings, allow_nan=False))
        policy = python_policy(work); policy['command'][-1] = str(work / 'harness.py'); policy['command'].append(str(work / 'settings.json'))
        policy['memory_bytes'] = 512 * 1024**2; policy['cpu_seconds'] = max(1, math.ceil(seconds))
        observed = run_bounded(policy, b'', seconds=seconds, output_bytes=4 * 1024**2, python=True)
        execution = {k: v for k, v in observed.items() if k not in {'stdout', 'stderr'}}
        execution.update(stdout_sha256=hashlib.sha256(observed['stdout']).hexdigest(),
            stderr=observed['stderr'][-1600:].decode(errors='replace'))
        base['execution'] = execution
        if observed['status'] != 'completed': return dict(base, status='runtime_or_resource_failure')
        try:
            raw = json.loads(observed['stdout']); rows = raw['results']
            if (set(raw) != {'format', 'results'} or raw['format'] != 'own-ordinary-Python-function-raw-values-v1'
                    or not isinstance(rows, list) or len(rows) != len(cases)
                    or any(set(row) != {'value', 'seconds'} or type(row['seconds']) not in [int, float]
                        or not math.isfinite(row['seconds']) or row['seconds'] < 0 for row in rows)):
                raise ValueError('Every complete raw function result required')
        except (KeyError, TypeError, ValueError, RecursionError): return dict(base, status='incomplete_or_invalid_observations')
        flags = [row['value'] == case['expected'] for row, case in zip(rows, cases)]
        return dict(base, status='all_finite_cases_pass' if all(flags) else 'finite_case_mismatch',
            all_cases_observed=True, functional_passed=all(flags), passed_cases=sum(flags), case_passes=flags, raw_observations=raw)


def implementation_sources():
    paths = [Path(__file__), HARNESS, ROOT / 'src/slm_lab/standalone_program_v1.py', ROOT / 'src/slm_lab/native_code.py',
        ROOT / 'scripts/python_program_sandbox_v1.py', ROOT / 'scripts/native_sandbox.py']
    return {str(p): file_hash(p) for p in paths}
