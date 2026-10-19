"""Full-case local APPS execution using existing OS confinement.

No source admission, gradients, model generation, preference creation or
official APPS score. Expected outputs are compared only outside confinement.
Unknown schemas and incomplete executions remain explicit qualifications.
"""
import hashlib
import json
import math
from pathlib import Path
import tempfile

from .corpus import file_hash
from .standalone_program_v1 import python_policy, run_bounded

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / 'scripts/apps_case_harness_v2.py'
MAX_PAYLOAD = 4 * 1024**2
MAX_PROGRAM = 512 * 1024
MAX_CASES = 1000
# Common names supplied by the publisher's wrapper, limited to stdlib. This is
# declared runtime context, never a modification of a solution's source bytes.
PRELUDE = '''import sys, time, itertools, collections, functools, math, fractions, typing, random, heapq
from itertools import accumulate, product, permutations, combinations
from collections import Counter, OrderedDict, deque, defaultdict, ChainMap
from functools import lru_cache
from math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2
from typing import List, Tuple
from heapq import *
stdin, stdout = sys.stdin, sys.stdout
'''


def contains_dict(value):
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            return True
        if isinstance(item, list):
            pending.extend(item)
    return False


def contract(raw):
    if not isinstance(raw, bytes) or len(raw) > MAX_PAYLOAD:
        raise ValueError('original_case_payload_requires_review')
    def invalid(value):
        raise ValueError('nonfinite_original_JSON_number_requires_review')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate_original_JSON_key_requires_review')
            result[key] = value
        return result
    try:
        value = json.loads(raw, parse_constant=invalid, object_pairs_hook=pairs)
        # JSON's exponent notation can overflow a float without using NaN or
        # Infinity tokens. It still needs an explicit source review.
        json.dumps(value, allow_nan=False)
    except (ValueError, RecursionError) as error:
        raise ValueError('original_JSON_requires_review') from error
    if not isinstance(value, dict):
        raise ValueError('original_case_schema_requires_review')
    inputs, outputs, name = value.get('inputs'), value.get('outputs'), value.get('fn_name')
    if (not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs)
            or not 1 <= len(inputs) <= MAX_CASES):
        raise ValueError('original_complete_case_schema_or_size_requires_review')
    if name is not None:
        if (not isinstance(name, str) or not name.isidentifier() or name.startswith('__')
                or not all(isinstance(args, list) for args in inputs)):
            raise ValueError('original_function_interface_requires_review')
        if contains_dict(inputs) or contains_dict(outputs):
            raise ValueError('publisher_dictionary_key_conversion_requires_review')
        mode = 'function'
    else:
        mode = 'stdin'
        def text(item):
            if isinstance(item, str):
                return item
            if isinstance(item, list) and all(isinstance(x, str) for x in item):
                return '\n'.join(item)
            raise ValueError('original_standard_io_schema_requires_review')
        inputs = [text(item) for item in inputs]
        outputs = [text(item) for item in outputs]
    return dict(mode=mode, function=name, inputs=inputs, outputs=outputs,
        original_tests_sha256=hashlib.sha256(raw).hexdigest(), total_cases=len(inputs))


def compare(observed, expected, mode):
    if mode == 'stdin':
        # Complete ordered output with outside whitespace ignored, a sufficient
        # exact comparison also accepted by the original publisher comparator.
        return isinstance(observed, str) and observed.strip() == expected.strip()
    return observed == expected or (isinstance(expected, list) and bool(expected) and observed == expected[0])


def execute(program, cases, seconds=20):
    if not isinstance(program, str) or not program or len(program.encode()) > MAX_PROGRAM or '\0' in program:
        raise ValueError('Complete bounded original Python source required')
    if not isinstance(cases, dict) or not 0 < seconds <= 120:
        raise ValueError('Explicit complete source contract and bounded wall time required')
    # Revalidate the caller's schema independently of any claimed total/role.
    validated = contract(json.dumps(dict(inputs=cases['inputs'], outputs=cases['outputs'],
        fn_name=cases['function']), allow_nan=False).encode())
    if any(validated[k] != cases[k] for k in ['mode', 'function', 'inputs', 'outputs', 'total_cases']):
        raise ValueError('Actual complete source case contract differs')
    base = dict(original_program_sha256=hashlib.sha256(program.encode()).hexdigest(),
        original_tests_sha256=cases['original_tests_sha256'], mode=cases['mode'],
        total_cases=cases['total_cases'], passed_cases=0, all_cases_observed=False,
        all_finite_published_cases_pass=False, official_APPS_runtime=False)
    try:
        compile(program, '<original-program>', 'exec')
    except (SyntaxError, ValueError, RecursionError) as error:
        return dict(**base, syntax_valid=False, status='full_compile_or_scope_qualification', diagnostic=str(error)[:300])
    with tempfile.TemporaryDirectory(prefix='slm-APPS-original-cases-v2-') as temporary:
        work = Path(temporary)
        (work/'program.py').write_text(program)
        (work/'prelude.py').write_text(PRELUDE)
        (work/'harness.py').write_bytes(HARNESS.read_bytes())
        (work/'settings.json').write_text(json.dumps(dict(program=str(work/'program.py'),
            prelude=str(work/'prelude.py'), mode=cases['mode'], function=cases['function'], inputs=cases['inputs'])))
        policy = python_policy(work)
        # The trusted harness and inputs are confined with the candidate;
        # expected outputs and repository/credential paths are never granted.
        policy['command'][-1] = str(work/'harness.py')
        policy['command'].append(str(work/'settings.json'))
        policy['memory_bytes'] = 768 * 1024**2
        policy['cpu_seconds'] = max(1, math.ceil(seconds))
        observed = run_bounded(policy, b'', seconds=seconds, output_bytes=MAX_PAYLOAD, python=True)
        retained = {k:v for k,v in observed.items() if k not in {'stdout', 'stderr'}}
        retained.update(stdout_sha256=hashlib.sha256(observed['stdout']).hexdigest(),
            stderr=observed['stderr'][-2000:].decode(errors='replace'))
        base.update(syntax_valid=True, execution=retained)
        if observed['status'] != 'completed':
            return dict(**base, status='runtime_or_resource_qualification')
        try:
            result = json.loads(observed['stdout'])
            rows = result['results']
            if (set(result) != {'format', 'results'} or result['format'] != 'own-APPS-raw-results-v2'
                    or not isinstance(rows, list) or len(rows) != cases['total_cases']
                    or any(set(row) != {'value', 'seconds'} or type(row['seconds']) not in {float,int}
                        or not math.isfinite(row['seconds']) or row['seconds'] < 0 for row in rows)):
                raise ValueError('Every complete original case result required')
        except (ValueError, TypeError, KeyError, RecursionError):
            return dict(**base, status='incomplete_or_invalid_raw_results_qualification')
        flags = [compare(row['value'], expected, cases['mode']) for row, expected in zip(rows, cases['outputs'])]
        base.update(passed_cases=sum(flags), all_cases_observed=True, all_finite_published_cases_pass=all(flags),
            raw_results=result, case_passes=flags)
        return dict(**base, status='all_finite_published_cases_pass' if all(flags) else 'completed_published_case_mismatch')


def implementation_sources():
    paths = [Path(__file__).resolve(), HARNESS, ROOT/'src/slm_lab/standalone_program_v1.py',
        ROOT/'src/slm_lab/native_code.py', ROOT/'scripts/python_program_sandbox_v1.py', ROOT/'scripts/native_sandbox.py']
    return {str(path): file_hash(path) for path in paths}
