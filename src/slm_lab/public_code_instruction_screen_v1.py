"""Conservative execution screen for whole public synthetic code examples.

Published tests are parsed as data, never executed as trusted code. Only
single literal equality assertions against one module-level function are
supported. Unsupported tests abstain the whole example. Passing these tests
does not prove the question or the teacher-generated tests are correct.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

from .standalone_program_v1 import python_policy, run_bounded

HARNESS = Path(__file__).resolve().parents[2] / 'scripts/public_code_literal_observations_v1.py'


class UnsupportedExample(ValueError):
    pass


def literal(node):
    if sum(1 for _ in ast.walk(node)) > 20000:
        raise UnsupportedExample('Literal input exceeds the bounded data domain')
    try:
        result = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, RecursionError) as error:
        raise UnsupportedExample('Every argument and expected value must be a literal') from error
    def check(value):
        if type(value) in {str, bytes, int, bool, type(None)}:
            return
        if type(value) in {float, complex}:
            if not math.isfinite(value.real) or not math.isfinite(value.imag):
                raise UnsupportedExample('Nonfinite literal')
            return
        if type(value) in {list, tuple, set, dict}:
            for part in (list(value.keys()) + list(value.values()) if type(value) is dict else value):
                check(part)
            return
        raise UnsupportedExample('Only finite built-in literal data is supported')
    check(result)
    if len(repr(result).encode()) > 256 * 1024:
        raise UnsupportedExample('Literal size budget exceeded')
    return result


def parse_whole(row):
    if not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in ['input','output','unit_tests','id']):
        raise UnsupportedExample('Original source identity, question, response and tests required')
    if len(row['output'].encode()) > 60000 or len(row['unit_tests'].encode()) > 1024**2:
        raise UnsupportedExample('Complete source size budget exceeded')
    match = re.fullmatch(r'\s*```(?:python|py)\r?\n([\s\S]*?)\r?\n```\s*', row['output'])
    if not match:
        raise UnsupportedExample('Exactly one complete Python code fence required')
    program = match.group(1)
    try:
        tree = ast.parse(program)
        tests = json.loads(row['unit_tests'])
    except (ValueError, SyntaxError, TypeError, RecursionError) as error:
        raise UnsupportedExample('Original program and test-list syntax required') from error
    functions = {item.name for item in tree.body if isinstance(item, ast.FunctionDef)}
    if not isinstance(tests, list) or not 1 <= len(tests) <= 128 or any(not isinstance(t, str) for t in tests):
        raise UnsupportedExample('Every original assertion string is required')
    cases = []
    names = set()
    for test in tests:
        try:
            statements = ast.parse(test).body
        except (SyntaxError, ValueError, RecursionError) as error:
            raise UnsupportedExample('Original assertion syntax required') from error
        if (len(statements) != 1 or not isinstance(statements[0], ast.Assert)
                or not isinstance(statements[0].test, ast.Compare)):
            raise UnsupportedExample('Only a single equality assertion is supported per original test')
        compare = statements[0].test
        if len(compare.ops) != 1 or not isinstance(compare.ops[0], ast.Eq) or len(compare.comparators) != 1:
            raise UnsupportedExample('One literal equality required')
        call = compare.left
        if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
                or call.func.id not in functions or call.func.id.startswith('__')
                or any(k.arg is None for k in call.keywords)):
            raise UnsupportedExample('Tests must call one original module-level function directly')
        names.add(call.func.id)
        args = [literal(node) for node in call.args]
        kwargs = {node.arg: literal(node.value) for node in call.keywords}
        if len(kwargs) != len(call.keywords):
            raise UnsupportedExample('Repeated keyword argument')
        expected = literal(compare.comparators[0])
        cases.append(dict(args_literal=repr(args), kwargs_literal=repr(kwargs), expected_literal=repr(expected)))
    if len(names) != 1:
        raise UnsupportedExample('All original tests must use one original function')
    return dict(program=program, function=next(iter(names)), cases=cases,
        complete_original_tests=len(tests), whole_original_response_unchanged=True,
        extraction='Exact interior of the sole Python fence; original training response remains untouched')


def observe(parsed, *, seconds=3):
    cases = parsed['cases']
    base = dict(all_cases_observed=False, all_published_tests_passed=False, expected_outputs_sent_to_worker=False,
        whole_original_tests=len(cases), published_tests_are_synthetic=True,
        independent_semantic_correctness_not_certified=True,
        program_sha256=hashlib.sha256(parsed['program'].encode()).hexdigest(),
        code_bytes_unchanged_after_fence_extraction=True, OS_confinement='Landlock and seccomp before interpreter startup')
    if type(seconds) not in {int,float} or not math.isfinite(seconds) or not 0 < seconds <= 10:
        raise ValueError('Bounded execution time required')
    with tempfile.TemporaryDirectory(prefix='slm-public-code-literal-v1-') as temporary:
        work = Path(temporary)
        (work/'program.py').write_text(parsed['program'])
        (work/'harness.py').write_bytes(HARNESS.read_bytes())
        # No gold literal is copied into the confined directory or settings.
        settings = dict(program=str(work/'program.py'), function=parsed['function'],
            inputs=[{k:c[k] for k in ['args_literal','kwargs_literal']} for c in cases])
        (work/'settings.json').write_text(json.dumps(settings))
        policy = python_policy(work)
        policy['command'][-1] = str(work/'harness.py')
        policy['command'].append(str(work/'settings.json'))
        policy['memory_bytes'] = 512*1024**2
        policy['cpu_seconds'] = max(1, math.ceil(seconds))
        actual = run_bounded(policy, b'', seconds=seconds, output_bytes=1024**2, python=True)
    execution = {k:v for k,v in actual.items() if k not in {'stdout','stderr'}}
    execution.update(stderr=actual['stderr'][-1200:].decode(errors='replace'),
        stdout_sha256=hashlib.sha256(actual['stdout']).hexdigest())
    base['execution'] = execution
    if actual['status'] != 'completed':
        return dict(base, status='runtime_resource_or_confinement_abstention')
    try:
        raw = json.loads(actual['stdout'])
        if (set(raw) != {'format','actual_value_literals'} or raw['format'] != 'public-code-literal-values-v1'
                or len(raw['actual_value_literals']) != len(cases)):
            raise ValueError('Every actual returned literal required')
        values = [literal(ast.parse(text,mode='eval').body) for text in raw['actual_value_literals']]
        golds = [ast.literal_eval(c['expected_literal']) for c in cases]
        flags = [actual == expected for actual,expected in zip(values,golds)]
    except (ValueError, TypeError, SyntaxError, KeyError, RecursionError):
        return dict(base,status='unsupported_or_incomplete_raw_output_abstention')
    return dict(base,status='all_published_tests_pass' if all(flags) else 'published_test_mismatch',
        all_cases_observed=True, all_published_tests_passed=all(flags),
        passed_tests=sum(flags), test_passes=flags, actual_value_literals=raw['actual_value_literals'])
