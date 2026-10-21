"""Observe explicit operator derivatives against all original public examples.

The original candidate contract is validated without pretending a derivative
is publisher code. Expected comparison text stays in the parent; original
public docstrings may themselves contain examples. The legacy confined raw
harness and compile label are reused for protocol equivalence.
"""
import json
import math
from pathlib import Path
import tempfile

from .documented_doctest_v1 import contract, compare, sha, PRELUDE, HARNESS
from .execution_operator_weights_v1 import derivative
from .standalone_program_v1 import python_policy, run_bounded


def observe(candidate, mutation=None, *, seconds=5):
    tests = contract(candidate)
    if not 0 < seconds <= 30:
        raise ValueError('An explicit bounded complete-example wall budget is required')
    original = candidate['program']; program = original if mutation is None else derivative(original, mutation)
    base = dict(original_source_id=candidate['source']['id'], original_public_program_sha256=sha(original),
        executed_program_sha256=sha(program), mutation_id=None if mutation is None else mutation['mutation_id'],
        original_doctest_text_sha256=candidate['doctest_text_sha256'], declared_prelude_sha256=sha(PRELUDE),
        original_example_count=len(tests), original_source_and_examples_unchanged=True,
        expected_outputs_not_sent_as_case_metadata=True, public_docstrings_may_contain_original_examples=True,
        all_original_examples_observed=False, all_original_examples_passed=False,
        eligible_value_discrimination=False, model_rollout=False, human_preference=False, data_admitted_to_training=False)
    with tempfile.TemporaryDirectory(prefix='slm-operator-observation-') as tmp:
        work = Path(tmp); (work/'program.py').write_text(program); (work/'prelude.py').write_text(PRELUDE)
        (work/'harness.py').write_bytes(HARNESS.read_bytes())
        settings = dict(program=str(work/'program.py'), prelude=str(work/'prelude.py'), function=candidate['function'],
                        examples=[t.source for t in tests])
        (work/'settings.json').write_text(json.dumps(settings))
        policy = python_policy(work); policy['command'][-1] = str(work/'harness.py')
        policy['command'].append(str(work/'settings.json')); policy['memory_bytes'] = 512*1024**2
        policy['cpu_seconds'] = max(1, math.ceil(seconds))
        raw = run_bounded(policy, b'', seconds=seconds, output_bytes=1024**2, python=True)
    execution = {k:v for k, v in raw.items() if k not in ('stdout', 'stderr')}
    execution['stderr'] = raw['stderr'][-2000:].decode(errors='replace')
    if raw['status'] != 'completed':
        return dict(**base, status='runtime_or_resource_population_qualification', execution=execution)
    try:
        result = json.loads(raw['stdout']); rows = result['results']
        if (set(result) != {'format', 'results'} or result['format'] != 'own-documented-doctest-raw-output-v1'
                or len(rows) != len(tests) or any(set(r) != {'stdout', 'exception_only', 'entry_body_calls'}
                    or not isinstance(r['stdout'], str) or type(r['entry_body_calls']) is not int or r['entry_body_calls'] < 0
                    or r['exception_only'] is not None and not isinstance(r['exception_only'], str) for r in rows)):
            raise ValueError('Every original raw public-example observation must be retained')
    except (ValueError, TypeError, KeyError):
        return dict(**base, status='incomplete_raw_output_qualification', execution=execution)
    checks = [compare(t, r) for t, r in zip(tests, rows)]
    calls = sum(r['entry_body_calls'] for r in rows)
    normal_failures = [i for i, (t, r, passed) in enumerate(zip(tests, rows, checks))
                       if not passed and t.exc_msg is None and r['exception_only'] is None]
    unexpected = [i for i, (t, r) in enumerate(zip(tests, rows)) if t.exc_msg is None and r['exception_only'] is not None]
    exception_differences = [i for i, (r, passed) in enumerate(zip(rows, checks)) if not passed and r['exception_only'] is not None]
    passed = all(checks) and calls > 0
    eligible = mutation is not None and calls > 0 and bool(normal_failures) and not exception_differences
    if not calls:
        status = 'original_examples_do_not_exercise_entrypoint'
    elif passed:
        status = 'reference_all_original_examples_passed' if mutation is None else 'not_distinguished_by_original_examples'
    elif eligible:
        status = 'operator_value_discriminated_by_original_examples'
    else:
        status = 'completed_exception_or_context_difference_qualification'
    return dict(**{**base, 'all_original_examples_observed':True, 'all_original_examples_passed':passed,
                   'eligible_value_discrimination':eligible}, status=status, execution=execution, raw_results=rows,
                example_passes=checks, entry_body_calls=calls, passed_examples=sum(checks),
                normal_output_mismatch_example_positions=normal_failures, unexpected_exception_example_positions=unexpected,
                failed_exception_comparison_positions=exception_differences,
                finite_public_example_discrimination_not_general_semantic_correctness=True)
