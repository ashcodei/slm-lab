import ast
from copy import deepcopy
import hashlib
import json

import pytest

from slm_lab.parameter_binding_curriculum_v1 import (
    UnsupportedBinding, argument_names, check, prepare, transform,
)


def row(code='def add(n, k):\n    return n + k', name='add', args='(2, 3)', expected='5'):
    return {'id': 'public-source', 'group_id': 'publisher-file', 'function': name,
            'original_program_sha256': hashlib.sha256(code.encode()).hexdigest(),
            'variant_index': 0, 'source': 'public-test-fixture', 'license': 'test-fixture',
            'prompt': 'Implement this Python function:\n\n' + ast.unparse(ast.parse(code).body[0]).splitlines()[0]
                      + '\n\nOriginal public description.\n\nPublished behavior examples:\n'
                      + name + args + ' -> ' + expected,
            'response': '```python\n' + code + '\n```',
            'verification_cases': [{'args_literal': args, 'expected_literal': expected}]}


def test_new_signature_and_body_pass_original_examples_without_mutating_source():
    original = row(); saved = deepcopy(original)
    result = transform(original, {'n': 'input_0', 'k': 'input_1'})
    assert original == saved
    assert 'def add(input_0, input_1):' in result['prompt']
    assert 'n -> input_0, k -> input_1' in result['prompt']
    assert 'return input_0 + input_1' in result['response']
    for key in ['group_id', 'original_program_sha256', 'verification_cases', 'source', 'license']:
        assert result[key] == original[key]
    assert check(result)['functional_passed']


def test_recursive_keywords_follow_the_new_interface():
    original = row('def fact(n):\n    if n == 0:\n        return 1\n    return n * fact(n=n - 1)',
                   'fact', '(6,)', '720')
    changed = transform(original, {'n': 'input_0'})
    assert 'fact(input_0=input_0 - 1)' in changed['response']
    assert check(changed)['functional_passed']


def test_comprehension_shadowing_and_external_keyword_names_keep_behavior():
    original = row('def calculate(n):\n    outer = n\n    values = [n * 2 for n in range(n)]\n    return sum(values) + outer',
                   'calculate', '(10,)', '100')
    assert check(transform(original, {'n': 'input_0'}))['functional_passed']
    original = row('def ordering(reverse):\n    return sorted([3, 1, 2], reverse=reverse)',
                   'ordering', '(True,)', '[3, 2, 1]')
    changed = transform(original, {'reverse': 'input_0'})
    assert 'reverse=input_0' in changed['response']
    assert check(changed)['functional_passed']


@pytest.mark.parametrize('mapping', [
    {'n': 'input_0'}, {'n': 'k', 'k': 'input_1'},
    {'n': 'input_0', 'k': 'input_0'}, {'n': 'class', 'k': 'input_1'},
])
def test_incomplete_colliding_or_invalid_name_maps_are_rejected(mapping):
    with pytest.raises(UnsupportedBinding): transform(row(), mapping)


@pytest.mark.parametrize('code', [
    'def f(n):\n    return locals()["n"]',
    'def f(n):\n    def helper():\n        return n\n    return helper()',
    'def f(n):\n    import math as n\n    return n.pi',
    'def f(n):\n    return f(**{"n": n - 1})',
    'def f(n):\n    alias = f\n    return alias(n=n)',
])
def test_unsafe_scope_reflection_or_indirect_recursion_is_excluded(code):
    with pytest.raises(UnsupportedBinding): transform(row(code, 'f', '(2,)', '2'), {'n': 'input_0'})


def test_example_types_are_only_a_naming_style_and_defaults_are_preserved():
    original = row('def f(n: int = 3):\n    return n + 1', 'f', '()', '4')
    assert argument_names(original, 'example_type') == {'n': 'value_0'}
    changed = transform(original, {'n': 'input_0'})
    assert 'input_0: int=3' in changed['response'] or 'input_0: int = 3' in changed['response']
    assert check(changed)['functional_passed']


def source(tmp_path):
    p = tmp_path / 'original.jsonl'; p.write_text(json.dumps(row()) + '\n')
    (tmp_path / 'manifest.json').write_text(json.dumps({'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()}}))
    return p


def test_preparation_reexecutes_original_and_variants_and_keeps_exact_counts(tmp_path, monkeypatch):
    import slm_lab.parameter_binding_curriculum_v1 as module
    monkeypatch.setattr(module, 'complete_encoding', lambda r, e: {'context_tokens': 100, 'target_tokens': 40})
    receipt = prepare(source(tmp_path), tmp_path / 'new-data', object())
    assert receipt['transformed_training_rows'] == 2
    assert receipt['original_algorithms_reexecuted'] == receipt['distinct_original_algorithms'] == 1
    assert receipt['complete_response_targets'] == 80
    assert not receipt['trained'] and not receipt['automatically_admitted_to_live_training']
    records = [json.loads(line) for line in (tmp_path / 'new-data/verification.jsonl').read_text().splitlines()]
    assert all(r['original']['functional_passed'] and r['transformed']['functional_passed'] for r in records)
    rows = [json.loads(line) for line in (tmp_path / 'new-data/train.jsonl').read_text().splitlines()]
    assert [r['variant_index'] for r in rows] == [1, 2]
    assert all(r['parent_variant_index'] == 0 for r in rows)


def test_source_mutation_during_verification_is_rejected(tmp_path, monkeypatch):
    import slm_lab.parameter_binding_curriculum_v1 as module
    p = source(tmp_path)
    monkeypatch.setattr(module, 'complete_encoding', lambda r, e: {'context_tokens': 100, 'target_tokens': 40})
    def changed_check(r):
        (tmp_path / 'manifest.json').write_text('{}')
        return {'functional_passed': True}
    monkeypatch.setattr(module, 'check', changed_check)
    with pytest.raises(ValueError, match='source changed'): prepare(p, tmp_path / 'new-data', object(), workers=1)
    assert not (tmp_path / 'new-data').exists()
