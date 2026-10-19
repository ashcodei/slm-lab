"""Real lexical edits and actual published-versus-finite execution controls."""
import ast
import copy

import pytest
from tokenizers import Tokenizer

from slm_lab.public_edge_case_hard_negatives_v1 import candidates, pair, sha
from slm_lab.public_function_contract_census_v2 import observe
from slm_lab.public_function_rollout_observation_v1 import observe as published


def fixture():
    doc = 'Café: distinguish prime numbers.\n>>> is_prime(3)\nTrue\n>>> is_prime(4)\nFalse\n'
    program = ('def is_prime(n: int) -> bool:\n    ' + repr(doc) + '\n'
               '    # Preserve this comment: < + True 99\n'
               '    if n < 2:\n        return False\n'
               '    return all(n % divisor != 0 for divisor in range(2, math.isqrt(n) + 1))\n')
    return dict(id='actual-unit-prime', group_id='unit-prime', split='train', prompt='Return whether n is prime.',
                response=program, original_public_URL='unit-fixture', rights={'unit_fixture': True},
                source_candidate=dict(id='actual-unit-prime', role='train', function='is_prime',
                                      original_source_git_path='maths/prime_check.py', original_function_program=program,
                                      original_literal_docstring=doc, whole_original_example_count=2))


def boundary_mutation(row):
    return next(r for r in candidates(row, maximum=64) if r['patch']['old'] == '<' and r['patch']['new'] == '<=')


def test_source_comments_unicode_documentation_annotations_and_deterministic_selection_are_preserved():
    row = fixture()
    choices = candidates(row, maximum=64)
    assert len(choices) > 5 and choices == candidates(row, maximum=64)
    assert candidates(row, maximum=3) == choices[:3]
    original = ast.parse(row['response']).body[0]
    for value in choices:
        patch = value['patch']; start, end = patch['start_character'], patch['end_character']
        assert value['program'] == row['response'][:start] + patch['new'] + row['response'][end:]
        assert '# Preserve this comment: < + True 99' in value['program']
        assert ast.get_docstring(ast.parse(value['program']).body[0], clean=False) == ast.get_docstring(original, clean=False)
        assert ast.dump(ast.parse(value['program']).body[0].args) == ast.dump(original.args)
        assert patch['original_prefix_bytes'] >= start
    held = copy.deepcopy(row); held['split'] = 'validation'
    with pytest.raises(ValueError):
        candidates(held)


@pytest.fixture(scope='module')
def actual():
    row = fixture(); mutation = boundary_mutation(row)
    return row, mutation, observe(row), observe(row, mutation['program']), published(row, mutation['program'])


def test_actual_program_passes_every_public_example_but_fails_the_prime_two_edge_case(actual):
    row, mutation, positive, negative, public = actual
    assert positive['all_cases_passed'] and positive['all_cases_observed']
    assert public['all_original_examples_passed'] and public['entry_body_calls'] > 0
    assert negative['all_cases_observed'] and not negative['all_cases_passed']
    tokenizer = Tokenizer.from_file('data/direct-algorithm-function-contracts-admitted-v1/tokenizer.json')
    result = pair(row, mutation, positive, negative, public, tokenizer)
    assert result['counterexample']['case']['args'] == [2]
    assert result['counterexample']['case']['expected'] is True
    assert result['prompt'] == row['prompt'] and result['chosen'] == row['response']
    assert result['chosen_complete_response_EOS_ids'][-1] == result['rejected_complete_response_EOS_ids'][-1] == 2
    assert tokenizer.decode(result['rejected_complete_response_EOS_ids'][:-1], skip_special_tokens=False) == mutation['program']
    assert not result['optimization_admitted'] and not result['human_preferences'] and not result['on_policy_RL']


def test_missing_cases_resource_failures_forged_public_passes_and_nonlexical_patches_are_rejected(actual):
    row, mutation, positive, negative, public = actual
    tokenizer = Tokenizer.from_file('data/direct-algorithm-function-contracts-admitted-v1/tokenizer.json')
    for field, value in [('all_cases_observed', False), ('raw_results', []), ('case_passes', [True]*negative['cases'])]:
        bad = copy.deepcopy(negative); bad[field] = value
        with pytest.raises(ValueError):
            pair(row, mutation, positive, bad, public, tokenizer)
    bad = copy.deepcopy(public); bad['all_original_examples_observed'] = False
    with pytest.raises(ValueError):
        pair(row, mutation, positive, negative, bad, tokenizer)
    bad = copy.deepcopy(mutation); bad['patch']['new'] = '<= 1 or n <'
    patch = bad['patch']; bad['program'] = row['response'][:patch['start_character']] + patch['new'] + row['response'][patch['end_character']:]
    bad['program_sha256'] = sha(bad['program'])
    with pytest.raises(ValueError):
        pair(row, bad, positive, negative, public, tokenizer)
