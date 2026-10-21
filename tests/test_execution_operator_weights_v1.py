import copy

import pytest
import torch
from torch.nn import functional as F

from slm_lab.code_target_regions_v1 import weighted_CE
from slm_lab.documented_operator_observation_v1 import observe
from slm_lab.execution_operator_weights_v1 import catalogue, derivative, candidate_target_weights
from test_code_target_regions_v1 import ByteOffsetsTokenizer
from test_documented_doctest_v1 import candidate


def test_UTF8_CRLF_chains_and_docstrings_have_exact_operator_spans_and_positive_complete_targets():
    program = 'def transform(π, y):\r\n    """é + <= and must stay."""\r\n    # preserve +\r\n    return -π + y if 0 <= π < y and y != 0 else π - y\r\n'
    inventory = catalogue(program); mutations = inventory['mutations']
    assert len(mutations) == 7 and not inventory['lexical_qualifications']
    assert {m['original_operator'] for m in mutations} == {'-', '+', '<=', '<', 'and', '!='}
    for m in mutations:
        a, b = m['program_char_start'], m['program_char_end']; changed = derivative(program, m)
        assert program[a:b] == m['original_operator']
        assert changed == program[:a]+m['replacement_operator']+program[b:]
        assert program.encode()[m['program_UTF8_byte_start']:m['program_UTF8_byte_end']].decode() == program[a:b]
    selected = next(m for m in mutations if m['operator'] == 'Add')
    response = '```python\n'+program+'```'; tokenizer = ByteOffsetsTokenizer()
    result = candidate_target_weights(response, program, mutations, [selected['mutation_id']], tokenizer)
    assert tokenizer.decode(result['response_EOS_ids'][:-1]) == response
    assert result['positive_target_weights'][-1] == 1 and min(result['positive_target_weights']) >= 1
    offsets = tokenizer.encode(response).offsets
    weighted = [response[a:b] for (a, b), w in zip(offsets, result['positive_target_weights']) if w > 1]
    assert weighted == ['+']
    before = copy.deepcopy(mutations); mutations[0]['program_char_start'] = 0
    assert catalogue(program)['mutations'] == before  # Cached inventory cannot be mutated by its caller.
    with pytest.raises(ValueError):
        derivative(program, mutations[0])


def test_fstring_AST_operator_without_lexical_span_is_qualified_and_never_rewrites_literal_text():
    source = 'def transform(x):\n    return f"literal + {x+1}"\n'
    result = catalogue(source)
    assert not result['mutations'] and len(result['lexical_qualifications']) == 1
    assert result['entire_declared_operator_population_enumerated']


def test_actual_whole_public_stateful_examples_discriminate_value_changes_without_changing_source():
    text = ">>> values = [1, 3]\n>>> transform(values[0])\n2\n>>> values.append(7)\n>>> transform(values[-1])\n8\n"
    original = candidate('def transform(x):\n    return x + 1', text); before = copy.deepcopy(original)
    mutation = catalogue(original['program'])['mutations'][0]
    reference = observe(original); changed = observe(original, mutation)
    assert reference['all_original_examples_passed'] and reference['original_example_count'] == 4
    assert changed['all_original_examples_observed'] and changed['eligible_value_discrimination']
    assert changed['normal_output_mismatch_example_positions'] == [1, 3]
    assert original == before and not changed['human_preference'] and not changed['model_rollout']
    assert not changed['data_admitted_to_training']


def test_unseen_boundary_and_runtime_differences_are_not_semantic_or_value_weight_certificates(tmp_path):
    c = candidate('def transform(x):\n    return 0 if x < 0 else 1', '>>> transform(1)\n1\n>>> transform(3)\n1\n')
    unseen = observe(c, catalogue(c['program'])['mutations'][0])
    assert unseen['status'] == 'not_distinguished_by_original_examples' and not unseen['eligible_value_discrimination']
    c = candidate('def transform(x):\n    return x * 0', '>>> transform(1)\n0\n>>> transform(3)\n0\n')
    failure = observe(c, catalogue(c['program'])['mutations'][0])
    assert failure['all_original_examples_observed'] and failure['unexpected_exception_example_positions'] == [0, 1]
    assert not failure['eligible_value_discrimination']
    c = candidate('def transform(x):\n    return x + 1',
        ">>> transform('a')\nTraceback (most recent call last):\n...\nTypeError: can only concatenate str (not \"int\") to str\n>>> transform(3)\n4\n")
    mixed = observe(c, catalogue(c['program'])['mutations'][0])
    assert mixed['normal_output_mismatch_example_positions'] == [1]
    assert mixed['failed_exception_comparison_positions'] == [0] and not mixed['eligible_value_discrimination']
    secret = tmp_path/'private.txt'; secret.write_text('private-fixture-value')
    c = candidate('def transform(x):\n    if x == 1: return 2\n    return open('+repr(str(secret))+').read()',
                  '>>> transform(1)\n2\n>>> transform(1)\n2\n')
    confined = observe(c, catalogue(c['program'])['mutations'][0])
    assert not confined['eligible_value_discrimination']
    assert all('PermissionError' in r['exception_only'] for r in confined['raw_results'])
    c['source']['original_partition'] = 'test'
    with pytest.raises(ValueError):
        observe(c)


def test_actual_weighted_gradient_matches_independent_logprob_objective_for_all_original_targets_and_EOS():
    source = 'def transform(x):\n    return x + 1'; tokenizer = ByteOffsetsTokenizer()
    mutations = catalogue(source)['mutations']; target = candidate_target_weights(source, source, mutations,
        [mutations[0]['mutation_id']], tokenizer)
    ids = [1, 7, *target['response_EOS_ids']]; labels = torch.tensor([[-100, -100, *target['response_EOS_ids']]])
    weights = torch.tensor([[0., 0., *target['positive_target_weights']]])
    logits = torch.linspace(-2, 2, len(ids)*300).reshape(1, len(ids), 300).requires_grad_()
    numerator, denominator = weighted_CE(logits, labels, weights); (numerator/denominator).backward(); actual = logits.grad.clone()
    independent = logits.detach().clone().requires_grad_(); lp = F.log_softmax(independent[0, :-1], -1)
    objective = sum(-lp[i-1, ids[i]]*weights[0, i] for i in range(2, len(ids)))/weights.sum()
    objective.backward()
    torch.testing.assert_close(numerator/denominator, objective)
    torch.testing.assert_close(actual, independent.grad)
