import ast

import pytest
import torch

from slm_lab.code_specification_likelihood_v1 import comparison, score
from slm_lab.code_tasks import TASKS, inputs, reference
from slm_lab.inference_rotary_cache_v1 import inference_cache
from slm_lab.native_greedy_observer_v1 import signature
from test_own_dense_growth_v1 import model as own_model


def test_complete_gold_programs_and_conflicting_specs_preserve_interfaces_without_target_changes():
    for task in TASKS:
        for language in ['Python', 'JavaScript']:
            for frame in ['completion', 'instruction']:
                pair = comparison(task, language, frame)
                other = next(t for t in TASKS if t[0] == pair['conflicting_specification_task'])
                assert other != task and other[1] == task[1]
                assert pair['conflicting_prompt'].replace(other[2], task[2]) == pair['original_prompt']
                assert pair['evaluation_only_optimization_forbidden']
                if language == 'Python':
                    expected = reference(task)
                    for condition in ['original', 'conflicting']:
                        code = pair[condition+'_prompt'] + pair['response'] if frame == 'completion' else pair['response']
                        ast.parse(code); namespace = {}; exec(code, namespace)
                        assert all(namespace[task[0]](*args) == expected(*args) for args in inputs(task[0]))
                elif frame == 'completion':
                    assert pair['response'] == task[5].split('{', 1)[1]
                else:
                    assert pair['response'] == task[5]


def test_response_EOS_scoring_matches_independent_cached_next_token_oracle_and_ignores_prompt_labels():
    torch.set_num_threads(1); model, _ = own_model(); model.eval()
    prompt = [1, 7, 4]; target = [8, 5, 2]
    before = signature(model); rng = torch.get_rng_state().clone()
    actual = score(model, prompt, target)
    # Independently advance a causal cache along the complete gold prefix;
    # score() instead makes one full-sequence forward and slices response rows.
    expected = []; top = []; rank = []
    with torch.no_grad(), inference_cache():
        state = model.network.prefill(torch.tensor([prompt]))
        for token in target:
            logits = state['logits'][0, -1].double()
            probabilities = torch.softmax(logits, -1)
            expected.append(float(-probabilities[token].log()))
            top.append(int(logits.argmax())); rank.append(int((logits > logits[token]).sum()) + 1)
            state = model.network.decode(torch.tensor([[token]]), state['cache'])
    assert actual['token_NLL'] == pytest.approx(expected, abs=1e-6)
    assert actual['teacher_prefix_top1_ids'] == top and actual['target_best_tied_rank'] == rank
    assert actual['response_EOS_targets'] == len(target)
    assert actual['EOS_NLL'] == actual['token_NLL'][-1]
    assert signature(model) == before and torch.equal(torch.get_rng_state(), rng)
    for bad_prompt, bad_target in [([7, 4], target), (prompt, target[:-1]), (prompt, [2, 8, 2])]:
        with pytest.raises(ValueError):
            score(model, bad_prompt, bad_target)
