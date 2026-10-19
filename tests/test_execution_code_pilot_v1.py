"""Independent full scalar/autograd oracle for the actual bounded pilot update."""
from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from slm_lab.code_residual_pilot_v1 import PROFILES, learning_rate
from slm_lab.execution_code_pilot_v1 import update
from slm_lab.execution_preference_v1 import reference_scores
from test_code_prompt_pilot_v1 import own_model, examples, OriginalReplayFixture


def inputs():
    code = examples()
    positives = [dict(id='verified:' + e['id'], split='train', human_preferences=False,
        feedback_source='machine_execution_of_original_publisher_tests', sequence=e['positive']) for e in code]
    pairs = []
    for e in code:
        rejected = deepcopy(e['positive'])
        start = next(i for i, v in enumerate(rejected[1]) if v != -100)
        rejected[0][start] += 1; rejected[1][start] += 1
        pairs.append(dict(id='pair:' + e['id'], split='train', human_preferences=False,
            feedback_source='machine_execution_of_original_publisher_tests', finite_execution_verified=True,
            chosen=e['positive'], rejected=rejected))
    return code, positives, pairs


@pytest.mark.parametrize('alpha', [0., .1])
def test_complete_CE_replay_and_repeated_pair_mean_match_independent_autograd(alpha):
    torch.set_num_threads(1)
    model = own_model(); oracle = deepcopy(model); code, positives, pairs = inputs()
    reference = reference_scores(model.code, pairs, policy_sha256='a' * 64, device='cpu', amp=nullcontext)
    repeated = [pairs[0], pairs[0], pairs[1]]
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=3e-5, weight_decay=.01, foreach=False)
    expected_optimizer = torch.optim.AdamW(oracle.adapter_parameters(), lr=3e-5, weight_decay=.01, foreach=False)
    for step in range(2):
        expected_optimizer.param_groups[0]['lr'] = learning_rate(PROFILES[1], step)
        expected_optimizer.zero_grad(set_to_none=True)
        def value(sequence):
            ids, labels = sequence
            return oracle.code(torch.tensor([ids]), labels=torch.tensor([labels]))['loss']
        ce = 0.
        for sequences in [[e['positive'] for e in code], [e['sequence'] for e in positives]]:
            n = [sum(v != -100 for v in labels[1:]) for _, labels in sequences]
            ce = ce + sum(count * value(seq) for count, seq in zip(n, sequences)) / sum(n)
        ce = ce + .5 * oracle.code(**OriginalReplayFixture().batch([{}], 'response', 'cpu'))['loss']
        preference = 0.
        for pair in repeated:
            chosen = -sum(v != -100 for v in pair['chosen'][1][1:]) * value(pair['chosen'])
            rejected = -sum(v != -100 for v in pair['rejected'][1][1:]) * value(pair['rejected'])
            ref = reference['records'][pair['id']]
            preference = preference + F.softplus(-.1 * (chosen - rejected - ref['chosen_logp'] + ref['rejected_logp'])) / len(repeated)
        objective = ce + alpha * preference
        objective.backward(); torch.nn.utils.clip_grad_norm_(oracle.adapter_parameters(), 1., error_if_nonfinite=True)
        expected_optimizer.step()
        result = update(model, optimizer, code, positives, repeated, reference, policy_sha256='a' * 64,
            replay_encoder=OriginalReplayFixture(), replay_row={}, family='generic_visual', step=step,
            alpha=alpha, device='cpu', amp=nullcontext)
        assert result['summed_objective'] == pytest.approx(float(objective.detach()), abs=2e-6)
        assert len(result['DPO']['records']) == 3 and result['pair_source_ids'][0] == result['pair_source_ids'][1]
        for actual, expected in zip(model.residuals.parameters(), oracle.residuals.parameters()):
            torch.testing.assert_close(actual, expected, atol=2e-8, rtol=1e-5)
        for actual, expected in zip(model.base.parameters(), oracle.base.parameters()): assert torch.equal(actual, expected)
        for actual, expected in zip(optimizer.state.values(), expected_optimizer.state.values()):
            torch.testing.assert_close(actual['exp_avg'], expected['exp_avg'], atol=1e-7, rtol=2e-5)


def test_held_source_and_changed_reference_reject_before_updates_or_gradients():
    for kind in ['code', 'positive', 'pair', 'reference']:
        model = own_model(); code, positives, pairs = inputs()
        reference = reference_scores(model.code, pairs, policy_sha256='a' * 64, device='cpu', amp=nullcontext)
        if kind == 'code': code[0]['role'] = 'validation'
        elif kind == 'positive': positives[0]['split'] = 'valid'
        elif kind == 'pair': pairs[0]['split'] = 'validation'
        else: reference['records'][pairs[0]['id']]['chosen_targets'] += 1
        optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=3e-5, foreach=False)
        before = {n: t.clone() for n, t in model.state_dict().items()}
        with pytest.raises(ValueError):
            update(model, optimizer, code, positives, pairs, reference, policy_sha256='a' * 64,
                replay_encoder=OriginalReplayFixture(), replay_row={}, family='generic_visual', step=0,
                alpha=.1, device='cpu', amp=nullcontext)
        assert not optimizer.state and all(p.grad is None for p in model.parameters())
        assert all(torch.equal(t, model.state_dict()[n]) for n, t in before.items())
