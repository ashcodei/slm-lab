"""Independent per-token value/gradient oracles, complete masks and tails."""
import copy

import pytest
import torch
from torch.nn import functional as F

from slm_lab.complete_response_stopping_loss_v1 import PROFILES, complete_response_CE_sums, metadata


def tensors():
    labels = torch.tensor([[-100,-100,5,2,-100],[-100,-100,6,7,2],[-100,-100,-100,5,2]])
    mask = torch.tensor([[1,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1]], dtype=torch.bool)
    logits = torch.linspace(-2, 2, 3*5*11).reshape(3,5,11).requires_grad_()
    return logits, labels, mask


@pytest.mark.parametrize('profile', PROFILES)
def test_every_token_value_and_logit_gradient_matches_independent_oracle(profile):
    logits, labels, mask = tensors()
    factor = 1 if profile == 'plain_tokens' else 4
    denominator = 7+(factor-1)*3
    result = complete_response_CE_sums(logits, labels, mask, profile=profile)
    assert result['original_target_counts'].tolist() == [2,3,2]
    assert result['genuine_EOS_counts'].tolist() == [1,1,1]
    assert result['weighted_target_counts'].sum().item() == denominator
    weighted = result['weighted_CE_sums'].sum()/denominator
    expected = logits.new_tensor(0.)
    gradient = torch.zeros_like(logits)
    eos = []
    for row in range(3):
        for position in range(1, 5):
            token = int(labels[row,position])
            if token == -100:
                continue
            local_weight = factor if token == 2 else 1
            score = -logits[row,position-1].log_softmax(-1)[token]
            expected = expected+score*local_weight/denominator
            gradient[row,position-1] = logits[row,position-1].detach().softmax(-1)
            gradient[row,position-1,token] -= 1
            gradient[row,position-1] *= local_weight/denominator
            if token == 2:
                eos.append(score)
    torch.testing.assert_close(weighted, expected)
    torch.testing.assert_close(result['EOS_CE_sums'], torch.stack(eos))
    weighted.backward()
    torch.testing.assert_close(logits.grad, gradient, atol=2e-8, rtol=2e-6)
    if factor == 1:
        ordinary = F.cross_entropy(logits[:,:-1].reshape(-1,11), labels[:,1:].reshape(-1), ignore_index=-100)
        torch.testing.assert_close(weighted, ordinary)


def test_padding_microbatches_and_three_empty_tail_ranks_preserve_global_normalization():
    logits, labels, mask = tensors()
    whole = complete_response_CE_sums(logits, labels, mask, profile='eos_weight4')['weighted_CE_sums'].sum()/16
    whole.backward()
    wanted = logits.grad.clone()
    local, local_labels, local_mask = tensors()
    # Three populated microbatches, one actual rank without a local source.
    for rank in range(4):
        positions = list(range(3))[rank::4]
        if positions:
            partial = complete_response_CE_sums(local[positions], local_labels[positions], local_mask[positions], profile='eos_weight4')
            (partial['weighted_CE_sums'].sum()/16).backward()
    torch.testing.assert_close(local.grad, wanted)
    tail = complete_response_CE_sums(logits[2:3], labels[2:3], mask[2:3], profile='eos_weight4')
    assert tail['original_target_counts'].item() == 2 and tail['weighted_target_counts'].item() == 5
    # A one-source global tail has a denominator of five, never the previous
    # batch's sixteen or a rounded/repeated four-source population.
    independent_tail = (-logits[2,2].log_softmax(-1)[5]-4*logits[2,3].log_softmax(-1)[2])/5
    torch.testing.assert_close(tail['weighted_CE_sums'].sum()/5, independent_tail)


def test_missing_duplicate_EOS_holes_padding_prompt_and_nonfinite_fail_closed():
    for corruption in ['missing_EOS','duplicate_EOS','hole','left_padding','prompt_target','bad_id','nonfinite']:
        logits, labels, mask = tensors()
        if corruption == 'missing_EOS': labels[0,3] = 4
        if corruption == 'duplicate_EOS': labels[1,2] = 2
        if corruption == 'hole': labels[1,3] = -100
        if corruption == 'left_padding': mask[0,0] = False
        if corruption == 'prompt_target': labels[0,0] = 4
        if corruption == 'bad_id': labels[1,3] = 11
        if corruption == 'nonfinite':
            with torch.no_grad(): logits[0,2,0] = float('nan')
        with pytest.raises(ValueError):
            complete_response_CE_sums(logits, labels, mask, profile='eos_weight4')
    logits, labels, mask = tensors()
    with pytest.raises(ValueError): complete_response_CE_sums(logits, labels, mask, profile='unregistered_weight')
    with pytest.raises(ValueError): complete_response_CE_sums(logits.double(), labels, mask, profile='plain_tokens')


def test_global_metadata_retains_all_views_and_exact_original_target_denominators():
    infos = [dict(position=i, source_role='train', view='reasoning' if i == 1 else 'code_only',
        supervision='public_synthetic_weak_SFT_only', use_as_RL_reward_oracle_forbidden=True,
        complete_context_tokens=length, original_prompt_positions=start, response_EOS_targets=length-start)
        for i, (length, start) in enumerate([(4,2),(5,2),(5,3)])]
    original = copy.deepcopy(infos)
    plain, stop = [metadata(infos, profile) for profile in PROFILES]
    assert plain['response_EOS_targets'] == stop['response_EOS_targets'] == 7
    assert plain['actual_global_weighted_targets'] == 7 and stop['actual_global_weighted_targets'] == 16
    assert stop['genuine_EOS_targets'] == 3 and stop['non_EOS_response_targets'] == 4
    assert stop['complete_sources'] == plain['complete_sources'] == original
    stop['complete_sources'][0]['view'] = 'changed'
    assert infos == original
    for field, value in [('source_role','reserved'),('use_as_RL_reward_oracle_forbidden',False),('response_EOS_targets',99)]:
        bad = copy.deepcopy(infos)
        bad[0][field] = value
        with pytest.raises(ValueError): metadata(bad, 'eos_weight4')
    with pytest.raises(ValueError): metadata([infos[0],infos[0]], 'eos_weight4')
