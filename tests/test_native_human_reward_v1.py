"""Actual small native reward learning, independent oracles and sharing checks."""
import copy
import math

import pytest
import torch

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.native_human_reward_v1 import (FrozenSharedNativeReward, NativeRewardFeatures,
    NativeRewardHeadStage, collect_features, native_features, evaluate_reward)
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_dense_human_preference_stage_v1 import pairs
from test_own_aligned_assistant_v1 import original_signature
from test_own_alignment_checkpoint_v1 import assert_nested_equal
from test_own_dense_growth_v1 import model as native_model


def setup(role='train'):
    base = native_model()[0].eval(); data = pairs(); data.role = role
    data.purpose = 'optimization' if role == 'train' else 'evaluation'
    data.manifest = dict(config=dict(max_text_length=base.config.max_text_length))
    for row in data.rows: row['split'] = role; row['original_split'] = role; row['source'] = 'synthetic CPU control'
    sources = dict(native_checkpoint_sha256='a'*64, human_manifest_sha256='b'*64, human_admission_sha256='c'*64,
        frozen_native_encoder_sha256=tensor_digest(base.network.named_parameters()))
    shards = [collect_features(base, data, rank, 2, sources, token_budget=base.config.max_text_length,
        maximum_sequences=3, allow_mechanics=True) for rank in range(2)]
    features = NativeRewardFeatures(base, data, shards, sources, allow_mechanics=True)
    setting = dict(seed=998171, steps=math.ceil(len(data)/2), batch_size=2, learning_rate=.003,
        betas=[.9, .95], weight_decay=.01, gradient_clip=1.)
    optimizer = torch.optim.AdamW(base.heads['reward'].parameters(), lr=setting['learning_rate'],
        betas=tuple(setting['betas']), weight_decay=setting['weight_decay'], foreach=False)
    return base, data, shards, features, setting, optimizer


def stage():
    base, _, _, features, setting, optimizer = setup()
    return NativeRewardHeadStage(base, optimizer, features, setting)


def test_actual_native_features_use_complete_unpadded_EOS_and_match_every_individual_response():
    base, data, shards, features, _, _ = setup(); before = original_signature(base); rng = torch.get_rng_state().clone()
    for p in range(len(data)):
        for field, slot in [('chosen', 0), ('rejected', 1)]:
            ids = torch.tensor([data.sequence(p, field)[0]])
            with torch.no_grad(): direct = base.network(ids, return_hidden=True, return_logits=False)['hidden'][0, -1]
            assert torch.allclose(features.values[p, slot], direct, atol=1e-6, rtol=1e-5)
    assert sum(s['complete_context_tokens'] for s in shards) == 55
    assert before == original_signature(base) and torch.equal(rng, torch.get_rng_state())
    assert not features.full and features.values.shape == (5, 2, base.config.hidden_size)
    with pytest.raises(ValueError, match='actual final EOS'): native_features(base, torch.tensor([[1, 4, 7]]))
    with pytest.raises(ValueError, match='right-padded'):
        native_features(base, torch.tensor([[1, 4, 2]]), torch.tensor([[True, False, True]]))


def test_partial_or_wrong_basis_sources_and_corrupt_shards_never_admit_as_full_production():
    base, data, shards, _, _, _ = setup()
    sources = shards[0]['sources']
    with pytest.raises(ValueError, match='all85786train/4642held'): collect_features(base, data, 0, 1, sources)
    for change in ['role', 'positions', 'fingerprint', 'feature']:
        altered = copy.deepcopy(shards)
        if change == 'role': altered[0]['role'] = 'valid'
        elif change == 'positions': altered[0]['positions'][0] = 1
        elif change == 'fingerprint': altered[0]['source_fingerprints'][0] = '0'*64
        else: altered[0]['features'][0, 0, 0] = float('nan')
        with pytest.raises(ValueError): NativeRewardFeatures(base, data, altered, sources, allow_mechanics=True)
    with torch.no_grad(): base.network.norm.weight.add_(.1)
    with pytest.raises(ValueError, match='feature basis'):
        collect_features(base, data, 0, 1, sources, allow_mechanics=True)


def test_reward_head_update_matches_independent_scalar_pair_loss_and_Adam_oracle():
    learner = stage(); head = copy.deepcopy(learner.base.heads['reward'])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003, betas=(.9, .95), weight_decay=.01, foreach=False)
    values = learner.features.values[learner.positions(0)]
    predictions = head(values).squeeze(-1)
    loss = sum(torch.logaddexp(torch.zeros(()), row[1]-row[0]) for row in predictions)/len(predictions)
    loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.); optimizer.step()
    fixed = [(p, p.detach().clone(), p._version) for _, p in learner.fixed]
    actual = learner.update()
    assert math.isclose(actual['pair_mean_nll'], float(loss.detach()), abs_tol=1e-7, rel_tol=1e-6)
    assert all(torch.allclose(p, head.get_parameter(n), atol=2e-7, rtol=2e-5)
        for n, p in learner.base.heads['reward'].named_parameters())
    assert all(torch.equal(p, old) and p._version == version and p.grad is None for p, old, version in fixed)
    assert len(learner.optimizer.state) == 3


def test_complete_unrepeated_epoch_tail_and_exact_next_head_Adam_RNG_recovery():
    learner = stage(); learner.update(); saved = learner.state(); pristine = copy.deepcopy(saved)
    expected = learner.update(); learned = copy.deepcopy(learner.base.heads['reward'].state_dict())
    adam = copy.deepcopy(learner.optimizer.state_dict()); rng = value_digest(capture_rng('cpu'))
    learner.resume(saved); assert learner.update() == expected
    assert_nested_equal(learned, learner.base.heads['reward'].state_dict()); assert_nested_equal(adam, learner.optimizer.state_dict())
    assert_nested_equal(saved, pristine); assert value_digest(capture_rng('cpu')) == rng
    tail = learner.update(); assert tail['pairs'] == 1
    assert sorted(p for row in learner.metrics for p in row['source_positions']) == list(range(5))
    completed = learner.finish(); assert completed['actual_train_pairs'] == 5 and completed['actual_updates'] == 3
    assert not completed['full_original_train_epoch_complete'] and completed['useful_own_policy_RLHF_unproved']


def test_bad_sources_metrics_Adam_head_or_RNG_reject_before_any_live_mutation():
    learner = stage(); learner.update(); saved = learner.state(); head = copy.deepcopy(learner.base.heads['reward'].state_dict())
    adam = copy.deepcopy(learner.optimizer.state_dict()); rng = value_digest(capture_rng('cpu'))
    mutations = [lambda x: x['sources'].update(native_checkpoint_sha256='0'*64),
        lambda x: x['metrics'][0]['source_positions'].reverse(),
        lambda x: x['optimizer']['state'][0]['exp_avg'].fill_(float('nan')),
        lambda x: x['head']['0.weight'].fill_(float('nan')),
        lambda x: x['rng'].update(torch=torch.zeros(4, dtype=torch.uint8))]
    for mutate in mutations:
        altered = copy.deepcopy(saved); mutate(altered); before = original_signature(learner.base)
        with pytest.raises(ValueError): learner.resume(altered)
        assert before == original_signature(learner.base); assert_nested_equal(head, learner.base.heads['reward'].state_dict())
        assert_nested_equal(adam, learner.optimizer.state_dict()); assert value_digest(capture_rng('cpu')) == rng


def test_held_rankings_and_modified_cache_cannot_supply_reward_training():
    base, _, _, held, setting, optimizer = setup('valid')
    with pytest.raises(ValueError, match='train-only'): NativeRewardHeadStage(base, optimizer, held, setting)
    before = original_signature(base); result = evaluate_reward(base, held, batch_size=2)
    assert before == original_signature(base) and result['overall']['pairs'] == 5
    assert [r['source_position'] for r in result['results']] == list(range(5))
    assert result['response_length_controls']['chosen_shorter']['pairs'] == 5
    for row in result['results']:
        assert row['correct'] == (row['chosen_score'] > row['rejected_score'])
    learner = stage(); before = original_signature(learner.base)
    learner.features.values[0, 0, 0] += 1
    with pytest.raises(ValueError, match='unchanged'): learner.update()
    assert original_signature(learner.base) == before and not learner.optimizer.state


def test_trained_reward_shares_counted_native_and_stays_fixed_during_real_actor_critic_learning():
    learner = stage()
    for _ in range(learner.options['steps']): learner.update()
    receipt = learner.finish(); base = learner.base
    with pytest.raises(ValueError, match='actually completed'): FrozenSharedNativeReward(base, receipt)
    scorer = FrozenSharedNativeReward(base, receipt, allow_mechanics=True)
    policy = OwnDenseActor(base, rank=2).eval(); ids = torch.tensor([[1, 4, 8, 9, 2]])
    before = scorer.score(ids); fixed = original_signature(base)
    assert torch.equal(before, base.preference_score(ids).reshape(()))
    assert scorer.base is policy.base and scorer.counts()['additional_values_over_same_native_policy'] == 0
    union = {id(p): p for m in [policy, scorer] for p in m.parameters()}
    assert sum(p.numel() for p in union.values()) == policy.counts()['actual_unique']
    optimizer = torch.optim.AdamW([p for _, p in policy.named_selected(('actor', 'critic'))], lr=.003, foreach=False)
    policy.train(); output = policy(ids, labels=ids); loss = output['loss']+(output['values']-1).square().mean()
    loss.backward(inputs=[p for _, p in policy.named_selected(('actor', 'critic'))]); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    assert torch.equal(before, scorer.score(ids)) and fixed == original_signature(base)
    assert policy.parts['critic'].weight.count_nonzero() > 0
    with torch.no_grad(): base.heads['reward'][-1].weight.add_(.01)
    with pytest.raises(ValueError, match='remain fixed'): scorer.score(ids)
