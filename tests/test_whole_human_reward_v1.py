import copy
import math

import pytest
import torch

from slm_lab.own_aligned_assistant_v2 import OwnAlignedAssistantV2
from slm_lab.whole_human_reward_v1 import (
    RewardFeatures, RewardHeadStage, collect_features, evaluate_reward,
    general_features, pairwise_loss, scores,
)
from test_own_aligned_assistant_v1 import fixture, original_signature
from test_own_alignment_checkpoint_v1 import assert_nested_equal


SOURCES = dict(source_manifest_sha256='a'*64, source_admission_sha256='b'*64,
               frozen_encoder_sha256='c'*64)


class Data:
    family = 'preferences'
    manifest = dict(config=dict(max_text_length=16))

    def __init__(self, role='train'):
        self.role = role; self.purpose = 'optimization' if role == 'train' else 'evaluation'
        self.rows = [dict(id=f'{role}-{i}', source_record_sha256=f'{i:064x}', split=role,
            source='helpful' if i%2 else 'harmless', native={
                'chosen':dict(length=4+i, target_start=2),
                'rejected':dict(length=5+i, target_start=2)}) for i in range(5)]

    def __len__(self):
        return len(self.rows)

    def sequence(self, position, field):
        size = self.rows[position]['native'][field]['length']
        ids = [1] + [4+position+(field == 'rejected')]*(size-2) + [2]
        return ids, [-100, -100]+ids[2:]


def setup(role='train'):
    model = OwnAlignedAssistantV2(fixture(), rank=2, reward_hidden=8, word_temporal=4).eval()
    data = Data(role)
    shards = [collect_features(model, data, rank, 2, SOURCES, token_budget=16, max_sequences=3)
              for rank in range(2)]
    features = RewardFeatures(data, shards, SOURCES)
    options = dict(seed=921, steps=math.ceil(len(data)/2), batch_size=2, learning_rate=.003,
                   betas=[.9, .95], weight_decay=.01, gradient_clip=1.)
    optimizer = torch.optim.AdamW(model.extension_parameters('reward'), lr=options['learning_rate'],
                                  betas=tuple(options['betas']), weight_decay=.01, foreach=False)
    return model, data, shards, features, options, optimizer


def test_actual_frozen_general_reward_features_and_gradients_do_not_follow_actor():
    model, data, shards, features, options, optimizer = setup()
    signature = original_signature(model.original)
    ids = torch.tensor([[1, 4, 7, 2], [1, 6, 2, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    before = general_features(model, ids, mask)
    with torch.no_grad():
        direct = model.original.language.base(ids, attention_mask=mask,
            return_hidden=True, return_logits=False)['hidden']
        assert torch.equal(before, direct[torch.arange(2), torch.tensor([3, 2])])
        expected = model(ids)['logits'].clone()
        for p in model.extension_parameters('language'):
            p.add_(.1)
        assert not torch.equal(expected, model(ids)['logits'])
    assert torch.equal(before, general_features(model, ids, mask))
    pairwise_loss(scores(model, features.batch([0, 1], 'cpu'))).backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.extension_parameters('reward'))
    assert all(p.grad is None for p in model.extension_parameters('language', 'vision', 'critic', 'word'))
    assert all(p.grad is None for p in model.original.parameters())
    assert signature == original_signature(model.original)
    with pytest.raises(ValueError, match='boolean attention mask'):
        general_features(model, ids, mask.long())
    with pytest.raises(ValueError, match='right-padded'):
        general_features(model, ids, torch.tensor([[1, 0, 1, 0], [1, 1, 1, 0]], dtype=torch.bool))
    with pytest.raises(ValueError, match='detached'):
        scores(model, features.batch([0], 'cpu').requires_grad_())


def test_every_whole_original_pair_shard_and_source_role_validated_before_optimization():
    model, data, shards, features, options, optimizer = setup()
    assert features.values.shape == (5, 2, model.config.hidden_size)
    assert sum(s['complete_context_tokens'] for s in shards) == sum(
        row['native'][field]['length'] for row in data.rows for field in ('chosen', 'rejected'))
    # Independently run every unpadded complete original response, so a
    # swapped chosen/rejected slot or a padded-token feature cannot pass.
    for p in range(len(data)):
        for slot, field in enumerate(('chosen', 'rejected')):
            ids, _ = data.sequence(p, field)
            actual = general_features(model, torch.tensor([ids]))[0]
            assert torch.allclose(actual, features.values[p, slot], atol=1e-6, rtol=1e-5)
    for corrupt in ('fingerprint', 'positions', 'nonfinite', 'wrongrole'):
        altered = copy.deepcopy(shards)
        if corrupt == 'fingerprint': altered[0]['source_fingerprints'][0] = 'd'*64
        elif corrupt == 'positions': altered[0]['positions'][0] = 1
        elif corrupt == 'nonfinite': altered[0]['features'][0, 0, 0] = float('nan')
        else: altered[0]['role'] = 'valid'
        with pytest.raises(ValueError, match='contracts'):
            RewardFeatures(data, altered, SOURCES)
    valid_model, valid_data, valid_shards, held, _, _ = setup('valid')
    with pytest.raises(ValueError, match='train-only'):
        RewardHeadStage(model, optimizer, held, options, SOURCES)
    report = evaluate_reward(valid_model, held)
    assert report['overall']['pairs'] == 5 and len(report['results']) == 5
    assert set(report['by_source']) == {'helpful', 'harmless'}
    assert report['response_length_controls']['chosen_shorter']['pairs'] == 5
    assert report['response_length_controls']['chosen_longer']['pairs'] == 0


def test_real_reward_updates_exact_recovery_and_unrepeated_final_pair():
    model, data, shards, features, options, optimizer = setup()
    stage = RewardHeadStage(model, optimizer, features, options, SOURCES)
    fixed = [(p, p.detach().clone(), p._version) for p in model.parameters()
             if id(p) not in {id(x) for x in stage.parameters}]
    first = stage.update(); state = stage.state()
    expected = stage.update(); weights = model.extension_state(); adam = copy.deepcopy(optimizer.state_dict())
    stage.resume(state)
    assert stage.update() == expected
    assert_nested_equal(weights, model.extension_state()); assert_nested_equal(adam, optimizer.state_dict())
    last = stage.update()
    assert last['pairs'] == 1
    draws = [p for metric in stage.metrics for p in metric['source_positions']]
    assert len(draws) == len(set(draws)) == len(data)
    assert sorted(draws) == list(range(len(data)))
    assert all(torch.equal(p, original) and p._version == version for p, original, version in fixed)
    with pytest.raises(ValueError, match='bounded'):
        stage.update()
    for corrupt in ('draw', 'unused', 'adam'):
        bad = copy.deepcopy(state)
        if corrupt == 'draw': bad['core']['metrics'][0]['source_positions'].reverse()
        elif corrupt == 'unused': bad['core']['model']['extensions']['critic.weight'].add_(1)
        else: next(iter(bad['core']['optimizer']['state'].values()))['exp_avg'].fill_(float('nan'))
        before = model.extension_state(); rng = torch.get_rng_state().clone()
        with pytest.raises(ValueError):
            stage.resume(bad)
        assert_nested_equal(before, model.extension_state()); assert torch.equal(rng, torch.get_rng_state())
