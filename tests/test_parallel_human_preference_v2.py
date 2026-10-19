import copy

import pytest
import torch

from slm_lab.dense_human_preference_v1 import preference_gradients
from slm_lab.dense_human_preference_stage_v1 import chosen_gradients
from slm_lab.parallel_human_preference_v2 import ParallelHumanPreferenceStage, local_gradients
from test_dense_human_preference_v1 import pairs
from test_own_dense_actor_v1 import actor


@pytest.mark.parametrize('DPO_weight', [0, 1])
def test_global_unequal_response_masks_and_empty_tail_match_serial_whole_pair_gradient(DPO_weight):
    torch.set_num_threads(1)
    data = pairs()
    original = actor()
    oracle = copy.deepcopy(original)
    setting = dict(DPO_beta=.1, chosen_CE_weight=.2)
    if DPO_weight:
        preference_gradients(oracle, data, [0, 1], beta=.1, chosen_CE_weight=.2)
    else:
        chosen_gradients(oracle, data, [0, 1], setting)
    gradients = []
    records = []
    for rank in range(4):
        model = copy.deepcopy(original)
        record = local_gradients(model, data, [0, 1], rank=rank, world=4,
                                 DPO_weight=DPO_weight, beta=.1, chosen_CE_weight=.2)
        records.extend(record)
        gradients.append([p.grad for _, p in model.named_selected(('actor',))])
        assert all(p.grad is None for p in model.base.parameters())
        assert model.parts['critic'].weight.grad is None
        if rank >= 2:
            assert not record and all(bool((g == 0).all()) for g in gradients[-1])
    assert [r['original_train_position'] for r in records] == [0, 1]
    assert sum(r['chosen_response_EOS_targets'] for r in records) == 5
    for index, (_, parameter) in enumerate(oracle.named_selected(('actor',))):
        assert torch.allclose(sum(row[index] for row in gradients), parameter.grad, atol=3e-7, rtol=3e-5)


def test_machine_labels_duplicate_pairs_and_held_data_reject_before_gradient():
    model = actor()
    data = pairs()
    with pytest.raises(ValueError, match='distinct train-only'):
        local_gradients(model, data, [0, 0])
    data.role = 'valid'
    with pytest.raises(ValueError, match='train-only'):
        local_gradients(model, data, [0])
    data.role = 'train'
    data.rows[0]['feedback_source'] = 'machine'
    with pytest.raises(ValueError, match='human-ranked'):
        local_gradients(model, data, [0])
    assert all(p.grad is None for p in model.parameters())


def test_actual_stage_next_Adam_RNG_and_weights_recover_and_tampered_replica_counts_reject():
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    model = actor()
    data = pairs()
    options = dict(seed=958993, epochs=1, steps=2, microbatch=1, accumulation=1,
                   learning_rate=.001, betas=[.9, .95], weight_decay=.01,
                   gradient_clip=1., DPO_beta=.1, DPO_weight=1, chosen_CE_weight=.1)
    optimizer = torch.optim.AdamW([p for _, p in model.named_selected(('actor',))],
        lr=.001, betas=(.9, .95), weight_decay=.01, foreach=False)
    stage = ParallelHumanPreferenceStage(model, optimizer, data, options, {'test_fixture': 'a'*64})
    stage.update()
    saved = stage.state()
    expected = stage.update()
    weights = tensor_digest(model.parts.named_parameters())
    moments = copy.deepcopy(optimizer.state_dict())
    RNG = torch.get_rng_state()
    bad = copy.deepcopy(saved)
    bad['core']['core']['metrics'][0]['per_replica_original_pairs'] = [0]
    with pytest.raises(ValueError, match='replica count'):
        stage.resume(bad)
    assert weights == tensor_digest(model.parts.named_parameters())
    assert torch.equal(RNG, torch.get_rng_state())
    stage.resume(saved)
    assert stage.update() == expected
    assert weights == tensor_digest(model.parts.named_parameters())
    assert torch.equal(RNG, torch.get_rng_state())
    for key, state in optimizer.state_dict()['state'].items():
        assert all(torch.equal(value, moments['state'][key][name]) for name, value in state.items())
