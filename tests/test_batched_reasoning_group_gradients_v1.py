import copy
import json

import pytest
import torch

from slm_lab.batched_reasoning_group_gradients_v1 import group_gradients
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference
from slm_lab.own_dense_actor_checkpoint_v1 import make_state, restore_state
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_public_reasoning_group_gradients_v1 import data, model, optimizer, equal


def setup(data, only_EOS=False):
    position = next(p for p in data.slots if data.row(p)['response'] == 'True')
    ids, labels = data.sequence(position)
    start = next(i for i, t in enumerate(labels) if t != -100)
    student = model((ids[start-1], ids[start], only_EOS))
    reference = FrozenOwnActorReference(student, sources=dict(start_sha256='a' * 64))
    return student, reference, position


def gradients(student, reference, data, position, profile, seeds=None):
    student.zero_grad(set_to_none=True)
    return group_gradients(student, data, position, reference, profile=profile, seeds=seeds or list(range(8)),
        purpose='CPU-mechanics', maximum_cache_bytes=8 * 1024**2, CPU_mechanics_horizon=2)


@pytest.mark.parametrize('only_EOS', [False, True])
def test_actual_batched_public_verifier_rewards_change_RL_gradients_and_zero_groups_keep_CE(data, only_EOS, tmp_path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    student, reference, position = setup(data, only_EOS)
    fixed = tensor_digest(student.base.named_parameters())
    critic = student.parts['critic'].weight.detach().clone()
    rng = capture_rng('cpu')
    control = gradients(student, reference, data, position, 'anchor_CE')
    control_gradients = [p.grad.clone() for _, p in student.named_selected(('actor',))]
    rl = gradients(student, reference, data, position, 'anchor_RL')
    actual_gradients = [p.grad.clone() for _, p in student.named_selected(('actor',))]
    assert control['trajectories'] == rl['trajectories']
    assert control['unconditional_original_answer_CE_anchor'] == rl['unconditional_original_answer_CE_anchor']
    if only_EOS:
        assert rl['actual_strict_rewards'] == [0.] * 8
        assert rl['actual_group_relative_advantages'] == [0.] * 8
        assert all(torch.equal(a, b) for a, b in zip(actual_gradients, control_gradients))
    else:
        assert set(rl['actual_strict_rewards']) == {0., 1.}
        assert any(not torch.equal(a, b) for a, b in zip(actual_gradients, control_gradients))
    assert equal(capture_rng('cpu'), rng)
    for row in rl['trajectories']:
        assert data.score(position, row['actual_generated_token_ids'], horizon=2) == row['actual_strict_source_verifier']
    adam = optimizer(student)
    old = tensor_digest(student.parts.named_parameters())
    adam.step()
    adam.zero_grad(set_to_none=True)
    assert tensor_digest(student.parts.named_parameters()) != old
    assert tensor_digest(student.base.named_parameters()) == fixed
    assert torch.equal(student.parts['critic'].weight, critic)
    reference.assert_bound(student)
    assert not torch.cuda.is_initialized()
    (tmp_path / 'actual-batched-public-RL-mechanics.json').write_text(json.dumps(dict(
        actual_small_student_values=student.counts()['actual_unique'], deliberately_initialized_fixture=True,
        actual_public_train_source_id=data.row(position)['id'], only_EOS_fixture=only_EOS,
        actual_rewards=rl['actual_strict_rewards'], actual_advantages=rl['actual_group_relative_advantages'],
        actual_batched_group_size=8, matched_control_and_RL_samples_and_CE_identical=True,
        actual_actor_values_changed_after_one_test_Adam_step=True, original_fixed_base_and_critic_preserved=True,
        actual_global_RNG_unchanged=True, full_size_GPUcapacity_and_model_quality_pending=True), indent=2))


def test_actual_next_batched_policy_update_replays_weights_Adam_all_observations_and_private_draws(data, tmp_path):
    student, reference, position = setup(data)
    adam = optimizer(student)
    sources = dict(experiment_sha256='b' * 64)
    def update(seeds):
        row = gradients(student, reference, data, position, 'anchor_RL', seeds=seeds)
        torch.nn.utils.clip_grad_norm_([p for _, p in student.named_selected(('actor',))], 1., error_if_nonfinite=True)
        adam.step()
        adam.zero_grad(set_to_none=True)
        return row
    first = update(list(range(8)))
    cursor = dict(complete_source_position=position, completed=1, next_private_group_seeds=list(range(8, 16)))
    saved = copy.deepcopy(make_state(student, adam, groups=('actor',), sources=sources, completed=1,
        planned=2, cursor=cursor, metrics=[first], device='cpu'))
    expected = update(list(range(8, 16)))
    weights, moments, rng = tensor_digest(student.parts.named_parameters()), copy.deepcopy(adam.state_dict()), capture_rng('cpu')
    restore_state(student, adam, saved, groups=('actor',), expected_sources=sources, planned=2, expected_cursor=cursor, device='cpu')
    actual = update(list(range(8, 16)))
    assert actual == expected and tensor_digest(student.parts.named_parameters()) == weights
    assert equal(adam.state_dict(), moments) and equal(capture_rng('cpu'), rng)
    (tmp_path / 'actual-next-batched-RL-update-recovery.json').write_text(json.dumps(dict(
        actual_small_student_values=student.counts()['actual_unique'], actual_test_Adam_updates=2,
        exact_next_actions_verifiers_probabilities_gradients_weights_Adam_and_private_RNG=True,
        production_optimizer_updates=0, full_size_GPUcapacity_and_durable_four_rank_recovery_still_pending=True), indent=2))


def test_invalid_profile_or_horizon_cannot_mutate_parameters_or_leave_partial_gradients(data):
    student, reference, position = setup(data)
    weights, rng = tensor_digest(student.parts.named_parameters()), capture_rng('cpu')
    for options in [dict(profile='unknown', CPU_mechanics_horizon=2),
            dict(profile='anchor_RL', CPU_mechanics_horizon=1000)]:
        with pytest.raises(ValueError):
            group_gradients(student, data, position, reference, seeds=list(range(8)), purpose='CPU-mechanics',
                maximum_cache_bytes=8 * 1024**2, **options)
        assert tensor_digest(student.parts.named_parameters()) == weights and equal(capture_rng('cpu'), rng)
        assert all(p.grad is None for p in student.parameters())
