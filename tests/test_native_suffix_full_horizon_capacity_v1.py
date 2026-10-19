"""Full-batch fixed native histories allocate; no synthetic action earns reward."""
import copy

import pytest
import torch

from slm_lab.native_suffix_full_horizon_capacity_v1 import stress
from slm_lab.native_suffix_train_data_v1 import NativeSuffixTrainData
from slm_lab.native_suffix_validation_stage_v2 import NativeSuffixValidationStageV2
from slm_lab.native_suffix_capacity_controller_v1 import live
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from test_parallel_native_suffix_stage_v1 import ROOT, CONTROL, OPTIONS, SOURCES, student
from test_public_reasoning_group_gradients_v1 import optimizer


def setup(**changes):
    public = NativeSuffixTrainData.CPU_mechanics(ROOT, CONTROL)
    own = student(public)
    return NativeSuffixValidationStageV2(own, optimizer(own), public, [[0], [1]], 'anchor_execution',
        dict(OPTIONS, **changes), SOURCES, mode='mechanics')


def test_full_eight_history_native_cache_and_original_CE_reference_gradients_preserve_existing_Adam_and_cursor(tmp_path):
    torch.set_num_threads(1)
    learner = setup()
    learner.update()
    assert len(learner.optimizer.state) == 17
    before = live(learner)
    weights = copy.deepcopy(learner.model.state_dict())
    flags = [(id(m), m.training) for m in learner.model.modules()]
    checkpointing = learner.model.policy.gradient_checkpointing
    row = stress(learner, 0, allow_mechanics=True)
    assert row['actual_native_group_size'] == 8 and row['actual_fixed_non_EOS_actions_per_history'] == 5
    assert row['full_original_action_horizon'] == 512 and row['actual_complete_original_CE_targets'] == 68
    assert row['actual_prefill_batch_sizes'] == [8] and row['actual_decode_batch_sizes'] == [8] * 4
    assert row['actual_peak_logical_KV_bytes'] == row['exact_full_group_horizon_logical_KV_bytes'] > 0
    assert len(row['actual_fixed_actions']) == 8 and all(len(a) == 5 and all(t >= 4 for t in a) for a in row['actual_fixed_actions'])
    assert row['actual_finite_actor_gradient_tensors'] == 17 and row['no_optimizer_step']
    assert row['fixed_actions_not_sampled_trajectories'] and row['no_execution_reward_claimed']
    assert row['source_cursor_Adam_weights_and_all_private_RNG_unchanged']
    assert not row['full_size_GPU_horizon_allocation_passed'] and not row['production_training_admitted']
    assert live(learner) == before
    assert all(torch.equal(t, learner.model.state_dict()[name]) for name, t in weights.items())
    assert flags == [(id(m), m.training) for m in learner.model.modules()]
    assert learner.model.policy.gradient_checkpointing == checkpointing
    assert all(p.grad is None for p in learner.model.parameters()) and not torch.cuda.is_initialized()
    import json
    (tmp_path / 'actual-fixed-native-full-group-allocation.json').write_text(json.dumps(row, indent=2, allow_nan=False))


def test_declared_cache_overflow_rejects_without_shrinking_or_leaving_gradients():
    learner = setup(maximum_cache_bytes=1)
    before = live(learner)
    with pytest.raises(ValueError, match='without shortening'): stress(learner, 0, allow_mechanics=True)
    assert live(learner) == before and not learner.optimizer.state
    assert all(p.grad is None for p in learner.model.parameters())


def test_CPU_fixture_or_unselected_source_cannot_gain_full_native_capacity():
    learner = setup()
    before = live(learner)
    with pytest.raises(ValueError): stress(learner, 0)
    with pytest.raises(ValueError): stress(learner, 8, allow_mechanics=True)
    with pytest.raises(ValueError): stress(learner, True, allow_mechanics=True)
    assert live(learner) == before and not learner.optimizer.state
