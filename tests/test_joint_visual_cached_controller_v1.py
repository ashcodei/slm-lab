"""Actual connected small controllers plus explicitly synthetic full-scope controls."""
from datetime import timedelta
import json
import os
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.joint_visual_cached_controller_v1 import run_capacity, require_full_capacity, FORMAT
from slm_lab.joint_visual_private_checkpoint_v2 import digest
from test_joint_visual_controller_v1 import nine_stage, independent, reports as original_reports
from test_joint_visual_cold_overlay_v1 import inputs


def test_actual_connected_all9family_cached_capacity_keeps_original_controller_and_native_cold(tmp_path):
    stage, batch, plan = nine_stage(bf16=True)
    result = run_capacity(stage, tmp_path / 'private', capacity_plan=plan, independent_native=independent,
        cold_inputs=inputs(batch), maximum_bytes=8 * 1024**2, minimum_free_bytes=0, allow_mechanics=True)
    assert result['actual_total_capacity_physical_optimizer_calls'] == 17
    assert result['actual_original_controller_physical_optimizer_calls'] == 11
    assert result['actual_separate_equivalence_physical_optimizer_calls'] == 6
    assert result['original_complete_capacity']['actual_complete_capacity_entries'] == 10
    assert result['original_complete_capacity']['actual_capacity_source_presentations'] == 13
    assert result['original_complete_capacity']['independent_native_cold']['actual_text_photo_logits_and_word_readout_bit_exact']
    assert not result['actual_cast_prefix_equivalence']['full_size_GPU_equivalence']
    assert not result['original_complete_capacity']['full_size_GPU_capacity_passed']
    assert all(getattr(m.forward, '__func__', None) is type(m).forward for m in stage.view.modules()
        if type(m).__name__ in ['_FrozenLinear', '_FrozenHead'])
    assert not torch.cuda.is_initialized()
    (tmp_path / 'actual-cached-controller.json').write_text(json.dumps(result, indent=2))


def reports():
    # Schema controls only; these declared counts never represent loaded models.
    result = []
    for rank, original in enumerate(original_reports()):
        inventory = dict(purpose='GPU-runtime', device='cuda:' + str(rank),
            cached_original_parameter_tensors=246, actual_conversions=246, patched_detach_only_wrappers=492,
            added_learned_parameters=0, cache_excluded_from_state_dict_and_export=True,
            additional_derived_values=900000000, additional_derived_bytes=1800000000)
        proof = dict(complete=True, passed=True, rank=rank, world=4, mode='capacity', actual_device_type='cuda',
            actual_device_index=rank, actual_research_parameters=999706442, actual_native_parameters=984399690,
            actual_source_prefix_sha256='e' * 64, actual_complete_prefix_presentations=192,
            actual_separate_physical_capacity_updates=6, actual_logical_completed_after_restoration=0,
            all_three_objectives_and_Adam_input_gradients_and_weights_Adam_metrics_private_RNG_exact=True,
            exact_durable_initializer_restored_before_full_capacity=True, full_size_GPU_equivalence=True,
            new_production_optimizer_updates=0, cache_inventory=inventory,
            actual_Adam_input_gradient_observations=[dict(actual_selected_leaves=130, actual_active_gradient_leaves=n,
                actual_Adam_input_gradient_sha256='f' * 64) for n in [112, 114, 128]])
        result.append(dict(format=FORMAT, complete=True, passed=True, original_complete_capacity=original,
            actual_cast_prefix_equivalence=proof, final_cache_inventory=dict(inventory),
            actual_original_controller_physical_optimizer_calls=11, actual_separate_equivalence_physical_optimizer_calls=6,
            actual_total_capacity_physical_optimizer_calls=17, additional_learned_values=0, production_updates=0))
    return result


def admission(rows):
    return require_full_capacity(rows, expected_bindings=dict(source='a' * 64),
        expected_capacity_plan_sha256='b' * 64, expected_prefix_sha256='e' * 64)


def test_both_complete_full_size_gates_are_required_without_a_GPU_speed_or_quality_claim():
    result = admission(reports())
    assert result['actual_all_four_full_size_cast_equivalence_gates_passed']
    assert result['actual_separate_physical_optimizer_calls_per_GPU'] == 17
    assert result['actual_GPU_speedup_not_established'] and not result['goal_achieved']


@pytest.mark.parametrize('case', ['CPU_equivalence', 'short_prefix', 'changed_source', 'missing_word_gradient',
    'hidden_extra_weights', 'partial_original_capacity', 'hidden_physical_updates'])
def test_incomplete_cast_or_original_capacity_cannot_unlock_full_training(case):
    rows = reports()
    if case == 'CPU_equivalence':
        rows[1]['actual_cast_prefix_equivalence']['actual_device_type'] = 'cpu'
    elif case == 'short_prefix':
        rows[1]['actual_cast_prefix_equivalence']['actual_complete_prefix_presentations'] = 3
    elif case == 'changed_source':
        rows[1]['actual_cast_prefix_equivalence']['actual_source_prefix_sha256'] = 'a' * 64
    elif case == 'missing_word_gradient':
        rows[1]['actual_cast_prefix_equivalence']['actual_Adam_input_gradient_observations'][-1]['actual_active_gradient_leaves'] = 112
    elif case == 'hidden_extra_weights':
        rows[1]['final_cache_inventory']['added_learned_parameters'] = 1
    elif case == 'partial_original_capacity':
        rows[1]['original_complete_capacity']['actual_capacity_source_presentations'] = 13
    else:
        rows[1]['actual_total_capacity_physical_optimizer_calls'] = 11
    with pytest.raises(ValueError):
        admission(rows)


def worker(rank, folder):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.nice(0) >= 19
    dist.init_process_group('gloo', init_method='file://' + str(Path(folder) / 'rendezvous'), rank=rank,
        world_size=4, timeout=timedelta(minutes=5))
    try:
        stage, batch, plan = nine_stage(rank=rank, world=4, bf16=True)
        # Independent actual private seeds, without changing any other worker.
        __import__('random').seed(15000 + rank)
        __import__('numpy').random.seed(16000 + rank)
        torch.manual_seed(17000 + rank)
        result = run_capacity(stage, Path(folder) / 'private', capacity_plan=plan, independent_native=independent,
            cold_inputs=inputs(batch), maximum_bytes=8 * 1024**2, minimum_free_bytes=0, allow_mechanics=True)
        assert result['actual_total_capacity_physical_optimizer_calls'] == 17
        assert result['original_complete_capacity']['independent_native_cold']['actual_text_photo_logits_and_word_readout_bit_exact']
        assert result['actual_cast_prefix_equivalence']['all_three_objectives_and_Adam_input_gradients_and_weights_Adam_metrics_private_RNG_exact']
        assert not result['original_complete_capacity']['full_size_GPU_capacity_passed']
        assert not torch.cuda.is_initialized()
        (Path(folder) / (str(rank) + '.json')).write_text(json.dumps(result, indent=2))
    finally:
        dist.destroy_process_group()


def test_four_actual_CPUreplicas_complete_both_cache_prefix_and_all_original_objectives_and_cold(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    rows = [json.loads((tmp_path / (str(rank) + '.json')).read_text()) for rank in range(4)]
    assert all(r['passed'] and r['actual_total_capacity_physical_optimizer_calls'] == 17 for r in rows)
    assert len({r['original_complete_capacity']['final_actual_shared_checkpoint_sha256'] for r in rows}) == 1
