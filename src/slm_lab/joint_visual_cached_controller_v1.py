"""Require both actual cast equivalence and the intact original full capacity."""
import copy

from .frozen_linear_cast_cache_v1 import FrozenLinearCastCache
from .joint_visual_cast_equivalence_v1 import prove_prefix
from .joint_visual_controller_v1 import run_capacity as original_capacity, require_full_capacity as original_require, run_training as original_training
from .joint_visual_private_checkpoint_v2 import _agree

FORMAT = 'own-visual-derived-cast-plus-original-full-capacity-v1'


def run_capacity(stage, directory, *, capacity_plan, independent_native, cold_inputs,
                 maximum_bytes, minimum_free_bytes, allow_mechanics=False):
    cache, equivalence = prove_prefix(stage, directory, maximum_bytes=maximum_bytes,
        minimum_free_bytes=minimum_free_bytes, allow_mechanics=allow_mechanics)
    try:
        original = original_capacity(stage, directory, capacity_plan=capacity_plan,
            independent_native=independent_native, cold_inputs=cold_inputs,
            maximum_bytes=maximum_bytes, minimum_free_bytes=minimum_free_bytes, allow_mechanics=allow_mechanics)
        cache.assert_current()
        return dict(format=FORMAT, complete=True, passed=True, original_complete_capacity=original,
            actual_cast_prefix_equivalence=equivalence, final_cache_inventory=cache.inventory(),
            actual_original_controller_physical_optimizer_calls=original['actual_physical_optimizer_calls'],
            actual_separate_equivalence_physical_optimizer_calls=6,
            actual_total_capacity_physical_optimizer_calls=original['actual_physical_optimizer_calls'] + 6,
            additional_learned_values=0, production_updates=0, candidate_promoted=False, goal_achieved=False)
    finally:
        cache.close()


def require_full_capacity(proofs, *, expected_bindings, expected_capacity_plan_sha256, expected_prefix_sha256):
    if not isinstance(proofs, list) or len(proofs) != 4:
        raise ValueError('All four actual cached and original full-size GPUcapacity reports are required')
    originals = []
    for rank, report in enumerate(proofs):
        if (report.get('format') != FORMAT or report.get('complete') is not True or report.get('passed') is not True
                or report.get('actual_original_controller_physical_optimizer_calls') != 11
                or report.get('actual_separate_equivalence_physical_optimizer_calls') != 6
                or report.get('actual_total_capacity_physical_optimizer_calls') != 17
                or report.get('additional_learned_values') != 0 or report.get('production_updates') != 0):
            raise ValueError('Preserve the full original11capacity calls and all6additional exact-equivalence calls')
        proof = report['actual_cast_prefix_equivalence']
        if (proof.get('complete') is not True or proof.get('passed') is not True
                or proof.get('rank') != rank or proof.get('world') != 4 or proof.get('mode') != 'capacity'
                or proof.get('actual_device_type') != 'cuda' or proof.get('actual_device_index') != rank
                or proof.get('actual_research_parameters') != 999706442 or proof.get('actual_native_parameters') != 984399690
                or proof.get('actual_source_prefix_sha256') != expected_prefix_sha256
                or proof.get('actual_complete_prefix_presentations') != 192
                or proof.get('actual_separate_physical_capacity_updates') != 6
                or proof.get('actual_logical_completed_after_restoration') != 0
                or proof.get('all_three_objectives_and_Adam_input_gradients_and_weights_Adam_metrics_private_RNG_exact') is not True
                or proof.get('exact_durable_initializer_restored_before_full_capacity') is not True
                or proof.get('full_size_GPU_equivalence') is not True or proof.get('new_production_optimizer_updates') != 0):
            raise ValueError('Actual full-size three-objective GPUgradient/Adam/RNG equivalence must pass on every rank')
        gradients = proof['actual_Adam_input_gradient_observations']
        if (len(gradients) != 3 or [r['actual_active_gradient_leaves'] for r in gradients] != [112, 114, 128]
                or any(r['actual_selected_leaves'] != 130 or len(r['actual_Adam_input_gradient_sha256']) != 64 for r in gradients)):
            raise ValueError('Every actual native/photo/word Adam-input gradient tensor must be checked')
        for inventory in [proof['cache_inventory'], report['final_cache_inventory']]:
            if (inventory.get('purpose') != 'GPU-runtime' or inventory.get('device') != 'cuda:' + str(rank)
                    or inventory.get('cached_original_parameter_tensors') != 246
                    or inventory.get('actual_conversions') != 246
                    or inventory.get('patched_detach_only_wrappers') != 492
                    or inventory.get('added_learned_parameters') != 0
                    or inventory.get('cache_excluded_from_state_dict_and_export') is not True
                    or inventory.get('additional_derived_bytes') != 2 * inventory.get('additional_derived_values', 0)
                    or not 0 < inventory.get('additional_derived_values', 0) < 984399690):
                raise ValueError('Count all actual derived CUDAstorage and preserve every registered/exported learned value')
        originals.append(report['original_complete_capacity'])
    original = original_require(originals, expected_bindings=expected_bindings,
        expected_capacity_plan_sha256=expected_capacity_plan_sha256)
    return dict(original, actual_all_four_full_size_cast_equivalence_gates_passed=True,
        actual_separate_physical_optimizer_calls_per_GPU=17,
        expected_prefix_sha256=expected_prefix_sha256, actual_GPU_speedup_not_established=True)


def run_training(stage, directory, journal, *, capacity_proofs, capacity_plan_sha256,
                 expected_capacity_prefix_sha256, **kwargs):
    bound = {k: v for k, v in stage._sources.items() if k != 'run_plan_sha256'}
    require_full_capacity(capacity_proofs, expected_bindings=bound,
        expected_capacity_plan_sha256=capacity_plan_sha256, expected_prefix_sha256=expected_capacity_prefix_sha256)
    error = None
    cache = None
    try:
        try:
            cache = FrozenLinearCastCache(device=stage.device).attach(stage.view)
        except Exception as exc:
            error = repr(exc)
        _agree(error, None, stage.group, stage.world)
        original_preflight = kwargs['preflight']
        def preflight():
            cache.assert_current()
            original_preflight()
        result = original_training(stage, directory, journal,
            capacity_proofs=[p['original_complete_capacity'] for p in capacity_proofs],
            capacity_plan_sha256=capacity_plan_sha256, **dict(kwargs, preflight=preflight))
        cache.assert_current()
        return dict(result, actual_final_frozen_cast_inventory=copy.deepcopy(cache.inventory()),
            all_actual_uncached_cached_full_size_equivalence_gates_passed_before_training=True)
    finally:
        if cache is not None:
            cache.close()
