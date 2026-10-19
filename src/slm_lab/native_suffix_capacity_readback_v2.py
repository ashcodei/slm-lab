"""Fresh indexed capacity-state readback with no optimizer calls or new actions."""
import copy
from pathlib import Path
from .corpus import file_hash

from .native_suffix_capacity_controller_v2 import _cold, history_inventory
from .native_suffix_capacity_evidence_v2 import require_capacity
from .codenet_checkpoint_v1 import capture_rng
from .parallel_function_repair_stage_v1 import value_digest
from .parallel_repair_durable_state_v1 import read_latest, restore_latest
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-native-suffix-indexed-fresh-complete-capacity-state-readback-v2'

def capacity_reports(stage, reports, horizon_reports, plan, allow_mechanics):
    return require_capacity(stage, reports, horizon_reports, plan, allow_mechanics=allow_mechanics)


def read_capacity(stage, directory, *, reports, horizon_reports, capacity_plan, independent_actor, cold_inputs, allow_mechanics=False):
    """A newly constructed stage reads completed capacity without any Adam call."""
    error = None
    try:
        capacity_reports(stage, reports, horizon_reports, capacity_plan, allow_mechanics)
        if (stage.sources.get(str(Path(__file__).resolve())) != file_hash(Path(__file__).resolve())
                or stage.completed or stage.failed or stage.optimizer.state or stage.mode not in ['capacity', 'mechanics']
                or stage.entries != capacity_plan['capacity_entries']):
            raise ValueError('Fresh capacity replicas and their exact whole source plan are required')
        stage.intact()
        common = dict(reports=value_digest(reports), horizon_reports=value_digest(horizon_reports), plan=value_digest(capacity_plan), directory=str(Path(directory).resolve()))
    except Exception as exc: error = repr(exc)
    stage.coordinated(error, 'fresh capacity readback preflight')
    if any(v != common for v in stage.gather(common)):
        raise ValueError('Every fresh replica must independently read the same original capacity proof/state')
    pointer = reports[0]['final_actual_pointer_sha256']
    saved = receipt = actual_pointer = None
    error = None
    try:
        saved, receipt, actual_pointer = read_latest(directory, expected_pointer_sha256=pointer)
        if receipt['checkpoint_sha256'] != reports[0]['final_actual_shared_checkpoint_sha256']:
            raise ValueError('The actual original capacity tensor file must match all retained reports')
    except Exception as exc: error = repr(exc)
    stage.coordinated(error, 'fresh capacity bytes before mutation')
    restore_latest(stage, directory, expected_pointer_sha256=pointer, allow_completed=True)
    error = None
    actor = tensor_digest(stage.model.parts.named_parameters())
    adam = value_digest(stage.optimizer.state_dict())
    rng = value_digest(capture_rng(stage.device))
    try:
        expected = reports[stage.rank]['actual_per_rank_intervals'][stage.rank]
        if (stage.completed != len(capacity_plan['capacity_entries']) or stage.data.budget(stage.entries) != reports[stage.rank]['actual_source_budget']
                or actor != expected['actor_sha256'] or adam != expected['Adam_sha256'] or rng != expected['private_RNG_sha256']):
            raise ValueError('Every fresh actor, Adam clock/value, private generator and original capacity cursor must match')
        stage.intact()
    except Exception as exc: error = repr(exc)
    stage.coordinated(error, 'fresh capacity all actual state values')
    error = None
    try: cold = _cold(stage, independent_actor, saved, cold_inputs)
    except Exception as exc: error = repr(exc)
    stage.coordinated(error, 'fresh indexed complete actor cold outputs')
    error = None
    try: inventory = history_inventory(stage)
    except Exception as exc: error = repr(exc)
    stage.coordinated(error, 'fresh all committed prefix and independent failure allocations')
    return dict(format=FORMAT, complete=True, rank=stage.rank, world=4, mode=stage.mode, profile=stage.profile,
        actual_research_actor_parameters=stage.model.counts()['actual_unique'], actual_native_parameters=stage.model.base.counts()['total_unique'],
        original_source_bindings=copy.deepcopy(stage.sources), original_objective_options=copy.deepcopy(stage.options),
        original_own_reference_binding=copy.deepcopy(stage.reference_binding), capacity_plan_sha256=value_digest(capacity_plan),
        actual_capacity_reports_sha256=value_digest(reports), actual_horizon_reports_sha256=value_digest(horizon_reports), actual_pointer_sha256=actual_pointer,
        actual_checkpoint_sha256=receipt['checkpoint_sha256'], actual_complete_capacity_updates=stage.completed,
        actual_actor_sha256=actor, actual_Adam_sha256=adam, actual_private_RNG_sha256=rng,
        actual_fixed_base_sha256=stage.base_digest, actual_Adam_states=len(stage.optimizer.state),
        actual_durable_indexed_history_inventory=inventory,
        independent_complete_actor_cold=cold, new_optimizer_calls=0, new_generated_actions=0,
        fresh_stage_was_empty_before_readback=True, every_complete_original_capacity_record_revalidated=True,
        production_training_admitted=False, goal_achieved=False)


def check_readbacks(stage, reports, horizon_reports, readbacks, plan, allow_mechanics):
    capacity_reports(stage, reports, horizon_reports, plan, allow_mechanics)
    if not isinstance(readbacks, list) or len(readbacks) != 4:
        raise ValueError('All four fresh actual capacity readbacks are required')
    for rank, proof in enumerate(readbacks):
        expected = reports[rank]['actual_per_rank_intervals'][rank]
        if (proof.get('format') != FORMAT or proof.get('complete') is not True or proof.get('rank') != rank
                or proof.get('world') != 4 or proof.get('mode') != reports[rank]['mode'] or proof.get('profile') != stage.profile
                or proof.get('actual_research_actor_parameters') != stage.model.counts()['actual_unique']
                or proof.get('actual_native_parameters') != stage.model.base.counts()['total_unique']
                or proof.get('original_source_bindings') != stage.sources or proof.get('original_objective_options') != stage.options
                or proof.get('original_own_reference_binding') != stage.reference_binding
                or proof.get('capacity_plan_sha256') != value_digest(plan) or proof.get('actual_capacity_reports_sha256') != value_digest(reports)
                or proof.get('actual_horizon_reports_sha256') != value_digest(horizon_reports)
                or proof.get('actual_pointer_sha256') != reports[rank]['final_actual_pointer_sha256']
                or proof.get('actual_checkpoint_sha256') != reports[rank]['final_actual_shared_checkpoint_sha256']
                or proof.get('actual_fixed_base_sha256') != stage.base_digest
                or proof.get('actual_actor_sha256') != expected['actor_sha256'] or proof.get('actual_Adam_sha256') != expected['Adam_sha256']
                or proof.get('actual_private_RNG_sha256') != expected['private_RNG_sha256']
                or proof.get('actual_complete_capacity_updates') != len(plan['capacity_entries'])
                or proof.get('actual_durable_indexed_history_inventory') != reports[rank].get('actual_durable_indexed_history_inventory')
                or proof.get('new_optimizer_calls') != 0 or proof.get('new_generated_actions') != 0
                or proof.get('fresh_stage_was_empty_before_readback') is not True
                or proof.get('every_complete_original_capacity_record_revalidated') is not True):
            raise ValueError('Every exact fresh capacity source/state/private RNG readback is required before training')
        cold = proof.get('independent_complete_actor_cold', {})
        if (cold.get('complete') is not True or not all(cold.get(k) is True for k in
                ['every_base_head_actor_critic_parameter_equal', 'no_shared_parameter_storage',
                 'complete_text_photo_word_outputs_bit_exact', 'all_private_RNGs_unchanged'])
                or stage.mode == 'train' and (cold.get('actual_text_positions') != 4096
                    or cold.get('actual_photo_side') != 448 or cold.get('actual_word_shape') != [64, 448])):
            raise ValueError('Every fresh full native text/photo/word capacity cold check must pass')

