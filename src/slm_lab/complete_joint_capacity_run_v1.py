"""Actual joint capacity updates, all-rank private state and exact next replay.

Execute every selected intact original source batch. A CPU mechanics result
can never satisfy the full-size GPU check. Native cold export, complete
training, ownership/funding and all final evaluations remain separate gates.
"""
import copy
from pathlib import Path

from .codenet_checkpoint_v1 import capture_rng
from .complete_joint_capacity_stage_v1 import CompleteJointCapacityStageV1
from .joint_indexed_checkpoint_v3 import restore_state
from .joint_indexed_durable_state_v3 import read_latest
from .parallel_function_repair_stage_v1 import value_digest
from .shared_joint_checkpoint_v2 import agree
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-complete-original-joint-capacity-private-replay-result-v1'


def live(stage):
    return dict(selected_values_sha256=tensor_digest(stage.view.named_selected()),
        Adam_sha256=value_digest(stage.optimizer.state_dict()), private_RNG_sha256=value_digest(capture_rng(stage.device)),
        completed=stage.completed, original_receipts_sha256=value_digest(stage.receipts))


def run(stage, directory, *, maximum_bytes, minimum_free_bytes, preflight, progress, allow_mechanics=False):
    error = identity = None
    try:
        counts = stage.view.actor.counts()
        full = stage.device.type == 'cuda' and counts['actual_unique'] == 999706442 and counts['frozen_dense_and_all_heads'] == 984399690
        if (type(stage) is not CompleteJointCapacityStageV1 or stage.completed or stage.failed or stage.world != 4
                or len(stage.entries) < 3 or not callable(preflight) or not callable(progress)
                or not full and not (allow_mechanics is True and stage.device.type == 'cpu' and counts['actual_unique'] < 10**7)
                or minimum_free_bytes < 40 * 1024**3):
            raise ValueError('All actual original joint capacity cases and four full-size GPUs or an explicitly small CPU control are required')
        preflight()
        identity = value_digest(dict(plan=stage.capacity_plan_sha256, sources=stage.sources,
            directory=str(Path(directory).absolute()), maximum_bytes=maximum_bytes, minimum_free_bytes=minimum_free_bytes,
            actual_full_size_GPU=full))
    except Exception as exc:
        error = repr(exc)
    agree(error, identity, stage.control_group, stage.world)
    stage.open_history()
    first = stage.train_next()
    first_receipt = stage.commit(directory, maximum_bytes=maximum_bytes, minimum_free_bytes=minimum_free_bytes)
    error = None
    try:
        preflight()
    except Exception as exc:
        error = repr(exc)
    agree(error, None, stage.control_group, stage.world)
    second = stage.train_next(); expected = live(stage)
    error = identity = shared = committed = pointer_sha256 = None
    try:
        shared, committed, pointer_sha256 = read_latest(directory)
        if first_receipt != committed or len({r['private_RNG_sha256'] for r in shared['private_states']}) != stage.world:
            raise ValueError('Every actual rank must retain its original independently seeded private generators')
        identity = value_digest(dict(receipt=committed, pointer_sha256=pointer_sha256))
    except Exception as exc:
        error = repr(exc)
    agree(error, identity, stage.control_group, stage.world)
    # A foreign rank's otherwise valid RNG must fail the externally committed
    # local identity before changing any learned value, Adam or generator.
    foreign = shared['private_states'][(stage.rank + 1) % stage.world]
    bad = dict(shared['common'], **shared['private_states'][stage.rank])
    bad['private_RNG'] = copy.deepcopy(foreign['private_RNG'])
    bad['private_RNG_sha256'] = foreign['private_RNG_sha256']
    rejected = False
    try:
        restore_state(stage.view, stage.optimizer, stage.guard.fixed, bad, sources=stage.sources, entries=stage.entries,
            history_configuration=stage.history, source_files=stage.source_files, resource_guard=stage.resource_guard,
            expected_common_sha256=committed['semantic_manifest']['actual_common_sha256'],
            expected_private_RNG_sha256=committed['semantic_manifest']['all_actual_private_RNG_sha256'][stage.rank],
            expected_completed=1, rank=stage.rank, world=stage.world, device=stage.device, control_group=stage.control_group)
    except ValueError:
        rejected = True
    error = None
    try:
        if not rejected or live(stage) != expected:
            raise ValueError('A foreign private generator must be rejected before any actual joint state mutation')
    except Exception as exc:
        error = repr(exc)
    agree(error, None, stage.control_group, stage.world)
    del bad, shared
    stage.restore(directory, expected_pointer_sha256=pointer_sha256)
    stage.open_history(); replayed = stage.train_next()
    error = None
    try:
        actual = live(stage)
        # The failed/uncommitted first attempt is preserved, so receipt
        # addresses differ. Neural state, RNG and complete raw metric must not.
        if (replayed != second or any(actual[k] != expected[k] for k in
                ['selected_values_sha256', 'Adam_sha256', 'private_RNG_sha256', 'completed'])):
            raise ValueError('The actual original next full joint batch,selected values,Adam,private generators and metric must replay exactly')
    except Exception as exc:
        error = repr(exc)
    agree(error, None, stage.control_group, stage.world)
    while stage.completed < len(stage.entries):
        error = None
        try:
            preflight()
        except Exception as exc:
            error = repr(exc)
        agree(error, None, stage.control_group, stage.world)
        metric = stage.train_next()
        error = None
        try:
            progress(stage.completed, metric)
        except Exception as exc:
            error = repr(exc)
        agree(error, None, stage.control_group, stage.world)
    final_receipt = stage.commit(directory, maximum_bytes=maximum_bytes, minimum_free_bytes=minimum_free_bytes)
    expected_final = live(stage)
    stage.restore(directory, allow_completed=True)
    error = None
    try:
        if live(stage) != expected_final:
            raise ValueError('The complete actual selected capacity state and every original source receipt must reload exactly')
        stage.reference.assert_bound(full=True); stage.guard.fixed.check(stage.view, stage.optimizer, full=True)
        preflight()
    except Exception as exc:
        error = repr(exc)
    agree(error, None, stage.control_group, stage.world)
    return dict(format=FORMAT, complete=True, rank=stage.rank, world=stage.world,
        actual_device_type=stage.device.type, actual_device_index=stage.device.index,
        actual_research_parameters=counts['actual_unique'], actual_native_parameters=counts['frozen_dense_and_all_heads'],
        capacity_plan_sha256=stage.capacity_plan_sha256, original_full_joint_steps=5362,
        source_bindings=copy.deepcopy(stage.sources), actual_complete_capacity_cases=stage.completed,
        actual_physical_AdamW_calls=stage.completed + 1,
        actual_selected_parameter_leaves=len(stage.view.named_selected()), actual_Adam_states=len(stage.optimizer.state),
        actual_final_sparse_parameter_clocks=copy.deepcopy(stage.clocks), final_actual_state=expected_final,
        exact_next_source_values_Adam_private_RNG_and_full_metric_replay=True,
        foreign_actual_rank_private_RNG_rejected_before_any_mutation=True,
        all_original_visual_cases_corrective_batches_and_real_empty_tasks_preserved=True,
        every_actual_complete_capacity_record_read_and_final_state_restored=True,
        first_actual_metric_sha256=value_digest(first), actual_replayed_second_metric_sha256=value_digest(second),
        actual_final_checkpoint_sha256=final_receipt['checkpoint_sha256'], full_size_GPU_capacity_passed=full,
        native_cold_export_and_owned_training_admission_still_required=True,
        production_updates=0, candidate_promoted=False, goal_achieved=False)
