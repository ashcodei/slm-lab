"""Complete indexed native training, durable scalar events and fresh attempts.

The exact selected capacity/full-horizon/fresh cold evidence remains required.
All complete neural records are durable before Adam in the indexed stage;
the bounded event stream preserves controller timing without copying them.
This controller grants no GPU lease, model promotion or quality claim.
"""
import copy
from pathlib import Path
import time

from .bounded_gzip_observation_stream_v1 import Writer, RawBudget, NEXT_RECORD
from .bounded_disk_reserve_wait_v1 import wait_for_reserve
from .codenet_checkpoint_v1 import capture_rng
from .native_allocated_history_guard_v1 import attach as attach_allocated_guard
from .native_history_summary_v1 import summarize
from .native_suffix_batched_history_stage_v5 import NativeSuffixBatchedHistoryStageV5, PROTOCOL as BATCH
from .native_suffix_capacity_controller_v2 import sync
from .native_indexed_history_inventory_v3 import history_inventory
from . import native_indexed_history_inventory_v3 as inventory_code
from .native_suffix_capacity_readback_v2 import check_readbacks
from .native_suffix_indexed_stage_v4 import HISTORY
from .indexed_native_history_v1 import fsync_directory
from .native_training_runtime_bindings_v1 import RuntimeBindings, PROTOCOL as RUNTIME
from .parallel_function_repair_stage_v1 import value_digest
from .parallel_repair_durable_state_v1 import commit, read_latest, restore_latest
from .whole_human_sft_evidence_v1 import tensor_digest

TRAINING = 'own-native-suffix-indexed-complete-training-controller-v3'
EVENT_BYTES = 4096


def run_training(stage, directory, history_directory, event_path, *, capacity_plan, reports, horizon_reports,
                 readbacks, runtime_sources, checkpoint_every, maximum_state_bytes, minimum_free_bytes,
                 maximum_event_raw_bytes, preflight, progress=None, expected_resume_pointer_sha256=None,
                 allow_mechanics=False):
    error = common = runtime = None
    try:
        if (type(stage) is not NativeSuffixBatchedHistoryStageV5 or stage.world != 4
                or stage.mode not in ['train', 'mechanics'] or stage.completed or stage.failed or stage.optimizer.state
                or any(type(v) is not int or v < 1 for v in [checkpoint_every, maximum_state_bytes, maximum_event_raw_bytes])
                or type(minimum_free_bytes) is not int or minimum_free_bytes < 0
                or minimum_free_bytes != stage.history_budgets['minimum_free_bytes']
                or maximum_event_raw_bytes < EVENT_BYTES * len(stage.entries)
                or not callable(preflight) or progress is not None and not callable(progress)
                or Path(history_directory).exists() or Path(event_path).exists()
                or any(Path(directory).glob('*.pending'))):
            raise ValueError('A fresh exact four-replica indexed stage, unused attempt and complete original budgets are required')
        runtime = RuntimeBindings(runtime_sources, required=[__file__, inventory_code.__file__])
        runtime.combine(stage.sources)
        check_readbacks(stage, reports, horizon_reports, readbacks, capacity_plan, allow_mechanics)
        if stage.mode == 'train' and stage.entries != stage.data.complete_schedule(
                epochs=stage.options['epochs'], width=4, seed=stage.options['seed']):
            raise ValueError('The complete original production source schedule cannot be narrowed')
        if expected_resume_pointer_sha256 is None and Path(directory, 'LATEST.json').exists():
            raise ValueError('Existing complete state requires its retained independent pointer before recovery')
        stage.intact();runtime.check()
        common = dict(binding=stage.initial_binding, plan=value_digest(capacity_plan), reports=value_digest(reports),
            horizon_reports=value_digest(horizon_reports), readbacks=value_digest(readbacks), runtime_sources=runtime.identity,
            directory=str(Path(directory).resolve()), history_directory=str(Path(history_directory).absolute()),
            event_path=str(Path(event_path).resolve()), checkpoint_every=checkpoint_every,
            maximum_state_bytes=maximum_state_bytes, minimum_free_bytes=minimum_free_bytes,
            maximum_event_raw_bytes=maximum_event_raw_bytes, resume_pointer=expected_resume_pointer_sha256)
    except Exception as exc: error = repr(exc)
    stage.coordinated(error, 'complete indexed training preflight')
    if any(v != common for v in stage.gather(common)):
        raise ValueError('Every actual replica must retain the exact same complete training attempt')
    if expected_resume_pointer_sha256 is not None:
        restore_latest(stage, directory, expected_pointer_sha256=expected_resume_pointer_sha256)
    stage.open_history(history_directory)
    attach_allocated_guard(stage)
    start = stage.completed
    events = final_receipt = None
    last_row = last_elapsed = None

    def checked_preflight():
        runtime.check();preflight()

    def root_action(action, phase):
        error = None
        if stage.rank == 0:
            try: action()
            except Exception as exc: error = repr(exc)
        stage.coordinated(error, phase)

    def open_events():
        nonlocal events
        path = Path(event_path);path.parent.mkdir(parents=True, exist_ok=True)
        events = Writer(path, root=path.parent, budget=RawBudget(maximum_event_raw_bytes),
            reserve_bytes=minimum_free_bytes, guard=checked_preflight)
        fsync_directory(path.parent)
        fsync_directory(path.parent.parent)
    root_action(open_events, 'new bounded controller event stream')
    try:
        while stage.completed < len(stage.entries):
            error = None
            try: checked_preflight()
            except Exception as exc: error = repr(exc)
            stage.coordinated(error, 'owned indexed training before source sampling')
            def fund_event():
                if events.budget.used + EVENT_BYTES > events.budget.maximum:
                    raise ValueError('The complete next scalar event must fit before any neural sampling')
                wait_for_reserve(events.root, minimum_bytes=minimum_free_bytes + NEXT_RECORD, guard=checked_preflight)
            root_action(fund_event, 'fund next full controller event before neural sampling')
            sync(stage);began = time.monotonic()
            last_row = stage.update()
            sync(stage);last_elapsed = time.monotonic() - began
            def append_event():
                event = dict(actual_complete_update=stage.completed, actual_seconds=last_elapsed,
                    attempt_started_at_complete_update=start,
                    indexed_observation_receipt_sha256=value_digest(stage.metrics[-1]),
                    original_expanded_observation_sha256=stage.metrics[-1]['record']['original_expanded_sha256'],
                    original_source_presentations=len(last_row['actual_original_source_positions']),
                    complete_original_context_tokens=last_row['complete_original_context_tokens'],
                    original_response_EOS_targets=last_row['original_response_EOS_targets'])
                from .native_reference_records_v1 import canonical
                if len(canonical(event)) > EVENT_BYTES:
                    raise ValueError('The complete controller event exceeded its predeclared full bound')
                events.append(event);events.fsync()
                if progress is not None: progress(last_row, last_elapsed)
            root_action(append_event, 'complete scalar timing and original indexed observation receipt')
            if stage.completed % checkpoint_every == 0 or stage.completed == len(stage.entries):
                final_receipt = commit(stage, directory, maximum_bytes=maximum_state_bytes, minimum_free_bytes=minimum_free_bytes)
        root_action(events.close if events is not None else lambda: None, 'complete fsynced controller event footer')
        error = None
        try:
            runtime.check()
            saved, receipt, pointer = read_latest(directory)
            if (receipt != final_receipt or saved['completed'] != len(stage.entries)
                    or saved['global_cursor'] != stage.cursor(stage.completed)
                    or value_digest(saved['metrics']) != value_digest(stage.metrics)
                    or tensor_digest(saved['model']['parts'].items()) != tensor_digest(stage.model.parts.named_parameters())
                    or value_digest(saved['optimizer']) != value_digest(stage.optimizer.state_dict())):
                raise ValueError('Every complete source receipt, actor, Adam state and cursor must be independently durable')
            stage.intact()
            with stage.checked_validation_sources():
                summary = summarize(stage.expanded_metrics(), group_size=stage.options['group_size'])
            budget = stage.data.budget(stage.entries)
            if (summary['actual_complete_updates'] != budget['updates']
                    or summary['actual_source_presentations'] != budget['source_presentations']
                    or summary['actual_unconditional_original_CE_targets'] != budget['original_response_EOS_targets']
                    or summary['actual_original_context_tokens'] != budget['complete_original_context_tokens']):
                raise ValueError('The complete original training population and every context/CE target must reconcile')
            inventory = history_inventory(stage)
        except Exception as exc: error = repr(exc)
        stage.coordinated(error, 'independent complete indexed training state and whole-source readback')
        event_receipts = stage.gather(events.receipt() if stage.rank == 0 else None)
        return dict(format=TRAINING, complete=True, rank=stage.rank, world=4, mode=stage.mode, profile=stage.profile,
            original_complete_training_budget=budget, actual_complete_updates=stage.completed,
            actual_new_optimizer_calls=stage.completed - start, actual_attempt_started_at_update=start,
            actual_research_actor_parameters=stage.model.counts()['actual_unique'], actual_native_parameters=stage.model.base.counts()['total_unique'],
            actual_capacity_readbacks_sha256=value_digest(readbacks), actual_horizon_reports_sha256=value_digest(horizon_reports),
            indexed_history_protocol=HISTORY, source_checked_history_batch_protocol=BATCH,
            runtime_binding_protocol=RUNTIME, additional_runtime_sources=copy.deepcopy(runtime.sources),
            actual_runtime_sources_sha256=runtime.identity, actual_history_budgets=copy.deepcopy(stage.history_budgets),
            actual_durable_indexed_history_inventory=inventory, actual_controller_event_stream=event_receipts[0],
            actual_streamed_complete_history_summary=summary,
            actual_checkpoint_sha256=receipt['checkpoint_sha256'], actual_pointer_sha256=pointer,
            actual_state_semantic_manifest=copy.deepcopy(receipt['semantic_manifest']),
            actual_final_actor_sha256=tensor_digest(stage.model.parts.named_parameters()),
            actual_final_Adam_sha256=value_digest(stage.optimizer.state_dict()),
            actual_final_private_RNG_sha256=value_digest(capture_rng(stage.device)),
            every_original_complete_source_and_CE_target_retained=True,
            production_updates=stage.completed if stage.mode == 'train' else 0,
            final_native_export_independent_value_readback_and_full_assessments_still_required=True,
            candidate_promoted=False, goal_achieved=False)
    except BaseException as exc:
        stage.failed = True;stage.optimizer.zero_grad(set_to_none=True)
        retained = retention_error = None
        if stage.rank == 0:
            try:
                if stage.history_writer.failure_artifact is None and stage.history_writer.failure_slot is not None:
                    retained = stage.history_writer.preserve_failure(dict(phase='indexed training controller',
                        original_exception=repr(exc), actual_completed_updates=stage.completed,
                        complete_last_actual_neural_row=last_row, actual_last_update_seconds=last_elapsed,
                        complete_coordinated_failure_observations=stage.failure_observations))
                else: retained = stage.history_writer.failure_artifact
            except Exception as failure: retention_error = repr(failure)
        stage.coordinated(retention_error, 'retain complete actual indexed controller failure')
        stage.last_controller_failure_receipt = stage.gather(retained)[0]
        raise
    finally:
        close_error = None
        if events is not None and not events.closed:
            try: events.close()
            except Exception as exc: close_error = repr(exc)
        stage.coordinated(close_error, 'fsync complete or interrupted controller event stream')
