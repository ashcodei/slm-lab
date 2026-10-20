"""Fresh capacity readback and complete public reasoned-code supervised training.

This controller owns no resources. Full-size admission additionally belongs to
the future owned launcher. Explicit CPU mechanics can never admit production.
Each run attempt has a new lossless journal; earlier attempts remain intact.
"""
import copy
from pathlib import Path
import time

import torch

from .public_reasoned_code_capacity_cases_v1 import choose
from .public_reasoned_code_capacity_controller_v2 import require_full_capacity_reports
from .batched_reasoning_capacity_controller_v1 import sync
from .complete_policy_cold_v1 import _cold
from .bounded_gzip_observation_stream_v1 import Writer, RawBudget
from .codenet_checkpoint_v1 import capture_rng
from .parallel_public_reasoned_code_SFT_stage_v1 import ParallelPublicReasonedCodeSFTStage
from .parallel_function_repair_stage_v1 import value_digest
from .parallel_repair_durable_state_v1 import commit, read_latest, restore_latest
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-public-reasoned-code-SFT-fresh-capacity-state-readback-v1'
TRAINING = 'own-public-reasoned-code-SFT-complete-source-training-controller-v1'


def capacity_reports(stage, reports, plan, allow_mechanics):
    if type(stage) is not ParallelPublicReasonedCodeSFTStage or stage.world != 4:
        raise ValueError('Exactly four typed own replicas are required')
    if stage.mode != 'mechanics':
        require_full_capacity_reports(reports, capacity_plan=plan, profile=stage.profile,
            expected_sources=stage.sources, expected_options=stage.options)
        if plan != choose(stage.data, seed=stage.options['seed'], epochs=stage.options['epochs']):
            raise ValueError('The complete original source capacity plan must reconstruct')
    elif allow_mechanics is not True or plan.get('mechanics_only') is not True:
        raise ValueError('CPU mechanics needs explicit opt-in and never grants production admission')
    if not isinstance(reports, list) or len(reports) != 4:
        raise ValueError('Every actual replica capacity report is required')
    for rank, proof in enumerate(reports):
        if (proof.get('rank') != rank or proof.get('world') != 4 or not proof.get('passed') or not proof.get('complete')
                or proof.get('mode') != ('mechanics' if stage.mode == 'mechanics' else 'capacity')
                or proof.get('profile') != stage.profile or proof.get('original_source_bindings') != stage.sources
                or proof.get('original_objective_options') != stage.options
                or proof.get('capacity_plan_sha256') != value_digest(plan)
                or proof.get('actual_complete_capacity_source_positions') != plan['capacity_entries']
                or proof.get('actual_complete_capacity_updates') != len(plan['capacity_entries'])
                or proof.get('actual_research_actor_parameters') != stage.model.counts()['actual_unique']
                or proof.get('actual_native_parameters') != stage.model.base.counts()['total_unique']):
            raise ValueError('Every actual capacity source, initializer, objective and model must match')
    for key in ['final_actual_shared_checkpoint_sha256', 'final_actual_pointer_sha256']:
        if len({p[key] for p in reports}) != 1: raise ValueError('All capacity workers must bind the same durable state')


def read_capacity(stage, directory, *, reports, capacity_plan, independent_actor, cold_inputs, allow_mechanics=False):
    """A newly constructed stage reads completed capacity without any Adam call."""
    error = None
    try:
        capacity_reports(stage, reports, capacity_plan, allow_mechanics)
        if (stage.completed or stage.failed or stage.optimizer.state or stage.mode not in ['capacity', 'mechanics']
                or stage.entries != capacity_plan['capacity_entries']):
            raise ValueError('Fresh capacity replicas and their exact whole source plan are required')
        stage.intact()
        initial_actor_sha256 = tensor_digest(stage.model.parts.named_parameters())
        common = dict(reports=value_digest(reports), plan=value_digest(capacity_plan), directory=str(Path(directory).resolve()),
            initial_actor_sha256=initial_actor_sha256)
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
    cold = _cold(stage, independent_actor, saved, cold_inputs)
    return dict(format=FORMAT, complete=True, rank=stage.rank, world=4, mode=stage.mode, profile=stage.profile,
        actual_research_actor_parameters=stage.model.counts()['actual_unique'], actual_native_parameters=stage.model.base.counts()['total_unique'],
        original_source_bindings=copy.deepcopy(stage.sources), original_objective_options=copy.deepcopy(stage.options),
        capacity_plan_sha256=value_digest(capacity_plan),
        actual_capacity_reports_sha256=value_digest(reports), actual_pointer_sha256=actual_pointer,
        initial_actor_sha256=initial_actor_sha256,
        actual_checkpoint_sha256=receipt['checkpoint_sha256'], actual_complete_capacity_updates=stage.completed,
        actual_actor_sha256=actor, actual_Adam_sha256=adam, actual_private_RNG_sha256=rng,
        actual_fixed_base_sha256=stage.base_digest, actual_Adam_states=len(stage.optimizer.state),
        independent_complete_actor_cold=cold, new_optimizer_calls=0, new_generated_actions=0,
        fresh_stage_was_empty_before_readback=True, every_complete_original_capacity_record_revalidated=True,
        production_training_admitted=False, goal_achieved=False)


def check_readbacks(stage, reports, readbacks, plan, allow_mechanics):
    capacity_reports(stage, reports, plan, allow_mechanics)
    if not isinstance(readbacks, list) or len(readbacks) != 4:
        raise ValueError('All four fresh actual capacity readbacks are required')
    for rank, proof in enumerate(readbacks):
        expected = reports[rank]['actual_per_rank_intervals'][rank]
        if (proof.get('format') != FORMAT or proof.get('complete') is not True or proof.get('rank') != rank
                or proof.get('world') != 4 or proof.get('mode') != reports[rank]['mode'] or proof.get('profile') != stage.profile
                or proof.get('actual_research_actor_parameters') != stage.model.counts()['actual_unique']
                or proof.get('actual_native_parameters') != stage.model.base.counts()['total_unique']
                or proof.get('original_source_bindings') != stage.sources or proof.get('original_objective_options') != stage.options
                or proof.get('capacity_plan_sha256') != value_digest(plan) or proof.get('actual_capacity_reports_sha256') != value_digest(reports)
                or proof.get('actual_pointer_sha256') != reports[rank]['final_actual_pointer_sha256']
                or proof.get('actual_checkpoint_sha256') != reports[rank]['final_actual_shared_checkpoint_sha256']
                or proof.get('actual_fixed_base_sha256') != stage.base_digest
                or proof.get('initial_actor_sha256') != tensor_digest(stage.model.parts.named_parameters())
                or proof.get('actual_actor_sha256') != expected['actor_sha256'] or proof.get('actual_Adam_sha256') != expected['Adam_sha256']
                or proof.get('actual_private_RNG_sha256') != expected['private_RNG_sha256']
                or proof.get('actual_complete_capacity_updates') != len(plan['capacity_entries'])
                or proof.get('new_optimizer_calls') != 0 or proof.get('new_generated_actions') != 0
                or proof.get('fresh_stage_was_empty_before_readback') is not True
                or proof.get('every_complete_original_capacity_record_revalidated') is not True):
            raise ValueError('Every exact fresh capacity source/state/private RNG readback is required before training')
        cold = proof.get('independent_complete_actor_cold', {})
        if (cold.get('complete') is not True or not all(cold.get(k) is True for k in
                ['every_base_head_actor_critic_parameter_equal', 'no_shared_parameter_storage',
                 'complete_text_photo_word_outputs_bit_exact', 'all_private_RNGs_unchanged'])
                or stage.mode == 'train' and (cold.get('actual_text_positions') != 4096
                    or cold.get('actual_photo_side') != 448 or cold.get('actual_word_shape') != [64, 448]
                    or cold.get('actual_word_batch_size') != 16 or cold.get('actual_word_readout_shape') != [16, 140, 257])):
            raise ValueError('Every fresh full native text/photo/word capacity cold check must pass')


def run_training(stage, directory, journal_path, *, capacity_plan, reports, readbacks,
                 checkpoint_every, maximum_state_bytes, minimum_free_bytes, maximum_journal_raw_bytes,
                 preflight, progress=None, expected_resume_pointer_sha256=None, allow_mechanics=False):
    """Commit the exact declared schedule; optional explicit recovery keeps attempts."""
    error = None
    try:
        check_readbacks(stage, reports, readbacks, capacity_plan, allow_mechanics)
        if (stage.mode not in ['train', 'mechanics'] or stage.completed or stage.failed or stage.optimizer.state
                or any(type(v) is not int or v < 1 for v in [checkpoint_every, maximum_state_bytes, maximum_journal_raw_bytes])
                or type(minimum_free_bytes) is not int or minimum_free_bytes < 0
                or not callable(preflight) or progress is not None and not callable(progress)
                or Path(journal_path).exists() or any(Path(directory).glob('*.pending'))):
            raise ValueError('A fresh initializer, unused attempt journal, explicit complete budgets and owned preflight are required')
        if stage.mode == 'train' and stage.entries != stage.data.complete_schedule(
                epochs=stage.options['epochs'], width=stage.options['width'], seed=stage.options['seed']):
            raise ValueError('The whole original production source schedule cannot be narrowed')
        if expected_resume_pointer_sha256 is None and Path(directory, 'LATEST.json').exists():
            raise ValueError('Existing durable training state requires its explicitly retained pointer before recovery')
        if (expected_resume_pointer_sha256 is None and Path(directory).exists()
                and any(Path(directory).iterdir())):
            raise ValueError('An uncommitted nonempty training directory requires inspection; preserve every prior artifact')
        stage.intact()
        common = dict(binding=stage.initial_binding, plan=value_digest(capacity_plan), reports=value_digest(reports),
            readbacks=value_digest(readbacks), journal=str(Path(journal_path).resolve()), directory=str(Path(directory).resolve()),
            checkpoint_every=checkpoint_every, maximum_state_bytes=maximum_state_bytes, minimum_free_bytes=minimum_free_bytes,
            maximum_journal_raw_bytes=maximum_journal_raw_bytes, resume_pointer=expected_resume_pointer_sha256)
    except Exception as exc: error = repr(exc)
    stage.coordinated(error, 'whole training controller preflight')
    if any(v != common for v in stage.gather(common)):
        raise ValueError('Every actual replica must retain the same complete owned training attempt')
    if expected_resume_pointer_sha256 is not None:
        restore_latest(stage, directory, expected_pointer_sha256=expected_resume_pointer_sha256)
    start = stage.completed
    writer = receipt = None
    def root_action(action, phase):
        error = None
        if stage.rank == 0:
            try: action()
            except Exception as exc: error = repr(exc)
        stage.coordinated(error, phase)
    def open_journal():
        nonlocal writer
        path = Path(journal_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = Writer(path, root=path.parent, budget=RawBudget(maximum_journal_raw_bytes),
            reserve_bytes=minimum_free_bytes, guard=preflight)
    root_action(open_journal, 'new complete-attempt journal')
    try:
        while stage.completed < len(stage.entries):
            error = None
            try: preflight()
            except Exception as exc: error = repr(exc)
            stage.coordinated(error, 'owned training before source sampling')
            sync(stage)
            began = time.monotonic()
            row = stage.update()
            sync(stage)
            elapsed = time.monotonic() - began
            def append():
                writer.append(dict(actual_complete_update=row, actual_seconds=elapsed,
                    attempt_started_at_complete_update=start, actual_complete_updates=stage.completed))
                writer.fsync()
                if progress is not None: progress(row, elapsed)
            root_action(append, 'retain every complete actual training update')
            if stage.completed % checkpoint_every == 0 or stage.completed == len(stage.entries):
                receipt = commit(stage, directory, maximum_bytes=maximum_state_bytes, minimum_free_bytes=minimum_free_bytes)
        root_action(writer.close if writer is not None else lambda: None, 'close complete-attempt journal')
        error = None
        try:
            saved, actual_receipt, pointer = read_latest(directory)
            if (receipt != actual_receipt or saved['completed'] != len(stage.entries)
                    or saved['global_cursor'] != stage.cursor(stage.completed)
                    or value_digest(saved['metrics']) != value_digest(stage.metrics)
                    or tensor_digest(saved['model']['parts'].items()) != tensor_digest(stage.model.parts.named_parameters())
                    or value_digest(saved['optimizer']) != value_digest(stage.optimizer.state_dict())):
                raise ValueError('The complete source schedule, all selected weights/Adam and every actual update must be durable')
            stage.intact()
        except Exception as exc: error = repr(exc)
        stage.coordinated(error, 'complete original training state readback')
        streams = stage.gather(writer.receipt() if stage.rank == 0 else None)
        return dict(format=TRAINING, complete=True, rank=stage.rank, world=4, mode=stage.mode, profile=stage.profile,
            original_complete_training_budget=stage.data.budget(stage.entries), actual_complete_updates=stage.completed,
            actual_new_optimizer_calls=stage.completed - start, actual_attempt_started_at_update=start,
            actual_research_actor_parameters=stage.model.counts()['actual_unique'],
            actual_native_parameters=stage.model.base.counts()['total_unique'],
            actual_capacity_readbacks_sha256=value_digest(readbacks), actual_complete_journal=streams[0],
            actual_checkpoint_sha256=receipt['checkpoint_sha256'], actual_pointer_sha256=pointer,
            actual_final_actor_sha256=tensor_digest(stage.model.parts.named_parameters()),
            actual_final_Adam_sha256=value_digest(stage.optimizer.state_dict()),
            actual_final_private_RNG_sha256=value_digest(capture_rng(stage.device)),
            every_original_complete_source_and_CE_target_retained=True,
            production_updates=stage.completed if stage.mode == 'train' else 0,
            final_native_export_independent_value_readback_and_full_assessments_still_required=True,
            candidate_promoted=False, goal_achieved=False)
    except BaseException:
        stage.failed = True
        stage.optimizer.zero_grad(set_to_none=True)
        raise
    finally:
        if writer is not None and not writer.closed: writer.close()
