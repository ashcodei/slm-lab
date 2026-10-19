"""Indexed whole-source capacity with immutable replay attempts and actor cold load.

This controller uses the qualified native stage, including real code workers,
complete CE fallback and real empty-rank tails. It owns no resource lease and
cannot authorize production. Early EOS in actual sampling does not prove
worst-case full-horizon allocation; a separate synthetic capacity probe and
complete source-state readback remain required before production admission.
"""
import copy
import re
import time

import torch

from .batched_reasoning_capacity_controller_v1 import _cold, draws, sync
from .codenet_checkpoint_v1 import capture_rng
from .native_suffix_capacity_cases_v1 import choose_admitted
from .native_suffix_validation_stage_v2 import PROTOCOL
from .native_suffix_indexed_stage_v4 import HISTORY
from .native_suffix_batched_history_stage_v5 import NativeSuffixBatchedHistoryStageV5, PROTOCOL as HISTORY_BATCH
from .native_history_summary_v1 import summarize
from .native_allocated_history_guard_v1 import attach as attach_allocated_guard
from .corpus import file_hash
from pathlib import Path
from .parallel_function_repair_stage_v1 import manifest, value_digest
from .parallel_native_suffix_stage_v1 import CE_ONLY, KERNEL, record_semantics
from .parallel_repair_durable_state_v1 import commit, read_latest, restore_latest
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-native-suffix-indexed-selected-capacity-private-replay-actor-cold-v2'



def live(stage):
    return dict(actor=tensor_digest(stage.model.parts.named_parameters()),
        Adam=value_digest(stage.optimizer.state_dict()), RNG=value_digest(capture_rng(stage.device)),
        completed=stage.completed, failed=stage.failed, receipts=value_digest(stage.metrics),
        active_directory=str(stage.history_reader.root) if stage.history_reader is not None else None,
        active_writer_identity=id(stage.history_writer),
        optimizer_tensor_identities=[(id(p), key, id(t), t.data_ptr(), t._version)
            for p, values in stage.optimizer.state.items() for key, t in values.items()])


def history_inventory(stage):
    """Check real retained success/failure allocations without reading all rows."""
    from .indexed_native_history_v1 import RECEIPT_LIMIT
    directories = {(r['directory'], r['header_sha256']) for r in stage.metrics}
    if stage.history_reader is not None:
        directories.add((str(stage.history_reader.root), stage.history_reader.header_sha256))
    result = []
    for directory, header in sorted(directories):
        reader = stage.history_file_reader(directory, header)
        slot = reader.root / 'independent-failure.reserved'
        allocation = 2 * reader.budgets['maximum_expanded_record_bytes'] + 2 * RECEIPT_LIMIT + 64 * 1024
        if slot.is_symlink() or not slot.is_file() or slot.stat().st_blocks * 512 < allocation:
            raise ValueError('Each retained attempt must keep its physically funded independent failure slot')
        seen, allocated, files = set(), reader.root.stat().st_blocks * 512, 0
        for path in reader.root.iterdir():
            if path.is_symlink() or not path.is_file(): raise ValueError('Only original regular indexed history artifacts are eligible')
            if (path.name not in ['reference.json.gz', 'independent-failure.reserved']
                    and re.fullmatch(r'record-\d{12}\.(json\.gz|receipt\.json|reserved)', path.name) is None
                    and re.fullmatch(r'failure-step-\d{12}\.(pt|receipt\.json)', path.name) is None):
                raise ValueError('An unknown writer cannot add artifacts to an exclusively owned indexed history')
            stat = path.stat(); files += 1
            identity = stat.st_dev, stat.st_ino
            if identity not in seen:
                seen.add(identity); allocated += stat.st_blocks * 512
        if allocated > reader.budgets['maximum_physical_bytes']:
            raise ValueError('The complete retained attempt exceeds its actual allocated physical budget')
        result.append(dict(directory=directory, header_sha256=header,
            starting_completed=reader.header['starting_completed'], actual_regular_files=files,
            actual_unique_inode_allocated_bytes=allocated,
            actual_independent_failure_slot_allocated_bytes=slot.stat().st_blocks * 512,
            original_indexed_history_budgets=copy.deepcopy(reader.budgets)))
    return result

def run_capacity(stage, directory, *, capacity_plan, independent_actor, cold_inputs,
                 maximum_bytes, minimum_free_bytes, history_directory, allow_mechanics=False):
    error = common = None
    try:
        if (type(stage) is not NativeSuffixBatchedHistoryStageV5 or stage.completed or stage.failed or stage.world != 4
                or len(stage.entries) < 2 or stage.entries != capacity_plan['capacity_entries']
                or not callable(independent_actor) or Path(history_directory).exists()
                or Path(directory, 'LATEST.json').exists() or any(Path(directory).glob('*.pending'))
                or stage.sources.get(str(Path(__file__).resolve())) != file_hash(Path(__file__).resolve())
                or stage.sources.get(str(Path(__file__).with_name('native_history_summary_v1.py').resolve())) != file_hash(Path(__file__).with_name('native_history_summary_v1.py').resolve())
                or stage.mode != 'capacity' and not (allow_mechanics is True and stage.mode == 'mechanics')):
            raise ValueError('A fresh four-replica optimized native-code capacity stage is required')
        if stage.mode == 'capacity':
            if capacity_plan != choose_admitted(stage.data, seed=stage.options['seed'], epochs=stage.options['epochs']):
                raise ValueError('Every complete actual admitted native worst-source batch and physical rotation must match')
        elif capacity_plan.get('mechanics_only') is not True:
            raise ValueError('Own-small controller fixtures must be explicitly mechanics-only')
        stage.intact()
        common = dict(plan=value_digest(capacity_plan), binding=stage.initial_binding,
            directory=str(Path(directory).resolve()), history_directory=str(Path(history_directory).absolute()))
    except Exception as exc:
        error = repr(exc)
    stage.coordinated(error, 'whole native capacity preflight')
    if any(v != common for v in stage.gather(common)):
        raise ValueError('Every actual rank must run the identical complete native workload')
    if stage.device.type == 'cuda': torch.cuda.reset_peak_memory_stats(stage.device)
    stage.open_history(Path(history_directory) / 'attempt-0001')
    attach_allocated_guard(stage)
    timings, physical_calls = [], 0

    def step():
        nonlocal physical_calls
        sync(stage)
        began = time.monotonic()
        value = stage.update()
        sync(stage)
        physical_calls += 1
        timings.append(dict(actual_global_step=stage.completed, actual_seconds=time.monotonic() - began,
            actual_generated_actions=sum(t['kernel'].get('actual_sampled_action_tokens', 0)
                for t in value['actual_rank_observations'] if t['kernel'] is not None)))
        return value

    def save(): return commit(stage, directory, maximum_bytes=maximum_bytes, minimum_free_bytes=minimum_free_bytes)

    step()
    first = save()
    error = None
    saved = receipt = pointer = None
    try:
        saved, receipt, pointer = read_latest(directory)
        if first != receipt: raise ValueError('The complete shared file and private-worker external receipt must reconcile')
    except Exception as exc:
        error = repr(exc)
    stage.coordinated(error, 'native first durable readback')
    wanted_draws, wanted = draws(stage.device), step()
    expected = live(stage)
    expected_semantic_metrics = summarize(stage.expanded_metrics(), group_size=stage.options['group_size'])
    for kind in ['invalid_CPU', 'invalid_CUDA', 'peer_foreign']:
        if kind == 'invalid_CUDA' and stage.device.type != 'cuda': continue
        bad = copy.deepcopy(saved)
        if kind == 'peer_foreign':
            if stage.rank == 2: bad['private_workers'][0]['rng'] = copy.deepcopy(saved['private_workers'][1]['rng'])
            expected_manifest = receipt['semantic_manifest']
        else:
            bad['private_workers'][3]['rng']['cuda' if kind == 'invalid_CUDA' else 'torch'] = torch.zeros(1, dtype=torch.uint8)
            expected_manifest = manifest(bad)
        rejected = False
        try: stage.resume(bad, expected_manifest=expected_manifest)
        except ValueError: rejected = True
        stage.coordinated(None if rejected and live(stage) == expected else 'Bad peer RNG mutated live native state',
            'native ' + kind + ' private RNG rejection')
    restore_latest(stage, directory, expected_pointer_sha256=pointer)
    stage.open_history(Path(history_directory) / 'attempt-0002')
    attach_allocated_guard(stage)
    actual_draws, actual = draws(stage.device), step()
    current = live(stage)
    error = None
    # Wall times/thread IDs remain in the raw evidence. Compare actual execution
    # semantics separately, preserving every program/test/output/verdict field.
    if (actual_draws != wanted_draws or record_semantics(actual) != record_semantics(wanted)
            or summarize(stage.expanded_metrics(), group_size=stage.options['group_size']) != expected_semantic_metrics
            or any(current[k] != expected[k] for k in ['actor', 'Adam', 'RNG', 'completed', 'failed'])):
        error = 'Next actual native actions/program verdicts/logps/gradients/weights/Adam/private RNG differ'
    stage.coordinated(error, 'exact next native source-group replay')
    while stage.completed < len(stage.entries): step()
    final = save()
    error = None
    last = last_receipt = final_pointer = None
    try:
        last, last_receipt, final_pointer = read_latest(directory)
        if (last_receipt != final or last['completed'] != len(stage.entries)
                or value_digest(last['metrics']) != value_digest(stage.metrics)
                or tensor_digest(last['model']['parts'].items()) != tensor_digest(stage.model.parts.named_parameters())):
            raise ValueError('Every final native source observation and selected value must equal its committed file')
    except Exception as exc:
        error = repr(exc)
    stage.coordinated(error, 'complete native final capacity readback')
    error = None
    try: cold = _cold(stage, independent_actor, last, cold_inputs)
    except Exception as exc: error = repr(exc)
    stage.coordinated(error, 'independent complete native actor cold outputs')
    error = None
    try:
        stage.intact()
        summary = summarize(stage.expanded_metrics(), group_size=stage.options['group_size'])
        budget = stage.data.budget(stage.entries)
        if (summary['actual_execution_eligible_presentations'] != budget['execution_eligible_presentations']
                or summary['actual_complete_CE_only_presentations'] != budget['CE_only_presentations']
                or summary['actual_unconditional_original_CE_targets'] != budget['original_response_EOS_targets']
                or summary['actual_source_presentations'] != budget['source_presentations']
                or summary['actual_original_context_tokens'] != budget['complete_original_context_tokens']):
            raise ValueError('All original native/CE-only source observations, contexts and unconditional targets must reconcile')
        inventory = history_inventory(stage)
    except Exception as exc: error = repr(exc)
    stage.coordinated(error, 'complete streamed source/target accounting and actual funded history')
    workers = stage.gather(dict(actual_rank=stage.rank, private_RNG_sha256=value_digest(capture_rng(stage.device)),
        actor_sha256=tensor_digest(stage.model.parts.named_parameters()), Adam_sha256=value_digest(stage.optimizer.state_dict()),
        actual_intervals=timings))
    if (len({v['private_RNG_sha256'] for v in workers}) != 4 or len({v['actor_sha256'] for v in workers}) != 1
            or len({v['Adam_sha256'] for v in workers}) != 1):
        raise ValueError('Four actual native workers require distinct private RNGs and identical model/Adam')
    return dict(format=FORMAT, complete=True, passed=True, rank=stage.rank, world=4, mode=stage.mode,
        actual_device_type=stage.device.type, actual_device_index=stage.device.index,
        actual_stage_class=type(stage).__name__, recorded_source_validation=PROTOCOL,
        actual_research_actor_parameters=stage.model.counts()['actual_unique'],
        actual_native_parameters=stage.model.base.counts()['total_unique'], profile=stage.profile,
        original_source_bindings=copy.deepcopy(stage.sources), original_objective_options=copy.deepcopy(stage.options),
        original_own_reference_binding=copy.deepcopy(stage.reference_binding), capacity_plan_sha256=value_digest(capacity_plan),
        actual_complete_capacity_updates=stage.completed, actual_physical_optimizer_calls=physical_calls,
        actual_source_budget=budget, actual_Adam_states=len(stage.optimizer.state),
        actual_complete_capacity_source_positions=copy.deepcopy(stage.entries),
        **{k: v for k, v in summary.items() if k.startswith('actual_') and k != 'actual_complete_updates'},
        actual_all_physical_global_generated_actions_including_replay=sum(t['actual_generated_actions'] for t in timings),
        actual_history_budgets=copy.deepcopy(stage.history_budgets),
        indexed_history_protocol=HISTORY, source_checked_history_batch_protocol=HISTORY_BATCH, actual_durable_indexed_history_inventory=inventory,
        actual_allocated_block_guard_reports=stage.gather(stage.history_writer.guard.report() if stage.rank == 0 else None),
        no_complete_neural_history_materialized_by_capacity_controller=True,
        actual_per_rank_intervals=workers, all_actual_next_actions_execution_semantics_unclipped_gradients_Adam_private_RNG_exact=True,
        execution_timings_and_thread_IDs_retained_outside_replay_equality=True,
        invalid_peer_CPU_RNG_rejected_before_mutation=True, invalid_peer_CUDA_RNG_rejected_before_mutation=stage.device.type == 'cuda',
        peer_only_valid_foreign_RNG_rejected_before_mutation=True, independent_complete_actor_cold=cold,
        final_actual_shared_checkpoint_sha256=final['checkpoint_sha256'], final_actual_pointer_sha256=final_pointer,
        actual_GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(stage.device) if stage.device.type == 'cuda' else None,
        actual_GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(stage.device) if stage.device.type == 'cuda' else None,
        full_size_actual_selected_workload_passed=stage.mode == 'capacity', full_size_GPU_capacity_passed=False,
        synthetic_worst_full_horizon_allocation_still_required=True, independent_complete_source_state_readback_still_required=True,
        production_training_admitted=False, production_updates=0, new_model_quality_gain=False,
        final_standalone_and_full_assessments_still_required=True, candidate_promoted=False, goal_achieved=False)
