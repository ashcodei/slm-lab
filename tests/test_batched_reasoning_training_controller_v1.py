"""Four actual own-small replicas; no GPU production admission is inferred."""
import copy
from datetime import timedelta
import json
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.batched_reasoning_capacity_controller_v1 import run_capacity, live
from slm_lab.batched_reasoning_training_controller_v1 import read_capacity, run_training
from slm_lab.bounded_gzip_observation_stream_v1 import lines
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.parallel_repair_durable_state_v1 import read_latest
from slm_lab.public_reasoning_train_data_v1 import PublicReasoningTrainData
from test_batched_reasoning_capacity_controller_v2 import setup, make_student, cold_inputs


def run(stage, folder, plan, reports, readbacks, *, progress=None, pointer=None, opt_in=True):
    return run_training(stage, folder / 'durable', folder / 'attempt.jsonl.gz', capacity_plan=plan,
        reports=reports, readbacks=readbacks, checkpoint_every=1, maximum_state_bytes=32 * 1024**2,
        minimum_free_bytes=0, maximum_journal_raw_bytes=32 * 1024**2,
        preflight=lambda: stage.data.guard.check(), progress=progress,
        expected_resume_pointer_sha256=pointer, allow_mechanics=opt_in)


def worker(rank, folder):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    folder = Path(folder)
    dist.init_process_group('gloo', init_method='file://' + str(folder / 'rendezvous'), rank=rank, world_size=4,
        timeout=timedelta(minutes=5))
    control = dist.new_group(backend='gloo', timeout=timedelta(minutes=5))
    try:
        data = PublicReasoningTrainData('data/public-reasoning-curriculum-v1',
            'reports/public-reasoning-curriculum-v1-readback-admission/complete.json')
        def independent():
            with torch.random.fork_rng(devices=[]): return make_student(data)
        for profile in ['anchor_CE', 'anchor_RL']:
            def fresh(): return setup(data, profile, rank, 4, control)
            initial = fresh()
            plan = dict(capacity_entries=copy.deepcopy(initial.entries), mechanics_only=True,
                complete_source_selection=False, actual_full_source_capacity_plan_not_used_by_this_small_fixture=True)
            cap_dir = folder / profile / 'capacity'
            capacity = run_capacity(initial, cap_dir, capacity_plan=plan, independent_actor=independent,
                cold_inputs=cold_inputs(data), maximum_bytes=32 * 1024**2, minimum_free_bytes=0, allow_mechanics=True)
            reports = initial.gather(capacity)
            # The second independent stage has fresh empty Adam and separate storage.
            reader = fresh()
            assert not ({p.data_ptr() for p in reader.model.parameters()} & {p.data_ptr() for p in initial.model.parameters()})
            before = live(reader)
            with pytest.raises(ValueError):
                read_capacity(reader, cap_dir, reports=reports, capacity_plan=plan,
                    independent_actor=independent, cold_inputs=cold_inputs(data))
            assert live(reader) == before
            bad_reports = copy.deepcopy(reports)
            for report in bad_reports: report['final_actual_pointer_sha256'] = 'e' * 64
            with pytest.raises(ValueError):
                read_capacity(reader, cap_dir, reports=bad_reports, capacity_plan=plan,
                    independent_actor=independent, cold_inputs=cold_inputs(data), allow_mechanics=True)
            assert live(reader) == before
            proof = read_capacity(reader, cap_dir, reports=reports, capacity_plan=plan,
                independent_actor=independent, cold_inputs=cold_inputs(data), allow_mechanics=True)
            readbacks = reader.gather(proof)
            assert proof['actual_complete_capacity_updates'] == 3 and proof['actual_Adam_states'] == 17
            assert proof['new_optimizer_calls'] == proof['new_generated_actions'] == 0
            assert proof['independent_complete_actor_cold']['complete_text_photo_word_outputs_bit_exact']

            trained = fresh()
            before = live(trained)
            broken = copy.deepcopy(readbacks)
            if rank == 2: broken[1]['actual_private_RNG_sha256'] = 'c' * 64
            with pytest.raises(ValueError): run(trained, folder / profile / 'forbidden', plan, reports, broken)
            assert live(trained) == before and not (folder / profile / 'forbidden').exists()
            with pytest.raises(ValueError): run(trained, folder / profile / 'no-opt-in', plan, reports, readbacks, opt_in=False)
            assert live(trained) == before
            trained_result = run(trained, folder / profile / 'complete', plan, reports, readbacks)
            expected = live(trained)
            assert trained_result['actual_complete_updates'] == 3 and trained_result['actual_new_optimizer_calls'] == 3
            assert trained_result['original_complete_training_budget']['source_presentations'] == 9
            assert trained_result['actual_complete_journal']['records'] == 3 and trained_result['actual_complete_journal']['complete']
            assert trained_result['production_updates'] == 0

            interrupted = fresh()
            attempts = folder / profile / 'recovery'
            def stop_after_second(row, elapsed):
                if row['step'] == 2: raise OSError('controlled root-only progress failure after actual Adam')
            with pytest.raises(ValueError, match='retain every complete actual training update'):
                run(interrupted, attempts, plan, reports, readbacks, progress=stop_after_second)
            assert interrupted.failed and interrupted.completed == 2
            assert all(p.grad is None for p in interrupted.model.parameters())
            saved, _, pointer = read_latest(attempts / 'durable')
            assert saved['completed'] == 1
            all_failed_rows = [r for _, r in lines(attempts / 'attempt.jsonl.gz')]
            assert [r['actual_complete_updates'] for r in all_failed_rows] == [1, 2]
            resumed = fresh()
            before = live(resumed)
            with pytest.raises(ValueError): run(resumed, attempts / 'missing-pin', plan, reports, readbacks, pointer='d' * 64)
            assert live(resumed) == before
            # A new journal retains the failed attempt; the same durable state is resumed.
            resumed_result = run_training(resumed, attempts / 'durable', attempts / 'second-attempt.jsonl.gz',
                capacity_plan=plan, reports=reports, readbacks=readbacks, checkpoint_every=1,
                maximum_state_bytes=32 * 1024**2, minimum_free_bytes=0, maximum_journal_raw_bytes=32 * 1024**2,
                preflight=lambda: data.guard.check(), expected_resume_pointer_sha256=pointer, allow_mechanics=True)
            actual = live(resumed)
            for key in ['actor', 'Adam', 'RNG', 'metrics', 'completed', 'failed']: assert actual[key] == expected[key]
            assert resumed_result['actual_attempt_started_at_update'] == 1
            assert resumed_result['actual_new_optimizer_calls'] == 2
            second_rows = [r for _, r in lines(attempts / 'second-attempt.jsonl.gz')]
            assert [r['actual_complete_updates'] for r in second_rows] == [2, 3]
            assert all_failed_rows[1]['actual_complete_update'] == second_rows[0]['actual_complete_update']
            assert (attempts / 'attempt.jsonl.gz').is_file()
            assert not torch.cuda.is_initialized()
            result = dict(profile=profile, rank=rank, actual_small_research_parameters=trained.model.counts()['actual_unique'],
                actual_native_parameters=trained.model.base.counts()['total_unique'],
                actual_capacity=capacity, actual_fresh_readback=proof, actual_complete_training=trained_result,
                actual_recovered_training=resumed_result, exact_complete_and_interrupted_recovered_final_state=True,
                actual_replayed_uncommitted_update_actions_verifiers_logps_gradients_equal=True,
                root_only_error_retains_both_observed_updates_and_old_durable_prefix=True,
                peer_only_bad_readback_and_missing_or_stale_pins_rejected_before_mutation=True,
                actual_failed_attempt_observed_updates=2, actual_recovery_new_updates=2,
                all_failed_and_recovery_journals_retained=True, full_size_GPUcapacity_or_quality=False,
                CPU_fixture_does_not_admit_production=True, production_updates=0, goal_achieved=False)
            (folder / (profile + '-rank-' + str(rank) + '.json')).write_text(json.dumps(result, indent=2, allow_nan=False))
    finally: dist.destroy_process_group()


def test_actual_four_replica_capacity_readback_training_and_preserved_attempt_recovery(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    for profile in ['anchor_CE', 'anchor_RL']:
        reports = [json.loads((tmp_path / (profile + '-rank-' + str(rank) + '.json')).read_text()) for rank in range(4)]
        assert len({r['actual_complete_training']['actual_checkpoint_sha256'] for r in reports}) == 1
        assert len({r['actual_fresh_readback']['actual_private_RNG_sha256'] for r in reports}) == 4
        assert all(r['exact_complete_and_interrupted_recovered_final_state'] for r in reports)
