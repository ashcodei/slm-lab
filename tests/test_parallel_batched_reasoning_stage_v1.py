"""Actual own-small replicas; no full-size GPU capacity or quality inferred."""
import copy
from datetime import timedelta
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab import parallel_repair_durable_state_v1 as durable
from slm_lab.batched_reasoning_group_gradients_v2 import group_gradients
from slm_lab.codenet_checkpoint_v1 import capture_rng, restore_rng
from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference
from slm_lab.parallel_batched_reasoning_stage_v1 import ParallelBatchedReasoningStage, private_seeds
from slm_lab.parallel_function_repair_stage_v1 import manifest, value_digest
from slm_lab.public_reasoning_train_data_v1 import PublicReasoningTrainData
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_public_reasoning_group_gradients_v1 import data, model, optimizer

SOURCES = {'own_small_initializer_sha256': 'a' * 64, 'original_public_source_plan_sha256': 'b' * 64}
OPTIONS = dict(learning_rate=1e-4, gradient_clip=1., group_size=4, KL_weight=.01, anchor_weight=1.,
    cache_absolute_bound=.001, maximum_cache_bytes=8 * 1024**2, gradient_bucket_values=4096,
    seed=989113, epochs=1, CPU_mechanics_horizon=2)


def positions(data):
    return [p for p in data.slots if data.row(p)['response'] == 'True'][:9]


def make_student(data):
    p = positions(data)[0]
    ids, _ = data.sequence(p)
    start = data.information(p)['original_prompt_positions']
    torch.manual_seed(989218)
    return model((ids[start-1], ids[start], False))


def setup(data, profile='anchor_RL', rank=0, world=1, control=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    student = make_student(data)
    entries = [[p] for p in positions(data)[:3]] if world == 1 else [positions(data)[:4], positions(data)[4:8], positions(data)[8:9]]
    random.seed(989313 + rank)
    np.random.seed(989313 + rank)
    torch.manual_seed(989313 + rank)
    return ParallelBatchedReasoningStage(student, optimizer(student), data, entries, profile, OPTIONS, SOURCES,
        mode='mechanics', rank=rank, world=world, control_group=control, gradient_group=None)


def draws():
    return [random.random(), float(np.random.random()), *torch.rand(4).tolist()]


def live(stage):
    return dict(actor=tensor_digest(stage.model.parts.named_parameters()),
        optimizer=value_digest(stage.optimizer.state_dict()), rng=value_digest(capture_rng('cpu')),
        completed=stage.completed, metrics=value_digest(stage.metrics), failed=stage.failed,
        tensor_identities=[(id(p), k, id(t), t.data_ptr(), t._version)
            for p, row in stage.optimizer.state.items() for k, t in row.items()])


def commit(stage, path):
    return durable.commit(stage, path, maximum_bytes=32 * 1024**2, minimum_free_bytes=0)


@pytest.mark.parametrize('profile', ['anchor_CE', 'anchor_RL'])
def test_real_single_worker_durable_next_actions_gradients_weights_Adam_and_RNG(data, tmp_path, profile):
    stage = setup(data, profile)
    fixed = tensor_digest(stage.model.base.named_parameters())
    initial_actor = tensor_digest(stage.model.parts.named_parameters())
    stage.update()
    first = commit(stage, tmp_path / 'durable')
    _, _, pointer = durable.read_latest(tmp_path / 'durable')
    wanted_draws = draws()
    wanted = stage.update()
    expected = live(stage)
    durable.restore_latest(stage, tmp_path / 'durable', expected_pointer_sha256=pointer)
    assert draws() == wanted_draws and stage.update() == wanted
    actual = live(stage)
    assert all(actual[k] == expected[k] for k in ['actor', 'optimizer', 'rng', 'completed', 'metrics'])
    assert actual['actor'] != initial_actor and tensor_digest(stage.model.base.named_parameters()) == fixed
    stage.update()
    final = commit(stage, tmp_path / 'durable')
    before = live(stage)
    with pytest.raises(ValueError):
        durable.restore_latest(stage, tmp_path / 'durable')
    assert live(stage) == before
    durable.restore_latest(stage, tmp_path / 'durable', allow_completed=True)
    assert stage.completed == 3 and first['generation'] == 1 and final['generation'] == 2
    assert all(float(s['step']) == 3 for s in stage.optimizer.state.values())
    wanted_actor = tensor_digest(stage.model.parts.named_parameters())
    wanted_Adam = value_digest(stage.optimizer.state_dict())
    wanted_RNG = value_digest(capture_rng('cpu'))
    cold = setup(data, profile)
    durable.restore_latest(cold, tmp_path / 'durable', allow_completed=True)
    assert tensor_digest(cold.model.parts.named_parameters()) == wanted_actor
    assert value_digest(cold.optimizer.state_dict()) == wanted_Adam and value_digest(capture_rng('cpu')) == wanted_RNG
    assert not torch.cuda.is_initialized()
    (tmp_path / 'actual-single-replica-recovery.json').write_text(json.dumps(dict(profile=profile,
        actual_student_values=stage.model.counts()['actual_unique'], actual_source_groups=3,
        exact_next_private_draws_native_samples_verifiers_unclipped_gradients_weights_Adam_and_RNG=True,
        full_size_GPU_capacity_or_model_quality=False, production_updates=0), indent=2))


def test_all_private_state_sources_masks_clocks_and_nonfinite_parts_reject_before_mutation(data):
    stage = setup(data)
    stage.update()
    state, receipt = stage.snapshot()
    stage.update()
    before = live(stage)
    for what in ['RNG', 'clock', 'missing_Adam', 'source', 'critic', 'reference', 'mask', 'nonfinite', 'cursor', 'flag']:
        bad = copy.deepcopy(state)
        if what == 'RNG': bad['private_workers'][0]['rng']['torch'] = torch.zeros(1, dtype=torch.uint8)
        if what == 'clock': bad['optimizer']['state'][0]['step'].fill_(44)
        if what == 'missing_Adam': del bad['optimizer']['state'][0]
        if what == 'source': bad['model']['sources']['original_public_source_plan_sha256'] = 'c' * 64
        if what == 'critic': bad['model']['parts']['critic.weight'].add_(.1)
        if what == 'reference': bad['reference_binding']['frozen_actor_parts_sha256'] = 'd' * 64
        if what == 'mask': bad['metrics'][0]['actual_rank_observations'][0]['kernel']['unconditional_original_answer_CE_anchor']['response_EOS_targets'] += 1
        if what == 'nonfinite': next(iter(bad['model']['parts'].values())).fill_(float('nan'))
        if what == 'cursor': bad['private_workers'][0]['local_source_cursor']['rank'] = 2
        if what == 'flag': bad['all_private_RNGs_and_source_cursors_retained'] = False
        with pytest.raises(ValueError):
            stage.resume(bad, expected_manifest=receipt)
        assert live(stage) == before
        # A freshly recomputed manifest cannot excuse structurally invalid state.
        if what in ['RNG', 'clock', 'missing_Adam', 'source', 'critic', 'reference', 'mask', 'cursor', 'flag']:
            with pytest.raises(ValueError):
                stage.resume(bad, expected_manifest=manifest(bad))
            assert live(stage) == before
    stage.data.purpose = 'source_preparation'
    try:
        with pytest.raises(ValueError): stage.resume(state, expected_manifest=receipt)
        with pytest.raises(ValueError): stage.update()
        assert live(stage) == before
    finally:
        del stage.data.purpose


def test_bad_settings_and_CPU_production_claims_reject_before_any_optimizer_update(data):
    for change in [dict(group_size=1), dict(seed=2**64), dict(CPU_mechanics_horizon=True),
            dict(CPU_mechanics_horizon=1000), dict(cache_absolute_bound=.01), dict(KL_weight=-1)]:
        student = make_student(data)
        adam = optimizer(student)
        values = tensor_digest(student.parts.named_parameters())
        with pytest.raises(ValueError):
            ParallelBatchedReasoningStage(student, adam, data, [[positions(data)[0]]], 'anchor_RL',
                dict(OPTIONS, **change), SOURCES, mode='mechanics')
        assert not adam.state and tensor_digest(student.parts.named_parameters()) == values
    for mode in ['capacity', 'train']:
        student = make_student(data)
        adam = optimizer(student)
        with pytest.raises(ValueError):
            ParallelBatchedReasoningStage(student, adam, data, [[positions(data)[0]]], 'anchor_RL',
                OPTIONS, SOURCES, mode=mode)
        assert not adam.state and not torch.cuda.is_initialized()


def test_new_full_data_parallel_schedule_preserves_every_original_source_and_target(data, tmp_path):
    entries = data.complete_schedule(epochs=1, width=4, seed=OPTIONS['seed'])
    budget = data.budget(entries)
    assert sorted(p for row in entries for p in row) == list(data.slots)
    assert len({data.row(p)['task'] for row in entries for p in row}) == 12
    assert budget == dict(updates=642, source_presentations=2568, complete_context_tokens=245190,
        response_EOS_targets=19080, full_native_generation_horizon_sum=242112)
    (tmp_path / 'actual-complete-source-budget.json').write_text(json.dumps(dict(**budget,
        original_train_task_families=12, all_whole_64_or_256_action_budgets_preserved=True,
        new_four_source_global_update_protocol=True, previous_single_source_RL_protocol_not_completed_by_this=True,
        original_queued_three_epoch_CE_warmup_not_replaced=True, optimizer_updates=0), indent=2))


def manual_source_mean_matches(stage):
    """Fresh serial local kernels check the actual four-source SUM denominator."""
    rng = capture_rng('cpu')
    try:
        student = make_student(stage.data)
        adam = optimizer(student)
        reference = FrozenOwnActorReference(student, sources=SOURCES)
        sums = [torch.zeros_like(p) for _, p in student.named_selected(('actor',))]
        for position in stage.entries[0]:
            student.zero_grad(set_to_none=True)
            group_gradients(student, stage.data, position, reference, profile=stage.profile,
                seeds=private_seeds(OPTIONS['seed'], 1, position, OPTIONS['group_size']), purpose='CPU-mechanics',
                maximum_cache_bytes=OPTIONS['maximum_cache_bytes'], CPU_mechanics_horizon=2,
                KL_weight=OPTIONS['KL_weight'], anchor_weight=OPTIONS['anchor_weight'])
            for total, (_, p) in zip(sums, student.named_selected(('actor',))):
                total.add_(p.grad / 4)
        for total, (_, p) in zip(sums, student.named_selected(('actor',))): p.grad = total
        norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in student.named_selected(('actor',))], 1.))
        assert norm == pytest.approx(stage.metrics[0]['gradient_norm_before_clip'], rel=5e-6, abs=1e-7)
        adam.step()
        assert all(torch.allclose(a, b, rtol=1e-6, atol=2e-8) for (_, a), (_, b)
            in zip(student.named_selected(('actor',)), stage.model.named_selected(('actor',))))
    finally:
        restore_rng(rng, 'cpu')


def worker(rank, folder):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    folder = Path(folder)
    dist.init_process_group('gloo', init_method='file://' + str(folder / 'rendezvous'), rank=rank,
        world_size=4, timeout=timedelta(minutes=5))
    control = dist.new_group(backend='gloo', timeout=timedelta(minutes=5))
    try:
        public = PublicReasoningTrainData('data/public-reasoning-curriculum-v1',
            'reports/public-reasoning-curriculum-v1-readback-admission/complete.json')
        for profile in ['anchor_CE', 'anchor_RL']:
            stage = setup(public, profile, rank, 4, control)
            stage.update()
            if rank == 0: manual_source_mean_matches(stage)
            directory = folder / profile / 'durable'
            first = commit(stage, directory)
            saved, receipt, pointer = durable.read_latest(directory)
            assert len({value_digest(w['rng']) for w in saved['private_workers']}) == 4
            wanted_draws = draws()
            wanted = stage.update()
            before = live(stage)
            bad = copy.deepcopy(saved)
            if rank == 2: bad['private_workers'][0]['rng'] = copy.deepcopy(saved['private_workers'][1]['rng'])
            with pytest.raises(ValueError): stage.resume(bad, expected_manifest=receipt['semantic_manifest'])
            assert live(stage) == before
            bad = copy.deepcopy(saved)
            bad['private_workers'][3]['rng']['torch'] = torch.zeros(1, dtype=torch.uint8)
            with pytest.raises(ValueError): stage.resume(bad, expected_manifest=manifest(bad))
            assert live(stage) == before
            # Root-only serialization failure must be coordinated, with no live mutation.
            original_export = stage.model.export
            def fail_export(*args, **kwargs): raise OSError('deliberate root snapshot serialization failure')
            if rank == 0: stage.model.export = fail_export
            with pytest.raises(ValueError, match='serialization'): stage.snapshot()
            if rank == 0: stage.model.export = original_export
            assert live(stage) == before
            durable.restore_latest(stage, directory, expected_pointer_sha256=pointer)
            assert draws() == wanted_draws and stage.update() == wanted
            assert all(live(stage)[k] == before[k] for k in ['actor', 'optimizer', 'rng', 'completed', 'metrics'])
            tail = stage.update()
            assert (tail['actual_rank_observations'][rank]['kernel'] is None) == (rank > 0)
            assert tail['actual_original_source_positions'] == positions(public)[8:9]
            final = commit(stage, directory)
            # A genuine peer-only NaN in the gradient rescore must retain every
            # sampled action on all ranks and abort before any shared Adam step.
            stage.resume(saved, expected_manifest=receipt['semantic_manifest'])
            pre_failure = live(stage)
            original_forward = stage.model.forward
            def nonfinite_training_only(*args, **kwargs):
                out = original_forward(*args, **kwargs)
                return dict(out, logits=torch.full_like(out['logits'], float('nan'))) if stage.model.training else out
            if rank == 2: stage.model.forward = nonfinite_training_only
            with pytest.raises(ValueError, match='whole source sampling'): stage.update()
            if rank == 2: stage.model.forward = original_forward
            now = live(stage)
            assert stage.failed and all(now[k] == pre_failure[k] for k in ['actor', 'optimizer', 'rng', 'completed', 'metrics', 'tensor_identities'])
            assert all(p.grad is None for p in stage.model.parameters())
            failure = stage.failure_observations[-1]
            observation = failure['actual_rank_errors'][2]['complete_actual_failure_observation']
            assert observation['nonfinite_gradient_forward_behavior_observed'] and len(observation['trajectories']) == 4
            json.dumps(failure, allow_nan=False)
            # Recover the final common durable file after the failed attempt.
            durable.restore_latest(stage, directory, allow_completed=True)
            assert not stage.failed and stage.completed == 3
            assert all(float(s['step']) == 3 for s in stage.optimizer.state.values())
            shared = stage.gather(dict(actor=tensor_digest(stage.model.parts.named_parameters()),
                Adam=value_digest(stage.optimizer.state_dict()), metrics=value_digest(stage.metrics)))
            assert all(v == shared[0] for v in shared) and not torch.cuda.is_initialized()
            (folder / (profile + '-rank-' + str(rank) + '.json')).write_text(json.dumps(dict(rank=rank, profile=profile,
                actual_student_values=stage.model.counts()['actual_unique'], actual_unique_global_source_groups=3,
                actual_source_presentations=9, genuine_final_local_source_count=1 if rank == 0 else 0,
                actual_ordinary_Adam_states=len(stage.optimizer.state), final_Adam_clocks=3,
                actual_physical_Adam_calls=4, root_serial_full_source_mean_matches=True,
                all_actual_actions_verifiers_unclipped_gradients_weights_Adam_private_RNG_replay_exact=True,
                distinct_private_global_RNGs_retained=4, peer_only_valid_RNG_swap_rejected=True,
                peer_only_nonfinite_gradient_forward_coordinated_before_Adam=True,
                complete_nonfinite_sample_observation_retained_on_every_rank=True,
                root_only_snapshot_error_coordinated_before_mutation=True,
                final_generation=final['generation'], final_checkpoint_sha256=final['checkpoint_sha256'],
                full_size_GPU_capacity_or_model_quality=False, production_updates=0), indent=2))
    finally:
        dist.destroy_process_group()


def test_actual_four_replicas_source_mean_empty_tail_private_recovery_and_peer_faults(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    proofs = [json.loads((tmp_path / (p + '-rank-' + str(r) + '.json')).read_text())
        for p in ['anchor_CE', 'anchor_RL'] for r in range(4)]
    assert len(proofs) == 8
    for profile in ['anchor_CE', 'anchor_RL']:
        assert len({p['final_checkpoint_sha256'] for p in proofs if p['profile'] == profile}) == 1
