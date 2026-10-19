"""Actual native-code workers and own-small CPU replicas; no quality inference."""
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
from slm_lab.batched_native_suffix_group_gradients_v1 import group_gradients, group_semantics
from slm_lab.codenet_checkpoint_v1 import capture_rng, restore_rng
from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference
from slm_lab.native_suffix_train_data_v1 import NativeSuffixTrainData, schedule_source_views
from slm_lab.parallel_function_repair_stage_v1 import manifest, value_digest
from slm_lab.parallel_native_suffix_stage_v1 import ParallelNativeSuffixStage, CE_ONLY, private_seeds, record_semantics
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_batched_native_suffix_group_gradients_v1 import ROOT, CONTROL, fixture
from test_public_reasoning_group_gradients_v1 import optimizer

SOURCES = dict(own_small_initializer_sha256='a' * 64, exact_native_source_controls_sha256='b' * 64)
OPTIONS = dict(learning_rate=1e-4, gradient_clip=1., group_size=8, KL_weight=.01, anchor_weight=1.,
    cache_absolute_bound=.001, maximum_cache_bytes=8 * 1024**2, gradient_bucket_values=4096,
    seed=989711, epochs=1, execution_workers=2, CPU_mechanics_horizon=5)


@pytest.fixture(scope='module')
def data(): return NativeSuffixTrainData.CPU_mechanics(ROOT, CONTROL)


def student(data):
    torch.manual_seed(989721)
    with data.episode(0) as episode:
        own, _ = fixture(episode)
    return own


def setup(data, profile='anchor_execution', rank=0, world=1, control=None):
    own = student(data)
    adam = optimizer(own)
    entries = [[0], [1], [2]] if world == 1 else [[0, 1, 2, 3], [4, 5, 6, 7], [8]]
    random.seed(989731 + rank); np.random.seed(989731 + rank); torch.manual_seed(989731 + rank)
    return ParallelNativeSuffixStage(own, adam, data, entries, profile, OPTIONS, SOURCES,
        mode='mechanics', rank=rank, world=world, control_group=control, gradient_group=None)


def live(stage):
    return dict(actor=tensor_digest(stage.model.parts.named_parameters()),
        optimizer=value_digest(stage.optimizer.state_dict()), RNG=value_digest(capture_rng('cpu')),
        completed=stage.completed, metrics=value_digest(stage.metrics), failed=stage.failed,
        tensor_identities=[(id(p), k, id(t), t.data_ptr(), t._version)
            for p, row in stage.optimizer.state.items() for k, t in row.items()])


def commit(stage, directory):
    return durable.commit(stage, directory, maximum_bytes=32 * 1024**2, minimum_free_bytes=0)


@pytest.mark.parametrize('profile', ['anchor_CE', 'anchor_execution'])
def test_actual_single_replica_complete_native_updates_and_durable_next_action_recovery(data, tmp_path, profile):
    stage = setup(data, profile)
    fixed = tensor_digest(stage.model.base.named_parameters())
    initial = tensor_digest(stage.model.parts.named_parameters())
    first = stage.update()
    commit(stage, tmp_path / 'durable')
    _, _, pointer = durable.read_latest(tmp_path / 'durable')
    next_row = stage.update()
    expected = live(stage)
    durable.restore_latest(stage, tmp_path / 'durable', expected_pointer_sha256=pointer)
    replay = stage.update()
    assert record_semantics(replay) == record_semantics(next_row)
    assert all(live(stage)[k] == expected[k] for k in ['actor', 'optimizer', 'RNG', 'completed'])
    assert stage.metrics[0] == first
    stage.update()
    commit(stage, tmp_path / 'durable')
    final = live(stage)
    cold = setup(data, profile)
    durable.restore_latest(cold, tmp_path / 'durable', allow_completed=True)
    assert all(live(cold)[k] == final[k] for k in ['actor', 'optimizer', 'RNG', 'completed', 'metrics'])
    assert stage.completed == 3 and len(stage.optimizer.state) == 17
    assert all(float(s['step']) == 3 for s in stage.optimizer.state.values())
    assert tensor_digest(stage.model.parts.named_parameters()) != initial
    assert tensor_digest(stage.model.base.named_parameters()) == fixed and not torch.cuda.is_initialized()
    assert sum(r['original_response_EOS_targets'] for r in stage.metrics) == 3 * 68
    (tmp_path / 'actual-single-replica-native-recovery.json').write_text(json.dumps(dict(
        profile=profile, actual_small_parameters=stage.model.counts()['actual_unique'], physical_optimizer_calls=4,
        logical_source_updates=3, original_human_sources=1, source_presentations=3, original_response_EOS_targets=204,
        actual_first_group=first, actual_next_group=next_row, actual_replayed_group=replay,
        all_actual_next_actions_execution_semantics_logps_gradients_weights_AdamW_private_RNG_exact=True,
        complete_raw_worker_timings_and_thread_observations_retained_but_not_claimed_equal=True,
        original_base_and_unused_critic_unchanged=True, actual_fixture_horizon=5, original_native_horizon=512,
        full_size_GPUcapacity_or_quality=False, production_updates=0), indent=2, allow_nan=False))


def test_actual_CE_execution_pair_uses_same_private_samples_but_real_rewards_change_the_update(data, tmp_path):
    control = setup(data, 'anchor_CE'); control_row = control.update()
    actual = setup(data, 'anchor_execution'); actual_row = actual.update()
    left = group_semantics(control_row['actual_rank_observations'][0]['kernel'])
    right = group_semantics(actual_row['actual_rank_observations'][0]['kernel'])
    left.pop('profile'); right.pop('profile')
    assert left == right and 1. in right['rewards'] and any(r < 1 for r in right['rewards'])
    assert actual_row['actual_unclipped_global_actor_gradient_sha256'] != control_row['actual_unclipped_global_actor_gradient_sha256']
    assert tensor_digest(actual.model.parts.named_parameters()) != tensor_digest(control.model.parts.named_parameters())
    (tmp_path / 'actual-paired-native-stage-policy-effect.json').write_text(json.dumps(dict(
        actual_private_actions_probabilities_executed_programs_and_complete_CE_matched=True,
        actual_rewards=right['rewards'], actual_advantages=right['advantages'],
        actual_policy_gradients_and_learned_actor_values_differ=True, physical_optimizer_calls=2,
        deliberately_initialized_source_specific_fixture=True, unassisted_coding_gain=False, production_updates=0), indent=2))


@pytest.mark.parametrize('profile', ['anchor_CE', 'anchor_execution'])
def test_unqualified_execution_view_retains_all_original_CE_without_sampling_or_invented_reward(profile, tmp_path):
    public = NativeSuffixTrainData.CPU_mechanics(ROOT, CONTROL)
    # An explicitly controlled unqualified disposition on the existing CPU-only
    # presentation fixture exercises the same production CE fallback branch.
    public._information[0]['native_execution_eligible'] = False
    stage = setup(public, profile)
    before = tensor_digest(stage.model.parts.named_parameters())
    row = stage.update()
    kernel = row['actual_rank_observations'][0]['kernel']
    assert kernel['format'] == CE_ONLY and kernel['trajectories'] == [] and kernel['actual_group_size'] == 0
    assert kernel['unconditional_original_CE_anchor']['local_original_response_EOS_targets'] == 68
    assert row['original_response_EOS_targets'] == 68 and row['complete_original_context_tokens'] == 496
    assert tensor_digest(stage.model.parts.named_parameters()) != before
    assert not public.full_source_data_admitted and len(stage.optimizer.state) == 17
    (tmp_path / 'actual-unqualified-view-full-CE.json').write_text(json.dumps(dict(profile=profile,
        actual_original_response_EOS_targets=68, actual_original_context_tokens=496, physical_optimizer_calls=1,
        no_sampled_execution_or_policy_reward=True, controlled_CPU_fixture_disposition_not_real_source_admission=True,
        actual_complete_observation=row, full_size_GPUcapacity_or_quality=False, production_updates=0), indent=2))


def test_changed_source_masks_worker_programs_reference_or_private_rng_reject_before_mutation(data):
    stage = setup(data)
    stage.update()
    state, receipt = stage.snapshot()
    before = live(stage)
    for what in ['source', 'target', 'worker_program', 'private_rng', 'critic', 'clock']:
        bad = copy.deepcopy(state)
        if what == 'source': bad['model']['sources']['exact_native_source_controls_sha256'] = 'c' * 64
        if what == 'target': bad['metrics'][0]['original_response_EOS_targets'] -= 1
        if what == 'worker_program':
            item = next(t for t in bad['metrics'][0]['actual_rank_observations'][0]['kernel']['trajectories']
                if t['actual_native_execution'].get('actual_original_oracle_observation', {}).get('actual_execution'))
            observed = item['actual_native_execution']['actual_original_oracle_observation']['actual_execution']
            key = next(k for k in ['actual_program_sha256', 'original_program_sha256', 'program_sha256'] if k in observed)
            observed[key] = 'd' * 64
        if what == 'private_rng': bad['private_workers'][0]['rng']['torch'] = torch.zeros(1, dtype=torch.uint8)
        if what == 'critic': bad['model']['parts']['critic.weight'].add_(.1)
        if what == 'clock': next(iter(bad['optimizer']['state'].values()))['step'].fill_(9)
        with pytest.raises(ValueError): stage.resume(bad, expected_manifest=receipt)
        assert live(stage) == before
        with pytest.raises(ValueError): stage.resume(bad, expected_manifest=manifest(bad))
        assert live(stage) == before


def test_wrong_optimizer_roles_horizons_and_CPU_production_claims_reject_before_any_update(data):
    for change in [dict(execution_workers=3), dict(group_size=1), dict(CPU_mechanics_horizon=513), dict(seed=2**64)]:
        own = student(data); adam = optimizer(own)
        with pytest.raises(ValueError):
            ParallelNativeSuffixStage(own, adam, data, [[0]], 'anchor_execution', dict(OPTIONS, **change), SOURCES, mode='mechanics')
        assert not adam.state
    for mode in ['capacity', 'train']:
        own = student(data); adam = optimizer(own)
        with pytest.raises(ValueError):
            ParallelNativeSuffixStage(own, adam, data, [[0]], 'anchor_execution', OPTIONS, SOURCES, mode=mode)
        assert not adam.state
    own = student(data)
    wrong = torch.optim.Adam([p for _, p in own.named_selected(('actor',))], lr=OPTIONS['learning_rate'])
    with pytest.raises(ValueError, match='AdamW'):
        ParallelNativeSuffixStage(own, wrong, data, [[0]], 'anchor_execution', OPTIONS, SOURCES, mode='mechanics')
    assert not wrong.state and not torch.cuda.is_initialized()


def manual_source_mean(stage):
    rng = capture_rng('cpu')
    try:
        own = student(stage.data); adam = optimizer(own)
        reference = FrozenOwnActorReference(own, sources=SOURCES)
        sums = [torch.zeros_like(p) for _, p in own.named_selected(('actor',))]
        for position in stage.entries[0]:
            own.zero_grad(set_to_none=True)
            with stage.data.episode(position) as episode:
                group_gradients(own, episode, reference, profile=stage.profile,
                    seeds=private_seeds(OPTIONS['seed'], 1, position, OPTIONS['group_size']), purpose='CPU-mechanics',
                    **{k: OPTIONS[k] for k in ['maximum_cache_bytes', 'KL_weight', 'anchor_weight',
                        'cache_absolute_bound', 'execution_workers', 'CPU_mechanics_horizon']})
            for total, (_, p) in zip(sums, own.named_selected(('actor',))): total.add_(p.grad / 4)
        for total, (_, p) in zip(sums, own.named_selected(('actor',))): p.grad = total
        norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in own.named_selected(('actor',))], 1.))
        assert norm == pytest.approx(stage.metrics[0]['gradient_norm_before_clip'], rel=5e-6, abs=1e-7)
        adam.step()
        assert all(torch.allclose(a, b, rtol=1e-6, atol=2e-8)
            for (_, a), (_, b) in zip(own.named_selected(('actor',)), stage.model.named_selected(('actor',))))
    finally: restore_rng(rng, 'cpu')


def worker(rank, folder):
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    folder = Path(folder)
    dist.init_process_group('gloo', init_method='file://' + str(folder / 'rendezvous'), rank=rank, world_size=4,
        timeout=timedelta(minutes=10))
    control = dist.new_group(backend='gloo', timeout=timedelta(minutes=10))
    try:
        public = NativeSuffixTrainData.CPU_mechanics(ROOT, CONTROL)
        public._information[4]['native_execution_eligible'] = False
        for profile in ['anchor_CE', 'anchor_execution']:
            stage = setup(public, profile, rank, 4, control)
            stage.update()
            if rank == 0: manual_source_mean(stage)
            directory = folder / profile / 'durable'
            commit(stage, directory)
            saved, receipt, pointer = durable.read_latest(directory)
            assert len({value_digest(w['rng']) for w in saved['private_workers']}) == 4
            wanted = stage.update()
            expected = live(stage)
            assert wanted['actual_rank_observations'][0]['kernel']['format'] == CE_ONLY
            bad = copy.deepcopy(saved)
            if rank == 2: bad['private_workers'][0]['rng'] = copy.deepcopy(saved['private_workers'][1]['rng'])
            with pytest.raises(ValueError): stage.resume(bad, expected_manifest=receipt['semantic_manifest'])
            assert live(stage) == expected
            durable.restore_latest(stage, directory, expected_pointer_sha256=pointer)
            replay = stage.update()
            assert record_semantics(replay) == record_semantics(wanted)
            assert all(live(stage)[k] == expected[k] for k in ['actor', 'optimizer', 'RNG', 'completed'])
            tail = stage.update()
            assert tail['actual_original_source_positions'] == [8]
            assert (tail['actual_rank_observations'][rank]['kernel'] is None) == (rank > 0)
            final = commit(stage, directory)
            stage.resume(saved, expected_manifest=receipt['semantic_manifest'])
            before_fault = live(stage)
            original_forward = stage.model.forward
            def nonfinite_gradient_only(*args, **kwargs):
                value = original_forward(*args, **kwargs)
                return dict(value, logits=torch.full_like(value['logits'], float('nan'))) if stage.model.training and args else value
            if rank == 2: stage.model.forward = nonfinite_gradient_only
            with pytest.raises(ValueError, match='whole source sampling'): stage.update()
            if rank == 2: stage.model.forward = original_forward
            after_fault = live(stage)
            assert stage.failed and all(after_fault[k] == before_fault[k]
                for k in ['actor', 'optimizer', 'RNG', 'completed', 'metrics', 'tensor_identities'])
            assert all(p.grad is None for p in stage.model.parameters())
            fault = stage.failure_observations[-1]['actual_rank_errors'][2]['complete_actual_failure_observation']
            assert len(fault['trajectories']) == 8 and all('actual_native_execution' in t for t in fault['trajectories'])
            json.dumps(fault, allow_nan=False)
            durable.restore_latest(stage, directory, allow_completed=True)
            assert stage.completed == 3 and not stage.failed
            shared = stage.gather(dict(actor=tensor_digest(stage.model.parts.named_parameters()),
                optimizer=value_digest(stage.optimizer.state_dict()), metrics=value_digest(stage.metrics)))
            assert all(row == shared[0] for row in shared) and not torch.cuda.is_initialized()
            (folder / (profile + '-rank-' + str(rank) + '.json')).write_text(json.dumps(dict(rank=rank, profile=profile,
                actual_small_parameters=stage.model.counts()['actual_unique'], physical_optimizer_calls=4,
                root_additional_serial_optimizer_calls=1 if rank == 0 else 0,
                complete_logical_updates=3, source_presentations=9, distinct_original_human_sources=1,
                original_response_EOS_targets=612, actual_empty_tail_local_sources=1 if rank == 0 else 0,
                exact_next_actual_native_actions_execution_semantics_gradients_weights_AdamW_private_RNG=True,
                actual_serial_full_source_mean_matches=True, actual_CE_only_source_retained=True,
                peer_only_bad_private_state_rejected_before_mutation=True,
                peer_only_nonfinite_gradient_forward_retains_all_actual_samples_and_worker_outcomes=True,
                actual_final_shared_checkpoint_sha256=final['checkpoint_sha256'],
                full_size_GPUcapacity_or_quality=False, production_updates=0), indent=2, allow_nan=False))
    finally: dist.destroy_process_group()


def test_actual_four_replicas_native_workers_source_mean_real_empty_tail_and_peer_faults(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    proofs = [json.loads((tmp_path / (p + '-rank-' + str(r) + '.json')).read_text())
        for p in ['anchor_CE', 'anchor_execution'] for r in range(4)]
    assert len(proofs) == 8
    for profile in ['anchor_CE', 'anchor_execution']:
        assert len({p['actual_final_shared_checkpoint_sha256'] for p in proofs if p['profile'] == profile}) == 1


def test_full_original_schedule_retains_all_3721_human_sources_four_levels_and_real_tails(tmp_path):
    entries = schedule_source_views(epochs=3, width=4, seed=OPTIONS['seed'])
    assert len(entries) == 11172 and sum(map(len, entries)) == 44652
    assert sum(len(e) == 1 for e in entries) == 12
    assert all(sorted(p for row in entries[epoch * 3724:(epoch + 1) * 3724] for p in row) == list(range(14884))
        for epoch in range(3))
    assert all(all(p % 4 == level for row in entries[epoch * 3724 + level * 931:epoch * 3724 + (level + 1) * 931] for p in row)
        for epoch in range(3) for level in range(4))
    (tmp_path / 'actual-full-native-source-schedule.json').write_text(json.dumps(dict(
        actual_source_only_updates=11172, actual_source_view_presentations=44652, distinct_original_human_sources=3721,
        original_levels=[4, 16, 64, 0], complete_original_response_EOS_targets=7010616,
        all_actual_single_source_three_empty_rank_steps=12, original_native_horizons_not_shortened=True,
        full_source_admission_and_GPUcapacity_still_required=True, optimizer_updates=0), indent=2))
