"""Actual small-own-neural selected checkpoint boundaries, never production admission."""
import copy
from datetime import timedelta
import io
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.joint_visual_private_checkpoint_v1 import FixedGuard, SOURCE_KEYS, make_state, restore_state, private_rng_sha256
from slm_lab.joint_visual_objective_update_v1 import update, named_groups, optimizer_clocks
from test_joint_visual_objective_update_v1 import fixture, payload, adam, OPTIONS, equal_optimizer
from test_global_word_microbatch_stage_v2 import spec as word_spec, subset


def prepare(batch, kind, rank=0, world=1, tail=False):
    if kind != 'word' or not tail:
        return payload(batch, kind, rank, world)
    native, source = payload(batch, 'native', rank, world)
    whole = subset(batch, [0])
    source = dict(word_spec(whole, [17]), family='words')
    if native is not None:
        native['word'] = dict(word_pixels=whole['word_pixels'], texts=whole['texts'])
    return native, source


def context(batch, world=1, tail=False):
    entries = []
    for step, kind in enumerate(['native', 'photo', 'word'], 1):
        _, source = prepare(batch, kind, 0, world, tail)
        positions = source['global_positions']
        entries.append(dict(step=step, family=source['family'], original_local_positions=positions,
            complete_context_tokens=sum(source['global_context_lengths']),
            response_EOS_targets=sum(source['global_response_EOS_targets']),
            proposed_rank_source_counts=[len(positions[r::world]) for r in range(world)]))
    return entries


def sources(fixed):
    return {k: fixed.sha256 if k == 'fixed_values_sha256' else 'a' * 64 for k in SOURCE_KEYS}


def serialized(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    stream.seek(0)
    return torch.load(stream, map_location='cpu', weights_only=True)


def draws():
    return random.random(), float(np.random.rand()), torch.rand(11)


def same_draws(left, right):
    return left[:2] == right[:2] and torch.equal(left[2], right[2])


@pytest.mark.parametrize('bf16', [False, True])
def test_selected_only_snapshot_restores_exact_next_cross_family_Adam_and_all_CPU_RNG(bf16):
    view, batch = fixture()
    optimizer = adam(view)
    fixed = FixedGuard(view, optimizer)
    bound = sources(fixed)
    entries = context(batch)
    options = dict(OPTIONS, precision='bf16' if bf16 else 'fp32')
    random.seed(731)
    np.random.seed(732)
    torch.manual_seed(733)
    metrics = []
    for kind in ['native', 'photo']:
        prepared, source = prepare(batch, kind)
        metrics.append(update(view, optimizer, prepared, source, options))
    state = serialized(make_state(view, optimizer, fixed, sources=bound, entries=entries,
        completed=2, metrics=metrics, mode='capacity'))
    committed_RNG_sha = state['private_RNG_sha256']
    selected = {n: p for rows in named_groups(view).values() for n, p in rows}
    assert state['optimized_names'] == list(selected) == list(state['selected_weights'])
    assert sum(v.numel() for v in state['selected_weights'].values()) == sum(p.numel() for p in selected.values())
    assert sum(v.numel() for v in state['selected_weights'].values()) < view.actor.counts()['actual_unique']
    assert not {'model', 'actor', 'backbone', 'network', 'frozen_weights'} & set(state)
    assert state['cursor']['group_clocks'] == dict(vision=2, pointing=1, word=0)
    prepared, source = prepare(batch, 'word')
    expected_record = update(view, optimizer, prepared, source, options)
    expected_draws = draws()
    resumed, _ = fixture()
    resumed_optimizer = adam(resumed)
    resumed_fixed = FixedGuard(resumed, resumed_optimizer)
    completed, recovered_metrics = restore_state(resumed, resumed_optimizer, resumed_fixed, state,
        sources=bound, entries=entries, mode='capacity', expected_completed=2,
        expected_private_RNG_sha256=committed_RNG_sha)
    assert completed == 2 and recovered_metrics == metrics
    assert update(resumed, resumed_optimizer, prepared, source, options) == expected_record
    assert same_draws(expected_draws, draws())
    for name, value in view.actor.state_dict().items():
        assert torch.equal(value, resumed.actor.state_dict()[name])
    equal_optimizer(optimizer, resumed_optimizer)
    fixed.check(view, optimizer)
    resumed_fixed.check(resumed, resumed_optimizer)
    assert optimizer_clocks(resumed, resumed_optimizer) == dict(vision=3, pointing=1, word=1)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('case', ['missing_moment', 'wrong_sparse_clock', 'invented_inactive_word_state',
    'nonfinite_weight', 'negative_second_moment', 'wrong_committed_cursor', 'wrong_source',
    'valid_foreign_RNG_and_rehashed_header', 'changed_future_schedule', 'capacity_snapshot_as_production'])
def test_invalid_private_envelopes_reject_before_weights_Adam_or_global_RNG_mutate(case):
    view, batch = fixture()
    optimizer = adam(view)
    fixed = FixedGuard(view, optimizer)
    bound = sources(fixed)
    entries = context(batch)
    prepared, source = prepare(batch, 'native')
    metric = update(view, optimizer, prepared, source, OPTIONS)
    state = serialized(make_state(view, optimizer, fixed, sources=bound, entries=entries,
        completed=1, metrics=[metric], mode='capacity'))
    expected_RNG = state['private_RNG_sha256']
    expected_completed, mode = 1, 'capacity'
    key = next(iter(state['optimizer']['state']))
    if case == 'missing_moment':
        del state['optimizer']['state'][key]
    elif case == 'wrong_sparse_clock':
        state['optimizer']['state'][key]['step'].add_(1)
    elif case == 'invented_inactive_word_state':
        word_key = state['optimizer']['param_groups'][2]['params'][0]
        state['optimizer']['state'][word_key] = copy.deepcopy(state['optimizer']['state'][key])
    elif case == 'nonfinite_weight':
        next(iter(state['selected_weights'].values())).reshape(-1)[0] = float('nan')
    elif case == 'negative_second_moment':
        state['optimizer']['state'][key]['exp_avg_sq'].reshape(-1)[0] = -1
    elif case == 'wrong_committed_cursor':
        expected_completed = 2
    elif case == 'wrong_source':
        bound = dict(bound, run_plan_sha256='b' * 64)
    elif case == 'valid_foreign_RNG_and_rehashed_header':
        state['private_RNG']['torch'] = torch.Generator().manual_seed(982173).get_state()
        state['private_RNG_sha256'] = private_rng_sha256(state['private_RNG'])
        assert state['private_RNG_sha256'] != expected_RNG
    elif case == 'changed_future_schedule':
        entries[-1]['original_local_positions'][0] += 100
    elif case == 'capacity_snapshot_as_production':
        mode = 'train'
    fresh, _ = fixture()
    fresh_optimizer = adam(fresh)
    fresh_fixed = FixedGuard(fresh, fresh_optimizer)
    before = {n: p.detach().clone() for n, p in fresh.named_parameters()}
    before_RNG = private_rng_sha256(capture_rng('cpu'))
    before_options = copy.deepcopy(fresh_optimizer.state_dict()['param_groups'])
    with pytest.raises(ValueError, match='Coordinated private visual checkpoint rejection'):
        restore_state(fresh, fresh_optimizer, fresh_fixed, state, sources=bound, entries=entries, mode=mode,
            expected_completed=expected_completed, expected_private_RNG_sha256=expected_RNG)
    assert all(torch.equal(before[n], p) and p.grad is None for n, p in fresh.named_parameters())
    assert not fresh_optimizer.state and fresh_optimizer.state_dict()['param_groups'] == before_options
    assert private_rng_sha256(capture_rng('cpu')) == before_RNG


def worker(rank, folder):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.nice(0) >= 19
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + str(Path(folder) / 'rendezvous'), rank=rank,
        world_size=4, timeout=timedelta(minutes=5))
    try:
        view, batch = fixture()
        optimizer = adam(view)
        fixed = FixedGuard(view, optimizer)
        bound = sources(fixed)
        entries = context(batch, 4, True)
        random.seed(800 + rank)
        np.random.seed(900 + rank)
        torch.manual_seed(1000 + rank)
        metrics = []
        for kind in ['native', 'photo']:
            prepared, source = prepare(batch, kind, rank, 4, True)
            metrics.append(update(view, optimizer, prepared, source, OPTIONS))
        state = serialized(make_state(view, optimizer, fixed, sources=bound, entries=entries,
            completed=2, metrics=metrics, mode='capacity', rank=rank, world=4))
        external_RNG_sha = state['private_RNG_sha256']
        peer_RNGs = [None] * 4
        dist.all_gather_object(peer_RNGs, state['private_RNG'])
        assert len({private_rng_sha256(r) for r in peer_RNGs}) == 4
        prepared, source = prepare(batch, 'word', rank, 4, True)
        expected_record = update(view, optimizer, prepared, source, OPTIONS)
        expected_draws = draws()
        resumed, _ = fixture()
        resumed_optimizer = adam(resumed)
        resumed_fixed = FixedGuard(resumed, resumed_optimizer)
        completed, restored_metrics = restore_state(resumed, resumed_optimizer, resumed_fixed, state,
            sources=bound, entries=entries, mode='capacity', rank=rank, world=4,
            expected_completed=2, expected_private_RNG_sha256=external_RNG_sha)
        assert completed == 2 and restored_metrics == metrics
        actual_record = update(resumed, resumed_optimizer, prepared, source, OPTIONS)
        assert actual_record == expected_record and same_draws(expected_draws, draws())
        assert actual_record['local_objective_record']['actual_native_forward_calls'] == (1 if rank == 0 else 0)
        assert all(torch.equal(value, resumed.actor.state_dict()[name]) for name, value in view.actor.state_dict().items())
        equal_optimizer(optimizer, resumed_optimizer)
        for control in ['valid_peer_RNG', 'missing_active_moment']:
            bad = copy.deepcopy(state)
            if rank == 1:
                if control == 'valid_peer_RNG':
                    bad['private_RNG'] = peer_RNGs[0]
                    bad['private_RNG_sha256'] = private_rng_sha256(bad['private_RNG'])
                else:
                    del bad['optimizer']['state'][next(iter(bad['optimizer']['state']))]
            before = {n: p.detach().clone() for n, p in resumed.named_parameters()}
            before_optimizer = copy.deepcopy(resumed_optimizer.state_dict())
            before_RNG = private_rng_sha256(capture_rng('cpu'))
            with pytest.raises(ValueError, match='Coordinated private visual checkpoint rejection'):
                restore_state(resumed, resumed_optimizer, resumed_fixed, bad, sources=bound, entries=entries,
                    mode='capacity', rank=rank, world=4, expected_completed=2,
                    expected_private_RNG_sha256=external_RNG_sha)
            assert all(torch.equal(before[n], p) for n, p in resumed.named_parameters())
            assert before_optimizer['param_groups'] == resumed_optimizer.state_dict()['param_groups']
            for key, value in before_optimizer['state'].items():
                for field, tensor in value.items():
                    assert torch.equal(tensor, resumed_optimizer.state_dict()['state'][key][field])
            assert private_rng_sha256(capture_rng('cpu')) == before_RNG
        assert not torch.cuda.is_initialized()
        (Path(folder) / (str(rank) + '.json')).write_text(json.dumps(dict(complete=True, rank=rank,
            actual_small_own_parameters=view.actor.counts()['actual_unique'],
            actual_saved_selected_values=sum(v.numel() for v in state['selected_weights'].values()),
            actual_private_RNG_sha256=external_RNG_sha, all_four_RNGs_distinct=True,
            exact_next_cross_family_Adam_weights_and_all_CPU_RNG=True, three_empty_tail_ranks_without_forward=True,
            two_one_rank_corruptions_rejected_everywhere_before_mutation=True,
            frozen_language_actor_critic_and_auxiliary_weights_not_copied=True,
            full_size_GPU_durable_commit_controller_and_quality_pending=True, production_updates=0), indent=2) + '\n')
    finally:
        dist.destroy_process_group()


def test_four_actual_own_replicas_selected_private_recovery_and_coordinated_rejection(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    proofs = [json.loads((tmp_path / (str(rank) + '.json')).read_text()) for rank in range(4)]
    assert all(p['complete'] and p['exact_next_cross_family_Adam_weights_and_all_CPU_RNG'] for p in proofs)
