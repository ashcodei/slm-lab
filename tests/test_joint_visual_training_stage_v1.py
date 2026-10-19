"""Actual own-small-neural sequencing, source rejection and durable continuation."""
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.corpus import file_hash
from slm_lab.joint_visual_objective_update_v1 import update, named_groups, optimizer_clocks
from slm_lab.joint_visual_private_checkpoint_v2 import FixedGuard, digest
from slm_lab.joint_visual_training_stage_v1 import JointVisualStage, make_plan
from test_joint_visual_private_checkpoint_v2 import (fixture, adam, prepare, context, sources,
    draws, same_draws, equal_optimizer, OPTIONS)


class OwnSmallMechanicsSources:
    """Explicit test-only original fixture delivery; never a full source adapter."""
    def __init__(self, batch, *, tail=False):
        self.batch, self.tail = batch, tail
        self.seen = []
        self.failure = None

    def prepare(self, entry, *, rank, world, device):
        assert torch.device(device).type == 'cpu'
        if self.failure == 'decode':
            raise ValueError('deliberate own-small-fixture decode failure')
        kind = {'images:additional_commons': 'native',
            'images:additional_photo_captions': 'photo', 'words': 'word'}[entry['family']]
        payload, source = prepare(self.batch, kind, rank, world, self.tail)
        self.seen.append(copy.deepcopy(entry))
        if self.failure == 'source':
            source['global_context_lengths'][0] += 1
        return payload, source


def stage_fixture(world=1, rank=0, tail=False, bf16=False, mode='mechanics'):
    view, batch = fixture()
    optimizer = adam(view)
    fixed = FixedGuard(view, optimizer)
    bound = sources(fixed)
    entries = context(batch, world, tail)
    options = dict(OPTIONS, precision='bf16' if bf16 else 'fp32')
    plan = make_plan(view, optimizer, fixed, sources={k: v for k, v in bound.items() if k != 'run_plan_sha256'},
        entries=entries, options=options, mode=mode)
    bound['run_plan_sha256'] = digest(plan)
    provider = OwnSmallMechanicsSources(batch, tail=tail)
    stage = JointVisualStage(view, optimizer, provider, plan=plan, sources=bound, entries=entries,
        rank=rank, world=world, fixed=fixed)
    return stage, batch


def weights(view):
    return {name: p.detach().clone() for name, p in view.named_parameters()}


def equal_weights(before, view):
    assert all(torch.equal(value, dict(view.named_parameters())[name]) for name, value in before.items())


@pytest.mark.parametrize('bf16', [False, True])
def test_actual_stage_matches_direct_objectives_and_exact_durable_next_source(tmp_path, bf16):
    stage, batch = stage_fixture(tail=True, bf16=bf16)
    reference, _ = fixture()
    reference_optimizer = adam(reference)
    selected = {id(p) for rows in named_groups(stage.view).values() for _, p in rows}
    frozen = {n: p.detach().clone() for n, p in stage.view.named_parameters() if id(p) not in selected}
    with torch.no_grad():
        text = stage.view.actor(batch['input_ids'], attention_mask=batch['attention_mask'])['logits'].clone()
    random.seed(420)
    np.random.seed(421)
    torch.manual_seed(422)
    stage.commit(tmp_path, maximum_bytes=4 * 1024**2, minimum_free_bytes=0)
    for kind in ['native', 'photo']:
        payload, source = prepare(batch, kind, tail=True)
        expected = update(reference, reference_optimizer, payload, source, stage._options)
        assert stage.next_update() == expected
    assert stage.completed == 2 and stage.next_entry['family'] == 'words'
    stage.commit(tmp_path, maximum_bytes=4 * 1024**2, minimum_free_bytes=0)
    pointer = file_hash(tmp_path / 'LATEST.json')
    metric = stage.next_update()
    expected_draws = draws()
    resumed, _ = stage_fixture(tail=True, bf16=bf16)
    resumed.restore(tmp_path, expected_pointer_sha256=pointer)
    assert resumed.completed == 2 and resumed.metrics == stage.metrics[:2]
    assert resumed.next_update() == metric and same_draws(expected_draws, draws())
    equal_weights(weights(stage.view), resumed.view)
    equal_optimizer(stage.optimizer, resumed.optimizer)
    equal_weights(frozen, stage.view)
    with torch.no_grad():
        assert torch.equal(text, stage.view.actor(batch['input_ids'], attention_mask=batch['attention_mask'])['logits'])
    assert stage.completed == 3 and stage.next_entry is None
    with pytest.raises(ValueError, match='healthy exact source-prefix'):
        stage.next_update()
    # Returned records/entries cannot rewrite the internal completed prefix.
    exported = stage.metrics
    exported[0]['family'] = 'words'
    assert stage.metrics[0]['family'] == 'images:additional_commons'
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('case', ['consistent_wrong_clock', 'metrics', 'learning_rate', 'options', 'future_source',
    'plan', 'bindings', 'mode', 'cursor', 'pending_gradient'])
def test_changed_boundary_rejects_before_any_neural_forward_or_optimizer_commit(case):
    stage, _ = stage_fixture()
    stage.next_update()
    if case == 'consistent_wrong_clock':
        for _, p in named_groups(stage.view)['vision']:
            stage.optimizer.state[p]['step'].add_(1)
    elif case == 'metrics':
        stage._metrics[0]['global_context_tokens'] += 1
    elif case == 'learning_rate':
        stage.optimizer.param_groups[0]['lr'] *= 2
    elif case == 'options':
        stage._options['matching_weight'] *= 2
    elif case == 'future_source':
        stage._entries[-1]['original_local_positions'][0] += 1
    elif case == 'plan':
        stage._plan['candidate_promoted'] = True
    elif case == 'bindings':
        stage._sources['base_checkpoint_sha256'] = 'b' * 64
    elif case == 'mode':
        stage.mode = 'train'
    elif case == 'cursor':
        stage.completed = 2
    elif case == 'pending_gradient':
        next(stage.view.parameters()).grad = torch.ones_like(next(stage.view.parameters()))
    before = weights(stage.view)
    before_optimizer = copy.deepcopy(stage.optimizer.state_dict())
    seen = list(stage.provider.seen)
    calls = []
    hook = stage.view.native.vision.register_forward_hook(lambda *args: calls.append(1))
    with pytest.raises(ValueError, match='Coordinated private visual checkpoint rejection'):
        stage.next_update()
    hook.remove()
    assert not calls and stage.provider.seen == seen
    equal_weights(before, stage.view)
    after = stage.optimizer.state_dict()
    assert before_optimizer['param_groups'] == after['param_groups']
    for key, row in before_optimizer['state'].items():
        assert all(torch.equal(value, after['state'][key][name]) for name, value in row.items())


@pytest.mark.parametrize('mode', ['capacity', 'train'])
def test_own_small_stage_cannot_be_presented_as_full_size_GPU_capacity_or_training(mode):
    with pytest.raises(ValueError, match='Coordinated private visual checkpoint rejection'):
        stage_fixture(mode=mode)
    assert not torch.cuda.is_initialized()


def test_changed_plan_and_settings_reject_before_durable_restore_mutates_values(tmp_path):
    stage, _ = stage_fixture()
    stage.next_update()
    stage.commit(tmp_path, maximum_bytes=4 * 1024**2, minimum_free_bytes=0)
    pointer = file_hash(tmp_path / 'LATEST.json')
    fresh, _ = stage_fixture()
    fresh._options['gradient_clip'] *= 2
    before = weights(fresh.view)
    with pytest.raises(ValueError, match='Coordinated private visual checkpoint rejection'):
        fresh.restore(tmp_path, expected_pointer_sha256=pointer)
    equal_weights(before, fresh.view)
    assert not fresh.optimizer.state and fresh.completed == 0


def worker(rank, folder):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.nice(0) >= 19
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + str(Path(folder) / 'rendezvous'),
        rank=rank, world_size=4, timeout=timedelta(minutes=5))
    try:
        stage, _ = stage_fixture(world=4, rank=rank, tail=True)
        random.seed(900 + rank)
        np.random.seed(901 + rank)
        torch.manual_seed(902 + rank)
        stage.next_update()
        stage.next_update()
        directory = Path(folder) / 'shared'
        receipt = stage.commit(directory, maximum_bytes=4 * 1024**2, minimum_free_bytes=0)
        pointer = file_hash(directory / 'LATEST.json')
        for control in ['wrong_clock', 'one_bad_pointer', 'decode', 'source']:
            if rank == 1 and control == 'wrong_clock':
                for _, p in named_groups(stage.view)['vision']:
                    stage.optimizer.state[p]['step'].add_(1)
            if rank == 2 and control in ['decode', 'source']:
                stage.provider.failure = control
            before = weights(stage.view)
            calls = []
            hook = stage.view.native.vision.register_forward_hook(lambda *args: calls.append(1))
            with pytest.raises(ValueError, match='Coordinated private visual checkpoint rejection'):
                if control == 'one_bad_pointer':
                    stage.restore(directory, expected_pointer_sha256='bad' if rank == 1 else pointer)
                else:
                    stage.next_update()
            hook.remove()
            assert not calls
            equal_weights(before, stage.view)
            assert stage.completed == 2
            stage.provider.failure = None
            stage.restore(directory, expected_pointer_sha256=pointer)
        actual = stage.next_update()
        expected_draws = draws()
        assert actual['local_objective_record']['actual_native_forward_calls'] == (1 if rank == 0 else 0)
        resumed, _ = stage_fixture(world=4, rank=rank, tail=True)
        resumed.restore(directory, expected_pointer_sha256=pointer)
        assert resumed.completed == 2 and resumed.next_entry['family'] == 'words'
        assert resumed.next_update() == actual and same_draws(expected_draws, draws())
        equal_weights(weights(stage.view), resumed.view)
        equal_optimizer(stage.optimizer, resumed.optimizer)
        assert optimizer_clocks(stage.view, stage.optimizer) == dict(vision=3, pointing=1, word=1)
        assert not torch.cuda.is_initialized()
        (Path(folder) / (str(rank) + '.json')).write_text(json.dumps(dict(complete=True, rank=rank,
            actual_own_small_parameters=stage.view.actor.counts()['actual_unique'],
            actual_completed_source_entries=stage.completed,
            exact_next_source_Adam_weights_metrics_and_private_RNG=True,
            actual_shared_checkpoint_sha256=receipt['checkpoint_sha256'],
            one_rank_wrong_clock_pointer_decode_and_source_errors_rejected_everywhere_before_forward=True,
            singleton_word_with_three_real_empty_ranks=True,
            new_full_size_visual_GPU_updates=0, full_size_GPU_capacity_and_runtime_admission_pending=True), indent=2) + '\n')
    finally:
        dist.destroy_process_group()


def test_four_real_own_small_stages_recover_and_coordinate_every_pre_kernel_failure(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    proofs = [json.loads((tmp_path / (str(rank) + '.json')).read_text()) for rank in range(4)]
    assert all(p['complete'] and p['exact_next_source_Adam_weights_metrics_and_private_RNG'] for p in proofs)
    assert len({p['actual_shared_checkpoint_sha256'] for p in proofs}) == 1
