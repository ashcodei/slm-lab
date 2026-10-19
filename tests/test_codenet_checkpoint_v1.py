from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from slm_lab.codenet_checkpoint_v1 import (FileBindings, atomic_checkpoint, capture_rng,
    restore_rng, save_collective, validate_resume)
from slm_lab.codenet_stage_v1 import EpochOrder, Objective, objective_gradients
from slm_lab.corpus import file_hash


def test_all_cpu_rng_streams_round_trip_without_reusing_the_first_draw():
    random.seed(7)
    np.random.seed(11)
    torch.manual_seed(13)
    before = (random.random(), np.random.random(), torch.rand(1).item())
    state = capture_rng('cpu')
    expected = (random.random(), np.random.random(), torch.rand(1).item())
    assert before != expected
    restore_rng(state, 'cpu')
    assert (random.random(), np.random.random(), torch.rand(1).item()) == expected


def test_failed_atomic_save_preserves_the_previous_checkpoint(tmp_path):
    path = tmp_path / 'resume.pt'
    atomic_checkpoint(path, {'value': torch.tensor([7.])}, 100_000, 0)
    before = path.read_bytes()
    with pytest.raises((ValueError, RuntimeError)):
        atomic_checkpoint(path, {'value': torch.zeros(100_000)}, 10_000, 0)
    assert path.read_bytes() == before
    assert not path.with_suffix('.pt.pending').exists()
    assert torch.load(path, weights_only=True)['value'].item() == 7.


def test_changed_plan_workers_or_completed_budget_cannot_resume():
    bundle = {'training_kind': 'complete_code_rehearsal_v1', 'run_plan_sha256': 'bound',
              'world_size': 2, 'planned_steps': 6, 'steps': 2,
              'worker_rng_states': [{}, {}], 'optimizer': {}}
    assert validate_resume(bundle, 'bound', 2, 6) == 2
    assert validate_resume({**bundle, 'steps': 6}, 'bound', 2, 6, allow_completed=True) == 6
    for key, value in [('run_plan_sha256', 'new'), ('world_size', 4), ('steps', 6),
                       ('planned_steps', 8), ('worker_rng_states', [{}])]:
        with pytest.raises(ValueError, match='Exact resumable'):
            validate_resume({**bundle, key: value}, 'bound', 2, 6)


def test_bound_source_change_stops_the_next_update(tmp_path):
    path = tmp_path / 'source'
    path.write_text('original')
    guard = FileBindings({str(path): file_hash(path)})
    guard.check(full=True)
    before = path.stat()
    path.write_text('modified')
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match='changed during training'):
        guard.check()


class NoiseLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(2, 4)
        self.dropout = torch.nn.Dropout(.35)
        self.last = torch.nn.Linear(4, 1)

    def forward(self, input_ids, labels):
        x = input_ids[:, :-1].float()
        features = torch.stack((x / 10, torch.sin(x)), dim=-1)
        predicted = self.last(self.dropout(self.first(features).tanh())).squeeze(-1)
        mask = labels[:, 1:] != -100
        return {'loss': (predicted[mask] - labels[:, 1:][mask].float()).square().mean()}


class Rows:
    def batch(self, rows, field, device):
        length = max(len(row) for row in rows) + 1
        ids = torch.zeros(len(rows), length, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = torch.tensor([p[0] for p in row])
            labels[i, 1:len(row) + 1] = torch.tensor([p[1] for p in row])
        return {'input_ids': ids.to(device), 'labels': labels.to(device)}


def resume_worker(rank, init_path, folder, fail_save):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + init_path, rank=rank, world_size=2)
    torch.manual_seed(19)
    model = DistributedDataParallel(NoiseLoss())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    torch.manual_seed(101 + rank)
    random.seed(201 + rank)
    np.random.seed(301 + rank)
    primary = [[(1, 2)], [(2, 3), (3, 4), (4, 5)], [(5, 1), (6, 2)], [(7, 3), (8, 4), (9, 5), (10, 6)]]
    replay = [[(3, 1)], [(8, 2), (9, 3)]]
    encoder = Rows()

    def update(step):
        sampler = EpochOrder(len(primary), 41)
        groups = [[primary[i] for i in group] for group in sampler.rank_groups(step, rank, 2, 1, 2)]
        optimizer.zero_grad(set_to_none=True)
        objective_gradients(model, [Objective('code', encoder, groups, 'response', 1.),
            Objective('replay', encoder, [[replay[rank]]], 'response', .5)], 'cpu', nullcontext, world=2)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()

    for step in range(2):
        update(step)

    def bundle(states):
        return {'training_kind': 'complete_code_rehearsal_v1', 'run_plan_sha256': 'bound',
                'world_size': 2, 'planned_steps': 6, 'steps': 2, 'worker_rng_states': states,
                'model': model.module.state_dict(), 'optimizer': optimizer.state_dict()}

    checkpoint = Path(folder) / 'resume.pt'
    if fail_save:
        with pytest.raises(ValueError, match='Collective snapshot failed'):
            save_collective(checkpoint, bundle, 'cpu', 2, rank, 128, 0)
        (Path(folder) / ('failed-' + str(rank))).write_text('same save failure reached this worker')
    else:
        save_collective(checkpoint, bundle, 'cpu', 2, rank, 1_000_000, 0)
        for step in range(2, 6):
            update(step)
        expected = copy.deepcopy(model.module.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        expected_draws = (random.random(), np.random.random(), torch.rand(1).item())
        model = DistributedDataParallel(NoiseLoss())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
        saved = torch.load(checkpoint, weights_only=True)
        start = validate_resume(saved, 'bound', 2, 6)
        assert not torch.equal(saved['worker_rng_states'][0]['torch'], saved['worker_rng_states'][1]['torch'])
        model.module.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['worker_rng_states'][rank], 'cpu')
        for step in range(start, 6):
            update(step)
        for name, value in model.module.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        for key, state in optimizer.state_dict()['state'].items():
            for name, value in state.items():
                torch.testing.assert_close(value, expected_optimizer['state'][key][name], rtol=0, atol=0)
        assert (random.random(), np.random.random(), torch.rand(1).item()) == expected_draws
        (Path(folder) / ('rank-' + str(rank) + '.json')).write_text(json.dumps({
            'exact_model_match': True, 'exact_adam_match': True, 'exact_next_rng_match': True,
            'worker': rank, 'completed_steps': 6}))
    dist.destroy_process_group()


def test_two_worker_restart_matches_uninterrupted_adam_and_rank_distinct_rng(tmp_path):
    mp.spawn(resume_worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path), False), nprocs=2, join=True)
    for rank in range(2):
        result = json.loads((tmp_path / ('rank-' + str(rank) + '.json')).read_text())
        assert result['exact_model_match'] and result['exact_adam_match'] and result['exact_next_rng_match']


def test_rank_zero_snapshot_error_reaches_every_worker(tmp_path):
    mp.spawn(resume_worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path), True), nprocs=2, join=True)
    assert all((tmp_path / ('failed-' + str(rank))).exists() for rank in range(2))
