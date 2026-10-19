from contextlib import nullcontext
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp

from slm_lab.codenet_stage_v1 import EpochOrder, Objective, exposure_budget, objective_gradients, rehearsal_selection


class LossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.7))

    def forward(self, input_ids, labels):
        mask = labels[:, 1:] != -100
        errors = self.weight * input_ids[:, :-1].float() - labels[:, 1:].float()
        return {'loss': errors[mask].square().mean()}


class Rows:
    def batch(self, rows, field, device):
        width = max(len(row) for row in rows) + 1
        ids = torch.zeros(len(rows), width, dtype=torch.long, device=device)
        labels = torch.full_like(ids, -100)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = torch.tensor([pair[0] for pair in row], device=device)
            labels[i, 1:len(row) + 1] = torch.tensor([pair[1] for pair in row], device=device)
        return {'input_ids': ids, 'labels': labels}


PRIMARY = [[(1, 2)], [(2, 5), (3, 7), (4, 10)], [(5, 7), (6, 9)], [(7, 10), (8, 12), (9, 13), (10, 14)]]
REPLAY = [[(3, 8), (4, 9)], [(8, 2)]]


def expected_gradient():
    def gradient(rows):
        pairs = [pair for row in rows for pair in row]
        return sum(2 * (.7 * x - y) * x for x, y in pairs) / len(pairs)
    return gradient(PRIMARY) + .5 * gradient(REPLAY)


def test_primary_and_rehearsal_use_independent_token_means_before_backward():
    model, encoder = LossModel(), Rows()
    result = objective_gradients(model, [Objective('code', encoder, [PRIMARY[:1], PRIMARY[1:]], 'response', 1.),
        Objective('replay', encoder, [REPLAY[:1], REPLAY[1:]], 'response', .5)], 'cpu', nullcontext)
    assert float(model.weight.grad) == pytest.approx(expected_gradient(), rel=1e-6)
    assert result['objectives']['code']['global_target_tokens'] == 10
    assert result['objectives']['replay']['global_target_tokens'] == 3
    assert float(model.weight.detach()) == pytest.approx(.7)  # Backward is not an update.


def distributed_worker(rank, init_path, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + init_path, rank=rank, world_size=2)
    model = DistributedDataParallel(LossModel())
    encoder = Rows()
    groups = [[PRIMARY[rank]], [PRIMARY[rank + 2]]]
    result = objective_gradients(model, [Objective('code', encoder, groups, 'response', 1.),
        Objective('replay', encoder, [[REPLAY[rank]]], 'response', .5)], 'cpu', nullcontext, world=2)
    Path(output + str(rank)).write_text(json.dumps({'gradient': float(model.module.weight.grad), 'counts': result}))
    dist.destroy_process_group()


def test_unequal_rank_and_microbatch_lengths_match_global_analytic_gradient(tmp_path):
    mp.spawn(distributed_worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path / 'rank-')), nprocs=2, join=True)
    for rank in range(2):
        row = json.loads((tmp_path / ('rank-' + str(rank))).read_text())
        assert row['gradient'] == pytest.approx(expected_gradient(), rel=1e-6)
        assert row['counts']['objectives']['code']['global_target_tokens'] == 10
        assert row['counts']['objectives']['replay']['global_target_tokens'] == 3


def test_source_sweeps_are_complete_rank_disjoint_and_resume_addressable():
    sampler = EpochOrder(97, 12)
    draws = []
    for step in range(5):
        ranks = [sampler.rank_groups(step, rank, 4, 2, 3) for rank in range(4)]
        draws.extend(ranks[rank][micro][item] for micro in range(3) for rank in range(4) for item in range(2))
    assert sorted(draws[:97]) == list(range(97))
    assert [EpochOrder(97, 12).at(i) for i in range(240, 264)] == [sampler.at(i) for i in range(240, 264)]
    assert sampler.rank_groups(10, 2, 4, 2, 3) == EpochOrder(97, 12).rank_groups(10, 2, 4, 2, 3)
    budget = exposure_budget(97, 3, 4, 2, 3)
    assert budget['actual_source_exposures'] == 312 and budget['rounding_exposures_above_requested_passes'] == 21


def test_rehearsal_families_rotate_without_skipping_their_own_exposure_order():
    names = ['code', 'math', 'images', 'dialogue']
    assert [rehearsal_selection(i, names) for i in range(8)] == [(name, epoch) for epoch in range(2) for name in names]


def test_invalid_objectives_fail_before_any_gradient_is_computed():
    model, encoder = LossModel(), Rows()
    for coefficient in [0., -1., float('nan'), float('inf')]:
        with pytest.raises(ValueError, match='Finite positive'):
            objective_gradients(model, [Objective('bad', encoder, [[PRIMARY[0]]], 'response', coefficient)], 'cpu', nullcontext)
    assert model.weight.grad is None
