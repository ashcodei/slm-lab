"""Real small-own-neural durable commits, injected I/O failures and four ranks."""
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

from slm_lab import joint_visual_durable_state_v2 as durable
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.corpus import file_hash
from slm_lab.joint_visual_private_checkpoint_v2 import FixedGuard, private_rng_sha256
from slm_lab.joint_visual_objective_update_v1 import update
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_joint_visual_private_checkpoint_v2 import fixture, adam, sources, context, prepare, draws, same_draws, OPTIONS


class Trial:
    def __init__(self, rank=0, world=1):
        self.view, self.batch = fixture()
        self.optimizer = adam(self.view)
        self.fixed = FixedGuard(self.view, self.optimizer)
        self.sources = sources(self.fixed)
        self.entries = context(self.batch, world, world == 4)
        self.rank, self.world = rank, world
        self.metrics = []

    def step(self):
        kind = ['native', 'photo', 'word'][len(self.metrics)]
        prepared, source = prepare(self.batch, kind, self.rank, self.world, self.world == 4)
        row = update(self.view, self.optimizer, prepared, source, OPTIONS)
        self.metrics.append(row)
        return row

    def commit(self, folder):
        return durable.commit(self.view, self.optimizer, self.fixed, folder, sources=self.sources,
            entries=self.entries, completed=len(self.metrics), metrics=self.metrics, mode='capacity',
            rank=self.rank, world=self.world, maximum_bytes=8 * 1024**2, minimum_free_bytes=0)

    def restore(self, folder, **kwargs):
        completed, self.metrics, receipt = durable.restore_latest(self.view, self.optimizer, self.fixed, folder,
            sources=self.sources, entries=self.entries, mode='capacity', rank=self.rank, world=self.world, **kwargs)
        assert completed == len(self.metrics)
        return receipt


def live(trial):
    return (tensor_digest(trial.view.actor.named_parameters()), value_digest(trial.optimizer.state_dict()),
        private_rng_sha256(capture_rng('cpu')))


def test_three_generations_keep_old_mmap_and_restore_all_actual_selected_Adam_and_RNG(tmp_path):
    trial = Trial()
    trial.step()
    first = trial.commit(tmp_path)
    old, _, old_pointer = durable.read_latest(tmp_path)
    old_values = tensor_digest(old['common']['selected_weights'].items())
    trial.step()
    second = trial.commit(tmp_path)
    trial.step()
    third = trial.commit(tmp_path)
    assert [x['generation'] for x in [first, second, third]] == [1, 2, 3]
    assert [x['slot'] for x in [first, second, third]] == [1, 0, 1]
    assert old['common']['cursor']['completed'] == 1
    assert tensor_digest(old['common']['selected_weights'].items()) == old_values
    expected = live(trial)
    cold = Trial()
    before = live(cold)
    with pytest.raises(ValueError):
        cold.restore(tmp_path, expected_pointer_sha256=old_pointer)
    assert live(cold) == before
    with pytest.raises(ValueError):
        cold.restore(tmp_path)
    assert live(cold) == before
    cold.restore(tmp_path, allow_completed=True)
    assert len(cold.metrics) == 3 and live(cold) == expected
    shared, receipt, _ = durable.read_latest(tmp_path)
    assert set(shared) == {'format', 'common', 'private_states'}
    assert len(shared['private_states']) == 1
    assert set(shared['private_states'][0]) == durable.PRIVATE_FIELDS
    assert receipt['common_selected_weights_and_Adam_stored_once']
    assert 'frozen_weights' not in shared['common']
    assert not torch.cuda.is_initialized()


def test_actual_next_source_update_from_file_is_exact(tmp_path):
    trial = Trial()
    trial.step()
    trial.step()
    trial.commit(tmp_path)
    wanted_record = trial.step()
    wanted_live = live(trial)
    wanted_draws = draws()
    resumed = Trial()
    resumed.restore(tmp_path)
    assert resumed.step() == wanted_record and live(resumed) == wanted_live
    assert same_draws(wanted_draws, draws())


@pytest.mark.parametrize('fail_at', ['receipt', 'pointer'])
def test_publication_failure_keeps_previous_complete_generation_and_live_neural_values(tmp_path, monkeypatch, fail_at):
    trial = Trial()
    trial.step()
    first = trial.commit(tmp_path)
    pointer = file_hash(tmp_path / 'LATEST.json')
    trial.step()
    before = live(trial)
    writer = durable.durable_json
    def fail(path, value, reserve):
        if (fail_at == 'receipt' and Path(path).name == 'receipt-slot-0.json'
                or fail_at == 'pointer' and Path(path).name == 'LATEST.json'):
            raise OSError('Deliberate own-test publication-fault')
        return writer(path, value, reserve)
    monkeypatch.setattr(durable, 'durable_json', fail)
    with pytest.raises(ValueError, match='publication-fault'):
        trial.commit(tmp_path)
    assert live(trial) == before and file_hash(tmp_path / 'LATEST.json') == pointer
    saved, receipt, _ = durable.read_latest(tmp_path)
    assert saved['common']['cursor']['completed'] == receipt['completed'] == 1
    assert receipt['checkpoint_sha256'] == first['checkpoint_sha256']
    monkeypatch.setattr(durable, 'durable_json', writer)
    assert trial.commit(tmp_path)['completed'] == 2


@pytest.mark.parametrize('artifact', ['receipt-slot-1.json', 'state-slot-1.pt'])
def test_corrupt_committed_bytes_reject_before_neural_or_private_mutation(tmp_path, artifact):
    trial = Trial()
    trial.step()
    trial.commit(tmp_path)
    trial.step()
    before = live(trial)
    path = tmp_path / artifact
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError):
        trial.restore(tmp_path)
    assert live(trial) == before


def worker(rank, folder):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.nice(0) >= 19
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + str(Path(folder) / 'rendezvous'),
        rank=rank, world_size=4, timeout=timedelta(minutes=5))
    try:
        trial = Trial(rank, 4)
        random.seed(1100 + rank)
        np.random.seed(1200 + rank)
        torch.manual_seed(1300 + rank)
        directory = Path(folder) / 'durable'
        trial.step()
        first = trial.commit(directory)
        first_pointer = file_hash(directory / 'LATEST.json')
        trial.step()
        before = live(trial)
        writer = durable.durable_json
        def fail(path, value, reserve):
            if Path(path).name == 'receipt-slot-0.json':
                raise OSError('Actual root-only publication-fault')
            return writer(path, value, reserve)
        if rank == 0:
            durable.durable_json = fail
        with pytest.raises(ValueError, match='publication-fault'):
            trial.commit(directory)
        assert live(trial) == before and file_hash(directory / 'LATEST.json') == first_pointer
        if rank == 0:
            durable.durable_json = writer
        second = trial.commit(directory)
        second_pointer = file_hash(directory / 'LATEST.json')
        assert first['generation'] == 1 and second['generation'] == 2
        expected_record = trial.step()
        expected_live = live(trial)
        expected_draws = draws()
        resumed = Trial(rank, 4)
        resumed.restore(directory, expected_pointer_sha256=second_pointer)
        assert resumed.step() == expected_record and live(resumed) == expected_live
        assert same_draws(expected_draws, draws())
        before = live(resumed)
        with pytest.raises(ValueError):
            resumed.restore(directory, expected_pointer_sha256='f' * 64 if rank == 1 else second_pointer)
        assert live(resumed) == before
        shared, receipt, _ = durable.read_latest(directory)
        assert len(shared['private_states']) == 4
        assert all(set(value) == durable.PRIVATE_FIELDS for value in shared['private_states'])
        hashes = {value['private_RNG_sha256'] for value in shared['private_states']}
        assert len(hashes) == 4
        assert receipt['checkpoint_sha256'] == second['checkpoint_sha256']
        assert not torch.cuda.is_initialized()
        (Path(folder) / (str(rank) + '.json')).write_text(json.dumps(dict(complete=True, rank=rank,
            actual_small_own_parameters=trial.view.actor.counts()['actual_unique'],
            common_selected_tensor_values_stored_once=sum(t.numel() for t in shared['common']['selected_weights'].values()),
            four_distinct_private_RNG_boundaries_preserved=True, exact_next_source_model_Adam_and_all_CPU_RNG=True,
            root_only_IO_failure_rejected_everywhere_before_any_mutation=True,
            old_complete_generation_preserved_on_publication_failure=True,
            one_rank_read_failure_coordinated_before_neural_mutation=True,
            committed_checkpoint_sha256=second['checkpoint_sha256'],
            full_size_GPU_durable_capacity_training_and_model_quality_pending=True, production_updates=0), indent=2) + '\n')
    finally:
        dist.destroy_process_group()


def test_four_actual_replicas_share_one_tensor_file_and_coordinate_failures_and_recovery(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    proofs = [json.loads((tmp_path / (str(rank) + '.json')).read_text()) for rank in range(4)]
    assert all(p['complete'] for p in proofs)
    assert len({p['committed_checkpoint_sha256'] for p in proofs}) == 1
