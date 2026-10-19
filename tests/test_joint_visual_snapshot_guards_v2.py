"""Reject invalid publication while retaining recovery from a valid checkpoint."""
import ast
from datetime import timedelta
import json
import os
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab import joint_visual_private_checkpoint_v2 as checkpoint
from slm_lab import joint_visual_durable_state_v2 as durable
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_joint_visual_private_checkpoint_v2 import fixture, adam, sources, context
from test_joint_visual_durable_state_v2 import Trial, live


@pytest.mark.parametrize('case', ['nonfinite_weight', 'negative_lr', 'zero_lr', 'zero_epsilon',
    'negative_decay', 'negative_beta', 'beta_above_one', 'incomplete_betas'])
def test_invalid_live_snapshot_cannot_publish_a_complete_pointer_or_change_any_live_value(tmp_path, case):
    trial = Trial()
    if case == 'nonfinite_weight':
        with torch.no_grad():
            trial.optimizer.param_groups[0]['params'][0].reshape(-1)[0] = float('nan')
    elif case == 'negative_lr':
        trial.optimizer.param_groups[0]['lr'] = -1.
    elif case == 'zero_lr':
        trial.optimizer.param_groups[0]['lr'] = 0.
    elif case == 'zero_epsilon':
        trial.optimizer.param_groups[0]['eps'] = 0.
    elif case == 'negative_decay':
        trial.optimizer.param_groups[0]['weight_decay'] = -1.
    elif case == 'negative_beta':
        trial.optimizer.param_groups[1]['betas'] = (-.1, .95)
    elif case == 'beta_above_one':
        trial.optimizer.param_groups[1]['betas'] = (.9, 1.1)
    else:
        trial.optimizer.param_groups[1]['betas'] = (.9,)
    before = live(trial)
    with pytest.raises(ValueError, match='Coordinated private visual checkpoint rejection'):
        trial.commit(tmp_path / 'unpublished')
    assert live(trial) == before
    assert not (tmp_path / 'unpublished' / 'LATEST.json').exists()
    assert not list((tmp_path / 'unpublished').glob('*.pt'))


def test_valid_committed_state_can_repair_damaged_current_selected_values(tmp_path):
    trial = Trial()
    trial.commit(tmp_path)
    wanted = live(trial)
    with torch.no_grad():
        trial.optimizer.param_groups[0]['params'][0].reshape(-1)[0] = float('nan')
    trial.restore(tmp_path)
    assert live(trial) == wanted


def test_every_original_private_and_durable_test_function_and_threshold_is_identical():
    root = Path(__file__).resolve().parents[1]
    for stem in ['joint_visual_private_checkpoint', 'joint_visual_durable_state']:
        trees = [ast.parse((root / 'tests' / ('test_' + stem + '_v' + str(version) + '.py')).read_text())
            for version in [1, 2]]
        def functions(tree):
            return [ast.dump(node) for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        assert functions(trees[0]) == functions(trees[1])


def worker(rank, folder):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.nice(0) >= 19
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + str(Path(folder) / 'rendezvous'),
        rank=rank, world_size=4, timeout=timedelta(minutes=5))
    try:
        trial = Trial(rank, 4)
        if rank == 1:
            with torch.no_grad():
                trial.optimizer.param_groups[0]['params'][0].reshape(-1)[0] = float('nan')
        before = live(trial)
        directory = Path(folder) / 'unpublished'
        with pytest.raises(ValueError, match='Coordinated private visual checkpoint rejection'):
            trial.commit(directory)
        assert live(trial) == before and not (directory / 'LATEST.json').exists()
        assert not list(directory.glob('*.pt'))
        assert not torch.cuda.is_initialized()
        (Path(folder) / (str(rank) + '.json')).write_text(json.dumps(dict(complete=True, rank=rank,
            one_rank_nonfinite_live_weight_rejected_by_every_actual_peer=True,
            no_tensor_checkpoint_receipt_or_pointer_published=True,
            actual_selected_weights_Adam_and_all_CPU_RNG_unchanged=True,
            no_existing_neural_job_changed=True, full_size_GPU_visual_capacity_pending=True), indent=2) + '\n')
    finally:
        dist.destroy_process_group()


def test_one_rank_invalid_live_weight_is_rejected_everywhere_before_publication(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    assert all(json.loads((tmp_path / (str(rank) + '.json')).read_text())['complete'] for rank in range(4))
