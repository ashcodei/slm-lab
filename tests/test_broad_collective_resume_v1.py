import copy
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import distributed as dist
from torch import multiprocessing as mp

from slm_lab.broad_anchored_execution_stage_v1 import BroadAnchoredExecutionStage, semantic_digest
from slm_lab.broad_collective_resume_v1 import restore
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.corpus import file_hash
from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_broad_interim_snapshot_reader_v1 import Corpus
from test_own_dense_actor_v1 import actor


def worker(rank, root, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=4)
    try:
        from slm_lab import anchored_execution_step_v2 as kernel
        kernel.ExpandedExecutionCorpus = Corpus  # Explicit tiny CPU mechanics corpus only.
        model = actor(); corpus = Corpus(); reference = FrozenOwnActorReference(model, sources={'unit': 'a'*64})
        optimizer = torch.optim.AdamW([p for _, p in model.named_selected(('actor',))], lr=.001,
                                     betas=(.9, .95), weight_decay=.01, foreach=False)
        stage = BroadAnchoredExecutionStage(model, optimizer, corpus, reference, root=Path(root), method='anchor_execution',
            seed=963117, sources={'unit': 'b'*64}, world=4, rank=rank,
            cache_absolute_bound=.001, cache_calibration_sha256='c'*64, CPU_mechanics_horizon=2)
        random.seed(700+rank); np.random.seed(700+rank); torch.manual_seed(700+rank)
        first = stage.update(); saved = stage.state(); path = Path(root)/f'rank-{rank}.pt'
        torch.save(saved, path)
        second = stage.update()
        expected = (tensor_digest(model.named_parameters()), value_digest(optimizer.state_dict()), value_digest(capture_rng('cpu')))
        for damage in ['moment', 'RNG', 'source_cursor', 'critic', 'reward']:
            bad = copy.deepcopy(saved)
            if rank == 2:
                if damage == 'moment':
                    bad['core']['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))
                elif damage == 'RNG':
                    bad['core']['rng']['torch'] = torch.zeros(3, dtype=torch.uint8)
                elif damage == 'source_cursor':
                    bad['core']['cursor']['source_order'][0] = 99
                elif damage == 'critic':
                    bad['core']['model']['parts']['critic.weight'].add_(1)
                else:
                    bad['core']['metrics'][0]['global_rewards'][0] = float('nan')
            wrong = Path(root)/f'rank-{rank}-{damage}.pt'; torch.save(bad, wrong)
            before = [(p._version, p.data_ptr()) for p in model.parameters()]
            with pytest.raises(ValueError, match='Collective'):
                restore(stage, wrong, expected_sha256=file_hash(wrong))
            assert before == [(p._version, p.data_ptr()) for p in model.parameters()]
            assert expected == (tensor_digest(model.named_parameters()), value_digest(optimizer.state_dict()), value_digest(capture_rng('cpu')))
        assert restore(stage, path, expected_sha256=file_hash(path)) == 1
        assert stage.metrics == saved['core']['metrics']
        actual = stage.update()
        assert semantic_digest(actual['trajectories']) == semantic_digest(second['trajectories'])
        assert actual['original_reference_CE_sum'] == second['original_reference_CE_sum']
        assert expected == (tensor_digest(model.named_parameters()), value_digest(optimizer.state_dict()), value_digest(capture_rng('cpu')))
        (Path(root)/f'broad-rank-{rank}.passed').write_text('actual coordinated rejection and exact next update passed\n')
    finally:
        dist.destroy_process_group()


def test_four_real_actor_replicas_reject_one_rank_corruption_before_mutation_and_restore_exactly(tmp_path):
    mp.spawn(worker, args=(str(tmp_path), str(tmp_path/'rendezvous')), nprocs=4, join=True)
    assert len(list(tmp_path.glob('broad-rank-*.passed'))) == 4
