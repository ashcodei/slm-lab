from pathlib import Path
import random
import threading

import numpy as np
import pytest
import torch
from torch import distributed as dist
from torch import multiprocessing as mp

from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint, capture_rng
from slm_lab.distributed_async_dense_state_v1 import DistributedAsyncDenseState
from slm_lab.distributed_dense_code_state_v1 import restore
from slm_lab.distributed_dense_code_step_v1 import gather, update
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_distributed_dense_code_v1 import adam, SOURCES
from test_own_dense_growth_v1 import model as own_model
from test_public_reasoned_code_train_data_v1 import packet


def worker(rank, root, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=4)
    try:
        random.seed(900+rank); np.random.seed(900+rank); torch.manual_seed(900+rank)
        data = PublicReasonedCodeSFTData(root); model, _ = own_model(); optimizer = adam(model)
        first = update(model, optimizer, data, [0, 1, 2], allow_mechanics=True)
        entered = threading.Event(); release = threading.Event()
        def delayed(path, state, maximum, reserve):
            entered.set(); assert release.wait(60)
            atomic_checkpoint(path, state, maximum, reserve)
        writer = DistributedAsyncDenseState(maximum_snapshot_tensor_bytes=10*1024**2, writer=delayed)
        path = Path(root)/'state.pt'
        writer.save(model, optimizer, path, sources=SOURCES, completed=1, planned=3,
                    metrics=[first], device='cpu', maximum_file_bytes=10*1024**2, minimum_free_bytes=0)
        assert not path.exists() and writer.check() is None
        if rank == 0:
            assert entered.wait(60)
        def draws():
            return random.random(), float(np.random.random()), float(torch.rand(()))
        expected_draws = draws()
        # Actually finish the next distributed Adam update while disk I/O is
        # gated, including the three empty source ranks of this real tail.
        second = update(model, optimizer, data, [2], allow_mechanics=True)
        assert second['rank_source_counts'] == [1, 0, 0, 0]
        target = (tensor_digest(model.named_parameters()), value_digest(optimizer.state_dict()), value_digest(capture_rng('cpu')))
        if rank == 0:
            assert writer.writer.pending is not None and not writer.writer.pending.done()
            release.set()
        receipt = writer.drain()
        assert receipt['completed'] == 1 and receipt['sources'] == SOURCES
        assert len(set(gather(receipt['sha256']))) == 1
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['common']['model']['network']['embedding.weight'].data_ptr() == saved['common']['model']['network']['lm_head.weight'].data_ptr()
        fresh, _ = own_model(); fresh_adam = adam(fresh)
        done, rows = restore(fresh, fresh_adam, saved, sources=SOURCES, planned=3, device='cpu')
        assert done == 1 and rows == [first] and draws() == expected_draws
        assert update(fresh, fresh_adam, data, [2], allow_mechanics=True) == second
        assert target == (tensor_digest(fresh.named_parameters()), value_digest(fresh_adam.state_dict()), value_digest(capture_rng('cpu')))
        # Reject a one-rank cursor or destination disagreement collectively,
        # before collecting a new private snapshot or changing any live state.
        for destination, completed in [(path, 3 if rank == 2 else 2),
                                       (None if rank == 1 else path, 2)]:
            with pytest.raises((ValueError, TypeError)):
                writer.save(fresh, fresh_adam, destination, sources=SOURCES, completed=completed, planned=3,
                            metrics=[first, second], device='cpu', maximum_file_bytes=10*1024**2, minimum_free_bytes=0)
            assert target == (tensor_digest(fresh.named_parameters()), value_digest(fresh_adam.state_dict()), value_digest(capture_rng('cpu')))
        # A later background disk failure reaches every rank; its prior durable
        # model/moments/cursor survive and the receipt remains at update one.
        original = path.read_bytes()
        writer.save(fresh, fresh_adam, path, sources=SOURCES, completed=2, planned=3,
                    metrics=[first, second], device='cpu', maximum_file_bytes=100, minimum_free_bytes=0)
        with pytest.raises(ValueError, match='Collective'):
            writer.drain()
        assert writer.last_commit == receipt and path.read_bytes() == original
        with pytest.raises(ValueError, match='Collective'):
            writer.close()
        assert target == (tensor_digest(fresh.named_parameters()), value_digest(fresh_adam.state_dict()), value_digest(capture_rng('cpu')))
        (Path(root)/f'async-rank-{rank}.passed').write_text('actual overlapped update, exact recovery and collective failure passed\n')
    finally:
        dist.destroy_process_group()


def test_all_four_real_replicas_advance_during_delayed_IO_then_recover_full_Adam_and_private_rng(tmp_path):
    packet(tmp_path)
    mp.spawn(worker, args=(str(tmp_path), str(tmp_path/'rendezvous')), nprocs=4, join=True)
    assert len(list(tmp_path.glob('async-rank-*.passed'))) == 4
