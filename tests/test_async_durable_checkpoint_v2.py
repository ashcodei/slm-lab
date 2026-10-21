import threading

import pytest
import torch

from slm_lab.async_durable_checkpoint_v2 import BoundedAsyncCheckpoint, freeze_cpu_bounded
from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint, capture_rng
from slm_lab.corpus import file_hash


def test_tied_vocabulary_aliases_remain_private_and_count_once_under_snapshot_budget(tmp_path):
    table = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    payload = dict(embedding=table, lm_head=table, settings={'depth': [3]})
    snapshot, size = freeze_cpu_bounded(payload, 48)
    assert size == 48 and snapshot['embedding'] is snapshot['lm_head']
    assert snapshot['embedding'].data_ptr() != table.data_ptr()
    table.zero_(); payload['settings']['depth'][0] = 8
    assert torch.equal(snapshot['embedding'], torch.arange(12).reshape(3, 4))
    assert snapshot['settings']['depth'] == [3]
    writer = BoundedAsyncCheckpoint(maximum_snapshot_tensor_bytes=48)
    writer.submit(tmp_path/'alias.pt', snapshot, 1024**2, 0)
    receipt = writer.flush(); writer.close()
    saved = torch.load(tmp_path/'alias.pt', weights_only=True)
    assert saved['embedding'].data_ptr() == saved['lm_head'].data_ptr()
    assert receipt['sha256'] == file_hash(tmp_path/'alias.pt')
    with pytest.raises(ValueError, match='budget'):
        freeze_cpu_bounded(snapshot, 47)
    circular = []; circular.append(circular)
    with pytest.raises(ValueError, match='Recursive'):
        freeze_cpu_bounded(circular, 100)


def test_failed_background_write_keeps_old_commit_and_never_advances_receipt(tmp_path):
    entered = threading.Event(); release = threading.Event()
    def delayed(path, state, maximum, reserve):
        entered.set(); assert release.wait(30)
        atomic_checkpoint(path, state, maximum, reserve)
    path = tmp_path/'state.pt'; atomic_checkpoint(path, {'step': 1}, 1024**2, 0)
    original = path.read_bytes(); before = capture_rng('cpu')
    writer = BoundedAsyncCheckpoint(maximum_snapshot_tensor_bytes=1024, writer=delayed)
    writer.submit(path, dict(step=2, large=torch.ones(64)), 100, 0)
    assert entered.wait(30) and writer.check() is None
    release.set()
    with pytest.raises((ValueError, RuntimeError)):
        writer.flush()
    assert writer.last_commit is None and path.read_bytes() == original
    with pytest.raises((ValueError, RuntimeError)):
        writer.close()
    after = capture_rng('cpu')
    assert before['python'] == after['python'] and before['numpy'] == after['numpy']
    assert torch.equal(before['torch'], after['torch'])
