import random
import threading

import pytest
import torch

from slm_lab.async_durable_checkpoint_v1 import AsyncCheckpoint, freeze_cpu
from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint, capture_rng


def test_background_writer_retains_original_weights_moments_metrics_and_rng(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    def gated(path, state, maximum, reserve):
        entered.set()
        assert release.wait(10)
        atomic_checkpoint(path, state, maximum, reserve)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    model(torch.ones(1, 3)).square().sum().backward(); optimizer.step()
    metrics = [dict(step=1, positions=[3, 4])]
    bundle = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), metrics=metrics, rng=capture_rng('cpu'))
    expected = freeze_cpu(bundle)
    writer = AsyncCheckpoint(writer=gated)
    path = tmp_path/'resume.pt'
    writer.submit(path, bundle, 1024**2, 0)
    assert entered.wait(10) and not path.exists()
    # Advance a real optimizer while its earlier snapshot has not been written.
    optimizer.zero_grad(); model(torch.ones(1, 3)).square().sum().backward(); optimizer.step()
    metrics[0]['positions'][0] = -1; bundle['rng']['torch'].zero_()
    release.set(); writer.close()
    actual = torch.load(path, weights_only=True)
    def equal(a, b):
        if isinstance(a, torch.Tensor): return torch.equal(a, b)
        if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (tuple, list)): return type(a) is type(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b
    assert equal(actual, expected)


def test_failed_background_atomic_write_preserves_previous_commit_and_propagates(tmp_path):
    path = tmp_path/'resume.pt'
    atomic_checkpoint(path, dict(step=1), 1024**2, 0)
    original = path.read_bytes()
    writer = AsyncCheckpoint()
    writer.submit(path, dict(step=2, large=torch.ones(1024)), 100, 0)
    with pytest.raises((ValueError, RuntimeError)): writer.flush()
    with pytest.raises((ValueError, RuntimeError)): writer.close()
    assert path.read_bytes() == original and not path.with_suffix('.pt.pending').exists()


def test_snapshot_does_not_advance_random_generators(tmp_path):
    before = capture_rng('cpu')
    writer = AsyncCheckpoint()
    writer.submit(tmp_path/'resume.pt', before, 1024**2, 0); writer.close()
    after = capture_rng('cpu')
    assert before['python'] == after['python'] and before['numpy'] == after['numpy']
    assert torch.equal(before['torch'], after['torch'])
    with pytest.raises(ValueError, match='closed'): writer.submit(tmp_path/'x', {}, 100, 0)
