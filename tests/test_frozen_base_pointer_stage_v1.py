import copy
import os
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.frozen_base_pointer_stage_v1 import FrozenBasePointerStage, make_head, make_optimizer
from slm_lab.own_code_delta_v1 import OwnCodeDelta
from slm_lab.point_grounding_evaluation_v1 import state_digest
from test_own_code_delta_v1 import own_models, components
from test_point_grounding_stage_v1 import options
from test_deterministic_vision_resize_v1 import ResizeFixture


def own_head():
    base, _ = own_models()
    return make_head(base, rank=8)


def settings(world=1, micro=2):
    return dict(options(world=world, micro=micro), vision_learning_rate=0., image_size=32)


def test_learning_and_reference_only_resume_preserve_general_images_code_and_existing_code_cache(tmp_path):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    base, target = own_models()
    model = OwnCodeDelta(base, components(base, target, 2), reserved_parameters=256)
    model.eval(); ids = torch.tensor([[1, 4, 7]]); pixels = torch.linspace(-1, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
    with torch.no_grad():
        original = {route: model.generate(ids, pixels, route=route, max_new_tokens=3).clone() for route in ['general', 'code']}
        cache = model.code.prefill(ids, pixels)['cache']
    head = make_head(model.base, rank=8)
    all_base = {n: t.clone() for n, t in head.base.state_dict().items()}
    versions = [p._version for p in head.base.parameters()]
    combined = torch.nn.ModuleDict(dict(language=model, grounding=head))
    assert sum(p.numel() for p in combined.parameters()) == model.counts()['total_with_reserved_head']
    opts = settings(); data = ResizeFixture()
    stage = FrozenBasePointerStage(head, make_optimizer(head, opts), data, opts, {'base_checkpoint_sha256': 'a' * 64}, 'cpu')
    initial = {n: t.clone() for n, t in head.state_dict().items() if not n.startswith('base.')}
    stage.update(); stage.save(tmp_path / 'resume.pt', 2 * 1024**2, 0)
    saved = torch.load(tmp_path / 'resume.pt', map_location='cpu', weights_only=True)
    assert set(saved['projections']) == {'query_projection.weight', 'patch_projection.weight'}
    assert 'model' not in saved and len(saved['optimizer']['state']) == 2
    first = stage.update(); expected = {n: t.clone() for n, t in head.state_dict().items()}
    stage.resume(saved); again = stage.update()
    assert first == again and all(torch.equal(t, head.state_dict()[n]) for n, t in expected.items())
    for _ in range(stage.completed, stage.steps):
        stage.update()
    assert all(torch.equal(t, head.base.state_dict()[n]) for n, t in all_base.items())
    assert versions == [p._version for p in head.base.parameters()]
    assert any(not torch.equal(t, head.state_dict()[n]) for n, t in initial.items())
    assert all(p.grad is None for p in model.base.parameters())
    assert all(p.grad is None for p in model.parts.parameters())
    model.assert_partition(); model.eval()
    with torch.no_grad():
        for route in ['general', 'code']:
            assert torch.equal(original[route], model.generate(ids, pixels, route=route, max_new_tokens=3))
        # A pre-existing code cache remains valid because no backbone tensor
        # is copied or restored during the head-only recovery.
        model.code.decode(torch.tensor([[8]]), cache)


def test_corrupt_projection_Adam_source_and_rng_reject_before_live_mutation(tmp_path):
    head = own_head(); data = ResizeFixture(); opts = settings()
    stage = FrozenBasePointerStage(head, make_optimizer(head, opts), data, opts, {'base_checkpoint_sha256': 'b' * 64}, 'cpu')
    stage.update(); stage.save(tmp_path / 'resume.pt', 2 * 1024**2, 0)
    saved = torch.load(tmp_path / 'resume.pt', weights_only=True); stage.update()
    expected = {n: t.clone() for n, t in head.state_dict().items()}; rng = torch.get_rng_state().clone()
    for mutate in [lambda s: s['sources'].update(base_checkpoint_sha256='c' * 64),
                   lambda s: s['projections']['query_projection.weight'].fill_(float('nan')),
                   lambda s: s['optimizer']['state'][0]['exp_avg_sq'].fill_(-1),
                   lambda s: s['sampler_state'].update(source_exposures=0),
                   lambda s: s['worker_rng_states'].pop()]:
        bad = copy.deepcopy(saved); mutate(bad)
        with pytest.raises(ValueError):
            stage.resume(bad)
        assert stage.completed == 2 and torch.equal(rng, torch.get_rng_state())
        assert all(torch.equal(t, head.state_dict()[n]) for n, t in expected.items())
    data.purpose = 'evaluation'
    with pytest.raises(ValueError, match='optimization role'):
        FrozenBasePointerStage(head, make_optimizer(head, opts), data, opts, {}, 'cpu')


def worker(rank, folder):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True); folder = Path(folder)
    dist.init_process_group('gloo', init_method='file://' + str(folder / 'rendezvous'), rank=rank, world_size=4)
    try:
        opts = settings(world=4); head = own_head(); before = state_digest(head.base)
        kernel = FrozenBasePointerStage(head, make_optimizer(head, opts), ResizeFixture(), opts,
                                        {'base_checkpoint_sha256': 'd' * 64}, 'cpu')
        kernel.update(); kernel.save(folder / 'resume.pt', 2 * 1024**2, 0)
        uninterrupted = kernel.update(); expected = {n: t.clone() for n, t in head.state_dict().items()}
        fresh = own_head()
        resumed = FrozenBasePointerStage(fresh, make_optimizer(fresh, opts), ResizeFixture(), opts,
                                         {'base_checkpoint_sha256': 'd' * 64}, 'cpu')
        resumed.resume(torch.load(folder / 'resume.pt', map_location='cpu', weights_only=True))
        assert resumed.update() == uninterrupted
        assert all(torch.equal(t, fresh.state_dict()[n]) for n, t in expected.items())
        assert state_digest(fresh.base) == before
        if rank == 0:
            torch.save(expected, folder / 'actual-four-worker.pt')
    finally:
        dist.destroy_process_group()


def test_actual_four_worker_reference_only_recovery_and_full_query_mean_match_global_batches(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    torch.set_num_threads(1); opts = settings(micro=8); head = own_head()
    stage = FrozenBasePointerStage(head, make_optimizer(head, opts), ResizeFixture(), opts,
                                  {'base_checkpoint_sha256': 'e' * 64}, 'cpu')
    for _ in range(stage.steps):
        stage.update()
    actual = torch.load(tmp_path / 'actual-four-worker.pt', weights_only=True)
    for name, value in head.state_dict().items():
        torch.testing.assert_close(value, actual[name], atol=2e-6, rtol=2e-5, msg=name)
