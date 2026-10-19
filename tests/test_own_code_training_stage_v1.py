import copy
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.own_code_delta_v1 import OwnCodeDelta
from slm_lab.own_code_training_stage_v1 import OwnCodeTrainingStage, make_optimizer, components
from test_own_code_delta_v1 import own_models, components as initial_components


class FixtureEncoder:
    def __init__(self, salt):
        self.salt = salt

    def batch(self, rows, field, device):
        assert field == 'response'
        encoded = []
        for row in rows:
            p = row['packed_code_index']
            prefix = [1, 4 + p % 8]
            body = [4 + (p * self.salt + i) % 24 for i in range(1 + (p + self.salt) % 5)] + [2]
            encoded.append((prefix + body, [-100] * len(prefix) + body))
        ids = torch.zeros(len(rows), max(len(pair[0]) for pair in encoded), dtype=torch.long, device=device)
        labels = torch.full_like(ids, -100)
        for i, (tokens, targets) in enumerate(encoded):
            ids[i, :len(tokens)] = torch.tensor(tokens, device=device)
            labels[i, :len(tokens)] = torch.tensor(targets, device=device)
        return dict(input_ids=ids, labels=labels, attention_mask=ids != 0)


class FixtureStream:
    role = 'train'
    purpose = 'optimization'

    def __init__(self, name, size, salt):
        self.name = name; self.size = size; self.encoder = FixtureEncoder(salt)

    def __len__(self):
        return self.size


def fixtures(world=1):
    general, target = own_models()
    model = OwnCodeDelta(general, initial_components(general, target, 2), reserved_parameters=256)
    model.code.gradient_checkpointing = True
    streams = {name: FixtureStream(name, n, i + 1)
               for i, (name, n) in enumerate([('generation', 9), ('repair', 24), ('variety', 31)])}
    streams['variety'].language_positions = {'FixtureA': list(range(15)), 'FixtureB': list(range(15, 31))}
    options = dict(seed=624319, world_size=world, microbatch_per_rank=4 // world, accumulation=2,
        global_examples_per_stream=8, steps=3, learning_rate=.0002, betas=[.9, .95], weight_decay=.01,
        gradient_norm_clip=1., warmup_fraction=.1, cosine_floor_fraction=.1,
        coefficients=dict(generation=1., repair=.5, variety=.25))
    return model, streams, options


def test_frozen_general_full_source_order_and_corrupt_resume_rejected_before_live_mutation(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    model, streams, options = fixtures()
    before = {n: t.clone() for n, t in model.base.state_dict().items()}
    stage = OwnCodeTrainingStage(model, make_optimizer(model, options), streams, options, {'plan': 'a' * 64}, 'cpu')
    all_positions = {name: [] for name in streams}
    for step in range(stage.steps):
        for name, groups in stage.positions(step, 0).items():
            all_positions[name].extend(p for group in groups for p in group)
    assert sorted(all_positions['repair']) == list(range(24))
    assert len(set(all_positions['generation'][:9])) == 9
    assert len(set(all_positions['variety'])) == 24
    stage.update(); stage.save(tmp_path / 'resume.pt', 8 * 1024**2, 0)
    saved = torch.load(tmp_path / 'resume.pt', weights_only=True)
    stage.update(); current = {n: t.clone() for n, t in model.parts.state_dict().items()}
    rng = torch.get_rng_state().clone()
    for mutate in [lambda s: s['sampler_state']['repair'].update(actual_exposures=0),
                   lambda s: s['worker_rng_states'].pop(),
                   lambda s: s['optimizer']['state'][0]['exp_avg'].fill_(float('nan')),
                   lambda s: s['options'].update(coefficients=dict(generation=2., repair=.5, variety=.25))]:
        bad = copy.deepcopy(saved); mutate(bad)
        with pytest.raises(ValueError):
            stage.resume(bad)
        assert stage.completed == 2 and all(torch.equal(t, model.parts.state_dict()[n]) for n, t in current.items())
        assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(t, model.base.state_dict()[n]) for n, t in before.items())
    # CUDA is hidden in this CPU suite. A private generator rejection is
    # injected to prove its device-format check precedes any live mutation;
    # this is explicitly not a real full-model CUDA recovery measurement.
    bad = copy.deepcopy(saved); bad['worker_rng_states'][0]['cuda'] = torch.zeros(1, dtype=torch.uint8)
    stage.device = torch.device('cuda')
    real_generator = torch.Generator
    class RejectingDeviceGenerator:
        def set_state(self, value):
            assert value.numel() == 1
            raise RuntimeError('invalid actual device RNG byte format')
    def checked_generator(*args, **kwargs):
        return RejectingDeviceGenerator() if torch.device(kwargs.get('device', 'cpu')).type == 'cuda' else real_generator(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(torch, 'Generator', checked_generator)
        with pytest.raises(ValueError, match='per-worker RNG'):
            stage.resume(bad)
    assert stage.completed == 2 and all(torch.equal(t, model.parts.state_dict()[n]) for n, t in current.items())
    assert torch.equal(rng, torch.get_rng_state())
    stage.device = torch.device('cpu')
    streams['repair'].role = 'validation'
    with pytest.raises(ValueError, match='optimization role'):
        OwnCodeTrainingStage(model, None, streams, options, {}, 'cpu')


def distributed_worker(rank, folder):
    folder = Path(folder)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + str(folder / 'rendezvous'), rank=rank, world_size=4)
    try:
        model, streams, options = fixtures(4)
        original = {n: t.clone() for n, t in model.base.state_dict().items()}
        stage = OwnCodeTrainingStage(model, make_optimizer(model, options), streams, options, {'plan': 'b' * 64}, 'cpu')
        first = stage.update()
        if rank == 0:
            torch.save(dict(parts={n: t.clone() for n, t in model.parts.state_dict().items()}, first=first), folder / 'first.pt')
        stage.save(folder / 'resume.pt', 8 * 1024**2, 0)
        second = stage.update(); expected = {n: t.clone() for n, t in model.parts.state_dict().items()}
        fresh, sources, settings = fixtures(4)
        resumed = OwnCodeTrainingStage(fresh, make_optimizer(fresh, settings), sources, settings, {'plan': 'b' * 64}, 'cpu')
        resumed.resume(torch.load(folder / 'resume.pt', map_location='cpu', weights_only=True))
        assert resumed.update() == second
        assert all(torch.equal(t, fresh.parts.state_dict()[n]) for n, t in expected.items())
        assert all(torch.equal(t, fresh.base.state_dict()[n]) for n, t in original.items())
        # Trained inference components reconstruct the same two explicit
        # routes, including cached code generation, from one original base.
        base, _ = own_models()
        reloaded = OwnCodeDelta(base, components(fresh), reserved_parameters=256)
        fresh.eval(); reloaded.eval(); ids = torch.tensor([[1, 4, 7]])
        with torch.no_grad():
            for route in ['general', 'code']:
                assert torch.equal(fresh(ids, route=route)['logits'], reloaded(ids, route=route)['logits'])
                assert torch.equal(fresh.generate(ids, route=route, max_new_tokens=3), reloaded.generate(ids, route=route, max_new_tokens=3))
    finally:
        dist.destroy_process_group()


def test_actual_four_worker_objectives_match_independent_serial_global_scalar_and_resume(tmp_path):
    spawn(distributed_worker, args=(str(tmp_path),), nprocs=4, join=True)
    torch.set_num_threads(1)
    model, streams, options = fixtures(1)
    model.code.train(); model.base.eval()
    stage = OwnCodeTrainingStage(model, make_optimizer(model, options), streams, options, {'unused': 'serial'}, 'cpu')
    expected_loss = 0.; counts = {}
    for name, stream in streams.items():
        positions = stage.positions(0, 0)[name]
        batches = [stream.encoder.batch([dict(packed_code_index=p) for p in group], 'response', 'cpu') for group in positions]
        sizes = [int((batch['labels'][:, 1:] != -100).sum()) for batch in batches]
        total = sum(sizes); counts[name] = total
        for batch, size in zip(batches, sizes):
            loss = model.code(**batch)['loss']
            coefficient = options['coefficients'][name] * size / total
            (loss * coefficient).backward(); expected_loss += float(loss.detach()) * coefficient
    norm = float(torch.nn.utils.clip_grad_norm_(model.code_parameters(), 1.))
    stage.optimizer.param_groups[0]['lr'] = options['learning_rate'] * stage.schedule(0)
    stage.optimizer.step()
    actual = torch.load(tmp_path / 'first.pt', weights_only=True)
    assert actual['first']['summed_objective'] == pytest.approx(expected_loss, abs=1e-6)
    assert actual['first']['gradient_norm_before_clipping'] == pytest.approx(norm, rel=1e-5, abs=1e-7)
    for name, count in counts.items():
        assert actual['first']['objectives'][name]['global_target_tokens'] == count
    for name, tensor in model.parts.state_dict().items():
        torch.testing.assert_close(tensor, actual['parts'][name], atol=2e-6, rtol=2e-5, msg=name)
