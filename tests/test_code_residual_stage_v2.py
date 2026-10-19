"""Actual four-rank kernel; small own architecture and explicit fixture data."""
from contextlib import nullcontext
import copy
from dataclasses import asdict
from datetime import timedelta
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import pytest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp

from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
from slm_lab.code_residual_export_v2 import FIELDS, export_inference, load_inference
from slm_lab.code_residual_stage_v2 import FULL_REPLAY_FAMILIES, ResidualStageKernel, validate_options
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.codenet_runner_v1 import VerifiedReplayEncoder
from slm_lab.codenet_stage_v1 import EpochOrder
from slm_lab.codenet_token_pack_v1 import packed_encoder
from slm_lab.corpus import file_hash
from slm_lab.recurrent_ablation import state_digest
from slm_lab.model import TinyMultimodalLM
from slm_lab.training import load_checkpoint

from test_codenet_runner_v1 import Tokenizer, small_config, options


class CompleteCode:
    def __init__(self, folder):
        self.folder = Path(folder)

    def __len__(self):
        return 76

    def sequence(self, position):
        response = [11 + position] * (35 if position == 0 else 1 + position % 7)
        return [1, 10] + response + [2], [-100, -100] + response + [2]


class AllReplay:
    families = list(FULL_REPLAY_FAMILIES)

    def __init__(self, image):
        from slm_lab.diverse_images import pixel_hash
        self.image = str(image)
        with Image.open(image) as picture:
            self.golden = pixel_hash(picture, 16)

    def size(self, name):
        return 4

    def row(self, name, position):
        row = dict(prompt='p', response='a' * (position + 1), split='train', id=f'{name}:{position}')
        if name in FULL_REPLAY_FAMILIES[13:] or name in {'generic_visual', 'image_reading'}:
            row.update(image=self.image, model_pixel_sha256_16=self.golden)
        return row


class Guard:
    fail = False

    def check(self, full=False):
        if self.fail:
            raise ValueError('Fixture source binding changed')


def worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4,
        timeout=timedelta(seconds=90))
    try:
        folder = Path(directory)
        config = small_config()
        token = folder / 'tokenizer.json'
        source = folder / 'base.pt'
        plan = folder / 'plan.json'
        torch.manual_seed(8911)
        base = TinyMultimodalLM(config)
        if rank == 0:
            token.write_bytes(b'own-fixture-tokenizer')
            torch.save(dict(config=asdict(config), model=base.state_dict(), tokenizer_sha256=file_hash(token)), source)
            manifest = folder / 'data-manifest.json'
            manifest.write_text(json.dumps(dict(fixture='76complete source programs and19replay families')))
            plan.write_text(json.dumps(dict(fixture='four-rank all19 kernel', production_stage_ready=False,
                base_checkpoint_sha256=file_hash(source), tokenizer_sha256=file_hash(token), code_manifest_sha256=file_hash(manifest))))
        dist.barrier()
        sources = dict(base_checkpoint_sha256=file_hash(source), tokenizer_sha256=file_hash(token), run_plan_sha256=file_hash(plan))
        code = CompleteCode(folder)
        replay = AllReplay(folder / 'image.png')
        tokenizer = Tokenizer()
        opts = {**options(), 'passes': 1, 'gradient_accumulation': 1, 'replay_accumulation': 1,
            'image_size': 16, 'log_every': 5}
        guard = Guard()

        def construct():
            torch.manual_seed(8911)
            raw = RoutedCodeResidual(TinyMultimodalLM(config), width=12)
            optimizer = torch.optim.AdamW(raw.adapter_parameters(), lr=opts['learning_rate'],
                weight_decay=opts['weight_decay'], foreach=False)
            kernel = ResidualStageKernel(raw, optimizer, code, packed_encoder(code, tokenizer, config),
                replay, VerifiedReplayEncoder(tokenizer, config, folder, image_size=16),
                opts, sources, 'cpu', nullcontext, guard=guard)
            return raw, optimizer, kernel

        raw, optimizer, kernel = construct()
        assert kernel.steps == 19
        base_before = state_digest(raw.base.state_dict())
        random.seed(9101 + rank)
        np.random.seed(9201 + rank)
        torch.manual_seed(9301 + rank)

        def state():
            return fingerprint(dict(model=raw.state_dict(), optimizer=optimizer.state_dict(),
                rng=capture_rng('cpu'), completed=kernel.completed, metrics=kernel.metrics))

        def update():
            draws = [random.random(), float(np.random.random()), float(torch.rand(()))]
            return draws, kernel.update()

        before = state()
        guard.fail = rank == 1
        with pytest.raises(ValueError, match='Collective preflight failed'):
            kernel.update()
        assert state() == before
        guard.fail = False
        code_seen = []
        for step in range(7):
            update()
            code_seen += [group[0] for group in EpochOrder(len(code), opts['seed']).rank_groups(step, rank, 4, 1, 1)]
        partial = kernel.snapshot(folder / 'resume.pt', maximum_bytes=1_000_000, minimum_free_bytes=0)
        assert len(kernel.metrics) == 7
        interrupted_progress = kernel.sampler_state(7)
        continuation = [update() for _ in range(7, 19)]
        expected = state()
        next_draws = [random.random(), float(np.random.random()), float(torch.rand(()))]
        raw, optimizer, kernel = construct()
        assert kernel.restore(folder / 'resume.pt', partial['checkpoint_sha256']) == (7, interrupted_progress)
        assert [update() for _ in range(7, 19)] == continuation
        assert state() == expected
        assert [random.random(), float(np.random.random()), float(torch.rand(()))] == next_draws
        assert len(kernel.metrics) == kernel.completed == 19
        assert set(row['replay_family'] for row in kernel.metrics) == set(FULL_REPLAY_FAMILIES)
        assert all(p.grad is None for p in raw.parameters())
        assert state_digest(raw.base.state_dict()) == base_before
        complete = kernel.snapshot(folder / 'complete.pt', maximum_bytes=1_000_000, minimum_free_bytes=0)
        before = state()
        with pytest.raises(ValueError, match='Collective preflight failed'):
            kernel.update()
        assert state() == before
        assert kernel.restore(folder / 'complete.pt', complete['checkpoint_sha256'], allow_completed=True)[0] == 19
        assert state() == before
        progress = kernel.sampler_state(19)
        assert progress['code_examples'] == 76 and progress['replay_examples'] == 76
        assert progress['code_epoch_and_cursor'] == [1, 0]
        assert all(f['updates'] == 1 and f['examples'] == 4 and f['epoch_and_cursor'] == [1, 0]
            for f in progress['replay_families'].values())
        (folder / f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank,
            actual_updates_in_uninterrupted_reference=19, actual_repeated_resume_updates=12,
            all19_replay_families_executed=True, full_per_update_history_with_log_every5=True,
            exact_residual_Adam_and_all_rank_RNG_continuation=True,
            exact_code_and_all19_family_progress=True, one_rank_source_guard_failure_rejected_without_mutation=True,
            frozen_base_unchanged=True, completed_budget_blocks_extra_update=True,
            model_counts=raw.parameter_counts(), scope='Small own architecture/fixture data only; not857M capacity or full-source training.')))
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope='module')
def completed_artifacts(tmp_path_factory):
    folder = tmp_path_factory.mktemp('all19-collective-stage')
    Image.new('RGB', (8, 4), (21, 55, 90)).save(folder / 'image.png')
    mp.spawn(worker, args=(str(folder / 'rendezvous'), str(folder)), nprocs=4, join=True)
    return folder


def test_four_actual_workers_use_complete_update_path_all19_families_and_exact_resume(completed_artifacts):
    for rank in range(4):
        assert json.loads((completed_artifacts / f'rank-{rank}.json').read_text())['all19_replay_families_executed']


@pytest.mark.parametrize('change', [{'seed': True}, {'replay_coefficient': float('nan')}, {'gradient_accumulation': 0}])
def test_invalid_options_reject_before_update(change):
    with pytest.raises(ValueError):
        validate_options({**options(), **change})


@pytest.fixture(scope='module')
def exported_artifacts(completed_artifacts):
    folder = completed_artifacts
    snapshot = folder / 'complete.pt'
    before = file_hash(snapshot)
    receipt = export_inference(snapshot, folder / 'base.pt', folder / 'tokenizer.json', folder / 'plan.json',
        folder / 'data-manifest.json', folder / 'inference.pt', expected_training_checkpoint_sha256=before,
        maximum_bytes=1_000_000, minimum_free_bytes=0)
    assert before == file_hash(snapshot) == receipt['complete_training_checkpoint_preserved_sha256']
    kwargs = dict(expected_checkpoint_sha256=receipt['checkpoint_sha256'], expected_base_sha256=file_hash(folder / 'base.pt'),
        expected_plan_sha256=file_hash(folder / 'plan.json'), expected_training_checkpoint_sha256=before,
        expected_data_sha256=file_hash(folder / 'data-manifest.json'))
    return folder, kwargs


def test_separate_inference_export_preserves_complete_state_and_exact_logits_cache_and_generation(exported_artifacts):
    folder, kwargs = exported_artifacts
    state = torch.load(folder / 'inference.pt', weights_only=True)
    assert set(state) == FIELDS and state['optimizer_or_rng_included'] is False
    assert not {'optimizer', 'worker_rng_states', 'rng'} & state.keys()
    assert (folder / 'inference.pt').stat().st_size < (folder / 'complete.pt').stat().st_size
    base, _ = load_checkpoint(folder / 'base.pt', 'cpu', training_state=False)
    original = RoutedCodeResidual(base, width=12)
    trained = torch.load(folder / 'complete.pt', weights_only=True)
    original.load_adapter_state(trained['residuals'])
    original.eval()
    policy, metadata = load_inference(folder / 'inference.pt', folder / 'base.pt', folder / 'tokenizer.json', **kwargs)
    ids = torch.tensor([[1, 10, 11, 12, 2]])
    with torch.no_grad():
        torch.testing.assert_close(policy(ids)['logits'], original(ids, route='code')['logits'], rtol=0, atol=0)
        cached = policy.decode(ids[:, 3:], policy.prefill(ids[:, :3])['cache'])
        torch.testing.assert_close(cached['logits'], policy(ids)['logits'][:, 3:], rtol=1e-5, atol=1e-6)
    assert torch.equal(policy.generate(ids[:, :2], max_new_tokens=4, use_cache=True),
        original.code.generate(ids[:, :2], max_new_tokens=4, use_cache=True))
    assert metadata['optimizer_loaded_on_GPU'] is False and metadata['route'] == 'code'
    assert metadata['model_counts'] == original.parameter_counts()


@pytest.mark.parametrize('fault', ['plan', 'tokenizer', 'count', 'order', 'missing_weight', 'nonfinite_weight',
    'partial', 'optimizer_extra', 'training_identity', 'data_identity'])
def test_inference_misbindings_and_corrupt_states_reject(exported_artifacts, tmp_path, fault):
    folder, expected = exported_artifacts
    kwargs = expected.copy()
    state = copy.deepcopy(torch.load(folder / 'inference.pt', weights_only=True))
    token = folder / 'tokenizer.json'
    if fault == 'plan':
        state['run_plan_sha256'] = 'd' * 64
    elif fault == 'tokenizer':
        token = tmp_path / 'changed-tokenizer.json'
        token.write_bytes(b'different-tokenizer')
    elif fault == 'count':
        state['model_counts']['total_unique'] += 1
    elif fault == 'order':
        state['residual_parameter_names'].reverse()
    elif fault == 'missing_weight':
        state['residuals'].pop(next(iter(state['residuals'])))
    elif fault == 'nonfinite_weight':
        next(iter(state['residuals'].values())).flatten()[0] = float('nan')
    elif fault == 'partial':
        state['steps'] -= 1
    elif fault == 'optimizer_extra':
        state['optimizer'] = {}
    elif fault == 'training_identity':
        state['training_checkpoint_sha256'] = 'd' * 64
    elif fault == 'data_identity':
        state['data_sha256'] = 'd' * 64
    path = tmp_path / 'bad.pt'
    torch.save(state, path)
    kwargs['expected_checkpoint_sha256'] = file_hash(path)
    with pytest.raises(ValueError):
        load_inference(path, folder / 'base.pt', token, **kwargs)


@pytest.mark.parametrize('fault', ['partial', 'Adam_corruption', 'missing_rank_RNG'])
def test_export_rejects_incomplete_training_state_without_writing(completed_artifacts, tmp_path, fault):
    folder = completed_artifacts
    state = copy.deepcopy(torch.load(folder / 'complete.pt', weights_only=True))
    if fault == 'partial':
        state['steps'] -= 1
    elif fault == 'Adam_corruption':
        state['optimizer']['state'][0]['exp_avg'].flatten()[0] += .01
    else:
        state['worker_rng_states'].pop()
    path = tmp_path / 'bad-training.pt'
    torch.save(state, path)
    output = tmp_path / 'must-not-exist.pt'
    with pytest.raises(ValueError):
        export_inference(path, folder / 'base.pt', folder / 'tokenizer.json', folder / 'plan.json', folder / 'data-manifest.json',
            output, expected_training_checkpoint_sha256=file_hash(path), maximum_bytes=1_000_000, minimum_free_bytes=0)
    assert not output.exists() and not output.with_suffix('.pt.pending').exists()
