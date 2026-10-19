"""Four actual workers, full visual rehearsal and exact complete-state recovery.

Source/image fixtures and a small own model verify mechanics. These are not
actual finalized-source admission, full706M capacity or capability evidence.
"""
from contextlib import nullcontext
from copy import deepcopy
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

from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
from slm_lab.codenet_checkpoint_v1 import FileBindings, capture_rng
from slm_lab.codenet_runner_v1 import VerifiedReplayEncoder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import pixel_hash
from slm_lab.finalized_visual_replay_v1 import FinalizedVisualReplay, VISUAL_FAMILIES
from slm_lab.model import TinyMultimodalLM
from slm_lab.paired_visual_training_v1 import PairedVisualKernel, freeze_for_visual_stage
from slm_lab.recurrent_ablation import state_digest
from test_codenet_runner_v1 import Tokenizer, small_config
from test_paired_visual_contrast_v1 import examples


class Pairs:
    def __init__(self, guard):
        self.rows = examples()
        for row in self.rows:
            row['images'] = [x.float() for x in row['images']]
            from slm_lab.paired_visual_contrast_v1 import pixel_digest
            row['pixel_sha256'] = [pixel_digest(x) for x in row['images']]
        self.contract = dict(fixture=True, real_source_admission=False,
            complete_original_fixture_fingerprint=fingerprint(self.rows))
        self.contract_sha256 = fingerprint(self.contract)
        for row in self.rows:
            row['source_contract_sha256'] = self.contract_sha256
        self.guard = guard

    def __len__(self):
        return len(self.rows)

    def example(self, position):
        return deepcopy(self.rows[position])


class Replay:
    families = VISUAL_FAMILIES

    def __init__(self, folder, guard):
        self.folder, self.guard = folder, guard
        self.contract = dict(fixture=True, real_source_admission=False,
            families=[dict(name=n, source_records=5) for n in self.families])
        self.contract_sha256 = fingerprint(self.contract)
        with Image.open(folder / 'image.png') as image:
            self.golden = pixel_hash(image, 16)

    def size(self, name):
        assert name in self.families
        return 5

    def row(self, name, position):
        assert name in self.families and 0 <= position < 5
        return dict(id=f'fixture:{name}:{position}', split='train', prompt='p', response='a' * (1 + position),
            image=str(self.folder / 'image.png'), model_pixel_sha256_16=self.golden)


def worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4,
        timeout=timedelta(seconds=180))
    try:
        folder = Path(directory)
        config, token = small_config(), Tokenizer()
        if rank == 0:
            Image.new('RGB', (16, 16), (35, 170, 60)).save(folder / 'image.png')
            (folder / 'source.json').write_text(json.dumps(dict(fixture=True, real_source_admission=False)))
        dist.barrier()
        guard = FileBindings({str(folder / 'source.json'): file_hash(folder / 'source.json'),
            str(folder / 'image.png'): file_hash(folder / 'image.png')})
        pairs, replay = Pairs(guard), Replay(folder, guard)
        opts = dict(steps=16, seed=438, pair_batch_size=1, pair_accumulation=1,
            replay_batch_size=1, replay_accumulation=1, image_size=16, learning_rate=.001,
            weight_decay=.01, gradient_clip=1., alpha=.3, beta=2., margin=.1, replay_coefficient=.5)
        sources = dict(base_checkpoint_sha256='a' * 64, tokenizer_sha256='b' * 64, run_plan_sha256='c' * 64)

        def construct():
            torch.manual_seed(49216)
            model = TinyMultimodalLM(config)
            optimizer = torch.optim.AdamW(freeze_for_visual_stage(model), lr=opts['learning_rate'],
                weight_decay=opts['weight_decay'], foreach=False)
            kernel = PairedVisualKernel(model, optimizer, pairs, replay,
                VerifiedReplayEncoder(token, config, folder, 16), opts, sources, 'cpu', nullcontext, guard=guard)
            return model, optimizer, kernel

        model, optimizer, kernel = construct()
        frozen = {n: p.clone() for n, p in model.state_dict().items() if not n.startswith('vision.')}
        vision_before = state_digest(model.vision.state_dict())
        ids = torch.tensor([[1, 5, 11, 2]])
        model.eval()
        with torch.no_grad():
            text_before = model(ids)['logits'].clone()
        random.seed(1000 + rank); np.random.seed(2000 + rank); torch.manual_seed(3000 + rank)
        expected_rngs = []
        seen = {name: set() for name in replay.families}
        for step in range(16):
            # Perturb and retain all three generators to make per-rank recovery substantive.
            random.random(); np.random.random(); torch.rand(3)
            result = kernel.update()
            name, occurrence = result['replay_family'], result['replay_occurrence']
            from slm_lab.codenet_stage_v1 import EpochOrder
            order = EpochOrder(5, opts['seed'] + 1000 + replay.families.index(name))
            seen[name].update(order.at(occurrence * 4 + j) for j in range(4))
            assert result['paired']['global_pair_exposures'] == 4
            assert result['proper_target_tokens'] > result['paired']['global_proper_target_tokens']
            expected_rngs.append(fingerprint(capture_rng('cpu')))
            if step == 6:
                saved = kernel.snapshot(folder / 'resume.pt', maximum_bytes=5_000_000, minimum_free_bytes=0)
        full = kernel.snapshot(folder / 'final.pt', maximum_bytes=5_000_000, minimum_free_bytes=0)
        assert all(population == set(range(5)) for population in seen.values())
        assert state_digest(model.vision.state_dict()) != vision_before
        assert all(torch.equal(p, model.state_dict()[n]) for n, p in frozen.items())
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(ids)['logits'], text_before, atol=0, rtol=0)
        expected = fingerprint(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            metrics=kernel.metrics, sampler=kernel.sampler_state(16), rng=capture_rng('cpu')))

        resumed, resumed_optimizer, recovered = construct()
        assert recovered.restore(folder / 'resume.pt', saved['checkpoint_sha256']) == 7
        assert fingerprint(capture_rng('cpu')) == expected_rngs[6]
        for step in range(7, 16):
            random.random(); np.random.random(); torch.rand(3)
            recovered.update()
            assert fingerprint(capture_rng('cpu')) == expected_rngs[step]
        actual = fingerprint(dict(model=resumed.state_dict(), optimizer=resumed_optimizer.state_dict(),
            metrics=recovered.metrics, sampler=recovered.sampler_state(16), rng=capture_rng('cpu')))
        assert actual == expected
        recovery = recovered.snapshot(folder / 'recovered.pt', maximum_bytes=5_000_000, minimum_free_bytes=0)
        if rank == 0:
            a = torch.load(folder / 'final.pt', weights_only=True)
            b = torch.load(folder / 'recovered.pt', weights_only=True)
            assert fingerprint(a) == fingerprint(b)
        dist.barrier()

        # A failed snapshot cannot erase the prior complete state.
        before_sha = file_hash(folder / 'recovered.pt')
        with pytest.raises(ValueError, match='Collective visual snapshot failed'):
            recovered.snapshot(folder / 'recovered.pt', maximum_bytes=1, minimum_free_bytes=0)
        assert file_hash(folder / 'recovered.pt') == before_sha
        assert not (folder / 'recovered.pt.pending').exists()

        # An accidentally changed frozen tensor on one rank blocks a new checkpoint everywhere.
        original = resumed.embedding.weight[0, 0].detach().clone()
        if rank == 3:
            with torch.no_grad():
                resumed.embedding.weight[0, 0].add_(.1)
        with pytest.raises(ValueError, match='Collective preflight'):
            recovered.snapshot(folder / 'recovered.pt', maximum_bytes=5_000_000, minimum_free_bytes=0)
        assert file_hash(folder / 'recovered.pt') == before_sha
        with torch.no_grad():
            resumed.embedding.weight[0, 0].copy_(original)

        # Metadata corruption, even with a new whole-file hash, rejects before applying anything.
        if rank == 0:
            bad = torch.load(folder / 'resume.pt', weights_only=True)
            bad['sampler_state']['paired']['draws'] += 1
            bad['state_sha256'] = fingerprint({k: v for k, v in bad.items() if k != 'state_sha256'})
            torch.save(bad, folder / 'corrupt.pt')
        dist.barrier()
        fresh, fresh_optimizer, fresh_kernel = construct()
        before = fingerprint(dict(model=fresh.state_dict(), optimizer=fresh_optimizer.state_dict(), rng=capture_rng('cpu')))
        with pytest.raises(ValueError, match='Collective preflight'):
            fresh_kernel.restore(folder / 'corrupt.pt', file_hash(folder / 'corrupt.pt'))
        assert fingerprint(dict(model=fresh.state_dict(), optimizer=fresh_optimizer.state_dict(), rng=capture_rng('cpu'))) == before

        # One worker losing the original source role must stop all before gradients/Adam.
        if rank == 2:
            for row in pairs.rows:
                row['split'] = 'valid'
        with pytest.raises(ValueError, match='Collective preflight'):
            fresh_kernel.update()
        assert fresh_kernel.completed == 0 and not fresh_optimizer.state
        assert all(p.grad is None for p in fresh.parameters())
        assert fingerprint(dict(model=fresh.state_dict(), optimizer=fresh_optimizer.state_dict(), rng=capture_rng('cpu'))) == before
        report = dict(rank=rank, actual_four_CPU_workers=True, complete_reference_updates=16,
            checkpoint_step=7, exact_resumed_updates=9, all_eight_visual_families_complete=True,
            unique_fixture_rows_per_family=5, global_pair_exposures=64,
            full_weights_Adam_each_rank_RNG_metrics_cursors_match=True,
            all_language_critic_tensors_and_text_logits_identical=True,
            failed_save_preserves_previous_complete_state=True, corrupt_resume_rejected_before_mutation=True,
            one_rank_frozen_weight_change_blocks_snapshot=True,
            one_rank_held_role_rejected_before_gradients=True, final_checkpoint=full, recovered_checkpoint=recovery,
            model_parameters=model.parameter_counts(),
            scope='Synthetic sources/images and own small model; no real finalized admission, full706M GPU capacity or capability claim.')
        (folder / f'rank-{rank}.json').write_text(json.dumps(report, indent=2))
    finally:
        dist.destroy_process_group()


def test_four_worker_complete_visual_training_and_recovery(tmp_path):
    mp.spawn(worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path)), nprocs=4, join=True)
    for rank in range(4):
        report = json.loads((tmp_path / f'rank-{rank}.json').read_text())
        assert report['full_weights_Adam_each_rank_RNG_metrics_cursors_match']


def test_real_visual_replay_factory_rejects_missing_final_sources_without_partial_reads(tmp_path):
    data = tmp_path / 'data/vqa-real-pairs-v1'
    data.mkdir(parents=True)
    (data / 'images.collected.jsonl').write_text('not training data')
    with pytest.raises(ValueError, match='Finalized complete30-family source bundle'):
        FinalizedVisualReplay(tmp_path, None, None)
    assert not (tmp_path / 'data/codenet-final-source-contract-v4').exists()
