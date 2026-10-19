"""Four actual workers, all prepared objective kinds, exact mixed-source resume."""
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import hashlib
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
from slm_lab.code_residual_stage_v3 import (EXPANDED_REPLAY_FAMILIES, ExpandedReplayEncoder,
    ExpandedResidualStageKernel, replay_field, replay_objective)
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.codenet_packed_replay_v1 import FAMILY as PACKED_FAMILY, OBJECTIVE as PACKED_OBJECTIVE
from slm_lab.codenet_runner_v1 import VerifiedReplayEncoder
from slm_lab.codenet_token_pack_v1 import packed_encoder
from slm_lab.corpus import file_hash
from slm_lab.model import TinyMultimodalLM
from slm_lab.recurrent_ablation import state_digest

from test_code_residual_stage_v2 import CompleteCode, AllReplay, Guard
from test_codenet_runner_v1 import Tokenizer, small_config, options


class Code(CompleteCode):
    def __len__(self):
        return 116


class ExpandedReplay(AllReplay):
    families = list(EXPANDED_REPLAY_FAMILIES)
    held = False

    def row(self, name, position):
        if name != PACKED_FAMILY:
            row = super().row(name, position)
        else:
            ids = list(range(3, 3 + [63, 7, 2, 12][position]))
            row = dict(tokens=ids, split='train', source_split='train',
                _replay_objective=PACKED_OBJECTIVE, _replay_context_tokens=len(ids),
                _replay_target_tokens=len(ids) - 1,
                _replay_input_sha256=hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest())
        if self.held:
            row['split'] = 'valid'
        return row


def worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4,
        timeout=timedelta(seconds=90))
    try:
        folder = Path(directory)
        config, tokenizer = small_config(), Tokenizer()
        token, source, plan = folder / 'tokenizer.json', folder / 'base.pt', folder / 'plan.json'
        torch.manual_seed(21891)
        if rank == 0:
            base = TinyMultimodalLM(config)
            token.write_bytes(b'own-fixture-tokenizer')
            torch.save(dict(config=asdict(config), model=base.state_dict(), tokenizer_sha256=file_hash(token)), source)
            plan.write_text(json.dumps(dict(fixture='four-rank29-family mixed objectives',
                production_stage_ready=False, base_checkpoint_sha256=file_hash(source), tokenizer_sha256=file_hash(token))))
        dist.barrier()
        sources = dict(base_checkpoint_sha256=file_hash(source), tokenizer_sha256=file_hash(token), run_plan_sha256=file_hash(plan))
        code, replay = Code(folder), ExpandedReplay(folder / 'image.png')
        opts = {**options(), 'passes': 2, 'gradient_accumulation': 1, 'replay_accumulation': 1,
            'image_size': 16, 'log_every': 7}
        guard = Guard()

        def construct():
            torch.manual_seed(21891)
            raw = RoutedCodeResidual(TinyMultimodalLM(config), width=12)
            optimizer = torch.optim.AdamW(raw.adapter_parameters(), lr=opts['learning_rate'],
                weight_decay=opts['weight_decay'], foreach=False)
            kernel = ExpandedResidualStageKernel(raw, optimizer, code, packed_encoder(code, tokenizer, config),
                replay, ExpandedReplayEncoder(tokenizer, config, folder, image_size=16),
                opts, sources, 'cpu', nullcontext, guard=guard)
            return raw, optimizer, kernel

        raw, optimizer, kernel = construct()
        assert kernel.steps == 58
        base_before = state_digest(raw.base.state_dict())
        random.seed(41901 + rank)
        np.random.seed(42901 + rank)
        torch.manual_seed(43901 + rank)

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
        replay.held = rank == 2
        with pytest.raises(ValueError, match='Collective preflight failed'):
            kernel.update()
        assert state() == before
        replay.held = False

        # Equal quotas put one full29-family cycle before this save, including
        # packed next-token replay; both kinds also occur after restoration.
        for _ in range(30):
            update()
        assert any(row['replay_response_field'] is None for row in kernel.metrics)
        partial = kernel.snapshot(folder / 'resume.pt', maximum_bytes=1_000_000, minimum_free_bytes=0)
        interrupted = kernel.sampler_state(30)
        expected_continuation = [update() for _ in range(30, 58)]
        expected_state = state()
        next_draws = [random.random(), float(np.random.random()), float(torch.rand(()))]
        raw, optimizer, kernel = construct()
        assert kernel.restore(folder / 'resume.pt', partial['checkpoint_sha256']) == (30, interrupted)
        assert [update() for _ in range(30, 58)] == expected_continuation
        assert state() == expected_state
        assert [random.random(), float(np.random.random()), float(torch.rand(()))] == next_draws
        assert len(kernel.metrics) == kernel.completed == 58
        assert set(row['replay_family'] for row in kernel.metrics) == set(EXPANDED_REPLAY_FAMILIES)
        for metric in kernel.metrics:
            name = metric['replay_family']
            assert metric['replay_response_field'] == replay_field(name)
            assert metric['replay_objective'] == replay_objective(name)
            expected_targets = 80 if name == PACKED_FAMILY else 14
            assert metric['objectives'][name]['global_target_tokens'] == expected_targets
        assert all(parameter.grad is None for parameter in raw.parameters())
        assert state_digest(raw.base.state_dict()) == base_before
        progress = kernel.sampler_state(58)
        assert progress['code_examples'] == progress['replay_examples'] == 232
        assert progress['code_epoch_and_cursor'] == [2, 0]
        assert progress['replay_objectives'][PACKED_FAMILY] == PACKED_OBJECTIVE
        assert all(value['updates'] == 2 and value['examples'] == 8 and value['epoch_and_cursor'] == [2, 0]
            for value in progress['replay_families'].values())
        complete = kernel.snapshot(folder / 'complete.pt', maximum_bytes=1_000_000, minimum_free_bytes=0)
        before = state()
        with pytest.raises(ValueError, match='Collective preflight failed'):
            kernel.update()
        assert state() == before
        assert kernel.restore(folder / 'complete.pt', complete['checkpoint_sha256'], allow_completed=True)[0] == 58
        assert state() == before

        # A valid previous19-family source reader/encoder cannot silently enter
        # this new objective layout; constructor errors propagate to every rank.
        encoder = VerifiedReplayEncoder(tokenizer, config, folder, image_size=16) if rank == 1 else kernel.replay_encoder
        with pytest.raises(ValueError, match='Collective preflight failed'):
            ExpandedResidualStageKernel(raw, optimizer, code, kernel.code_encoder,
                replay, encoder, opts, sources, 'cpu', nullcontext, guard=guard)
        assert state() == before
        (folder / f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank,
            actual_reference_updates=58, actual_repeated_resume_updates=28,
            all29_families_executed_twice=True, original_packed_causal_targets_per_global_update=80,
            original_supervised_response_targets_per_global_update=14,
            exact_residual_Adam_rank_RNG_history_and_next_draw_continuation=True,
            explicit_causal_and_response_objective_fields_preserved=True,
            one_rank_held_input_rejected_before_weight_Adam_RNG_or_history_changes=True,
            one_rank_source_guard_and_wrong_encoder_rejections=True,
            base_unchanged=True, every_family_cursor_exact=True, full_history_with_log_every7=True,
            model_counts=raw.parameter_counts(), scope='Own small CPU replica and explicit fixture data; not857M capacity or full-source neural training.')))
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope='module')
def completed_artifacts(tmp_path_factory):
    folder = tmp_path_factory.mktemp('all29-mixed-objectives')
    Image.new('RGB', (8, 4), (21, 55, 90)).save(folder / 'image.png')
    mp.spawn(worker, args=(str(folder / 'rendezvous'), str(folder)), nprocs=4, join=True)
    return folder


def test_four_actual_workers_execute_both_objectives_all29_families_and_exact_resume(completed_artifacts):
    for rank in range(4):
        proof = json.loads((completed_artifacts / f'rank-{rank}.json').read_text())
        assert proof['all29_families_executed_twice']
        assert proof['one_rank_held_input_rejected_before_weight_Adam_RNG_or_history_changes']


def test_unknown_source_cannot_silently_default_to_supervised_response():
    with pytest.raises(ValueError, match='admitted replay family'):
        replay_field('not-admitted')
