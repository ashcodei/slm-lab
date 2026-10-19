"""Actual four-worker30-family update/recovery and completed-state export."""
from contextlib import nullcontext
from copy import deepcopy
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
from slm_lab.code_residual_export_v2 import export_inference, load_inference
from slm_lab.code_residual_source_contract_v4 import (REQUIRED_FAMILIES, COMPONENT_NAMES,
    FIELDS, SourceContractResidualKernel, build_required_contract)
from slm_lab.code_residual_stage_v3 import ExpandedReplayEncoder
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.codenet_packed_replay_v1 import FAMILY as PACKED_FAMILY
from slm_lab.codenet_paired_photo_replay_v1 import FAMILY as PAIRED_FAMILY
from slm_lab.codenet_token_pack_v1 import packed_encoder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import pixel_hash
from slm_lab.model import TinyMultimodalLM
from slm_lab.recurrent_ablation import state_digest
from slm_lab.training import load_checkpoint

from test_code_residual_stage_v2 import Guard
from test_code_residual_stage_v3 import ExpandedReplay
from test_codenet_runner_v1 import Tokenizer, small_config, options


class Code:
    def __init__(self, folder):
        self.folder = folder

    def __len__(self):
        return 120

    def sequence(self, position):
        answer = [11 + position % 100] * (35 if position == 0 else 1 + position % 7)
        return [1, 10] + answer + [2], [-100, -100] + answer + [2]


class Replay(ExpandedReplay):
    families = list(REQUIRED_FAMILIES)

    def __init__(self, folder):
        super().__init__(folder / 'image.png')
        self.other = str(folder / 'other.png')
        with Image.open(self.other) as image:
            self.other_golden = pixel_hash(image, 16)

    def row(self, name, position):
        row = super().row(name, position)
        if name == PAIRED_FAMILY:
            row.update(prompt='p', response='a' if position % 2 == 0 else 'bb', task='visual_qa',
                image=self.image if position % 2 == 0 else self.other,
                model_pixel_sha256_16=self.golden if position % 2 == 0 else self.other_golden)
        return row


def worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4,
        timeout=timedelta(seconds=90))
    try:
        folder = Path(directory)
        config, tokenizer, replay, code = small_config(), Tokenizer(), Replay(folder), Code(folder)
        token, source, plan = folder / 'tokenizer.json', folder / 'base.pt', folder / 'plan.json'
        torch.manual_seed(50189)
        if rank == 0:
            token.write_bytes(b'own-fixture-tokenizer')
            torch.save(dict(config=asdict(config), model=TinyMultimodalLM(config).state_dict(),
                tokenizer_sha256=file_hash(token)), source)
        dist.barrier()
        contract = build_required_contract(replay, file_hash(token),
            {name: hashlib.sha256(('fixture:' + name).encode()).hexdigest() for name in COMPONENT_NAMES})
        if rank == 0:
            (folder / 'replay-contract.json').write_text(json.dumps(contract))
            manifest = folder / 'data-manifest.json'
            manifest.write_text(json.dumps(dict(fixture=True, real_source_admission=False,
                programs=120, replay_families=30, replay_contract_fingerprint=fingerprint(contract))))
            plan.write_text(json.dumps(dict(fixture=True, production_stage_ready=False,
                base_checkpoint_sha256=file_hash(source), tokenizer_sha256=file_hash(token),
                code_manifest_sha256=file_hash(manifest), replay_contract_fingerprint=fingerprint(contract))))
        dist.barrier()
        sources = dict(base_checkpoint_sha256=file_hash(source), tokenizer_sha256=file_hash(token), run_plan_sha256=file_hash(plan))
        opts = {**options(), 'passes': 2, 'gradient_accumulation': 1, 'replay_accumulation': 1,
            'image_size': 16, 'log_every': 9}
        guard = Guard()

        def construct():
            torch.manual_seed(50189)
            raw = RoutedCodeResidual(TinyMultimodalLM(config), width=12)
            optimizer = torch.optim.AdamW(raw.adapter_parameters(), lr=opts['learning_rate'],
                weight_decay=opts['weight_decay'], foreach=False)
            kernel = SourceContractResidualKernel(raw, optimizer, code, packed_encoder(code, tokenizer, config),
                replay, ExpandedReplayEncoder(tokenizer, config, folder, image_size=16), opts, sources,
                'cpu', nullcontext, replay_contract=contract, expected_contract_fingerprint=fingerprint(contract), guard=guard)
            return raw, optimizer, kernel

        raw, optimizer, kernel = construct()
        assert kernel.steps == 60
        base_before = state_digest(raw.base.state_dict())
        random.seed(5101 + rank)
        np.random.seed(5201 + rank)
        torch.manual_seed(5301 + rank)

        def state():
            return fingerprint(dict(model=raw.state_dict(), optimizer=optimizer.state_dict(),
                rng=capture_rng('cpu'), completed=kernel.completed, metrics=kernel.metrics))

        def update():
            draws = [random.random(), float(np.random.random()), float(torch.rand(()))]
            return draws, kernel.update()

        before = state()
        replay.held = rank == 2
        with pytest.raises(ValueError, match='Collective preflight failed'):
            kernel.update()
        assert state() == before
        replay.held = False
        for _ in range(31):
            update()
        assert {PACKED_FAMILY, PAIRED_FAMILY} <= {metric['replay_family'] for metric in kernel.metrics}
        partial = kernel.snapshot(folder / 'resume.pt', maximum_bytes=1_000_000, minimum_free_bytes=0)
        before = state()
        if rank == 1:
            kernel.contract['families'][-1]['source_records'] += 1
        with pytest.raises(ValueError, match='Collective preflight failed'):
            kernel.snapshot(folder / 'must-not-write.pt', maximum_bytes=1_000_000, minimum_free_bytes=0)
        assert not (folder / 'must-not-write.pt').exists() and state() == before
        kernel.contract = deepcopy(contract)
        interrupted = kernel.sampler_state(31)
        continuation = [update() for _ in range(31, 60)]
        expected, next_draws = state(), [random.random(), float(np.random.random()), float(torch.rand(()))]
        raw, optimizer, kernel = construct()
        assert kernel.restore(folder / 'resume.pt', partial['checkpoint_sha256']) == (31, interrupted)
        assert [update() for _ in range(31, 60)] == continuation
        assert state() == expected
        assert [random.random(), float(np.random.random()), float(torch.rand(()))] == next_draws
        assert len(kernel.metrics) == kernel.completed == 60
        assert state_digest(raw.base.state_dict()) == base_before
        assert all(parameter.grad is None for parameter in raw.parameters())
        for metric in kernel.metrics:
            name = metric['replay_family']
            objective = next(item['objective'] for item in contract['families'] if item['name'] == name)
            assert metric['replay_objective'] == objective and metric['replay_response_field'] == FIELDS[objective]
            target_tokens = 80 if name == PACKED_FAMILY else 10 if name == PAIRED_FAMILY else 14
            assert metric['objectives'][name]['global_target_tokens'] == target_tokens
        progress = kernel.sampler_state(60)
        assert progress['code_examples'] == progress['replay_examples'] == 240
        assert progress['replay_contract'] == contract and progress['replay_contract_fingerprint'] == fingerprint(contract)
        assert all(value['updates'] == 2 and value['examples'] == 8 and value['epoch_and_cursor'] == [2, 0]
            for value in progress['replay_families'].values())
        complete = kernel.snapshot(folder / 'complete.pt', maximum_bytes=1_000_000, minimum_free_bytes=0)
        before = state()
        with pytest.raises(ValueError, match='Collective preflight failed'):
            kernel.update()
        assert state() == before
        assert kernel.restore(folder / 'complete.pt', complete['checkpoint_sha256'], allow_completed=True)[0] == 60
        assert state() == before
        bad = deepcopy(contract)
        if rank == 3:
            bad['families'][28]['objective'] = 'response_only'
        with pytest.raises(ValueError, match='Collective preflight failed'):
            SourceContractResidualKernel(raw, optimizer, code, kernel.code_encoder, replay, kernel.replay_encoder,
                opts, sources, 'cpu', nullcontext, replay_contract=bad, expected_contract_fingerprint=fingerprint(bad))
        assert state() == before
        (folder / f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank,
            actual_reference_updates=60, actual_repeated_resume_updates=29,
            all30_original_required_families_executed_twice=True,
            paired_human_question_different_images_same_prompt_and_distinct_answers_executed=True,
            explicit_packed_response_and_image_targets_verified=True,
            exact_all_rank_RNG_residual_Adam_full_history_contract_and_cursors_resume=True,
            one_rank_held_input_rejected_before_updates=True,
            one_rank_changed_contract_blocks_snapshot_without_overwriting=True,
            one_rank_wrong_original_objective_rejects_before_mutation=True,
            frozen_backbone_unchanged=True, model_counts=raw.parameter_counts(),
            scope='Own8688-parameter CPU replica and explicit fixture examples; no real paired-source admission or full857M neural run.')))
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope='module')
def completed_artifacts(tmp_path_factory):
    folder = tmp_path_factory.mktemp('all30-source-contract')
    Image.new('RGB', (8, 4), (21, 55, 90)).save(folder / 'image.png')
    Image.new('RGB', (8, 4), (220, 85, 10)).save(folder / 'other.png')
    mp.spawn(worker, args=(str(folder / 'rendezvous'), str(folder)), nprocs=4, join=True)
    return folder


def test_four_actual_workers_keep_complete_source_contract_and_all30_objectives_through_resume(completed_artifacts):
    for rank in range(4):
        proof = json.loads((completed_artifacts / f'rank-{rank}.json').read_text())
        assert proof['all30_original_required_families_executed_twice']
        assert proof['one_rank_changed_contract_blocks_snapshot_without_overwriting']


def test_completed30_family_checkpoint_exports_without_losing_resumable_state_and_matches_logits(completed_artifacts):
    folder = completed_artifacts
    checkpoint = folder / 'complete.pt'
    before = file_hash(checkpoint)
    receipt = export_inference(checkpoint, folder / 'base.pt', folder / 'tokenizer.json', folder / 'plan.json',
        folder / 'data-manifest.json', folder / 'inference.pt', expected_training_checkpoint_sha256=before,
        maximum_bytes=1_000_000, minimum_free_bytes=0)
    assert file_hash(checkpoint) == before == receipt['complete_training_checkpoint_preserved_sha256']
    policy, metadata = load_inference(folder / 'inference.pt', folder / 'base.pt', folder / 'tokenizer.json',
        expected_checkpoint_sha256=receipt['checkpoint_sha256'], expected_base_sha256=file_hash(folder / 'base.pt'),
        expected_plan_sha256=file_hash(folder / 'plan.json'), expected_training_checkpoint_sha256=before,
        expected_data_sha256=file_hash(folder / 'data-manifest.json'))
    base, _ = load_checkpoint(folder / 'base.pt', 'cpu', training_state=False)
    original = RoutedCodeResidual(base, width=12)
    original.load_adapter_state(torch.load(checkpoint, weights_only=True)['residuals'])
    original.eval()
    ids = torch.tensor([[1, 10, 11, 12, 2]])
    with torch.no_grad():
        torch.testing.assert_close(policy(ids)['logits'], original(ids, route='code')['logits'], rtol=0, atol=0)
    assert metadata['optimizer_loaded_on_GPU'] is False
