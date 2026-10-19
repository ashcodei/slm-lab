"""Capacity evidence uses actual four-worker kernels, never a fixture as857M proof."""
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp

import slm_lab.code_residual_capacity_v4 as capacity
from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
from slm_lab.code_residual_source_contract_v4 import (COMPONENT_NAMES, SourceContractResidualKernel,
    build_required_contract)
from slm_lab.code_residual_stage_v3 import ExpandedReplayEncoder
from slm_lab.codenet_token_pack_v1 import packed_encoder
from slm_lab.recurrent_ablation import state_digest

from test_code_residual_source_contract_v4 import Code, Replay
from test_codenet_runner_v1 import Tokenizer, options, small_config


def probe_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4,
        timeout=timedelta(seconds=90))
    try:
        folder = Path(directory)
        config, token, code, replay = small_config(), Tokenizer(), Code(folder), Replay(folder)
        torch.manual_seed(23471)
        model = RoutedCodeResidual(capacity.load_checkpoint(folder / 'base.pt', 'cpu', training_state=False)[0], width=12)
        opts = {**options(), 'passes': 2, 'gradient_accumulation': 1, 'replay_accumulation': 1, 'image_size': 16}
        optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=opts['learning_rate'],
            weight_decay=opts['weight_decay'], foreach=False)
        digest = hashlib.sha256(b'fixture').hexdigest()
        sources = dict(base_checkpoint_sha256=digest, tokenizer_sha256=digest, run_plan_sha256=digest)
        contract = build_required_contract(replay, digest, {name: digest for name in COMPONENT_NAMES})
        kernel = SourceContractResidualKernel(model, optimizer, code, packed_encoder(code, token, config),
            replay, ExpandedReplayEncoder(token, config, folder, 16), opts, sources, 'cpu', nullcontext,
            replay_contract=contract, expected_contract_fingerprint=fingerprint(contract))
        before = fingerprint(dict(model=model.state_dict(), optimizer=optimizer.state_dict()))
        replay.held = rank == 1
        with pytest.raises(ValueError, match='Collective preflight failed'):
            capacity.exercise_capacity(kernel, 0, {}, 'response')
        assert fingerprint(dict(model=model.state_dict(), optimizer=optimizer.state_dict())) == before
        assert kernel.completed == 0
        replay.held = False
        base_before = state_digest(model.base.state_dict())
        image_row = replay.row('paired_human_photo_questions', rank % 2)
        proof = capacity.exercise_capacity(kernel, 0, image_row, 'response')
        assert proof['actual_full_kernel_optimizer_updates'] == kernel.completed == 1
        assert len(kernel.metrics) == 1  # Noncontiguous diagnostic updates never become a training history.
        assert proof['additional_noncontiguous_disposable_probe_updates'] == 29
        assert len(proof['all_declared_source_families_exercised']) == 30
        assert {m['replay_objective'] for m in proof['probe_metrics']} == {'causal_pretraining', 'response_only'}
        assert proof['actual_maximum_context_backward'] == 38
        assert proof['actual_image_batch_sizes'] == [[16, 16]]
        assert proof['combined_extreme_backward']['objectives']['maximum_complete_code']['global_target_tokens'] == 144
        assert proof['combined_extreme_backward']['objectives']['maximum_original_image']['global_target_tokens'] == 10
        assert state_digest(model.base.state_dict()) == base_before
        assert all(p.grad is None for p in model.parameters())
        assert set(p.name for p in folder.glob('*.pt')) == {'base.pt'}
        weights = [None] * 4
        dist.all_gather_object(weights, state_digest(model.residuals.state_dict()))
        assert len(set(weights)) == 1
        (folder / f'capacity-rank-{rank}.json').write_text(json.dumps(dict(**proof,
            model_counts=model.parameter_counts(), same_residuals_across_ranks=True,
            single_rank_held_input_rejected_before_update=True,
            actual857M_CUDA_capacity_verified=False)))
    finally:
        dist.destroy_process_group()


def test_actual_four_worker_capacity_protocol_exercises_all_sources_and_combined_extremes(tmp_path):
    from dataclasses import asdict
    from slm_lab.model import TinyMultimodalLM
    torch.manual_seed(6541)
    config = small_config()
    torch.save(dict(model=TinyMultimodalLM(config).state_dict(), config=asdict(config)), tmp_path / 'base.pt')
    Image.new('RGB', (8, 4), (21, 55, 90)).save(tmp_path / 'image.png')
    Image.new('RGB', (8, 4), (220, 85, 10)).save(tmp_path / 'other.png')
    mp.spawn(probe_worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path)), nprocs=4, join=True)
    for rank in range(4):
        proof = json.loads((tmp_path / f'capacity-rank-{rank}.json').read_text())
        assert proof['candidate_weights_saved'] is False
        assert proof['actual857M_CUDA_capacity_verified'] is False


def plan_fixture(tmp_path):
    digest = hashlib.sha256(b'fixture').hexdigest()
    plan = dict(format=capacity.FORMAT, world_size=4, residual_width=2048,
        expected_model_counts=capacity.MODEL_COUNTS, training_options={**options(), 'image_size': 448},
        candidate_export=False, replay_contract_fingerprint=digest,
        preceding_processes=[dict(role=role, pid=10 + i, start_ticks='1')
            for i, role in enumerate(('main', 'joint', 'feedback', 'resumer'))],
        source_files_sha256={str(capacity.PACKAGE / name): digest for name in capacity.RUNTIME_FILES}
            | {str(Path(capacity.__file__)): digest},
        storage=dict(maximum_report_bytes=1024**2, minimum_free_bytes=26 * 1024**3))
    for name in ('base_checkpoint', 'tokenizer', 'code_pack', 'source_bundle', 'final_base_binding', 'output'):
        plan[name] = str(tmp_path / name)
    for name in ('base_checkpoint', 'tokenizer', 'code_manifest', 'code_admission', 'source_bundle_manifest', 'final_base_binding'):
        plan[name + '_sha256'] = digest
    return plan


def test_unfinalized_sources_reject_before_gpu_query_or_output_creation(tmp_path, monkeypatch):
    plan = plan_fixture(tmp_path)
    bundle = Path(plan['source_bundle'])
    bundle.mkdir()
    (bundle / 'manifest.json').write_text(json.dumps(dict(format='own-complete-code-final-source-bundle-v4',
        all30_actual_prepared_families_opened=False)))
    plan['source_bundle_manifest_sha256'] = capacity.file_hash(bundle / 'manifest.json')
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    def forbidden(*args, **kwargs):
        pytest.fail('Unfinalized source touched GPU inspection or launched workers')
    monkeypatch.setattr(capacity.subprocess, 'run', forbidden)
    with pytest.raises(ValueError, match='Actual complete final-source bundle'):
        capacity.controller(path)
    assert not Path(plan['output']).exists()


def test_original_predecessor_blocks_capacity_before_gpu_inspection(tmp_path, monkeypatch):
    plan = plan_fixture(tmp_path)
    plan['output'] = str(capacity.ROOT / 'reports' / ('uncreated-capacity-fixture-' + tmp_path.name))
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(capacity, 'bind_inputs', lambda p: ({}, None, {}))
    monkeypatch.setattr(capacity, '_process', lambda pid, ticks: pid == plan['preceding_processes'][0]['pid'])
    monkeypatch.setattr(capacity.subprocess, 'run', lambda *a, **k: pytest.fail('Live predecessor reached GPU inspection'))
    with pytest.raises(ValueError, match='Original GPU predecessor remains live'):
        capacity.controller(path)
    assert not Path(plan['output']).exists()
