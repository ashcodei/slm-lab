"""Four actual CPU workers exercise the production loop and durable generations."""
from contextlib import nullcontext
from datetime import timedelta
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp

from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
from slm_lab.code_residual_lifecycle_v4 import FORMAT, checkpoint_layout, rank_zero, run_updates
from slm_lab.code_residual_production_v4 import (ControllerBoundGuard, MODEL_COUNTS, PACKAGE,
    RUNTIME_FILES, FORMAT as PLAN_FORMAT, finish_inference, validate_production_plan, _merge)
import slm_lab.code_residual_production_v4 as production
from slm_lab.code_residual_source_contract_v4 import SourceContractResidualKernel
from slm_lab.code_residual_stage_v3 import ExpandedReplayEncoder
from slm_lab.codenet_checkpoint_v1 import FileBindings, capture_rng
from slm_lab.codenet_token_pack_v1 import packed_encoder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import TinyMultimodalLM
from slm_lab.recurrent_ablation import state_digest
from slm_lab.training import load_checkpoint

from test_code_residual_source_contract_v4 import Code, Replay, completed_artifacts
from test_codenet_runner_v1 import Tokenizer, options


class StochasticCaller:
    """Reproduce the three external per-step draws in the independent reference."""
    def __init__(self, kernel):
        self.kernel = kernel

    def __getattr__(self, name):
        return getattr(self.kernel, name)

    def update(self):
        random.random()
        np.random.random()
        torch.rand(())
        return self.kernel.update()


def loop_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4,
        timeout=timedelta(seconds=120))
    try:
        folder, output = Path(directory), Path(directory) / 'lifecycle'
        contract = json.loads((folder / 'replay-contract.json').read_text())
        plan_path = folder / 'plan.json'
        sources = dict(base_checkpoint_sha256=file_hash(folder / 'base.pt'), tokenizer_sha256=file_hash(folder / 'tokenizer.json'),
            run_plan_sha256=file_hash(plan_path))
        reference = torch.load(folder / 'complete.pt', weights_only=True)
        config = load_checkpoint(folder / 'base.pt', 'cpu', training_state=False)[0].config
        code, replay, tokenizer = Code(folder), Replay(folder), Tokenizer()
        opts = {**options(), 'passes': 2, 'gradient_accumulation': 1, 'replay_accumulation': 1,
            'image_size': 16, 'log_every': 9}
        if rank == 0:
            output.mkdir()
            shutil.copy2(folder / 'resume.pt', output / 'resume.pt')
            atomic_json(output / 'checkpoint-ledger.json', dict(format=FORMAT, run_plan_sha256=file_hash(plan_path),
                latest=dict(checkpoint=str(output / 'resume.pt'), checkpoint_sha256=file_hash(output / 'resume.pt'),
                    checkpoint_bytes=(output / 'resume.pt').stat().st_size, steps=31), previous=None))
            (folder / 'code-pack').mkdir()
            shutil.copy2(folder / 'data-manifest.json', folder / 'code-pack/manifest.json')
        dist.barrier()

        def construct(checkpoint_sha256):
            torch.manual_seed(50189)
            base, _ = load_checkpoint(folder / 'base.pt', 'cpu', training_state=False)
            model = RoutedCodeResidual(base, width=12)
            optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=opts['learning_rate'], weight_decay=opts['weight_decay'], foreach=False)
            kernel = SourceContractResidualKernel(model, optimizer, code, packed_encoder(code, tokenizer, config),
                replay, ExpandedReplayEncoder(tokenizer, config, folder, image_size=16), opts, sources, 'cpu', nullcontext,
                replay_contract=contract, expected_contract_fingerprint=fingerprint(contract))
            kernel.restore(output / 'resume.pt', checkpoint_sha256)
            return model, optimizer, StochasticCaller(kernel)

        original_partial = file_hash(output / 'resume.pt')
        model, optimizer, kernel = construct(original_partial)
        partial = run_updates(kernel, output, maximum_checkpoint_bytes=1_000_000, minimum_free_bytes=0,
            checkpoint_every=5, log_every=7, until_step=43, maximum_report_bytes=1_000_000)
        assert partial['completed_steps'] == 43 and partial['planned_steps'] == 60 and partial['training_complete'] is False
        assert len(kernel.metrics) == 43 and not (output / 'training-completed.json').exists()
        ledger = rank_zero(lambda: checkpoint_layout(output, sources['run_plan_sha256']))
        assert ledger['latest']['steps'] == 43 and ledger['previous']['steps'] == 40
        before = {name: file_hash(output / name) for name in ['resume.pt', 'previous.pt', 'checkpoint-ledger.json']}
        with pytest.raises(ValueError, match='Collective snapshot failed'):
            run_updates(kernel, output, maximum_checkpoint_bytes=100, minimum_free_bytes=0,
                checkpoint_every=5, log_every=7, until_step=44, maximum_report_bytes=1_000_000)
        assert all(file_hash(output / name) == digest for name, digest in before.items())
        assert not (output / 'pending.pt').exists() and not (output / 'pending.pt.pending').exists()
        model, optimizer, kernel = construct(before['resume.pt'])
        result = run_updates(kernel, output, maximum_checkpoint_bytes=1_000_000, minimum_free_bytes=0,
            checkpoint_every=5, log_every=7, maximum_report_bytes=1_000_000)
        assert result['training_complete'] and result['completed_steps'] == result['planned_steps'] == 60
        assert state_digest(model.residuals.state_dict()) == state_digest(reference['residuals'])
        assert fingerprint(optimizer.state_dict()) == fingerprint(reference['optimizer'])
        assert kernel.metrics == reference['metrics']
        # The reference consumes one diagnostic next draw before its final save.
        random.random()
        np.random.random()
        torch.rand(())
        assert fingerprint(capture_rng('cpu')) == fingerprint(reference['worker_rng_states'][rank])
        ledger = rank_zero(lambda: checkpoint_layout(output, sources['run_plan_sha256']))
        assert ledger['latest']['steps'] == 60 and ledger['previous']['steps'] == 55
        assert {path.name for path in output.glob('*.pt')} == {'resume.pt', 'previous.pt'}
        final_hash = file_hash(output / 'resume.pt')
        inline_plan = dict(base_checkpoint=str(folder / 'base.pt'), tokenizer=str(folder / 'tokenizer.json'),
            base_checkpoint_sha256=sources['base_checkpoint_sha256'], code_manifest_sha256=file_hash(folder / 'data-manifest.json'),
            code_pack=str(folder / 'code-pack'), storage=dict(maximum_inference_bytes=1_000_000,
                minimum_free_bytes=0, maximum_report_bytes=1_000_000))
        receipt = rank_zero(lambda: finish_inference(output, inline_plan, plan_path, result))
        completion_hash = file_hash(output / 'training-completed.json')
        assert rank_zero(lambda: finish_inference(output, inline_plan, plan_path, result)) == receipt
        assert file_hash(output / 'training-completed.json') == completion_hash
        assert file_hash(output / 'resume.pt') == final_hash
        if rank == 0:
            exported = torch.load(output / 'inference.pt', weights_only=True)
            good = (output / 'inference.pt').read_bytes()
            (output / 'inference-good.fixture').write_bytes(good)
            next(iter(exported['residuals'].values())).flatten()[0] += .01
            exported['residual_state_sha256'] = state_digest(exported['residuals'])
            torch.save(exported, output / 'inference.pt')
        dist.barrier()
        bad_hash = file_hash(output / 'inference.pt')
        with pytest.raises(ValueError, match='Existing inference weights differ'):
            rank_zero(lambda: finish_inference(output, inline_plan, plan_path, result))
        assert file_hash(output / 'inference.pt') == bad_hash and file_hash(output / 'resume.pt') == final_hash
        dist.barrier()
        if rank == 0:
            (output / 'inference.pt').write_bytes((output / 'inference-good.fixture').read_bytes())
        dist.barrier()
        assert rank_zero(lambda: finish_inference(output, inline_plan, plan_path, result)) == receipt
        (folder / f'lifecycle-rank-{rank}.json').write_text(json.dumps(dict(rank=rank,
            actual_loop_updates=30, restored_partial_step=31, first_stop_step=43,
            failed_checkpoint_at_step44_preserves_latest_and_previous=True,
            explicit_same_checkpoint_resume_reproduces_independent_reference_weights_Adam_RNG_and_full_history=True,
            final_step=60, previous_generation_step=55, pending_file_absent_after_success=True,
            idempotent_completed_export_preserves_checkpoint_and_completion_report=True,
            unexplained_existing_inference_mutation_rejected_without_overwrite=True,
            scope='Own8688-parameter replica and CPU fixtures; production loop/exact checkpoints/export only, not actual857M training or ready final-source plan.')))
    finally:
        dist.destroy_process_group()


def test_actual_four_worker_production_loop_rotation_failure_explicit_resume_and_idempotent_export(completed_artifacts):
    folder = completed_artifacts
    mp.spawn(loop_worker, args=(str(folder / 'lifecycle-rendezvous'), str(folder)), nprocs=4, join=True)
    for rank in range(4):
        assert json.loads((folder / f'lifecycle-rank-{rank}.json').read_text())['failed_checkpoint_at_step44_preserves_latest_and_previous']


def plan_fixture(tmp_path):
    names = ['base_checkpoint', 'tokenizer', 'code_pack', 'source_bundle', 'final_base_binding', 'capacity_proof', 'evaluation_binding', 'output']
    plan = dict(format=PLAN_FORMAT, world_size=4, residual_width=2048, expected_model_counts=MODEL_COUNTS,
        training_options=options(), training_ready=True, automatic_candidate_promotion=False,
        gradient_reduction='fixed_parameter_order_worker_average_v1',
        replay_selection='full_pass_floor_then_uniform_extra_family_updates_v1', replay_contract_fingerprint='a' * 64,
        source_files_sha256={str(PACKAGE / name): 'a' * 64 for name in RUNTIME_FILES},
        storage=dict(maximum_checkpoint_bytes=2 * 1024**3, maximum_inference_bytes=768 * 1024**2,
            maximum_report_bytes=512 * 1024**2, minimum_free_bytes=26 * 1024**3),
        preceding_processes=[dict(role=role, pid=100 + i, start_ticks='123') for i, role in enumerate(['main', 'joint', 'feedback', 'resumer'])])
    plan.update({name: str(tmp_path / name) for name in names})
    for name in ['base_checkpoint', 'tokenizer', 'code_manifest', 'code_admission', 'source_bundle_manifest', 'final_base_binding', 'capacity_proof', 'evaluation_binding']:
        plan[name + '_sha256'] = 'a' * 64
    return plan


def test_incomplete_or_unfunded_production_plan_rejects_before_any_launch(tmp_path):
    plan = plan_fixture(tmp_path)
    assert validate_production_plan(plan) == plan['training_options']
    for field, value in [('training_ready', False), ('automatic_candidate_promotion', True), ('residual_width', 4096)]:
        bad = {**plan, field: value}
        with pytest.raises(ValueError):
            validate_production_plan(bad)
    bad = {**plan, 'storage': {**plan['storage'], 'minimum_free_bytes': 25 * 1024**3}}
    with pytest.raises(ValueError, match='26GiB'):
        validate_production_plan(bad)
    bad = {**plan, 'preceding_processes': plan['preceding_processes'][:-1]}
    with pytest.raises(ValueError, match='birth identities'):
        validate_production_plan(bad)


def test_conflicting_transitive_runtime_or_baseline_evidence_cannot_override_a_binding(tmp_path):
    path = str(tmp_path / 'source.py')
    bindings = {path: 'a' * 64}
    with pytest.raises(ValueError, match='Conflicting'):
        _merge(bindings, {path: 'b' * 64})
    assert bindings[path] == 'a' * 64


def test_controller_rejects_unfinalized_real_sources_before_GPU_queries_or_output_creation(tmp_path, monkeypatch):
    plan = plan_fixture(tmp_path)
    bundle = Path(plan['source_bundle'])
    bundle.mkdir()
    (bundle / 'manifest.json').write_text(json.dumps(dict(format='own-complete-code-final-source-bundle-v4',
        all30_actual_prepared_families_opened=False, full_original_paired_admission_completed=False)))
    plan['source_bundle_manifest_sha256'] = file_hash(bundle / 'manifest.json')
    plan_path = tmp_path / 'run-plan.json'
    plan_path.write_text(json.dumps(plan))
    def forbidden(*args, **kwargs):
        raise AssertionError('Unfinalized source evidence must reject before querying GPUs or launching workers')
    monkeypatch.setattr(production.subprocess, 'run', forbidden)
    with pytest.raises(ValueError, match='actual30-family'):
        production.controller(plan_path)
    assert not Path(plan['output']).exists()


def test_controller_rechecks_output_under_lease_before_launching_after_a_race(tmp_path, monkeypatch):
    plan = plan_fixture(tmp_path)
    plan['output'] = str(tmp_path / 'checkpoints/new-stage')
    (tmp_path / 'reports').mkdir()
    plan_path = tmp_path / 'run-plan.json'
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(production, 'ROOT', tmp_path)
    monkeypatch.setattr(production, 'bind_completed_inputs', lambda value: ({}, SimpleNamespace(check=lambda **kw: None), {}))
    monkeypatch.setattr(production, '_process', lambda *args: False)
    monkeypatch.setattr(production.shutil, 'disk_usage', lambda path: SimpleNamespace(free=100 * 1024**3))
    original_lock = production.fcntl.flock
    def interleaved_lock(fd, operation):
        original_lock(fd, operation)
        output = Path(plan['output'])
        output.mkdir(parents=True)
        (output / 'other-writer.fixture').write_bytes(b'owned test interleaving')
    monkeypatch.setattr(production.fcntl, 'flock', interleaved_lock)
    def forbidden(*args, **kwargs):
        raise AssertionError('A newly occupied output must reject before GPU query/worker launch')
    monkeypatch.setattr(production.subprocess, 'run', forbidden)
    with pytest.raises(ValueError, match='became occupied'):
        production.controller(plan_path)
    assert (Path(plan['output']) / 'other-writer.fixture').read_bytes() == b'owned test interleaving'


def test_guard_follows_real_parent_birth_and_rejects_after_parent_exits(tmp_path):
    source = tmp_path / 'source.py'
    source.write_text('fixture source\n')
    child = subprocess.Popen([sys.executable, '-c', 'import sys;sys.stdin.read()'], stdin=subprocess.PIPE)
    try:
        proc = Path('/proc') / str(child.pid)
        ticks = (proc / 'stat').read_text().split(') ', 1)[1].split()[19]
        owner = dict(pid=child.pid, start_ticks=ticks, parent_cmdline_sha256=file_hash(proc / 'cmdline'))
        guard = ControllerBoundGuard(FileBindings({str(source): file_hash(source)}), owner)
        guard.check(full=True)
        child.stdin.close()
        child.wait(timeout=5)
        with pytest.raises(ValueError, match='parent controller ended'):
            guard.check()
    finally:
        if child.poll() is None:
            child.stdin.close()
            child.wait(timeout=5)
