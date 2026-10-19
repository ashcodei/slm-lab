"""Real four-worker lifecycle/capacity mechanics and prelaunch data gates."""
from contextlib import nullcontext
from copy import deepcopy
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

from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
from slm_lab.code_residual_lifecycle_v4 import checkpoint_layout, rank_zero
from slm_lab.codenet_checkpoint_v1 import FileBindings, capture_rng
from slm_lab.codenet_runner_v1 import VerifiedReplayEncoder
from slm_lab.corpus import file_hash
from slm_lab.model import TinyMultimodalLM
from slm_lab.paired_visual_capacity_v1 import exercise_capacity
from slm_lab.paired_visual_lifecycle_v1 import run_visual_stage, finish_visual_export
from slm_lab.paired_visual_production_v1 import FORMAT, MODEL_COUNTS, PACKAGE, RUNTIME_FILES, validate_plan, validate_capacity_proof, bind_visual_baselines
import slm_lab.paired_visual_production_v1 as production
from slm_lab.paired_visual_training_v1 import PairedVisualKernel, freeze_for_visual_stage
from slm_lab.training import load_checkpoint
from test_codenet_runner_v1 import small_config, Tokenizer
from test_paired_visual_training_v1 import Pairs, Replay


def lifecycle_worker(rank, rendezvous, directory):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=180))
    try:
        folder, output = Path(directory), Path(directory) / 'lifecycle'
        if rank == 0:
            Image.new('RGB', (16, 16), (35, 170, 60)).save(folder / 'image.png')
            (folder / 'source.json').write_text(json.dumps(dict(fixture=True, real_source_admission=False)))
        dist.barrier()
        guard = FileBindings({str(folder / n): file_hash(folder / n) for n in ['source.json', 'image.png']})
        pairs, replay = Pairs(guard), Replay(folder, guard)
        opts = dict(steps=16, seed=438, pair_batch_size=1, pair_accumulation=1, replay_batch_size=1, replay_accumulation=1,
            image_size=16, learning_rate=.001, weight_decay=.01, gradient_clip=1., alpha=.3, beta=2., margin=.1, replay_coefficient=.5)
        sources = dict(base_checkpoint_sha256='a' * 64, tokenizer_sha256='b' * 64, run_plan_sha256='c' * 64)
        def construct():
            torch.manual_seed(49216)
            model = TinyMultimodalLM(small_config())
            optimizer = torch.optim.AdamW(freeze_for_visual_stage(model), lr=opts['learning_rate'], weight_decay=opts['weight_decay'], foreach=False)
            kernel = PairedVisualKernel(model, optimizer, pairs, replay,
                VerifiedReplayEncoder(Tokenizer(), model.config, folder, 16), opts, sources, 'cpu', nullcontext, guard=guard)
            random.seed(1000 + rank); np.random.seed(2000 + rank); torch.manual_seed(3000 + rank)
            return model, optimizer, kernel
        model, optimizer, reference = construct()
        for _ in range(16): reference.update()
        reference.snapshot(folder / 'reference.pt', maximum_bytes=5_000_000, minimum_free_bytes=0)
        expected = fingerprint(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), metrics=reference.metrics,
            sampler=reference.sampler_state(16), rng=capture_rng('cpu')))
        model, optimizer, kernel = construct()
        storage = dict(maximum_checkpoint_bytes=5_000_000, minimum_free_bytes=0,
            checkpoint_every=3, log_every=5, maximum_report_bytes=2_000_000)
        partial = run_visual_stage(kernel, output, until_step=7, **storage)
        assert partial['completed_steps'] == 7 and not partial['training_complete']
        with pytest.raises(ValueError, match='Complete original clean visual training'):
            finish_visual_export(kernel, output, partial, maximum_inference_bytes=5_000_000,
                minimum_free_bytes=0, maximum_report_bytes=2_000_000)
        assert not (output / 'inference.pt').exists()
        ledger = rank_zero(lambda: checkpoint_layout(output, sources['run_plan_sha256']))
        assert ledger['latest']['steps'] == 7 and ledger['previous']['steps'] == 6
        before = {n: file_hash(output / n) for n in ['resume.pt', 'previous.pt', 'checkpoint-ledger.json']}
        fresh_model, fresh_optimizer, fresh = construct()
        with pytest.raises(ValueError, match='requires explicit recovery'):
            run_visual_stage(fresh, output, **storage)
        with pytest.raises(ValueError, match='latest complete journaled state'):
            run_visual_stage(fresh, output, resume_sha256='f' * 64, **storage)
        with pytest.raises(ValueError, match='Collective visual snapshot failed'):
            run_visual_stage(fresh, output, resume_sha256=before['resume.pt'], until_step=8,
                **{**storage, 'maximum_checkpoint_bytes': 1})
        assert all(file_hash(output / n) == h for n, h in before.items())
        assert not (output / 'pending.pt').exists() and not (output / 'pending.pt.pending').exists()
        model, optimizer, kernel = construct()
        result = run_visual_stage(kernel, output, resume_sha256=before['resume.pt'], **storage)
        assert result['training_complete'] and kernel.completed == 16
        assert fingerprint(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), metrics=kernel.metrics,
            sampler=kernel.sampler_state(16), rng=capture_rng('cpu'))) == expected
        ledger = rank_zero(lambda: checkpoint_layout(output, sources['run_plan_sha256']))
        assert ledger['latest']['steps'] == 16 and ledger['previous']['steps'] == 15
        final_sha = file_hash(output / 'resume.pt')
        receipt = finish_visual_export(kernel, output, result, maximum_inference_bytes=5_000_000,
            minimum_free_bytes=0, maximum_report_bytes=2_000_000)
        completion_sha = file_hash(output / 'training-completed.json')
        assert finish_visual_export(kernel, output, result, maximum_inference_bytes=5_000_000,
            minimum_free_bytes=0, maximum_report_bytes=2_000_000) == receipt
        assert file_hash(output / 'training-completed.json') == completion_sha
        assert file_hash(output / 'resume.pt') == final_sha
        assert {p.name for p in output.glob('*.pt')} == {'resume.pt', 'previous.pt', 'inference.pt'}
        if rank == 0:
            native, metadata = load_checkpoint(output / 'inference.pt', 'cpu', training_state=False)
            native.eval(); model.eval(); ids = torch.tensor([[1, 5, 11, 2]])
            image = pairs.example(0)['images'][0][None]
            with torch.no_grad():
                for pixels in [None, image]:
                    torch.testing.assert_close(native(ids, pixel_values=pixels)['logits'], model(ids, pixel_values=pixels)['logits'], atol=0, rtol=0)
                torch.testing.assert_close(native.prefill(ids, pixel_values=image)['logits'], model.prefill(ids, pixel_values=image)['logits'], atol=0, rtol=0)
            exported = torch.load(output / 'inference.pt', weights_only=True)
            assert 'optimizer' not in exported and 'worker_rng_states' not in exported
            assert exported['complete_training_checkpoint_sha256'] == final_sha
            (folder / 'native-export-good.pt').write_bytes((output / 'inference.pt').read_bytes())
            exported['model']['vision.patch.weight'].flatten()[0] += .01
            torch.save(exported, output / 'inference.pt')
        dist.barrier()
        bad_sha = file_hash(output / 'inference.pt')
        with pytest.raises(ValueError, match='Existing visual inference differs'):
            finish_visual_export(kernel, output, result, maximum_inference_bytes=5_000_000,
                minimum_free_bytes=0, maximum_report_bytes=2_000_000)
        assert file_hash(output / 'inference.pt') == bad_sha and file_hash(output / 'resume.pt') == final_sha
        # Every rank must finish inspecting the deliberately corrupted bytes
        # before rank zero restores the test fixture.
        dist.barrier()
        if rank == 0:
            (output / 'inference.pt').write_bytes((folder / 'native-export-good.pt').read_bytes())
        dist.barrier()

        # Exercise the exact capacity helper, with honest small-model/CPU scope.
        disposable, _, capacity_kernel = construct()
        capacity = exercise_capacity(capacity_kernel, [0, 4], {n: 4 for n in replay.families})
        assert capacity['actual_full_kernel_optimizer_updates'] == 1 and len(capacity['probes']) == 16
        assert len(capacity['all_eight_visual_families_exercised']) == 8
        assert capacity['high_consensus_binary_contrast_exercised']
        assert capacity['original_language_critic_unchanged'] and not capacity['candidate_weights_saved']
        (folder / f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank,
            actual_four_CPU_workers=True, model_counts=model.parameter_counts(), complete_reference_updates=16,
            first_stop_step=7, failed_checkpoint_step8_preserves_latest_previous_and_ledger=True,
            implicit_restart_and_wrong_resume_rejected=True, exact_explicit_resume_matches_reference=True,
            final_step=16, previous_step=15, native_text_image_and_cached_logits_match_export=True,
            idempotent_export_preserves_training_and_completion_report=True,
            unexplained_inference_mutation_rejected_without_overwrite=True,
            disposable_capacity=capacity,
            scope='Own8304-parameter CPU model and synthetic source/image fixtures; not real finalized-source/full706M GPU training or capacity.'), indent=2))
    finally:
        dist.destroy_process_group()


def test_actual_four_worker_visual_lifecycle_and_disposable_capacity(tmp_path):
    mp.spawn(lifecycle_worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path)), nprocs=4, join=True)
    for rank in range(4):
        assert json.loads((tmp_path / f'rank-{rank}.json').read_text())['exact_explicit_resume_matches_reference']


def plan_fixture(tmp_path):
    opts = dict(steps=5000, seed=438, pair_batch_size=1, pair_accumulation=1, replay_batch_size=1, replay_accumulation=16,
        image_size=448, learning_rate=.0001, weight_decay=.01, gradient_clip=1., alpha=.3, beta=2., margin=.1, replay_coefficient=.5)
    paths = ['base_checkpoint', 'tokenizer', 'code_pack', 'source_bundle', 'final_base_binding', 'output',
        'capacity_proof', 'evaluation_binding', 'visual_evaluation_binding']
    plan = dict(format=FORMAT, world_size=4, expected_model_counts=MODEL_COUNTS, training_options=opts,
        automatic_candidate_promotion=False, training_ready=True, checkpoint_every=25, log_every=10,
        pair_contract_sha256='a' * 64, visual_replay_contract_sha256='a' * 64, replay_contract_fingerprint='a' * 64,
        source_files_sha256={str(PACKAGE / (n + '.py')): file_hash(PACKAGE / (n + '.py')) for n in RUNTIME_FILES},
        preceding_processes=[dict(role=n, pid=100 + i, start_ticks='123') for i, n in enumerate(['main', 'joint', 'feedback', 'resumer'])],
        storage=dict(maximum_checkpoint_bytes=4 * 1024**3, maximum_inference_bytes=3 * 1024**3,
            maximum_report_bytes=512 * 1024**2, minimum_free_bytes=26 * 1024**3))
    plan.update({n: str(tmp_path / n) for n in paths})
    for n in ['base_checkpoint', 'tokenizer', 'code_manifest', 'code_admission', 'source_bundle_manifest', 'final_base_binding',
            'paired_manifest', 'capacity_proof', 'evaluation_binding', 'visual_evaluation_binding']:
        plan[n + '_sha256'] = 'a' * 64
    return plan


def test_unfinalized_source_rejects_before_GPU_query_or_output(tmp_path, monkeypatch):
    plan = plan_fixture(tmp_path)
    data = Path(plan['source_bundle']); data.mkdir()
    (data / 'manifest.json').write_text(json.dumps(dict(format='own-complete-code-final-source-bundle-v4',
        all30_actual_prepared_families_opened=False, full_original_paired_admission_completed=False)))
    plan['source_bundle_manifest_sha256'] = file_hash(data / 'manifest.json')
    path = tmp_path / 'plan.json'; path.write_text(json.dumps(plan))
    def forbidden(*args, **kwargs): raise AssertionError('GPU query or worker launch before final-source admission')
    monkeypatch.setattr(production.subprocess, 'run', forbidden)
    with pytest.raises(ValueError, match='complete finalized30-family'):
        production.controller(path)
    assert not Path(plan['output']).exists()


def test_incomplete_underfunded_or_wrong_parameter_plan_rejects(tmp_path):
    plan = plan_fixture(tmp_path); validate_plan(plan)
    for field, value in [('training_ready', False), ('world_size', 1), ('automatic_candidate_promotion', True),
            ('expected_model_counts', dict(total=8304))]:
        with pytest.raises(ValueError): validate_plan({**plan, field: value})
    with pytest.raises(ValueError, match='4GiB'):
        validate_plan({**plan, 'storage': {**plan['storage'], 'maximum_checkpoint_bytes': 2 * 1024**3}})
    with pytest.raises(ValueError):
        validate_plan({**plan, 'training_options': {**plan['training_options'], 'image_size': 224}})
    with pytest.raises(ValueError):
        validate_plan({**plan, 'preceding_processes': plan['preceding_processes'][:-1]})


def test_CPU_or_missing_capacity_cannot_authorize_production(tmp_path):
    plan = plan_fixture(tmp_path)
    with pytest.raises(ValueError, match='full706M/four-L40'):
        validate_capacity_proof(dict(format=production.PROOF_FORMAT, passed=True, actual_world_size=4,
            model_counts=dict(total=8304), fixture=True), plan, {})


def test_actual_native_visual_baseline_binding_and_wrong_policy_coverage(tmp_path):
    root = Path(__file__).resolve().parents[1]
    reports = {n: root / 'reports/extended-curriculum-v1' / (n + '.json') for n in ['vision', 'ocr', 'objects']}
    baseline = json.loads(reports['vision'].read_text())
    binding = dict(format='own-paired-visual-baseline-binding-v1', baseline_completed=True,
        baseline_checkpoint_sha256=baseline['checkpoint_sha256'], reports={n: str(p) for n, p in reports.items()},
        files_sha256={str(p): file_hash(p) for p in reports.values()})
    path = tmp_path / 'binding.json'; path.write_text(json.dumps(binding))
    plan = dict(visual_evaluation_binding=str(path), visual_evaluation_binding_sha256=file_hash(path),
        base_checkpoint_sha256=baseline['checkpoint_sha256'], tokenizer_sha256=baseline['tokenizer_sha256'])
    bound = bind_visual_baselines(plan)
    assert all(str(p) in bound for p in reports.values())
    with pytest.raises(ValueError, match='exact-own-policy'):
        bind_visual_baselines({**plan, 'base_checkpoint_sha256': 'f' * 64})
    short = json.loads(reports['ocr'].read_text()); short['results'] = short['results'][:1]
    (tmp_path / 'short.json').write_text(json.dumps(short))
    binding['reports']['ocr'] = str(tmp_path / 'short.json')
    binding['files_sha256'][str(tmp_path / 'short.json')] = file_hash(tmp_path / 'short.json')
    path.write_text(json.dumps(binding)); plan['visual_evaluation_binding_sha256'] = file_hash(path)
    with pytest.raises(ValueError, match='Complete original448px'):
        bind_visual_baselines(plan)
