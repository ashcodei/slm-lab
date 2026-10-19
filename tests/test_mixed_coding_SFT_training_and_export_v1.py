"""Actual own-small four-rank coding controller, interrupted recovery and export.

The sources are real admitted train data; these small models test mechanics.
No full model, new GPUcontext, production admission or accuracy gain is implied.
"""
import copy
from datetime import timedelta
import json
from pathlib import Path
import random
import shutil

import numpy as np
import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.batched_reasoning_capacity_controller_v1 import live
from slm_lab.bounded_gzip_observation_stream_v1 import lines
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.corpus import file_hash
from slm_lab.deterministic_positions_v1 import install
from slm_lab.mixed_coding_SFT_capacity_controller_v1 import run_capacity
from slm_lab.mixed_coding_SFT_data_v1 import MixedCodingSFTData, PROFILES
from slm_lab.mixed_coding_SFT_training_controller_v1 import read_capacity, run_training
from slm_lab.mixed_coding_SFT_native_export_v1 import export, FULL
from slm_lab.own_dense_standalone_v1 import load
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.parallel_mixed_coding_SFT_stage_v1 import ParallelMixedCodingSFTStage
from slm_lab.parallel_repair_durable_state_v1 import read_latest
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_mixed_coding_SFT_capacity_controller_v1 import student as original_student
from test_parallel_mixed_coding_SFT_stage_v1 import ROOT, OPTIONS, SOURCES, source_positions
from test_public_reasoning_group_gradients_v1 import optimizer


def student():
    result = original_student()
    install(result.base)
    # The original dense value head must survive actor-only continuation;
    # the separate research critic starts and stays zero. These are fixtures.
    with torch.no_grad():
        result.base.network.value_head.weight.copy_(torch.linspace(-.02, .03, result.config.hidden_size).reshape(1, -1))
    return result


def run(stage, folder, plan, reports, readbacks, *, progress=None, pointer=None, opt_in=True):
    return run_training(stage, folder / 'durable', folder / 'attempt.jsonl.gz', capacity_plan=plan,
        reports=reports, readbacks=readbacks, checkpoint_every=1, maximum_state_bytes=32*1024**2,
        minimum_free_bytes=0, maximum_journal_raw_bytes=32*1024**2, preflight=lambda: stage.data.guard.check(),
        progress=progress, expected_resume_pointer_sha256=pointer, allow_mechanics=opt_in)


def native_export(trained, receipt, folder, inputs):
    state, _, _ = read_latest(folder / 'complete/durable')
    hot = trained.model.eval()
    original, rng = tensor_digest(hot.named_parameters()), value_digest(capture_rng('cpu'))
    original_critic = hot.base.network.value_head.weight.detach().clone()
    assert bool((original_critic != 0).any()) and bool((hot.parts['critic'].weight == 0).all())
    tokenizer = ROOT / 'data/public-weak-code-SFT-v1/tokenizer.json'
    native, payload, proof = export(hot, state, receipt, sources=SOURCES, tokenizer_sha256=file_hash(tokenizer),
        working_budget_bytes=256*1024**2, allow_mechanics=True)
    assert tensor_digest(hot.named_parameters()) == original
    assert torch.equal(native.network.value_head.weight, original_critic)
    assert native.counts()['total_unique'] == 267430 and hot.counts()['actual_unique'] == 267718
    assert not ({p.data_ptr() for p in native.parameters()} & {p.data_ptr() for p in hot.parameters()})
    assert proof['no_external_model_weights_or_live_teacher_API'] and proof['public_precomputed_teacher_text_in_SFT']
    assert proof['generated_synthetic_tests_never_used_as_RL_oracle'] and not proof['whole_production_source_schedule_completed']
    path = folder / 'native'
    path.mkdir()
    shutil.copyfile(tokenizer, path / 'tokenizer.json')
    torch.save(payload, path / 'model.pt')
    cold, metadata = load(path / 'model.pt', expected_checkpoint_sha256=file_hash(path / 'model.pt'), expected_sources=SOURCES)
    assert tensor_digest(native.named_parameters()) == tensor_digest(cold.named_parameters())
    assert metadata['no_old_base_actor_optimizer_or_training_reference_loaded']
    errors = {}
    with torch.no_grad(), torch.autocast('cpu', enabled=False):
        for name in ['text', 'photo']:
            a, b, c = native(**inputs[name])['logits'], cold(**inputs[name])['logits'], hot(**inputs[name])['logits']
            assert torch.equal(a, b) and bool(torch.isfinite(a).all())
            errors[name] = float((a-c).abs().max())
        a, b = native.word_logits(inputs['word']['word_pixels']), cold.word_logits(inputs['word']['word_pixels'])
        assert a.dtype == torch.float32 and torch.equal(a, b)
    for case in ['no_opt_in', 'partial', 'wrong_Adam', 'production_claim']:
        changed = copy.deepcopy(receipt)
        if case == 'partial': changed['actual_complete_updates'] -= 1
        if case == 'wrong_Adam': changed['actual_final_Adam_sha256'] = 'e'*64
        if case == 'production_claim': changed['mode'] = 'train'
        with pytest.raises(ValueError): export(hot, state, changed, sources=SOURCES,
            tokenizer_sha256=file_hash(tokenizer), working_budget_bytes=256*1024**2, allow_mechanics=case != 'no_opt_in')
    # Restore the actual old class position methods on a separate model. The
    # exporter must reject their different geometry without folding anything.
    with torch.random.fork_rng(devices=[]): old = student()
    old.import_export(state['model'], expected_sources=SOURCES)
    del old.base.network.vision.forward
    del old.base.patch_grid
    with pytest.raises(ValueError, match='position protocol'): export(old.eval(), state, receipt, sources=SOURCES,
        tokenizer_sha256=file_hash(tokenizer), working_budget_bytes=256*1024**2, allow_mechanics=True)
    assert tensor_digest(hot.named_parameters()) == original and value_digest(capture_rng('cpu')) == rng
    return dict(actual_research_parameters=267718, actual_native_parameters=267430,
        all_original_native_values_and_text_photo_word_cold_exact=True, source_model_and_private_RNG_unchanged=True,
        original_nonzero_dense_critic_preserved_and_unused_research_critic_excluded=True,
        actual_hot_to_folded_FP32_logit_errors=errors, hot_generation_equivalence_not_assumed=True,
        actual_native_checkpoint_sha256=file_hash(path / 'model.pt'), actual_tokenizer_sha256=file_hash(path / 'tokenizer.json'),
        source_canonical_position_protocol_required=True, full_model_GPUcapacity_training_or_quality=False,
        production_updates=0, goal_achieved=False)


def worker(rank, folder):
    folder = Path(folder)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + str(folder / 'rendezvous'), rank=rank, world_size=4,
        timeout=timedelta(minutes=10))
    control = dist.new_group(backend='gloo', timeout=timedelta(minutes=10))
    try:
        data = MixedCodingSFTData(ROOT)
        ps = source_positions(data)
        entries = [ps[:12], ps[12:17], ps[17:18]]
        ids, _ = data.sequence(ps[0])
        text, photo = torch.tensor([ids[:32]]), torch.tensor([ids[:8]])
        inputs = dict(text=dict(input_ids=text, attention_mask=torch.ones_like(text, dtype=torch.bool)),
            photo=dict(input_ids=photo, attention_mask=torch.ones_like(photo, dtype=torch.bool),
                pixel_values=torch.linspace(-1., 1., 3*32*32).reshape(1, 3, 32, 32)),
            word=dict(word_pixels=torch.linspace(-1., 1., 3*64*64).reshape(1, 3, 64, 64)))
        def independent():
            with torch.random.fork_rng(devices=[]): return student()
        for profile in PROFILES:
            def fresh():
                own = student()
                random.seed(989925 + rank)
                np.random.seed(989925 + rank)
                torch.manual_seed(989925 + rank)
                return ParallelMixedCodingSFTStage(own, optimizer(own), data, entries, profile, OPTIONS, SOURCES,
                    mode='mechanics', rank=rank, world=4, control_group=control, gradient_group=None)
            initial = fresh()
            plan = dict(capacity_entries=copy.deepcopy(entries), mechanics_only=True, complete_source_selection=False,
                full_original_capacity_plan_not_used_by_this_explicit_small_fixture=True)
            cap_dir = folder / profile / 'capacity'
            capacity = run_capacity(initial, cap_dir, capacity_plan=plan, independent_actor=independent,
                cold_inputs=inputs, maximum_bytes=32*1024**2, minimum_free_bytes=0, allow_mechanics=True)
            reports = initial.gather(capacity)
            reader = fresh()
            before = live(reader)
            with pytest.raises(ValueError): read_capacity(reader, cap_dir, reports=reports, capacity_plan=plan,
                independent_actor=independent, cold_inputs=inputs)
            assert live(reader) == before
            wrong = copy.deepcopy(reports)
            for r in wrong: r['final_actual_pointer_sha256'] = 'e'*64
            with pytest.raises(ValueError): read_capacity(reader, cap_dir, reports=wrong, capacity_plan=plan,
                independent_actor=independent, cold_inputs=inputs, allow_mechanics=True)
            assert live(reader) == before
            proof = read_capacity(reader, cap_dir, reports=reports, capacity_plan=plan,
                independent_actor=independent, cold_inputs=inputs, allow_mechanics=True)
            readbacks = reader.gather(proof)
            assert proof['new_optimizer_calls'] == 0 and proof['actual_Adam_states'] == 17
            assert proof['initial_actor_sha256'] != proof['actual_actor_sha256']
            trained = fresh()
            before = live(trained)
            broken = copy.deepcopy(readbacks)
            if rank == 2: broken[1]['actual_private_RNG_sha256'] = 'c'*64
            with pytest.raises(ValueError): run(trained, folder / profile / 'forbidden', plan, reports, broken)
            assert live(trained) == before and not (folder / profile / 'forbidden').exists()
            with pytest.raises(ValueError): run(trained, folder / profile / 'no-opt-in', plan, reports, readbacks, opt_in=False)
            assert live(trained) == before
            # Reusing learned capacity factors with a fresh optimizer is a
            # different initializer, even when parameter counts/roles match.
            stale = fresh()
            saved, _, _ = read_latest(cap_dir)
            with torch.no_grad():
                for n, p in stale.named: p.copy_(saved['model']['parts'][n])
            stale.intact()
            before_stale = live(stale)
            with pytest.raises(ValueError): run(stale, folder / profile / 'stale-actor', plan, reports, readbacks)
            assert live(stale) == before_stale and not (folder / profile / 'stale-actor').exists()
            orphan = folder / profile / 'orphan/durable'
            if rank == 0:
                orphan.mkdir(parents=True)
                (orphan / 'uncommitted-observation.bin').write_bytes(b'original interrupted observation')
            trained.gather(True)
            with pytest.raises(ValueError): run(trained, folder / profile / 'orphan', plan, reports, readbacks)
            assert (orphan / 'uncommitted-observation.bin').read_bytes() == b'original interrupted observation'
            assert live(trained) == before
            trained_result = run(trained, folder / profile / 'complete', plan, reports, readbacks)
            expected = live(trained)
            assert trained_result['actual_complete_updates'] == trained_result['actual_new_optimizer_calls'] == 3
            assert trained_result['original_complete_training_budget']['source_presentations'] == 18
            assert trained_result['actual_complete_journal']['records'] == 3 and trained_result['production_updates'] == 0
            interrupted = fresh()
            attempts = folder / profile / 'recovery'
            def stop_after_second(row, elapsed):
                if row['step'] == 2: raise OSError('controlled root-only observer failure after actual Adam')
            with pytest.raises(ValueError, match='retain every complete actual training update'):
                run(interrupted, attempts, plan, reports, readbacks, progress=stop_after_second)
            assert interrupted.failed and interrupted.completed == 2
            saved, _, pointer = read_latest(attempts / 'durable')
            failed_rows = [r for _, r in lines(attempts / 'attempt.jsonl.gz')]
            assert saved['completed'] == 1 and [r['actual_complete_updates'] for r in failed_rows] == [1, 2]
            resumed = fresh()
            before_resume = live(resumed)
            with pytest.raises(ValueError): run_training(resumed, attempts / 'durable', attempts / 'bad-pin.jsonl.gz',
                capacity_plan=plan, reports=reports, readbacks=readbacks, checkpoint_every=1,
                maximum_state_bytes=32*1024**2, minimum_free_bytes=0, maximum_journal_raw_bytes=32*1024**2,
                preflight=lambda: data.guard.check(), expected_resume_pointer_sha256='d'*64, allow_mechanics=True)
            assert live(resumed) == before_resume and not (attempts / 'bad-pin.jsonl.gz').exists()
            resumed_result = run_training(resumed, attempts / 'durable', attempts / 'second-attempt.jsonl.gz',
                capacity_plan=plan, reports=reports, readbacks=readbacks, checkpoint_every=1,
                maximum_state_bytes=32*1024**2, minimum_free_bytes=0, maximum_journal_raw_bytes=32*1024**2,
                preflight=lambda: data.guard.check(), expected_resume_pointer_sha256=pointer, allow_mechanics=True)
            actual = live(resumed)
            assert all(actual[k] == expected[k] for k in ['actor', 'Adam', 'RNG', 'metrics', 'completed', 'failed'])
            assert resumed_result['actual_attempt_started_at_update'] == 1 and resumed_result['actual_new_optimizer_calls'] == 2
            second = [r for _, r in lines(attempts / 'second-attempt.jsonl.gz')]
            assert [r['actual_complete_updates'] for r in second] == [2, 3]
            assert failed_rows[1]['actual_complete_update'] == second[0]['actual_complete_update']
            exported, error = None, None
            if rank == 0:
                try: exported = native_export(trained, trained_result, folder / profile, inputs)
                except Exception as exc: error = repr(exc)
            trained.coordinated(error, 'actual independent native export and complete cold load')
            exported = trained.gather(exported)[0]
            assert not torch.cuda.is_initialized()
            result = dict(profile=profile, rank=rank, actual_capacity=capacity, actual_fresh_readback=proof,
                actual_complete_training=trained_result, actual_recovered_training=resumed_result, actual_native_export=exported,
                actual_small_research_parameters=267718, actual_native_parameters=267430,
                exact_complete_and_interrupted_recovered_final_state=True, actual_replayed_uncommitted_losses_and_gradients_equal=True,
                root_only_error_retains_both_observed_updates_and_old_durable_prefix=True,
                peer_bad_readback_stale_actor_orphan_directory_and_stale_pins_rejected_before_mutation=True,
                all_failed_and_recovery_journals_retained=True, CPU_fixture_does_not_admit_production=True,
                full_model_GPUcapacity_training_or_quality=False, production_updates=0, goal_achieved=False)
            (folder / (profile + '-rank-' + str(rank) + '.json')).write_text(json.dumps(result, indent=2, allow_nan=False))
    finally: dist.destroy_process_group()


def test_actual_four_replica_capacity_readback_training_preserved_attempt_recovery_and_native_export(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    for profile in PROFILES:
        reports = [json.loads((tmp_path / (profile + '-rank-' + str(rank) + '.json')).read_text()) for rank in range(4)]
        assert len({r['actual_complete_training']['actual_checkpoint_sha256'] for r in reports}) == 1
        assert len({r['actual_fresh_readback']['actual_private_RNG_sha256'] for r in reports}) == 4
        assert all(r['exact_complete_and_interrupted_recovered_final_state'] for r in reports)
    assert FULL == dict(updates=5256, source_presentations=84084, complete_context_tokens=42271575,
        response_EOS_targets=21316734, family_presentations=dict(public_weak_synthetic_code=65217,
            original_human_code=11163, public_procedural_reasoning=7704))
