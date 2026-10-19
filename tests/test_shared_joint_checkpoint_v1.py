"""Actual own joint neural recovery, sparse heads, durable failures and Gloo.

Synthetic source controls prove mechanics only. Full-size original data,
history funding, CUDA/BF16 capacity and real quality remain unproved.
"""
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn
from torch.nn import functional as F

from slm_lab import shared_joint_checkpoint_v1 as private
from slm_lab import shared_joint_durable_state_v1 as durable
from slm_lab import shared_joint_gradient_reduce_v1 as reduction
from slm_lab.code_priority_gradient_blend_v1 import combine
from slm_lab.codenet_checkpoint_v1 import capture_rng, restore_rng
from slm_lab.corpus import file_hash
from slm_lab.own_dense_actor_fold_v1 import fold_actor
from slm_lab.own_dense_standalone_v1 import export, load
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_shared_joint_actor_vision_v1 import setup, gradients

RESERVE = 40 * 1024**3


def draws(): return [random.random(), float(np.random.random()), float(torch.rand(()))]


class Trial:
    def __init__(self, folder, rank=0, world=1):
        self.folder = Path(folder); self.folder.mkdir(parents=True, exist_ok=True)
        self.view, self.ids, self.pixels, self.words = setup(self.folder)
        self.optimizer = torch.optim.AdamW([p for _, p in self.view.named_selected()], lr=.001, weight_decay=.01, foreach=False)
        self.fixed = private.FixedGuard(self.view, self.optimizer)
        self.rank, self.world = rank, world; self.metrics = []
        def task(kind, count=2):
            word = kind == 'word_CTC'
            denominators = {'whole_words': count} if word else {'native_response_EOS': 3 * count}
            if kind == 'image_matching': denominators['paired_captions'] = count
            return dict(task=kind, source_family='synthetic_control:' + kind, source_role='train', source_sha256='a' * 64,
                original_positions=list(range(count)), complete_context_tokens=0 if word else 4 * count,
                response_EOS_targets=0 if word else 3 * count, word_UTF8_bytes=count if word else 0,
                normalization_denominators=denominators)
        self.entries = [dict(step=1, objectives=[task('text'), task('image_matching')]),
            dict(step=2, objectives=[task('text'), task('word_CTC')]), dict(step=3, objectives=[task('text', 1)])]
        self.sources = dict(text_initializer_sha256='a' * 64, visual_initializer_sha256='b' * 64,
            tokenizer_sha256=file_hash(self.folder / 'tokenizer.json'), run_plan_sha256='c' * 64,
            source_admission_sha256='d' * 64, complete_schedule_sha256=value_digest(self.entries), fixed_values_sha256=self.fixed.sha256)

    def reduced(self, loss, objective_index=0):
        named = self.view.named_selected()
        values = {n: None for n, _ in named} if loss is None else gradients(self.view, loss)
        total = 0. if loss is None else float(loss.detach())
        if self.world == 1 or objective_index >= len(self.entries[len(self.metrics)]['objectives']): return values, total
        objective = self.entries[len(self.metrics)]['objectives'][objective_index]
        gradients_for_input = values
        before = value_digest(values)
        values, proof = reduction.reduce(values, named_parameters=named, task_sha256=value_digest(objective),
            global_denominators=objective['normalization_denominators'], bucket_values=256)
        assert proof['all_actual_reduced_gradient_bytes_agree'] and proof['largest_host_bucket_values'] <= 256
        assert before == value_digest(gradients_for_input)
        scalar = torch.tensor(total, dtype=torch.float64); dist.all_reduce(scalar); dist.broadcast(scalar, src=0)
        return values, float(scalar)

    def step(self, *, stochastic_probe=True):
        probe = draws() if stochastic_probe else None
        index = len(self.metrics); row = self.entries[index]
        count = len(row['objectives'][0]['original_positions']); local = list(range(count))[self.rank::self.world]
        code = None
        if local:
            ids = self.ids[local]
            logits = self.view.text(input_ids=ids)['logits']
            code = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction='sum') / (3 * count)
        primary, primary_loss = self.reduced(code)
        secondary_loss = None
        if index == 0:
            # Each rank differentiates 1/world of the complete two-image
            # contrastive objective. This is an explicit replicated control,
            # not a claim of production sharded contrastive efficiency.
            secondary_loss = self.view.image_matching(dict(input_ids=self.ids, labels=self.ids, pixel_values=self.pixels),
                self.ids, torch.ones_like(self.ids, dtype=torch.bool), torch.eye(2, dtype=torch.bool))['loss'] / self.world
        elif index == 1 and local:
            logits = self.view.word_logits(self.words[local])
            secondary_loss = F.ctc_loss(logits.log_softmax(-1).transpose(0, 1), torch.tensor([97, 98])[local],
                torch.full((len(local),), logits.shape[1], dtype=torch.long), torch.ones(len(local), dtype=torch.long),
                blank=256, reduction='sum') / 2
        secondary, other_loss = self.reduced(secondary_loss, 1)
        mixed, proof = combine(primary, secondary, primary_names=self.view.task_names('text'),
            secondary_weight=.3, project_conflicts=True)
        named = self.view.named_selected(); active = [n for n in mixed if mixed[n] is not None]
        expected = set(n for objective in row['objectives'] for n in self.view.task_names(objective['task']))
        assert set(active) == expected
        for name, parameter in named: parameter.grad = mixed[name]
        norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.))
        self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)
        metric = dict(step=index + 1, entry_sha256=value_digest(row), active_parameter_names=active,
            normalized_task_losses=[primary_loss] + ([other_loss] if len(row['objectives']) > 1 else []),
            gradient_norm_before_clip=norm, globally_reduced_task_gradients_before_combining=True, gradient_interaction=proof)
        self.metrics.append(metric)
        return probe

    def snapshot(self):
        return private.make_state(self.view, self.optimizer, self.fixed, sources=self.sources, entries=self.entries,
            completed=len(self.metrics), metrics=self.metrics, rank=self.rank, world=self.world)

    def restore(self, state, **kwargs):
        completed, self.metrics = private.restore_state(self.view, self.optimizer, self.fixed, state,
            sources=self.sources, entries=self.entries, expected_common_sha256=private.common_digest(state),
            expected_private_RNG_sha256=state['private_RNG_sha256'], expected_completed=state['cursor']['completed'],
            rank=self.rank, world=self.world, **kwargs)
        assert completed == len(self.metrics)

    def commit(self, folder, **kwargs):
        return durable.commit(self.view, self.optimizer, self.fixed, folder, sources=self.sources, entries=self.entries,
            completed=len(self.metrics), metrics=self.metrics, maximum_bytes=4 * 1024**2,
            minimum_free_bytes=RESERVE, rank=self.rank, world=self.world, **kwargs)

    def recover(self, folder, **kwargs):
        completed, self.metrics, receipt = durable.restore_latest(self.view, self.optimizer, self.fixed, folder,
            sources=self.sources, entries=self.entries, rank=self.rank, world=self.world, **kwargs)
        assert completed == len(self.metrics)
        return receipt


def live(trial):
    return tensor_digest(trial.view.named_parameters()), value_digest(trial.optimizer.state_dict()), value_digest(capture_rng('cpu'))


def check_native(trial, folder):
    before = live(trial)
    folded, proof = fold_actor(trial.view.actor, sources=trial.sources, working_budget_bytes=256 * 1024**2)
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(trial.folder / 'tokenizer.json', folder / 'tokenizer.json')
    torch.save(export(folded, sources=trial.sources, provenance=proof, completed_updates=len(trial.metrics),
        tokenizer_sha256=trial.sources['tokenizer_sha256']), folder / 'model.pt')
    cold, metadata = load(folder / 'model.pt', expected_checkpoint_sha256=file_hash(folder / 'model.pt'), expected_sources=trial.sources)
    assert tensor_digest(cold.named_parameters()) == tensor_digest(folded.named_parameters())
    errors = []
    with torch.no_grad():
        for pixels in [None, trial.pixels]:
            hot = trial.view.actor(trial.ids, pixel_values=pixels)['logits']
            merged = folded(trial.ids, pixel_values=pixels)['logits']
            assert torch.allclose(hot, merged, atol=1e-5, rtol=1e-4)
            assert torch.equal(cold(trial.ids, pixel_values=pixels)['logits'], merged)
            errors.append(float((hot - merged).abs().max()))
        assert torch.equal(cold.word_logits(trial.words), trial.view.word_logits(trial.words))
    assert metadata['no_old_base_actor_optimizer_or_training_reference_loaded'] and live(trial) == before
    return dict(standalone_sha256=file_hash(folder / 'model.pt'), hot_folded_text_photo_maximum_errors=errors,
        all_native_values_and_folded_text_photo_word_outputs_cold_exact=True, native_count=cold.counts()['total_unique'])


def test_actual_joint_private_sparse_Adam_exact_next_update_and_nonaliasing_recovery(tmp_path):
    trial = Trial(tmp_path / 'source'); trial.step(); saved = trial.snapshot(); untouched = value_digest(saved)
    expected_probe = trial.step(); expected = live(trial); expected_metrics = copy.deepcopy(trial.metrics)
    trial.restore(saved); assert trial.step() == expected_probe and live(trial) == expected
    assert trial.metrics == expected_metrics and value_digest(saved) == untouched
    clocks = private.cursor(trial.view, trial.entries, len(trial.metrics), trial.metrics)['actual_per_parameter_Adam_steps']
    assert clocks['vision.project.weight'] == 1 and all(c == 2 for n, c in clocks.items() if n.startswith('actor.'))
    assert all(c == 1 for n, c in clocks.items() if n.startswith(('word.', 'pointing.')))
    trial.fixed.check(trial.view, trial.optimizer, full=True)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('case', ['missing_weight', 'nonfinite_weight', 'sparse_clock', 'missing_moment', 'negative_variance',
    'optimizer_settings', 'source', 'schedule_tail', 'metric', 'RNG', 'private_hash', 'partial_gradient', 'fixed_value'])
def test_joint_recovery_rejects_damage_before_any_live_weight_Adam_or_RNG_mutation(tmp_path, case):
    trial = Trial(tmp_path); trial.step(); state = trial.snapshot(); trial.step()
    if case == 'missing_weight': state['selected_weights'].pop(next(n for n in state['selected_weights'] if n.startswith('word.')))
    elif case == 'nonfinite_weight': next(iter(state['selected_weights'].values())).reshape(-1)[0] = float('nan')
    elif case == 'sparse_clock':
        index = state['optimized_names'].index('vision.project.weight'); state['optimizer']['state'][index]['step'].fill_(2)
    elif case == 'missing_moment': next(iter(state['optimizer']['state'].values())).pop('exp_avg')
    elif case == 'negative_variance': next(iter(state['optimizer']['state'].values()))['exp_avg_sq'].reshape(-1)[0] = -1.
    elif case == 'optimizer_settings': state['optimizer']['param_groups'][0]['lr'] *= 2
    elif case == 'source': state['sources']['source_admission_sha256'] = 'e' * 64
    elif case == 'schedule_tail': trial.entries[-1]['objectives'][0]['original_positions'] = [8]
    elif case == 'metric': state['metrics'][0]['active_parameter_names'].pop()
    elif case == 'RNG': state['private_RNG']['torch'] = torch.tensor([1], dtype=torch.uint8)
    elif case == 'private_hash': state['private_RNG_sha256'] = 'e' * 64
    elif case == 'partial_gradient': trial.view.named_selected()[0][1].grad = torch.zeros_like(trial.view.named_selected()[0][1])
    else:
        with torch.no_grad(): trial.view.actor.base.network.norm.weight.add_(.01)
    before = live(trial)
    with pytest.raises(ValueError, match='Coordinated joint checkpoint rejection'): trial.restore(state)
    assert live(trial) == before and not torch.cuda.is_initialized()


def test_three_joint_generations_keep_old_mmap_and_export_actual_final_text_photo_word(tmp_path):
    trial = Trial(tmp_path / 'source'); folder = tmp_path / 'commits'
    trial.step(); first = trial.commit(folder); old, _, old_pointer = durable.read_latest(folder)
    old_hash = private.common_digest(old['common'])
    trial.step(); second = trial.commit(folder); trial.step(); third = trial.commit(folder)
    assert [r['generation'] for r in [first, second, third]] == [1, 2, 3]
    assert private.common_digest(old['common']) == old_hash and old['common']['cursor']['completed'] == 1
    expected = live(trial); before = live(trial)
    with pytest.raises(ValueError): trial.recover(folder, expected_pointer_sha256=old_pointer)
    assert live(trial) == before
    with pytest.raises(ValueError): trial.recover(folder)
    assert live(trial) == before
    with torch.no_grad(): trial.view.named_selected()[0][1].reshape(-1)[0] = float('nan')
    trial.recover(folder, allow_completed=True); assert live(trial) == expected
    result = check_native(trial, tmp_path / 'standalone')
    result.update(actual_joint_logical_updates=3, original_small_source_training_calls=2,
        frozen_language_critic_other_heads_unchanged=True, one_common_tensor_Adam_copy=True,
        exact_next_private_Adam_RNG_recovery=True, synthetic_controls_only=True,
        full_source_4L40_training_or_quality_gain=False)
    (tmp_path / 'actual-small-joint-durable-and-native-cold.json').write_text(json.dumps(result, indent=2))


@pytest.mark.parametrize('target', ['receipt', 'pointer'])
def test_failed_joint_publication_keeps_the_preceding_complete_generation(tmp_path, monkeypatch, target):
    trial = Trial(tmp_path / 'source'); trial.step(); folder = tmp_path / 'commits'
    trial.commit(folder); _, first, pointer_sha = durable.read_latest(folder); trial.step(); expected = live(trial)
    original = durable.durable_json
    def failing(path, value, reserve):
        if (Path(path).name == 'LATEST.json') == (target == 'pointer'): raise OSError('injected joint publication failure')
        return original(path, value, reserve)
    with monkeypatch.context() as patch:
        patch.setattr(durable, 'durable_json', failing)
        with pytest.raises(ValueError, match='injected joint publication failure'): trial.commit(folder)
    assert live(trial) == expected
    _, recovered, _ = durable.read_latest(folder, expected_pointer_sha256=pointer_sha)
    assert recovered == first
    trial.recover(folder, expected_pointer_sha256=pointer_sha); trial.step(); assert live(trial) == expected
    assert (folder / 'state-slot-0.pt').is_file()


def test_serial_gradient_transport_preserves_noncontiguous_inputs_signed_zero_and_absent_heads():
    named = [('matrix', torch.nn.Parameter(torch.zeros(4, 3))), ('zero', torch.nn.Parameter(torch.zeros(2))),
        ('unused', torch.nn.Parameter(torch.zeros(3)))]
    values = dict(matrix=torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),
        zero=torch.tensor([-0., 0.]), unused=None)
    before = value_digest(values)
    actual, receipt = reduction.reduce(values, named_parameters=named, task_sha256='a' * 64,
        global_denominators={'native_response_EOS': 6, 'paired_captions': 2}, bucket_values=2)
    assert value_digest(actual) == before == value_digest(values)
    assert receipt['largest_host_bucket_values'] == 2 and receipt['actual_active_parameter_tensors'] == 2
    assert receipt['original_global_denominators'] == {'native_response_EOS': 6, 'paired_captions': 2}
    assert actual['matrix'].data_ptr() != values['matrix'].data_ptr() and actual['unused'] is None


@pytest.mark.parametrize('case', ['nonfinite_weight', 'invalid_Adam'])
def test_invalid_live_joint_boundary_cannot_publish_any_checkpoint(tmp_path, case):
    trial = Trial(tmp_path / 'source'); trial.step()
    if case == 'nonfinite_weight':
        with torch.no_grad(): trial.view.named_selected()[0][1].reshape(-1)[0] = float('nan')
    else: trial.optimizer.param_groups[0]['lr'] = -1.
    before = live(trial)
    with pytest.raises(ValueError, match='Coordinated joint checkpoint rejection'): trial.commit(tmp_path / 'unpublished')
    assert live(trial) == before and not list((tmp_path / 'unpublished').glob('*.pt'))
    assert not (tmp_path / 'unpublished' / 'LATEST.json').exists()


def worker(rank, folder):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.nice(0) >= 19
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    root = Path(folder)
    dist.init_process_group('gloo', init_method='file://' + str(root / 'rendezvous'), rank=rank, world_size=4, timeout=timedelta(minutes=4))
    try:
        trial = Trial(root / ('source-' + str(rank)), rank, 4)
        oracle = Trial(root / 'serial-oracle') if rank == 0 else None
        def compare_serial():
            if rank != 0: return
            saved_rng = capture_rng('cpu'); oracle.step(stochastic_probe=False); restore_rng(saved_rng, 'cpu')
            differences = {}
            for (name, parameter), (expected_name, expected) in zip(trial.view.named_selected(), oracle.view.named_selected()):
                assert name == expected_name
                if not torch.allclose(parameter, expected, atol=2e-7, rtol=2e-6):
                    differences[name] = float((parameter.detach() - expected.detach()).abs().max())
            if differences:
                torch.save(dict(actual={n: p.detach().clone() for n, p in trial.view.named_selected()},
                    oracle={n: p.detach().clone() for n, p in oracle.view.named_selected()},
                    actual_Adam=trial.optimizer.state_dict(), oracle_Adam=oracle.optimizer.state_dict(),
                    actual_metrics=trial.metrics, oracle_metrics=oracle.metrics),
                    root / ('serial-oracle-mismatch-step-' + str(len(oracle.metrics)) + '.pt'))
                raise AssertionError('Independent serial parameter mismatch: ' + repr(differences))
        random.seed(991700 + rank); np.random.seed(991700 + rank); torch.manual_seed(991700 + rank)
        trial.step()
        compare_serial()
        first = trial.commit(root / 'commits')
        shared, _, pointer = durable.read_latest(root / 'commits')
        assert len(set(first['semantic_manifest']['all_actual_private_RNG_sha256'])) == 4
        assert len(shared['private_states']) == 4 and set(shared['common']) == durable.COMMON_FIELDS
        expected_probe = trial.step(); expected = live(trial); expected_metrics = copy.deepcopy(trial.metrics)
        trial.recover(root / 'commits', expected_pointer_sha256=pointer)
        assert trial.step() == expected_probe and live(trial) == expected and trial.metrics == expected_metrics
        compare_serial()
        second = trial.commit(root / 'commits')
        local = trial.snapshot()
        if rank == 2: local['private_RNG']['torch'] = torch.tensor([1], dtype=torch.uint8)
        before = live(trial)
        with pytest.raises(ValueError, match='Coordinated joint checkpoint rejection'): trial.restore(local)
        assert live(trial) == before
        bad = {n: torch.zeros_like(p) for n, p in trial.view.named_selected()}
        if rank == 1: next(iter(bad.values())).reshape(-1)[0] = float('nan')
        with pytest.raises(ValueError, match='Coordinated joint checkpoint rejection'):
            reduction.reduce(bad, named_parameters=trial.view.named_selected(), task_sha256=value_digest(trial.entries[0]),
                global_denominators={'native_response_EOS': 6}, bucket_values=256)
        assert live(trial) == before
        # A true one-source text tail forwards on rank0 only; sparse visual
        # values/moments keep their earlier clocks on all four replicas.
        trial.step(); third = trial.commit(root / 'commits')
        compare_serial()
        trial.fixed.check(trial.view, trial.optimizer, full=True)
        result = dict(rank=rank, first=first['semantic_manifest'], second=second['semantic_manifest'], third=third['semantic_manifest'],
            exact_next_actual_joint_Adam_RNG_and_gradient_interaction_replay=True,
            one_bad_private_rank_rejected_before_all_peer_mutations=True, true_single_source_text_tail=True,
            one_bad_local_gradient_rejected_before_all_peer_mutations=True,
            actual_joint_physical_AdamW_calls=4, actual_joint_logical_updates=3,
            original_source_fixture_training_calls=2, frozen_language_critic_other_heads_unchanged=True,
            synthetic_controls_only=True, full_size_GPUcapacity_or_quality_gain=False)
        if rank == 0:
            result['serial_global_normalization_parameter_oracle_passed'] = True
            result['serial_global_normalization_parameter_oracle_updates'] = len(oracle.metrics)
            result['native_cold'] = check_native(trial, root / 'standalone')
        assert not torch.cuda.is_initialized()
        (root / ('actual-rank-' + str(rank) + '.json')).write_text(json.dumps(result, indent=2))
        dist.barrier()
    finally: dist.destroy_process_group()


def test_actual_four_replica_joint_durable_recovery_global_gradients_private_RNG_and_native_cold(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    rows = [json.loads((tmp_path / ('actual-rank-' + str(rank) + '.json')).read_text()) for rank in range(4)]
    assert all(row['third'] == rows[0]['third'] for row in rows)
    assert sum(row['actual_joint_physical_AdamW_calls'] for row in rows) == 16
    assert rows[0]['serial_global_normalization_parameter_oracle_passed']
