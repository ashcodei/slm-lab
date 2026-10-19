"""Real admitted sources and actual own-small CPU gradient/recovery checks."""
from collections import Counter
import copy
from datetime import timedelta
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn
from torch.nn import functional as F

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.mixed_coding_SFT_data_v1 import MixedCodingSFTData, FAMILIES, PROFILES, EXPECTED
from slm_lab.parallel_function_repair_stage_v1 import manifest, value_digest
from slm_lab.parallel_mixed_coding_SFT_stage_v1 import ParallelMixedCodingSFTStage
from slm_lab import parallel_repair_durable_state_v1 as durable
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_public_reasoning_group_gradients_v1 import model, optimizer

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = dict(learning_rate=1e-4, gradient_clip=1., maximum_microbatch_examples=2,
    maximum_padded_tokens=4096, gradient_bucket_values=4096, epochs=3, width=16, seed=989923)
SOURCES = dict(own_small_initializer_sha256='a' * 64, complete_mixed_source_plan_sha256='b' * 64)


@pytest.fixture(scope='module')
def data(): return MixedCodingSFTData(ROOT)


def source_positions(data):
    families = [[p for p in data.slots if data.information(p)['family'] == family] for family in FAMILIES]
    ordered = [sorted(rows, key=lambda p: (data.information(p)['complete_context_tokens'], p))[:6] for rows in families]
    return [p for i in range(6) for rows in ordered for p in rows[i:i+1]]


def setup(data, profile='plain_tokens', rank=0, world=1, control=None, entries=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(989924)
    student = model()
    ps = source_positions(data)
    entries = entries or [ps[:12], ps[12:17], ps[17:18]]
    random.seed(989925 + rank)
    np.random.seed(989925 + rank)
    torch.manual_seed(989925 + rank)
    return ParallelMixedCodingSFTStage(student, optimizer(student), data, entries, profile, OPTIONS, SOURCES,
        mode='mechanics', rank=rank, world=world, control_group=control, gradient_group=None)


def live(stage):
    return dict(parts=tensor_digest(stage.model.parts.named_parameters()), base=tensor_digest(stage.model.base.named_parameters()),
        optimizer=value_digest(stage.optimizer.state_dict()), RNG=value_digest(capture_rng('cpu')),
        cursor=stage.completed, metrics=value_digest(stage.metrics), failed=stage.failed,
        tensor_identities=[(id(p), k, id(t), t.data_ptr(), t._version)
            for p, row in stage.optimizer.state.items() for k, t in row.items()])


def draws(): return [random.random(), float(np.random.random()), *torch.rand(3).tolist()]


def test_all_actual_original_sources_complete_masks_epoch_coverage_and_nominal_weights(data, tmp_path):
    totals, family_targets = Counter(), Counter()
    for p in data.slots:
        ids, labels = data.sequence(p)
        info = data.information(p)
        start = next(i for i, t in enumerate(labels) if t != -100)
        assert ids[0] == 1 and ids[-1] == 2 and labels == [-100] * start + ids[start:]
        assert len(ids) <= 4096 and info['complete_context_tokens'] == len(ids)
        assert info['response_EOS_targets'] == len(ids) - start and info['source_role'] == 'train'
        totals.update(source_presentations=1, complete_context_tokens=len(ids), response_EOS_targets=len(ids)-start)
        family_targets[info['family']] += len(ids) - start
    assert dict(totals) == EXPECTED
    assert family_targets == dict(zip(FAMILIES, [6502280, 584218, 19080]))
    nominal = [family_targets[f] * data.family_weights[f] for f in FAMILIES]
    assert [v / sum(nominal) for v in nominal] == pytest.approx([.5, .4, .1])
    before = value_digest(capture_rng('cpu'))
    one = data.complete_schedule(epochs=1, width=16, seed=OPTIONS['seed'])
    three = data.complete_schedule(epochs=3, width=16, seed=OPTIONS['seed'])
    assert value_digest(capture_rng('cpu')) == before
    assert Counter(p for e in one for p in e) == Counter(data.slots)
    assert Counter(p for e in three for p in e) == Counter({p: 3 for p in data.slots})
    assert len(one) == 1752 and len(three) == 5256 and len(one[-1]) == 12
    assert data.budget(three) == dict(updates=5256, source_presentations=84084, complete_context_tokens=42271575,
        response_EOS_targets=21316734, family_presentations=dict(zip(FAMILIES, [65217, 11163, 7704])))
    # Construct every actual full-source train batch; compare all IDs/masks/EOS
    # and every padded cell, without any neural forward or source truncation.
    for e in one:
        batch = data.batch(e)
        for j, p in enumerate(e):
            ids, labels = data.sequence(p)
            assert batch['input_ids'][j, :len(ids)].tolist() == ids
            assert batch['labels'][j, :len(ids)].tolist() == labels
            assert bool(batch['attention_mask'][j, :len(ids)].all())
            assert not bool(batch['attention_mask'][j, len(ids):].any())
            assert not bool((batch['labels'][j, len(ids):] != -100).any())
            assert not bool((batch['input_ids'][j, len(ids):] != 0).any())
    for entries in [[], [[]], [[0, 0]], [[28028]], [[True]]]:
        with pytest.raises(ValueError): data.budget(entries)
    with pytest.raises(ValueError): data.score(0, [2])
    assert not torch.cuda.is_initialized()
    (tmp_path / 'actual-complete-mixed-source-readback.json').write_text(json.dumps(dict(
        actual_one_epoch_budget=data.budget(one), actual_three_epoch_budget=data.budget(three),
        actual_all1752native_batches_checked=True, nominal_whole_epoch_weighted_target_mass=[.5, .4, .1],
        per_batch_normalization_does_not_guarantee_epoch_loss_fraction=True, synthetic_test_reward_forbidden=True,
        source_only=True, neural_forwards=0, GPUcontexts=0, goal_achieved=False), indent=2))


@pytest.mark.parametrize('profile', PROFILES)
def test_actual_padded_microbatch_gradient_matches_independent_serial_full_source_loss(data, tmp_path, profile):
    stage = setup(data, profile)
    serial = setup(data, profile)
    positions = stage.entries[0]
    denom = sum(data.weight(profile, p) * sum(v != -100 for v in data.sequence(p)[1]) for p in positions)
    serial.model.train()
    serial.model.policy.gradient_checkpointing = True
    sums = []
    for p in positions:
        ids, labels = data.sequence(p)
        logits = serial.model(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.bool))['logits']
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 32768), torch.tensor(labels[1:]), reduction='sum')
        (loss * data.weight(profile, p) / denom).backward()
        sums.append(float(loss.detach()))
    norm = float(torch.nn.utils.clip_grad_norm_(serial.parameters, OPTIONS['gradient_clip'], error_if_nonfinite=True))
    expected_grad = [p.grad.clone() for p in serial.parameters]
    serial.optimizer.step()
    actual_grad = []
    ordinary = stage.optimizer.step
    def observed_step(*args, **kwargs):
        actual_grad.extend(p.grad.clone() for p in stage.parameters)
        return ordinary(*args, **kwargs)
    stage.optimizer.step = observed_step
    row = stage.update()
    errors = [float((a-b).abs().max()) for a, b in zip(actual_grad, expected_grad)]
    assert max(errors) < 2e-6
    assert row['gradient_norm_before_clip'] == pytest.approx(norm, abs=2e-6, rel=2e-6)
    assert row['original_token_mean_NLL'] == pytest.approx(sum(sums) / row['source_metadata']['response_EOS_targets'], abs=2e-6)
    assert all(torch.allclose(a, b, atol=2e-6, rtol=2e-6) for a, b in zip(stage.parameters, serial.parameters))
    assert len(stage.optimizer.state) == len(stage.parameters) == 17
    assert {i['family'] for i in row['source_metadata']['complete_sources']} == set(FAMILIES)
    assert not torch.cuda.is_initialized()
    (tmp_path / ('actual-serial-gradient-' + profile + '.json')).write_text(json.dumps(dict(profile=profile,
        actual_source_positions=positions, actual_complete_response_EOS_targets=row['source_metadata']['response_EOS_targets'],
        actual_original_context_tokens=row['source_metadata']['complete_context_tokens'],
        actual_maximum_serial_vs_microbatch_clipped_gradient_error=max(errors), actual_source_row=row,
        own_small_student_values=stage.model.counts()['actual_unique'], actual_selected_Adam_states=17,
        full_model_training=False, quality_gain_established=False, goal_achieved=False), indent=2))


def test_full_size_CPU_claim_bad_settings_and_relabelled_sources_reject_before_Adam(data):
    for options in [dict(OPTIONS, maximum_padded_tokens=512), dict(OPTIONS, width=0), dict(OPTIONS, epochs=True),
            dict(OPTIONS, learning_rate=float('nan'))]:
        torch.manual_seed(989924)
        student = model()
        adam = optimizer(student)
        before = tensor_digest(student.parts.named_parameters())
        with pytest.raises(ValueError): ParallelMixedCodingSFTStage(student, adam, data, [[0]], 'plain_tokens', options, SOURCES, mode='mechanics')
        assert not adam.state and tensor_digest(student.parts.named_parameters()) == before
    for mode in ['capacity', 'train']:
        student = model()
        adam = optimizer(student)
        with pytest.raises(ValueError): ParallelMixedCodingSFTStage(student, adam, data, [[0]], 'plain_tokens', OPTIONS, SOURCES, mode=mode)
        assert not adam.state
    stage = setup(data)
    before = live(stage)
    data.role = 'reserved'
    try:
        with pytest.raises(ValueError): stage.update()
        assert live(stage) == before
    finally: del data.role


def worker(rank, folder):
    folder = Path(folder)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + str(folder / 'rendezvous'), rank=rank, world_size=4,
        timeout=timedelta(minutes=10))
    control = dist.new_group(backend='gloo', timeout=timedelta(minutes=10))
    try:
        public = MixedCodingSFTData(ROOT)
        for profile in PROFILES:
            stage = setup(public, profile, rank, 4, control)
            initial, fixed = live(stage)['parts'], live(stage)['base']
            row1 = stage.update()
            durable.commit(stage, folder / profile, maximum_bytes=32 * 1024**2, minimum_free_bytes=0)
            state, receipt, pointer = durable.read_latest(folder / profile)
            wanted_draws = draws()
            wanted_row = stage.update()
            expected = live(stage)
            durable.restore_latest(stage, folder / profile, expected_pointer_sha256=pointer)
            assert draws() == wanted_draws and stage.update() == wanted_row
            got = live(stage)
            assert all(got[k] == expected[k] for k in ['parts', 'optimizer', 'RNG', 'cursor', 'metrics'])
            before = live(stage)
            for what in ['peer_RNG', 'peer_cursor', 'clock', 'source', 'CE_mask', 'critic', 'nonfinite', 'flag']:
                bad = copy.deepcopy(state)
                if what == 'peer_RNG': bad['private_workers'][2]['rng']['torch'] = torch.zeros(1, dtype=torch.uint8)
                if what == 'peer_cursor': bad['private_workers'][2]['local_source_cursor']['rank'] = 1
                if what == 'clock': bad['optimizer']['state'][0]['step'].fill_(99)
                if what == 'source': bad['model']['sources']['complete_mixed_source_plan_sha256'] = 'c' * 64
                if what == 'CE_mask': bad['metrics'][0]['source_metadata']['response_EOS_targets'] += 1
                if what == 'critic': bad['model']['parts']['critic.weight'].add_(.1)
                if what == 'nonfinite': next(iter(bad['model']['parts'].values())).fill_(float('nan'))
                if what == 'flag': bad['all_private_RNGs_and_source_cursors_retained'] = False
                for retained in [receipt['semantic_manifest'], manifest(bad)]:
                    with pytest.raises(ValueError): stage.resume(bad, expected_manifest=retained)
                    assert live(stage) == before
            tail = stage.update()
            assert [r['actual_positions'] for r in tail['actual_rank_observations']] == [stage.entries[-1], [], [], []]
            assert [r['microbatches'] for r in tail['actual_rank_observations']][1:] == [[], [], []]
            durable.commit(stage, folder / profile, maximum_bytes=32 * 1024**2, minimum_free_bytes=0)
            before = live(stage)
            with pytest.raises(ValueError): durable.restore_latest(stage, folder / profile)
            assert live(stage) == before
            cold = setup(public, profile, rank, 4, control)
            durable.restore_latest(cold, folder / profile, allow_completed=True)
            assert all(live(cold)[k] == before[k] for k in ['parts', 'base', 'optimizer', 'RNG', 'cursor', 'metrics'])
            assert before['parts'] != initial and before['base'] == fixed
            assert all(float(s['step']) == 3 for s in stage.optimizer.state.values())
            # A peer-only nonfinite forward must clear partial rank gradients
            # before reduction or Adam. The unmodified live workers are separate.
            fault = setup(public, profile, rank, 4, control)
            prior = live(fault)
            hook = None
            if rank == 2:
                def poison(module, args, output): return dict(output, logits=output['logits'] * float('nan'))
                hook = fault.model.register_forward_hook(poison)
            with pytest.raises(ValueError, match='before reduction'): fault.update()
            if hook is not None: hook.remove()
            after = live(fault)
            assert all(after[k] == prior[k] for k in ['parts', 'base', 'optimizer', 'RNG', 'cursor', 'metrics'])
            assert after['failed'] and all(p.grad is None for p in fault.model.parameters())
            assert fault.failure_observations[-1]['actual_rank_errors'][2] is not None
            assert not torch.cuda.is_initialized()
            (folder / (profile + '-rank-' + str(rank) + '.json')).write_text(json.dumps(dict(profile=profile, rank=rank,
                actual_own_small_values=stage.model.counts()['actual_unique'], actual_selected_Adam_states=len(stage.parameters),
                complete_global_source_budget=public.budget(stage.entries), actual_all_source_rows=stage.metrics,
                actual_independent_four_replica_gradient_SUM=True, actual_next_update_weights_Adam_private_RNG_bit_exact=True,
                actual_cold_checkpoint_reload_bit_exact=True, peer_only_corrupt_recovery_rejected_before_mutation=True,
                peer_only_nonfinite_forward_rejected_and_all_partial_gradients_cleared=True,
                actual_empty_tail_has_three_ranks_without_forward=True, original_full_source_training_tail_is12not1=True,
                first_actual_unclipped_gradient_sha256=row1['actual_unclipped_global_actor_gradient_sha256'],
                full_model_GPUcapacity_or_training=False, model_quality_gain=False, goal_achieved=False), indent=2))
    finally: dist.destroy_process_group()


def test_actual_four_replica_source_SUM_shared_durable_recovery_and_peer_faults(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    reports = [json.loads((tmp_path / (p + '-rank-' + str(r) + '.json')).read_text()) for p in PROFILES for r in range(4)]
    for profile in PROFILES:
        rows = [r for r in reports if r['profile'] == profile]
        assert len({r['first_actual_unclipped_gradient_sha256'] for r in rows}) == 1
    assert reports[0]['first_actual_unclipped_gradient_sha256'] != reports[4]['first_actual_unclipped_gradient_sha256']
