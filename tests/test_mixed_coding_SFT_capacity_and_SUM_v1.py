"""Independent complete source maxima and actual four-rank serial CE comparison."""
from collections import Counter
from datetime import timedelta
import json
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn
from torch.nn import functional as F

from slm_lab.mixed_coding_SFT_data_v1 import MixedCodingSFTData, FAMILIES, PROFILES
from slm_lab.mixed_coding_SFT_capacity_cases_v1 import choose
from test_parallel_mixed_coding_SFT_stage_v1 import ROOT, OPTIONS, setup, data
from test_public_reasoning_group_gradients_v1 import model, optimizer


def test_every_actual_native_source_maximum_full_batch_and_rank_rotation(data, tmp_path):
    plan = choose(data, epochs=3, seed=OPTIONS['seed'])
    full = data.complete_schedule(epochs=3, width=16, seed=OPTIONS['seed'])
    native = {}
    for p in data.slots:
        ids, labels = data.sequence(p)
        start = next(i for i, v in enumerate(labels) if v != -100)
        native[p] = dict(context=len(ids), prefix=start, targets=len(ids)-start, family=data.locate(p)[0])
    assert Counter(p for e in full for p in e) == Counter({p: 3 for p in data.slots})
    selected = {p for b in plan['selected_intact_original_batches'] for p in b['intact_original_positions']}
    for f in FAMILIES:
        rows = [r for r in native.values() if r['family'] == f]
        for k in ['context', 'prefix', 'targets']:
            assert max(r[k] for r in rows) == max(native[p][k] for p in selected if native[p]['family'] == f)
    for k in ['context', 'targets']:
        actual_max = max(sum(native[p][k] for p in e) for e in full)
        assert any(sum(native[p][k] for p in b['intact_original_positions']) == actual_max for b in plan['selected_intact_original_batches'])
        rank_max = max(sum(native[p][k] for p in e[r::4]) for e in full for r in range(4))
        assert any(sum(native[p][k] for p in e[r::4]) == rank_max for e in plan['capacity_entries'] for r in range(4))
    for b in plan['selected_intact_original_batches']:
        actual = full[b['original_global_step'] - 1]
        assert b['intact_original_positions'] == actual
        maps = [m for m in plan['capacity_case_mapping'] if m.get('original_global_step') == b['original_global_step']]
        assert len(maps) == 4
        for r in range(4):
            assert Counter(p for m in maps for p in m['actual_original_positions'][r::4]) == Counter(actual)
    assert all(any(b['original_global_step'] == i for b in plan['selected_intact_original_batches']) for i in [1752, 3504, 5256])
    assert plan['capacity_case_mapping'][-1]['actual_rank_source_counts'] == [1, 0, 0, 0]
    assert plan['capacity_budget'] == data.budget(plan['capacity_entries'])
    assert not plan['training_admitted'] and not plan['full_size_capacity_passed']
    assert not torch.cuda.is_initialized()
    (tmp_path / 'actual-complete-source-capacity-plan.json').write_text(json.dumps(plan, indent=2))


def worker(rank, folder):
    folder = Path(folder)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + str(folder / 'SUM-rendezvous'), rank=rank, world_size=4,
        timeout=timedelta(minutes=10))
    control = dist.new_group(backend='gloo', timeout=timedelta(minutes=10))
    try:
        public = MixedCodingSFTData(ROOT)
        for profile in PROFILES:
            stage = setup(public, profile, rank, 4, control)
            # Compute a separate, serial full-response CE gradient at the same
            # initial weights. No stage method or reduction is reused here.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(989924)
                serial = model().train()
            adam = optimizer(serial)
            positions = stage.entries[0]
            denominator = sum(public.weight(profile, p) * sum(v != -100 for v in public.sequence(p)[1]) for p in positions)
            for p in positions:
                ids, labels = public.sequence(p)
                logits = serial(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.bool))['logits']
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, 32768), torch.tensor(labels[1:]), reduction='sum')
                (loss * public.weight(profile, p) / denominator).backward()
            named = serial.named_selected(('actor',))
            norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in named], OPTIONS['gradient_clip'], error_if_nonfinite=True))
            expected = [p.grad.clone() for _, p in named]
            adam.step()
            actual = []
            ordinary = stage.optimizer.step
            def capture(*args, **kwargs):
                actual.extend(p.grad.clone() for p in stage.parameters)
                return ordinary(*args, **kwargs)
            stage.optimizer.step = capture
            row = stage.update()
            maximum = max(float((a-b).abs().max()) for a, b in zip(actual, expected))
            assert maximum < 2e-6
            assert row['gradient_norm_before_clip'] == pytest.approx(norm, abs=2e-6, rel=2e-6)
            assert all(torch.allclose(p, s, atol=2e-6, rtol=2e-6) for p, (_, s) in zip(stage.parameters, named))
            (folder / (profile + '-serial-SUM-rank-' + str(rank) + '.json')).write_text(json.dumps(dict(
                profile=profile, rank=rank, actual_world_size=4, actual_positions=positions,
                actual_maximum_four_replica_vs_independent_serial_gradient_error=maximum,
                actual_unclipped_global_norm=row['gradient_norm_before_clip'], independent_serial_norm=norm,
                exactly_one_global_weighted_denominator_and_gradient_SUM_verified=True,
                additional_world_average_would_fail_this_comparison=True,
                actual_all_original_native_contexts_and_EOS_labels_used=True,
                own_small_CPUmechanics=True, production_GPUupdates=0, quality_gain_established=False,
                goal_achieved=False), indent=2))
        assert not torch.cuda.is_initialized()
    finally: dist.destroy_process_group()


def test_actual_four_rank_weighted_and_plain_gradients_match_independent_serial(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    reports = [json.loads((tmp_path / (p + '-serial-SUM-rank-' + str(r) + '.json')).read_text()) for p in PROFILES for r in range(4)]
    assert all(r['exactly_one_global_weighted_denominator_and_gradient_SUM_verified'] for r in reports)
