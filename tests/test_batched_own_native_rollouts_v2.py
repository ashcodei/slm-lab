from dataclasses import replace
import json

import pytest
import torch

from slm_lab.batched_own_native_rollouts_v2 import sample, select_cache, BehaviorComparisonError
from test_frozen_linear_cast_cache_v2 import small


def serial(actor, prefix, seeds, horizon):
    records = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            state = actor.policy.prefill(torch.tensor([prefix]))
            tokens, logps = [], []
            for step in range(horizon):
                logits = state['logits'][0, -1].float()
                token = int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
                tokens.append(token)
                logps.append(float(logits.log_softmax(-1)[token]))
                if token == 2 or step + 1 == horizon:
                    break
                state = actor.policy.decode(torch.tensor([[token]]), state['cache'])
        records.append(dict(actions=tokens, logps=logps, generator_state=generator.get_state()))
    return records


def test_actual35layer_four_attempts_match_independent_serial_tokens_behavior_and_private_draws(tmp_path):
    actor = small()
    prefix, seeds, horizon = [1, 5, 7], [19, 31, 47, 79], 12
    before = torch.get_rng_state().clone()
    expected = serial(actor, prefix, seeds, horizon)
    report = sample(actor, prefix, seeds=seeds, horizon=horizon, purpose='CPU-mechanics', maximum_cache_bytes=8 * 1024**2)
    from slm_lab.batched_own_native_rollouts_v2 import state_sha
    for actual, wanted in zip(report['trajectories'], expected):
        assert actual['actual_generated_token_ids'] == wanted['actions']
        assert max(abs(a - b) for a, b in zip(actual['actual_cached_behavior_logps'], wanted['logps'])) < 1e-5
        assert actual['final_private_generator_sha256'] == state_sha(wanted['generator_state'])
    assert report['actual_prefill_batch_sizes'] == [4]
    assert max(report['actual_decode_batch_sizes']) <= 4
    assert report['cached_full_max_absolute_difference'] <= .001
    assert torch.equal(torch.get_rng_state(), before) and all(p.grad is None for p in actor.parameters())
    assert sample(actor, prefix, seeds=seeds, horizon=horizon, purpose='CPU-mechanics', maximum_cache_bytes=8 * 1024**2) == report
    assert not torch.cuda.is_initialized()
    (tmp_path / 'actual35layer-four-private-attempts.json').write_text(json.dumps(report, indent=2))


def test_finished_real_neural_histories_leave_batch_without_extra_actions_or_private_RNG_draws(tmp_path):
    actor = small()
    seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    report = sample(actor, [1, 5, 7], seeds=seeds, horizon=24, purpose='CPU-mechanics', maximum_cache_bytes=16 * 1024**2)
    counts = [r['actual_native_action_count'] for r in report['trajectories']]
    assert len(set(counts)) > 1 and min(counts) < 24
    assert report['actual_decode_batch_sizes'] == sorted(report['actual_decode_batch_sizes'], reverse=True)
    assert min(report['actual_decode_batch_sizes']) < 8
    expected = serial(actor, [1, 5, 7], seeds, 24)
    from slm_lab.batched_own_native_rollouts_v2 import state_sha
    for actual, wanted in zip(report['trajectories'], expected):
        assert actual['actual_generated_token_ids'] == wanted['actions']
        assert actual['final_private_generator_sha256'] == state_sha(wanted['generator_state'])
    (tmp_path / 'actual35layer-variable-stops.json').write_text(json.dumps(report, indent=2))


@pytest.mark.parametrize('case', ['short_cache', 'duplicate_seed', 'wrong_purpose', 'unbounded_horizon', 'autocast'])
def test_invalid_runtime_or_capacity_rejects_before_any_prefill(case):
    actor = small()
    kwargs = dict(seeds=[19, 31], horizon=12, purpose='CPU-mechanics', maximum_cache_bytes=8 * 1024**2)
    if case == 'short_cache':
        kwargs['maximum_cache_bytes'] = 1
    elif case == 'duplicate_seed':
        kwargs['seeds'] = [19, 19]
    elif case == 'wrong_purpose':
        kwargs['purpose'] = 'GPU-runtime'
    elif case == 'unbounded_horizon':
        kwargs['horizon'] = 64
    called = []
    original = actor.policy.prefill
    actor.policy.prefill = lambda *a, **k: (called.append(True), original(*a, **k))[1]
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=case == 'autocast'), pytest.raises(ValueError):
        sample(actor, [1, 5, 7], **kwargs)
    assert not called and not torch.cuda.is_initialized()


def test_foreign_and_changed_native_cache_cannot_be_attached_to_active_histories():
    actor, other = small(), small()
    with torch.no_grad():
        cache = actor.policy.prefill(torch.tensor([[1, 5], [1, 5]]))['cache']
    with pytest.raises(ValueError):
        select_cache(other.policy, cache, [0])
    bad = replace(cache, layers=((cache.layers[0][0][:, :, :-1], cache.layers[0][1]), *cache.layers[1:]))
    with pytest.raises(ValueError):
        select_cache(actor.policy, bad, [0])
    with pytest.raises(ValueError):
        select_cache(actor.policy, cache, [0, 0])
    with torch.no_grad():
        next(actor.parts.parameters()).add_(.01)
    with pytest.raises(ValueError):
        select_cache(actor.policy, cache, [0])


def test_tighter_behavior_guard_preserves_all_actual_failure_observations_without_an_update():
    actor = small()
    before = torch.get_rng_state().clone()
    with pytest.raises(BehaviorComparisonError) as failure:
        sample(actor, [1, 5, 7], seeds=[19, 31, 47, 79], horizon=12,
            purpose='CPU-mechanics', maximum_cache_bytes=8 * 1024**2, cache_full_absolute_bound=1e-10)
    report = failure.value.report
    assert len(report['trajectories']) == 4 and report['all_actual_actions_and_probabilities_preserved']
    assert not report['behavior_rescore_passed'] and report['model_updates'] == 0
    assert torch.equal(torch.get_rng_state(), before) and all(p.grad is None for p in actor.parameters())


def test_nonfinite_full_rescore_is_rejected_with_all_original_samples_and_JSON_safe_evidence(monkeypatch):
    actor = small()
    original = actor.forward
    def nonfinite(*args, **kwargs):
        result = original(*args, **kwargs)
        return dict(result, logits=torch.full_like(result['logits'], float('nan')))
    monkeypatch.setattr(actor, 'forward', nonfinite)
    before = torch.get_rng_state().clone()
    with pytest.raises(BehaviorComparisonError) as failure:
        sample(actor, [1, 5, 7], seeds=[19, 31, 47, 79], horizon=3,
            purpose='CPU-mechanics', maximum_cache_bytes=8 * 1024**2)
    report = failure.value.report
    assert report['nonfinite_full_forward_behavior_observed'] and not report['behavior_rescore_passed']
    assert report['cached_full_max_absolute_difference'] is None and len(report['trajectories']) == 4
    assert all(t['actual_nonfinite_full_forward_behavior_observed'] for t in report['trajectories'])
    assert all(p == 'NaN' for t in report['trajectories'] for p in t['actual_full_forward_behavior_logps'])
    json.dumps(report, allow_nan=False)
    assert torch.equal(torch.get_rng_state(), before) and all(p.grad is None for p in actor.parameters())
