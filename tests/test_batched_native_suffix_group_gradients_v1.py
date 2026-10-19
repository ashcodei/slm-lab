"""Real private samples and exact-source execution on an initialized CPUfixture.

The tiny model deliberately encodes a five-token source-specific transition
chain. This tests mechanics; it is not a trained production checkpoint or an
unassisted coding result. Every original public full-solution target remains.
"""
import copy
import json
import math
from pathlib import Path

import pytest
import torch

from slm_lab.batched_native_suffix_group_gradients_v1 import group_gradients, group_semantics, CodeGroupFailure
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.corpus import file_hash
from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference
from slm_lab.native_code_execution_oracle_v2 import NativeCodeExecutionOracle
from slm_lab.own_dense_actor_checkpoint_v1 import make_state, restore_state
from slm_lab.own_code_delta_v1 import norm_names
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.qualified_native_token_suffix_episode_v1 import QualifiedNativeTokenSuffixEpisode, CompleteLegacySuffixCatalogue
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_public_reasoning_group_gradients_v1 import model, optimizer, equal

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / 'reports/native-suffix-CPU-mechanics-qualification-v1/closed-qualified-controls.json'


@pytest.fixture(scope='module')
def episode():
    corpus = ExpandedExecutionCorpus(ROOT)
    oracle = NativeCodeExecutionOracle(corpus)
    return QualifiedNativeTokenSuffixEpisode(oracle, CONTROL)


def fixture(episode, only_EOS=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    student = model()
    network = student.base.network
    prepared = episode.rollout_view()
    tokens = [prepared['native_rollout_prefix_ids'][-1], *prepared['original_missing_suffix_and_EOS_ids']]
    assert len(tokens) == len(set(tokens)) == 6 and tokens[-1] == 2
    hidden, amplitude = 8, 20.
    scale = math.sqrt(hidden)
    gate = float(torch.nn.functional.silu(torch.tensor(1.)))
    with torch.no_grad():
        network.embedding.weight.zero_()
        for i, token in enumerate(tokens): network.embedding.weight[token, i] = amplitude
        for name in norm_names(student.config.num_layers): network.get_parameter(name).fill_(1.)
        for block in network.blocks:
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']: getattr(block.attn, name).weight.zero_()
            block.gate.weight.zero_(); block.up.weight.zero_(); block.down.weight.zero_()
            block.gate.weight[:hidden].fill_(1. / scale)
            block.up.weight[:hidden].copy_(torch.eye(hidden) / (scale * gate))
            for i in range(6):
                target = torch.zeros(hidden)
                target[5 if only_EOS and i == 0 else min(i + 1, 5)] = amplitude
                if i == 0 and not only_EOS: target[5] = amplitude
                target[i] -= amplitude
                block.down.weight[:, i].copy_(target)
    reference = FrozenOwnActorReference(student, sources=dict(own_fixture_sha256='a' * 64,
        source_controls_sha256=file_hash(CONTROL)))
    return student, reference


def gradients(student, episode, reference, profile, seeds=None):
    student.zero_grad(set_to_none=True)
    return group_gradients(student, episode, reference, profile=profile, seeds=list(range(8)) if seeds is None else seeds,
        purpose='CPU-mechanics', maximum_cache_bytes=8 * 1024**2, CPU_mechanics_horizon=5)


@pytest.mark.parametrize('only_EOS', [False, True])
def test_real_source_qualified_native_groups_actual_reward_policy_gradients_and_unconditional_full_CE(episode, only_EOS, tmp_path):
    student, reference = fixture(episode, only_EOS)
    fixed, critic = tensor_digest(student.base.named_parameters()), student.parts['critic'].weight.clone()
    rng = value_digest(capture_rng('cpu'))
    control = gradients(student, episode, reference, 'anchor_CE')
    wanted = [p.grad.clone() for _, p in student.named_selected(('actor',))]
    native = gradients(student, episode, reference, 'anchor_execution')
    actual = [p.grad.clone() for _, p in student.named_selected(('actor',))]
    a, b = group_semantics(control), group_semantics(native)
    a.pop('profile'); b.pop('profile')
    assert a == b
    if only_EOS:
        assert len(set(native['actual_rewards'])) == 1 and native['actual_advantages'] == [0.] * 8
        assert all(torch.equal(x, y) for x, y in zip(wanted, actual))
    else:
        assert 1. in native['actual_rewards'] and any(r < 1 for r in native['actual_rewards'])
        assert any(not torch.equal(x, y) for x, y in zip(wanted, actual))
    assert native['unconditional_original_CE_anchor']['local_original_response_EOS_targets'] == episode.rollout_view()['original_response_EOS_targets']
    assert native['all_original_reference_CE_targets_unconditional'] and value_digest(capture_rng('cpu')) == rng
    assert all(t['actual_execution_thread']['actual_nice'] == 19 and t['actual_execution_thread']['actual_scheduler'] == 5
        and not t['actual_execution_thread']['thread_is_main'] for t in native['trajectories'])
    old = tensor_digest(student.parts.named_parameters())
    adam = optimizer(student)
    torch.nn.utils.clip_grad_norm_(list(p for _, p in student.named_selected(('actor',))), 1., error_if_nonfinite=True)
    adam.step(); adam.zero_grad(set_to_none=True)
    assert tensor_digest(student.parts.named_parameters()) != old
    assert tensor_digest(student.base.named_parameters()) == fixed and torch.equal(student.parts['critic'].weight, critic)
    assert not torch.cuda.is_initialized()
    proof = dict(actual_small_fixture_parameters=student.counts()['actual_unique'], deliberately_initialized_source_specific_fixture=True,
        actual_source_id=episode.corpus.row(episode.position)['id'], CPU_mechanics_horizon=5,
        complete_original_native_horizon=episode.rollout_view()['full_native_action_budget'], only_EOS_fixture=only_EOS,
        actual_native_rewards=native['actual_rewards'], actual_native_advantages=native['actual_advantages'],
        exact_paired_private_actions_probabilities_execution_outcomes_and_full_CE=True,
        actual_all_original_CE_targets=native['unconditional_original_CE_anchor']['local_original_response_EOS_targets'],
        actual_CE_and_execution_intervals_overlap=native['actual_CE_and_execution_task_intervals_overlap'],
        actual_complete_group=native, actual_test_Adam_updates=1, original_base_and_critic_unchanged=True,
        production_updates=0, unassisted_coding_score=False, full_size_GPUcapacity_or_quality_not_established=True)
    (tmp_path / 'actual-native-source-group-gradient-control.json').write_text(json.dumps(proof, indent=2, allow_nan=False))


def test_actual_next_private_group_verdicts_gradients_weights_Adam_and_RNG_replay_exact(episode, tmp_path):
    student, reference = fixture(episode)
    adam = optimizer(student)
    sources = dict(source_controls_sha256=file_hash(CONTROL), kernel_sha256=file_hash(ROOT / 'src/slm_lab/batched_native_suffix_group_gradients_v1.py'))
    def update(seeds):
        row = gradients(student, episode, reference, 'anchor_execution', seeds)
        digest = tensor_digest((n, p.grad) for n, p in student.named_selected(('actor',)))
        torch.nn.utils.clip_grad_norm_([p for _, p in student.named_selected(('actor',))], 1., error_if_nonfinite=True)
        adam.step(); adam.zero_grad(set_to_none=True)
        return row, dict(actual_semantics=group_semantics(row), all_actual_unclipped_actor_gradients_sha256=digest)
    first, _ = update(list(range(8)))
    cursor = dict(episode_binding=episode.binding(), completed=1, next_private_seeds=list(range(8, 16)))
    saved = copy.deepcopy(make_state(student, adam, groups=('actor',), sources=sources, completed=1,
        planned=2, cursor=cursor, metrics=[first], device='cpu'))
    next_row, expected = update(list(range(8, 16)))
    values, moments, rng = tensor_digest(student.parts.named_parameters()), copy.deepcopy(adam.state_dict()), capture_rng('cpu')
    restore_state(student, adam, saved, groups=('actor',), expected_sources=sources, planned=2, expected_cursor=cursor, device='cpu')
    replay_row, actual = update(list(range(8, 16)))
    assert expected == actual and tensor_digest(student.parts.named_parameters()) == values
    assert equal(adam.state_dict(), moments) and equal(capture_rng('cpu'), rng)
    (tmp_path / 'actual-next-native-code-group-recovery.json').write_text(json.dumps(dict(
        actual_small_fixture_parameters=student.counts()['actual_unique'], actual_physical_test_Adam_calls=3,
        actual_logical_trial_updates=2, actual_first_group=first, actual_next_group=next_row, actual_replayed_group=replay_row,
        exact_actual_actions_verdict_semantics_logps_unclipped_gradients_weights_Adam_private_RNG=True,
        every_actual_execution_timing_and_worker_observation_retained=True,
        timing_and_thread_ID_diagnostics_not_claimed_equal=True, production_updates=0,
        full_size_GPUcapacity_or_quality_not_established=True), indent=2, allow_nan=False))


def test_missing_verdict_groups_keep_CE_and_never_invent_zero_rewards(episode, monkeypatch):
    student, reference = fixture(episode)
    with monkeypatch.context() as changed:
        changed.setattr(episode, 'score', lambda *a, **k: dict(reward=None, status='controlled_test_abstention'))
        control = gradients(student, episode, reference, 'anchor_CE')
        expected = [p.grad.clone() for _, p in student.named_selected(('actor',))]
        actual = gradients(student, episode, reference, 'anchor_execution')
    assert actual['actual_rewards'] == [None] * 8 and actual['actual_advantages'] == [0.] * 8
    assert actual['policy_gradient_abstained_due_to_missing_verdict']
    assert actual['unconditional_original_CE_anchor'] == control['unconditional_original_CE_anchor']
    assert all(torch.equal(a, p.grad) for a, (_, p) in zip(expected, student.named_selected(('actor',))))


def test_nonfinite_gradient_rescore_retains_all_real_samples_and_verdicts_and_clears_partial_gradients(episode, monkeypatch):
    student, reference = fixture(episode)
    before, rng = tensor_digest(student.parts.named_parameters()), value_digest(capture_rng('cpu'))
    original = student.forward
    def nonfinite_only_policy_rescore(*args, **kwargs):
        result = original(*args, **kwargs)
        return dict(result, logits=torch.full_like(result['logits'], float('nan'))) if student.training and args else result
    monkeypatch.setattr(student, 'forward', nonfinite_only_policy_rescore)
    with pytest.raises(CodeGroupFailure) as failure: gradients(student, episode, reference, 'anchor_execution')
    report = failure.value.report
    assert report['phase'] == 'nonfinite_or_mismatched_gradient_behavior' and report['nonfinite_gradient_forward_behavior_observed']
    assert len(report['trajectories']) == 8 and all('actual_native_execution' in t for t in report['trajectories'])
    json.dumps(report, allow_nan=False)
    assert tensor_digest(student.parts.named_parameters()) == before and value_digest(capture_rng('cpu')) == rng
    assert all(p.grad is None for p in student.parameters())


def test_execution_exception_keeps_all_attempts_and_cannot_publish_partial_gradients(episode, monkeypatch):
    student, reference = fixture(episode)
    before = tensor_digest(student.parts.named_parameters())
    original = episode.score
    def fail_one_native_outcome(actions, *, horizon):
        if actions == [2]: raise OSError('controlled source worker failure')
        return original(actions, horizon=horizon)
    with monkeypatch.context() as changed:
        changed.setattr(episode, 'score', fail_one_native_outcome)
        with pytest.raises(CodeGroupFailure) as failure: gradients(student, episode, reference, 'anchor_execution')
    assert len(failure.value.report['trajectories']) == 8
    assert failure.value.report['actual_execution_errors']
    assert all('actual_native_execution' in t for t in failure.value.report['trajectories'])
    assert tensor_digest(student.parts.named_parameters()) == before and all(p.grad is None for p in student.parameters())


def test_CPUfixture_cannot_claim_full_catalogue_GPUeligibility_or_change_source_roles(episode):
    student, reference = fixture(episode)
    before = tensor_digest(student.parts.named_parameters())
    with pytest.raises(ValueError):
        group_gradients(student, episode, reference, profile='anchor_execution', seeds=list(range(8)),
            purpose='GPU-runtime', maximum_cache_bytes=8 * 1024**2)
    try:
        with pytest.raises(ValueError): CompleteLegacySuffixCatalogue(episode.oracle)
    except FileNotFoundError:
        raise AssertionError('An incomplete full catalogue must explicitly reject, not silently become eligible')
    assert tensor_digest(student.parts.named_parameters()) == before and all(p.grad is None for p in student.parameters())
