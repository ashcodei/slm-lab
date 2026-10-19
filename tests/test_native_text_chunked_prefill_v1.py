"""Actual own-small full-vocabulary/depth and full-context prefill controls.

No test uses full-size GPU weights or establishes coding accuracy or GPU speed.
"""
import json

import pytest
import torch

from slm_lab.batched_own_native_rollouts_v2 import sample, select_cache, signature
from slm_lab.deterministic_positions_v1 import install as positions
from slm_lab.model import ModelConfig
from slm_lab.native_text_chunked_prefill_v1 import install, PROTOCOL
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone


def student(*, layers=35, vocabulary=32768):
    config = ModelConfig(vocab_size=vocabulary, hidden_size=8, intermediate_size=16,
        num_layers=layers, num_heads=2, num_kv_heads=1, max_text_length=4096,
        vision_hidden_size=8, vision_intermediate_size=16, vision_layers=1,
        vision_heads=2, image_size=32, patch_size=4, pool_size=2)
    with torch.random.fork_rng(devices=[]):
        base = positions(OwnDenseAssistant(new_backbone(config, 990171), reward_hidden=4,
            word_temporal=4, pointing_rank=4, router_hidden=4, detector_hidden=4, word_expansion=1))
        actor = OwnDenseActor(base, rank=2, seed=990173).eval()
    with torch.no_grad():
        for name, p in actor.named_selected(('actor',)):
            if name.endswith('.up'): p.copy_(torch.linspace(-.03, .03, p.numel()).reshape(p.shape))
    return actor


def differences(actual, expected):
    logits = float((actual['logits'] - expected['logits']).abs().max())
    logps = float((actual['logits'].log_softmax(-1) - expected['logits'].log_softmax(-1)).abs().max())
    kv = max(float((a-b).abs().max()) for aa, bb in zip(actual['cache'].layers, expected['cache'].layers)
        for a, b in zip(aa, bb))
    assert logits < 2e-5 and logps < .001 and kv < 2e-5
    assert actual['cache'].bytes == expected['cache'].bytes
    return dict(maximum_logit_difference=logits, maximum_log_probability_difference=logps, maximum_KV_difference=kv)


def test_actual35layers_full_vocabulary_eight_histories_preserve_context_and_source_state(tmp_path):
    actor, peer = student(), student()
    ids = (torch.arange(8 * 257).reshape(8, 257) % 32764) + 4
    ids[:, 0] = 1
    before = signature(actor)
    tensors = {n: t.clone() for n, t in actor.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with torch.no_grad(): expected = actor.policy.prefill(ids)
    controller = install(actor, chunk_tokens=64, purpose='CPU-mechanics')
    actual = actor.policy.prefill(ids)
    errors = differences(actual, expected)
    assert actor.counts()['actual_unique'] < 10**6
    assert signature(actor) == before and torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(t, tensors[n]) for n, t in actor.state_dict().items())
    assert all(p.grad is None for p in actor.parameters())
    observed = controller.last_observation
    assert observed['actual_chunk_query_lengths'] == [64, 64, 64, 64, 1]
    assert observed['actual_chunk_key_lengths'] == [64, 128, 192, 256, 257]
    assert observed['actual_complete_native_KV_bytes'] == expected['cache'].bytes
    # Existing full-forward scoring is untouched, including its gradient path.
    with torch.no_grad():
        assert torch.equal(actor(ids[:, :19])['logits'], peer(ids[:, :19])['logits'])
        wanted = actor.policy.decode(torch.full((8, 1), 17), expected['cache'])
        got = actor.policy.decode(torch.full((8, 1), 17), actual['cache'])
    differences(got, wanted)
    with pytest.raises(ValueError): select_cache(peer.policy, actual['cache'], [0])
    selected = select_cache(actor.policy, actual['cache'], [1, 4, 7])
    assert selected.layers[0][0].shape[0] == 3 and selected.text_length == 257
    proof = dict(actual_small_research_parameters=actor.counts()['actual_unique'],
        actual_decoder_layers=35, actual_vocabulary=32768, **observed, **errors,
        actual_next_cached_decode_checked=True, unmodified_peer_and_full_forward_bit_exact=True,
        every_source_tensor_parameter_inventory_and_global_RNG_unchanged=True,
        actual_GPU_contexts=0, full_size_GPU_capacity_or_quality_claimed=False)
    (tmp_path / 'actual35layer-full-vocabulary-prefill.json').write_text(json.dumps(proof, indent=2))
    controller.close()
    with torch.no_grad(): assert torch.equal(actor.policy.prefill(ids)['logits'], expected['logits'])


def test_complete4096_context_eight_histories_reaches_every_original_KV_position(tmp_path):
    # Full context geometry with a declared two-layer/64-token CPU fixture.
    actor = student(layers=2, vocabulary=64)
    ids = (torch.arange(8 * 4095).reshape(8, 4095) % 60) + 4
    ids[:, 0] = 1
    with torch.no_grad():
        independent = [actor.policy.prefill(ids[r:r+1]) for r in range(8)]
    controller = install(actor, chunk_tokens=128, purpose='CPU-mechanics')
    actual = actor.policy.prefill(ids)
    observed = dict(controller.last_observation)
    errors = []
    for r, expected in enumerate(independent):
        row = dict(logits=actual['logits'][r:r+1], cache=select_cache(actor.policy, actual['cache'], [r]))
        errors.append(differences(row, expected))
    # The final original context position is genuinely stored, not padded away.
    with torch.no_grad(): final = actor.policy.decode(torch.full((8, 1), 17), actual['cache'])
    assert final['cache'].stored_tokens == final['cache'].text_length == 4096
    assert all(k.shape[0] == 8 and k.shape[2] == 4096 for pair in final['cache'].layers for k in pair)
    assert observed['actual_chunk_query_lengths'] == [128] * 31 + [127]
    assert observed['actual_chunk_key_lengths'][-1] == 4095
    assert max(observed['actual_explicit_causal_mask_elements']) <= 128 * 4095
    proof = dict(**observed, actual_small_research_parameters=actor.counts()['actual_unique'],
        actual_decoder_layers=2, actual_vocabulary=64, actual_final_context_positions=4096,
        actual_independent_serial_history_errors=errors, actual_GPU_contexts=0,
        full_size_GPU_capacity_or_quality_claimed=False)
    (tmp_path / 'actual-complete4096-eight-histories.json').write_text(json.dumps(proof, indent=2))


def test_native_sampler_still_checks_every_actual_action_and_image_path_remains_exact(tmp_path):
    actor = student()
    ids = torch.tensor([[1, 5, 7]])
    pixels = torch.linspace(-1., 1., 3 * 64 * 64).reshape(1, 3, 64, 64)
    with torch.no_grad(): expected = actor.policy.prefill(ids, pixel_values=pixels)
    controller = install(actor, chunk_tokens=16, purpose='CPU-mechanics')
    actual = actor.policy.prefill(ids, pixel_values=pixels)
    assert controller.last_observation['original_image_prefill_used']
    assert torch.equal(actual['logits'], expected['logits'])
    assert all(torch.equal(a, b) for aa, bb in zip(actual['cache'].layers, expected['cache'].layers) for a, b in zip(aa, bb))
    before, rng = signature(actor), torch.get_rng_state().clone()
    report = sample(actor, [1] + list(range(4, 68)), seeds=list(range(990201, 990209)), horizon=6,
        purpose='CPU-mechanics', maximum_cache_bytes=8 * 1024**2)
    assert report['behavior_rescore_passed'] and len(report['trajectories']) == 8
    assert report['cached_full_max_absolute_difference'] <= .001
    assert all(t['actual_generated_token_ids'] and t['actual_full_forward_behavior_logps'] for t in report['trajectories'])
    assert signature(actor) == before and torch.equal(torch.get_rng_state(), rng)
    (tmp_path / 'actual-eight-private-native-attempts.json').write_text(json.dumps(report, indent=2))
    with pytest.raises(ValueError): install(actor, chunk_tokens=16, purpose='CPU-mechanics')
    with pytest.raises(ValueError): install(student(), chunk_tokens=0, purpose='CPU-mechanics')
    with pytest.raises(ValueError): install(student(), chunk_tokens=16, purpose='GPU-runtime')
    actor.train()
    with pytest.raises(ValueError): actor.policy.prefill(ids)
    actor.eval()
    with torch.autocast('cpu', dtype=torch.bfloat16), pytest.raises(ValueError): actor.policy.prefill(ids)
    with torch.no_grad(): next(actor.parts.parameters()).add_(.01)
    with pytest.raises(ValueError): select_cache(actor.policy, expected['cache'], [0])
    # Real adapter updates are allowed; a following prefill binds a fresh cache.
    fresh = actor.policy.prefill(ids)
    assert fresh['cache'].parameter_signature == actor.policy.signature()
