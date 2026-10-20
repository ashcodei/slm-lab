"""Fresh frame autocasts must retain both real actor endpoints exactly."""
import torch

from slm_lab import code_quality, evaluation_provenance
from slm_lab.CPU_frozen_cast_cache_v1 import CPUFrozenCastCache
from slm_lab.function_repair_evaluation_CPU_v1 import legacy_runtime
from slm_lab.inference_rotary_cache_v1 import inference_cache
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_own_dense_actor_v1 import actor


def test_cached_BF16_frames_match_uncached_after_actual_actor_training_and_endpoint_swaps(tmp_path):
    torch.set_num_threads(1)
    model = actor()
    sources = {'own_source': 'a' * 64}
    ids = torch.tensor([[1, 4, 7, 8, 2]])
    initial = model.export(sources)
    native_sha = tensor_digest(model.base.named_parameters())
    parameters = [p for _, p in model.named_selected(('actor',))]
    optimizer = torch.optim.AdamW(parameters, lr=.03, foreach=False)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        model(ids, labels=ids)['loss'].backward(inputs=parameters)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    trained = model.export(sources)
    assert any(not torch.equal(initial['parts'][k], v) for k, v in trained['parts'].items())
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'exact immutable small own actor endpoint')

    expected = []
    for payload in (initial, trained):
        model.import_export(payload, expected_sources=sources)
        with legacy_runtime(model, checkpoint, {}):
            logits = model(ids)['logits'].clone()
            actions = model.generate(ids, max_new_tokens=8, eos_token_id=2, use_cache=True)
        expected.append((logits, actions))
    assert not torch.equal(expected[0][0], expected[1][0])

    rng = torch.get_rng_state().clone()
    counts = model.counts()
    cache = CPUFrozenCastCache()
    cache.attach(model.policy)
    try:
        with inference_cache():
            for payload, (wanted_logits, wanted_actions) in zip((initial, trained), expected):
                # Changing adapters outside autocast is required to avoid stale casts.
                model.import_export(payload, expected_sources=sources)
                for _ in ('completion', 'instruction'):
                    with legacy_runtime(model, checkpoint, {}):
                        torch.set_num_threads(2)  # Legacy evaluation requests two.
                        assert torch.get_num_threads() == 1
                        assert torch.equal(model(ids)['logits'], wanted_logits)
                        assert torch.equal(model.generate(ids, max_new_tokens=8, eos_token_id=2,
                                                          use_cache=True), wanted_actions)
                        report = {}
                        code_quality.seal(report, evaluation_provenance.capture(), device='cpu')
                        assert report['inference_precision'] == 'bf16'
                        assert report['actual_CPU_threads'] == 1
                    assert not torch.is_autocast_enabled('cpu')
                cache.assert_current()
        inventory = cache.inventory()
        assert inventory['actual_cache_hits'] > inventory['actual_conversions'] > 0
        assert inventory['added_learned_parameters'] == 0
        assert model.counts() == counts
        assert native_sha == tensor_digest(model.base.named_parameters())
        assert torch.equal(rng, torch.get_rng_state())
        assert all(p.grad is None for p in model.parameters())
    finally:
        cache.close()
    assert not cache.entries and not cache.patched
