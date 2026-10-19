"""Actual own-neural gradients/recovery for the three machine preference arms."""
import copy
from contextlib import nullcontext
import random

import numpy as np
import pytest
import torch

from slm_lab.CPU_frozen_cast_cache_v1 import CPUFrozenCastCache
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.model import ModelConfig
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone
from slm_lab.public_edge_preference_stage_v1 import EdgePreferences, EdgePreferenceStage, METHODS, fork, log_odds, schedule
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest


@pytest.fixture(scope='module')
def data():
    return EdgePreferences('data/public-edge-case-hard-negatives-admitted-v1',
                           'reports/public-edge-case-hard-negatives-v1-readback-admission/complete.json')


def model():
    config = ModelConfig(vocab_size=32768, hidden_size=8, intermediate_size=16, num_layers=1,
                         num_heads=2, num_kv_heads=1, max_text_length=4096,
                         vision_hidden_size=8, vision_intermediate_size=16, vision_layers=1,
                         vision_heads=2, image_size=16, patch_size=4, pool_size=2)
    base = OwnDenseAssistant(new_backbone(config, 980013), pointing_rank=4, reward_hidden=4,
                            word_temporal=4, router_hidden=4, detector_hidden=4, word_expansion=1)
    return OwnDenseActor(base, rank=2).eval()


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def test_all77_actual_native_pairs_three_complete_epochs_masks_and_shared_prefixes(data):
    entries = schedule(len(data.rows))
    assert len(data.rows) == 77 and len(entries) == 60
    assert data.budget(entries)['chosen_response_EOS_targets'] == 98106
    assert data.budget(entries)['rejected_response_EOS_targets'] == 98109
    for row in data.rows:
        branch = fork(row); d = branch['response_position']
        assert row['chosen_complete_response_EOS_ids'][:d] == row['rejected_complete_response_EOS_ids'][:d]
        assert branch['context'] == row['native_prompt_ids'] + row['chosen_complete_response_EOS_ids'][:d]
        assert branch['chosen'] != branch['rejected'] and branch['chosen'] >= 4 and branch['rejected'] >= 4
    for epoch in range(3):
        assert sorted(p for entry in entries[epoch*20:(epoch+1)*20] for p in entry) == list(range(77))
    bad = copy.deepcopy(data.rows[0]); bad['rejected_complete_response_EOS_ids'] = bad['chosen_complete_response_EOS_ids']
    with pytest.raises(ValueError):
        fork(bad)


def test_fork_log_odds_matches_full_normalized_probability_and_only_credits_the_two_critical_actions():
    logits = torch.tensor([[[5., 2., 1.], [1000., 997., 999.]]], requires_grad=True)
    odds = log_odds(logits, 0, 2)
    normalized = logits[0, -1].log_softmax(-1)
    assert torch.equal(odds, normalized[0] - normalized[2])
    loss = torch.nn.functional.softplus(-.1 * odds)
    loss.backward()
    assert not logits.grad[0, 0].any() and logits.grad[0, 1, 1] == 0
    assert logits.grad[0, 1, 0] < 0 < logits.grad[0, 1, 2]


@pytest.mark.parametrize('profile', METHODS)
@pytest.mark.parametrize('precision', ['CPU_FP32', 'CPU_BF16_cached'])
def test_actual_own_neural_selected_gradients_complete_sources_exact_recovery_and_cold_policy(data, profile, precision):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    student = model()
    selected = [p for _, p in student.named_selected(('actor',))]
    optimizer = torch.optim.AdamW(selected, lr=1e-4, betas=(.9, .95), weight_decay=.01, foreach=False)
    order = sorted(range(len(data.rows)), key=lambda p: data.lengths[p])[:2]
    entries = [[order[0]], [order[1]]]
    options = dict(learning_rate=1e-4, gradient_clip=1., beta=.1, preference_weight=1., precision=precision)
    sources = dict(experiment_sha256='a'*64, profile_sha256=__import__('hashlib').sha256(profile.encode()).hexdigest())
    cache = CPUFrozenCastCache() if precision == 'CPU_BF16_cached' else None
    if cache:
        cache.attach(student.policy)
    stage = EdgePreferenceStage(student, optimizer, data, entries, profile, options, sources, cache, 'capacity')
    fixed = tensor_digest(student.base.named_parameters())
    before = tensor_digest(student.parts.named_parameters())
    random.seed(980213); np.random.seed(980213); torch.manual_seed(980213)
    stage.update()
    saved = copy.deepcopy(stage.state())
    wanted = stage.update()
    parts = tensor_digest(student.parts.named_parameters())
    adam = lambda: tensor_digest((str(i)+'.'+k, value[k]) for i, value in enumerate(optimizer.state.values()) for k in ['step','exp_avg','exp_avg_sq'])
    wanted_adam = adam(); wanted_rng = capture_rng('cpu')
    assert parts != before and tensor_digest(student.base.named_parameters()) == fixed
    assert all(p.grad is None for p in student.base.parameters())
    bad = copy.deepcopy(saved); bad['metrics'][0]['complete_pairs'][0]['chosen_response_EOS_targets'] -= 1
    with pytest.raises(ValueError):
        stage.resume(bad)
    assert tensor_digest(student.parts.named_parameters()) == parts and adam() == wanted_adam and equal(capture_rng('cpu'), wanted_rng)
    bad = copy.deepcopy(saved); bad['rng']['torch'] = torch.zeros(1, dtype=torch.uint8)
    with pytest.raises(ValueError):
        stage.resume(bad)
    assert tensor_digest(student.parts.named_parameters()) == parts and adam() == wanted_adam and equal(capture_rng('cpu'), wanted_rng)
    stage.options['beta'] = .2
    with pytest.raises(ValueError):
        stage.resume(saved)
    assert tensor_digest(student.parts.named_parameters()) == parts and adam() == wanted_adam and equal(capture_rng('cpu'), wanted_rng)
    stage.options['beta'] = .1
    stage.resume(saved)
    assert stage.update() == wanted
    assert tensor_digest(student.parts.named_parameters()) == parts and adam() == wanted_adam and equal(capture_rng('cpu'), wanted_rng)
    other = OwnDenseActor(student.base, **student.settings).eval()
    other.import_export(student.export(sources), expected_sources=sources)
    if cache:
        cache.attach(other.policy)
    batch = data.batch(order[0], 'fork')
    student.eval()
    context = torch.autocast('cpu', dtype=torch.bfloat16) if cache else nullcontext()
    with torch.no_grad(), context:
        assert torch.equal(student(**batch)['logits'], other(**batch)['logits'])
    assert tensor_digest(student.base.named_parameters()) == fixed
    assert stage.reference.resident_inventory(student)['extra_reference_excluded_from_export']
    if cache:
        cache.close()
