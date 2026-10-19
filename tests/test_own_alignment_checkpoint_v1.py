import copy
import random

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from slm_lab.own_aligned_assistant_v1 import OwnAlignedAssistant, byte_ctc_loss, decode_bytes
from slm_lab.own_alignment_checkpoint_v1 import make_state, restore_state
from test_own_aligned_assistant_v1 import fixture, original_signature


GROUPS = ('language', 'vision', 'critic', 'reward', 'word')
SOURCES = dict(original_policy_sha256='a'*64, run_plan_sha256='b'*64, input_sha256='c'*64)


def update(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    draw = random.random() + float(np.random.random()) + float(torch.rand(()))
    ids = torch.tensor([[1, 4, 7, 2], [1, 6, 5, 2]])
    pixels = torch.arange(2*3*16*16, dtype=torch.float32).reshape(2, 3, 16, 16)/1000 - 1
    output = model(ids, pixels, labels=ids)
    scores = model.preference_score(ids)
    words = model.word_logits(torch.ones(2, 3, 8, 32)*draw)
    loss = (output['loss'] + F.softplus(-(scores[0]-scores[1]))
            + (output['values']-draw).square().mean() + .1*byte_ctc_loss(words, ['cat', 'ab']))
    loss.backward()
    params = model.extension_parameters(*GROUPS)
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
    assert all(p.grad is None for p in model.original.parameters())
    torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return dict(draw=draw, objective=float(loss.detach()))


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_nested_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def setup():
    model = OwnAlignedAssistant(fixture(), rank=2, reward_hidden=8, word_temporal=4).eval()
    optimizer = torch.optim.AdamW(model.extension_parameters(*GROUPS), lr=1e-3, foreach=False)
    metrics = [update(model, optimizer)]
    state = make_state(model, optimizer, groups=GROUPS, sources=SOURCES, completed_steps=1, planned_steps=3,
                       sampler_state=dict(next_source_position=1), metrics=metrics, device='cpu')
    kwargs = dict(groups=GROUPS, expected_sources=SOURCES, planned_steps=3,
                  expected_sampler_state=dict(next_source_position=1), device='cpu')
    return model, optimizer, state, kwargs


def test_all_new_modules_Adam_RNG_and_source_cursor_reproduce_actual_next_update(tmp_path):
    torch.use_deterministic_algorithms(True)
    model, optimizer, state, kwargs = setup()
    signature = original_signature(model.original)
    path = tmp_path / 'new-only.pt'; torch.save(state, path)
    saved = torch.load(path, weights_only=True)
    expected = update(model, optimizer)
    weights = model.extension_state(); adam = copy.deepcopy(optimizer.state_dict())
    restored_steps, metrics = restore_state(model, optimizer, saved, **kwargs)
    assert restored_steps == 1 and metrics == state['metrics']
    assert update(model, optimizer) == expected
    assert_nested_equal(weights, model.extension_state())
    assert_nested_equal(adam, optimizer.state_dict())
    assert signature == original_signature(model.original)
    assert not torch.cuda.is_initialized()


def test_incomplete_corrupt_Adam_rng_sources_and_cursors_reject_before_any_mutation():
    model, optimizer, state, kwargs = setup()
    update(model, optimizer)
    expected = model.extension_state(); adam = copy.deepcopy(optimizer.state_dict())
    for mutate in [lambda s: s['model']['sources'].update(input_sha256='d'*64),
                   lambda s: s['model']['extensions'][next(iter(s['model']['extensions']))].fill_(float('nan')),
                   lambda s: s['optimizer']['state'][0]['exp_avg_sq'].fill_(-1),
                   lambda s: s['optimizer']['state'].pop(0),
                   lambda s: s['optimizer']['param_groups'][0].update(lr=.5),
                   lambda s: s['rng'].update(torch=torch.zeros(4, dtype=torch.uint8)),
                   lambda s: s['sampler_state'].update(next_source_position=2),
                   lambda s: s.update(fixed_original_state_sha256='0'*64)]:
        bad = copy.deepcopy(state); mutate(bad)
        rng = torch.get_rng_state().clone(); signature = original_signature(model)
        with pytest.raises(ValueError):
            restore_state(model, optimizer, bad, **kwargs)
        assert signature == original_signature(model) and torch.equal(rng, torch.get_rng_state())
        assert_nested_equal(expected, model.extension_state())
        assert_nested_equal(adam, optimizer.state_dict())


def test_complete_byte_alphabet_and_repeated_character_CTC_capacity():
    assert decode_bytes([0, 0, 256, 0, 195, 169]) == ('\x00\x00é', True)
    assert decode_bytes([255]) == ('�', False)
    with pytest.raises(ValueError, match='time axis'):
        byte_ctc_loss(torch.zeros(1, 2, 257), ['aa'])
    loss = byte_ctc_loss(torch.zeros(1, 3, 257), ['aa'])
    assert bool(torch.isfinite(loss))
