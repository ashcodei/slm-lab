import copy
import math

import pytest
import torch
from torch.nn import functional as F

from slm_lab.own_aligned_assistant_v3 import OwnAlignedAssistantV3
from slm_lab.shared_word_data_v1 import SharedWordData
from slm_lab.shared_word_stage_v1 import SharedWordStage, accumulate_gradients, ctc_word_sum
from test_own_aligned_assistant_v1 import fixture
from test_own_alignment_checkpoint_v1 import assert_nested_equal

SOURCES = dict(synthetic_test_source='c'*64)


def model():
    result = OwnAlignedAssistantV3(fixture(), rank=2, reward_hidden=8, word_temporal=4).eval()
    result.aligned.gradient_checkpointing = True
    return result


def synthetic_words():
    # Small graph-only fixture. The separate full-model probe uses the real
    # admitted reader, original448px inputs and complete publisher transcripts.
    data = object.__new__(SharedWordData); data.role = 'train'; data.purpose = 'optimization'
    data.rows = [dict(response=t, byte_family='synthetic_fixture',
                      native=dict(length=5+i, target_start=3))
                 for i,t in enumerate(['aa', 'é', 'aba', '\x00a', 'aaa'])]
    def batch(positions, device='cpu'):
        size = max(data.rows[p]['native']['length'] for p in positions)
        ids = torch.zeros(len(positions), size, dtype=torch.long)
        labels = torch.full_like(ids, -100); mask = torch.zeros_like(ids, dtype=torch.bool)
        for i,p in enumerate(positions):
            n = data.rows[p]['native']['length']; ids[i,:n] = torch.tensor([1]+[4+p]*(n-2)+[2])
            labels[i,3:n] = ids[i,3:n]; mask[i,:n] = True
        pixels = torch.stack([torch.linspace(-1, 1, 3*32*32).reshape(3,32,32)*(1-.08*p) for p in positions])
        words = torch.stack([torch.linspace(-1, 1, 3*8*32).reshape(3,8,32)*(1-.08*p) for p in positions])
        return dict(input_ids=ids.to(device), labels=labels.to(device), attention_mask=mask.to(device),
                    pixel_values=pixels.to(device), word_pixels=words.to(device),
                    texts=[data.rows[p]['response'] for p in positions])
    data.batch = batch
    return data


def options(data, **changes):
    result = dict(seed=8613, epochs=2, steps=6, microbatch=1, accumulation=2,
                  learning_rate=.003, betas=[.9,.95], weight_decay=.01, gradient_clip=1.,
                  ctc_weight=.7, native_CE_weight=.4)
    result.update(changes)
    result['steps'] = result['epochs'] * math.ceil(len(data)/(result['microbatch']*result['accumulation']))
    return result


def stage(m, data, opts=None):
    opts = opts or options(data)
    optimizer = torch.optim.AdamW(m.extension_parameters('vision','word'), lr=opts['learning_rate'],
                                 betas=tuple(opts['betas']), weight_decay=opts['weight_decay'], foreach=False)
    return SharedWordStage(m, optimizer, data, opts, SOURCES)


def test_whole_UTF8_CTC_and_native_CE_microbatch_gradients_match_independent_joint_oracle():
    m = model(); oracle = copy.deepcopy(m); data = synthetic_words()
    m.aligned.train(); m.original.eval(); oracle.aligned.train(); oracle.original.eval()
    batches = [data.batch([0]), data.batch([1,2])]
    record = accumulate_gradients(m, batches, ctc_weight=.7, native_CE_weight=.4)
    # Independent whole-batch formula: no production loss/normalizer helper.
    full = data.batch([0,1,2]); logits = oracle.word_logits(full['word_pixels'])
    raw = [list(t.encode('utf-8')) for t in full['texts']]
    lengths = torch.tensor([len(r) for r in raw])
    ctc = F.ctc_loss(logits.log_softmax(-1).transpose(0,1), torch.tensor(sum(raw,[])),
                     torch.full((3,), logits.shape[1], dtype=torch.long), lengths,
                     blank=256, reduction='none', zero_infinity=False)
    ctc = (ctc/lengths).mean()
    native = oracle.aligned(full['input_ids'], full['pixel_values'], attention_mask=full['attention_mask'])['logits']
    ce = F.cross_entropy(native[:,:-1].reshape(-1,native.shape[-1]), full['labels'][:,1:].reshape(-1), ignore_index=-100)
    joint = .7*ctc + .4*ce; joint.backward()
    assert record['summed_objective'] == pytest.approx(float(joint.detach()), rel=2e-6)
    assert record['words'] == 3 and record['UTF8_byte_targets'] == 7
    assert record['response_EOS_targets'] == int((full['labels'][:,1:]!=-100).sum())
    for a,b in zip(m.extension_parameters('vision','word'), oracle.extension_parameters('vision','word')):
        assert a.grad is not None and bool(torch.isfinite(a.grad).all())
        # FP32 CTC recurrences and GRU kernels differ with batch shape. The
        # observed worst per-tensor norm difference is2.30e-5; a normalized
        # bound checks the full gradient without an unstable near-zero ratio.
        difference = a.grad-b.grad
        assert float(difference.norm()) <= 3e-5*float(b.grad.norm())+5e-7
        assert float(difference.abs().max()) <= 6e-5+3e-5*float(b.grad.abs().max())
    selected = {id(p) for p in m.extension_parameters('vision','word')}
    assert all(p.grad is None for p in m.parameters() if id(p) not in selected)
    assert any(p.grad is not None for p in oracle.extension_parameters('language'))
    with pytest.raises(ValueError, match='fit'):
        ctc_word_sum(torch.zeros(1,2,257), ['aaa'])
    with pytest.raises(ValueError, match='finite'):
        ctc_word_sum(torch.full((1,5,257),float('nan')), ['aa'])


def test_real_vision_word_Adam_recovery_cold_reload_and_complete_unrepeated_partial_batches():
    m = model(); data = synthetic_words(); learner = stage(m, data)
    selected = {id(p) for p in learner.parameters}
    fixed = [(p,p.detach().clone(),p._version,p.data_ptr(),p.requires_grad)
             for p in m.parameters() if id(p) not in selected]
    ids = torch.tensor([[1,4,6,2]])
    with torch.no_grad():
        cache = m.original.language.code.prefill(ids)['cache']
        cached = m.original.language.code.decode(torch.tensor([[7]]),cache)['logits'].clone()
        text = m(ids)['logits'].clone()
    first = learner.update(); saved = learner.state(); second = learner.update()
    weights = m.extension_state(); adam = copy.deepcopy(learner.optimizer.state_dict()); rng = torch.get_rng_state().clone()
    learner.resume(saved); assert learner.update() == second
    assert_nested_equal(weights,m.extension_state()); assert_nested_equal(adam,learner.optimizer.state_dict())
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(p,t) and (p._version,p.data_ptr(),p.requires_grad)==(v,ptr,flag)
               for p,t,v,ptr,flag in fixed)
    assert len(learner.optimizer.state) == len(learner.parameters)
    assert first['words'] == 2 and len(learner.positions(2)) == 1 and len(learner.positions(2)[0]) == 1
    for epoch in range(2):
        draws = [p for step in range(epoch*3,(epoch+1)*3) for group in learner.positions(step) for p in group]
        assert len(draws) == len(set(draws)) == len(data) and sorted(draws)==list(range(len(data)))
    assert learner.cursor(6)['actual_draws'] == 10 and learner.cursor(6)['repeated_rounding_sources'] == 0
    m.eval()
    with torch.no_grad():
        assert torch.equal(text,m(ids)['logits'])
        assert torch.equal(cached,m.original.language.code.decode(torch.tensor([[7]]),cache)['logits'])
        batch = data.batch([0]); before = m(**{k:batch[k] for k in ('input_ids','pixel_values','attention_mask')})['logits']
        words = m.word_logits(batch['word_pixels'])
    cold = OwnAlignedAssistantV3(m.original,rank=2,reward_hidden=8,word_temporal=4).eval()
    cold.import_export(m.export(SOURCES),expected_sources=SOURCES)
    with torch.no_grad():
        assert torch.equal(before,cold(**{k:batch[k] for k in ('input_ids','pixel_values','attention_mask')})['logits'])
        assert torch.equal(words,cold.word_logits(batch['word_pixels']))


def test_corrupt_source_normalizer_unused_language_and_Adam_reject_before_any_mutation():
    m = model(); data = synthetic_words(); learner = stage(m,data); learner.update(); saved = learner.state()
    weights = m.extension_state(); adam = copy.deepcopy(learner.optimizer.state_dict()); rng = torch.get_rng_state().clone()
    def unused_language(s):
        name = next(n for n in s['core']['model']['extensions'] if n.startswith('language_linears.'))
        s['core']['model']['extensions'][name].add_(1.)
    def moment(s):
        next(iter(s['core']['optimizer']['state'].values()))['exp_avg'].fill_(float('nan'))
    for mutate in [lambda s:s['core']['metrics'][0].update(UTF8_byte_targets=999),
                   lambda s:s['core']['sampler_state'].update(actual_draws=999), unused_language, moment,
                   lambda s:s['core']['model']['sources'].update(synthetic_test_source='d'*64)]:
        bad = copy.deepcopy(saved); mutate(bad)
        signature = [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in m.parameters()]
        with pytest.raises(ValueError): learner.resume(bad)
        assert_nested_equal(weights,m.extension_state()); assert_nested_equal(adam,learner.optimizer.state_dict())
        assert torch.equal(rng,torch.get_rng_state())
        assert signature == [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in m.parameters()]
    data.role = 'valid'
    with pytest.raises(ValueError,match='train-only'): stage(m,data)
