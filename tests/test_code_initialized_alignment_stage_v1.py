import copy
import math

import pytest
import torch
from torch.nn import functional as F

from slm_lab.code_initialized_alignment_stage_v1 import CodeInitializedAlignmentStage, objectives, source_batch, REFERENCE_ROUTES
from slm_lab.own_aligned_assistant_v3 import OwnAlignedAssistantV3
from test_own_aligned_assistant_v1 import fixture
from test_own_alignment_checkpoint_v1 import assert_nested_equal

SOURCES = dict(actual_source='a'*64)


class Human:
    family, role, purpose = 'dialogue','train','optimization'
    def __len__(self): return 7
    def batch(self,positions,field,device):
        sequences = [[1,4,*([5+p]*(1+p%3)),2] for p in positions]
        n = max(map(len,sequences)); ids = torch.zeros(len(positions),n,dtype=torch.long)
        labels = torch.full_like(ids,-100); mask = torch.zeros_like(ids,dtype=torch.bool)
        for i,seq in enumerate(sequences):
            ids[i,:len(seq)] = torch.tensor(seq); labels[i,2:len(seq)] = torch.tensor(seq[2:]); mask[i,:len(seq)] = True
        return dict(input_ids=ids.to(device),labels=labels.to(device),attention_mask=mask.to(device))


class Generation:
    name, role, purpose = 'generation','train','optimization'
    @property
    def encoder(self): return self
    def __len__(self): return 3
    def batch(self,positions,field,device): return Human().batch([r['packed_code_index'] for r in positions],field,device)


def setup(alpha=.1):
    model = OwnAlignedAssistantV3(fixture(),rank=2,reward_hidden=8,word_temporal=4).eval()
    streams = dict(human=Human(),generation=Generation())
    options = dict(seed=113,steps=3,human_per_update=4,programs_per_update=1,learning_rate=.001,
                   betas=[.9,.95],weight_decay=.01,gradient_clip=1e9,reference_KL_alpha=alpha)
    optimizer = torch.optim.AdamW(model.extension_parameters('language'),lr=.001,betas=(.9,.95),weight_decay=.01,foreach=False)
    return CodeInitializedAlignmentStage(model,optimizer,streams,options,SOURCES)


def test_actual_streamed_unequal_target_CE_and_full_vocabulary_KL_gradients_match_joint_autograd(monkeypatch):
    stage = setup(); expected = copy.deepcopy(stage.model); losses = []
    positions = stage.positions(0)
    for name in stage.streams:
        batches = [source_batch(stage.streams,name,[p],'cpu') for p in positions[name]]
        targets = sum(int((batch['labels'][:,1:]!=-100).sum()) for batch in batches)
        for batch in batches:
            inputs = {k:v for k,v in batch.items() if k!='labels'}
            student = expected.aligned(**inputs)['logits']
            with torch.no_grad(): teacher = expected.route(REFERENCE_ROUTES[name])(**inputs)['logits']
            shifted = batch['labels'][:,1:]; mask = shifted != -100
            logp = student[:,:-1].float()[mask].log_softmax(-1)
            logq = teacher[:,:-1].float()[mask].log_softmax(-1)
            ce = F.nll_loss(logp,shifted[mask],reduction='sum')
            kl = (logq.exp()*(logq-logp)).sum()
            losses.append((ce+.1*kl)/targets)
    sum(losses).backward(); captured = []
    monkeypatch.setattr(stage.optimizer,'step',lambda:captured.extend(p.grad.detach().clone() for p in stage.parameters))
    record = stage.update()
    for actual,parameter in zip(captured,expected.extension_parameters('language')):
        assert torch.allclose(actual,parameter.grad,atol=3e-7,rtol=3e-5)
    assert math.isclose(record['summed_objective'],float(sum(losses).detach()),rel_tol=2e-7)
    assert all(p.grad is None for p in expected.original.parameters())
    assert record['streams']['generation']['token_mean_reference_KL'] == 0


def test_complete_source_exposures_real_code_frozen_update_and_exact_recovery():
    stage = setup(); selected = {id(p) for p in stage.parameters}
    fixed = [(p,p.detach().clone(),p._version,p.data_ptr()) for p in stage.model.parameters() if id(p) not in selected]
    stage.update(); state = stage.state(); expected = stage.update()
    weights = stage.model.extension_state(); adam = copy.deepcopy(stage.optimizer.state_dict())
    stage.resume(state); assert stage.update() == expected
    assert_nested_equal(weights,stage.model.extension_state()); assert_nested_equal(adam,stage.optimizer.state_dict())
    stage.update()
    for name in stage.streams:
        draws = [p for r in stage.metrics for p in r['source_positions'][name]]
        assert set(draws) == set(range(len(stage.streams[name])))
    assert len([p for r in stage.metrics for p in r['source_positions']['human']]) == 12
    assert len([p for r in stage.metrics for p in r['source_positions']['generation']]) == 3
    assert all(torch.equal(p,t) and p._version==version and p.data_ptr()==pointer for p,t,version,pointer in fixed)
    for kind in ('target','reference','unused','moment'):
        bad = copy.deepcopy(state)
        if kind=='target': bad['core']['metrics'][0]['streams']['human']['response_EOS_targets'] += 1
        elif kind=='reference': bad['core']['metrics'][0]['streams']['generation']['reference_route'] = 'general'
        elif kind=='unused': bad['core']['model']['extensions']['critic.weight'].add_(1)
        else: next(iter(bad['core']['optimizer']['state'].values()))['exp_avg'].fill_(float('nan'))
        before = stage.model.extension_state(); rng = torch.get_rng_state().clone()
        with pytest.raises(ValueError): stage.resume(bad)
        assert_nested_equal(before,stage.model.extension_state()); assert torch.equal(rng,torch.get_rng_state())


def test_zero_anchor_control_never_queries_reference_and_held_stream_rejected(monkeypatch):
    stage = setup(alpha=0.)
    monkeypatch.setattr(stage.model,'route',lambda *a,**k:pytest.fail('No reference forward in the zero-KL control'))
    record = stage.update()
    assert all(r['token_mean_reference_KL']==0 for r in record['streams'].values())
    streams = dict(stage.streams); streams['human'] = Human(); streams['human'].role = 'valid'
    with pytest.raises(ValueError,match='train-only'):
        CodeInitializedAlignmentStage(stage.model,stage.optimizer,streams,stage.options,SOURCES)
