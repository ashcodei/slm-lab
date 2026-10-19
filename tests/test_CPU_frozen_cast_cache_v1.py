import torch
import pytest

from slm_lab.CPU_frozen_cast_cache_v1 import CPUFrozenCastCache
from slm_lab.own_dense_actor_v1 import _FrozenLinear,OwnDenseActor
from test_own_dense_standalone_v1 import fixture


def test_actual_CPU_bfloat_linear_outputs_and_hidden_gradients_match_and_original_weights_remain_frozen():
    original=torch.nn.Linear(64,32,bias=True);layer=_FrozenLinear(original)
    hidden=torch.linspace(-1,1,3*64).reshape(3,64).requires_grad_()
    with torch.autocast('cpu',dtype=torch.bfloat16):expected=layer(hidden)
    expected.square().sum().backward();gradient=hidden.grad.clone();hidden.grad=None
    keys=list(layer.state_dict());count=sum(p.numel() for p in layer.parameters())
    cache=CPUFrozenCastCache();cache.attach(layer)
    with torch.autocast('cpu',dtype=torch.bfloat16):actual=layer(hidden)
    actual.square().sum().backward()
    assert torch.equal(actual,expected) and torch.equal(hidden.grad,gradient)
    assert all(p.grad is None for p in original.parameters()) and keys==list(layer.state_dict())
    assert count==sum(p.numel() for p in layer.parameters()) and cache.inventory()['added_learned_parameters']==0
    with torch.autocast('cpu',dtype=torch.bfloat16):layer(hidden)
    assert cache.inventory()['actual_cache_hits']>=2
    cache.assert_current();cache.close()


def test_changed_source_invalidates_cast_and_FP32_execution_never_uses_cached_precision():
    original=torch.nn.Linear(16,8,bias=False);layer=_FrozenLinear(original);hidden=torch.randn(2,16)
    cache=CPUFrozenCastCache();cache.attach(layer)
    with torch.autocast('cpu',dtype=torch.bfloat16):before=layer(hidden)
    with torch.no_grad():original.weight.add_(1)
    with pytest.raises(ValueError,match='Frozen source changed'):cache.assert_current()
    with torch.autocast('cpu',dtype=torch.bfloat16):after=layer(hidden)
    assert not torch.equal(before,after) and cache.inventory()['actual_conversions']==2
    cache.assert_current();cache.close()
    expected=layer(hidden);cache.attach(layer)
    assert torch.equal(layer(hidden),expected)
    cache.close()


def test_whole_actual_own_actor_gradients_and_exported_parameter_inventory_match(tmp_path):
    base,ids,_,_=fixture(tmp_path);actor=OwnDenseActor(base,rank=2,seed=979211).eval()
    parameters=[p for _,p in actor.named_selected(('actor',))]
    with torch.autocast('cpu',dtype=torch.bfloat16):loss=actor(ids,labels=ids)['loss']
    loss.backward(inputs=parameters);wanted=[p.grad.clone() for p in parameters]
    actor.zero_grad(set_to_none=True);keys=list(actor.state_dict());counts=actor.counts()
    cache=CPUFrozenCastCache();cache.attach(actor.policy)
    with torch.autocast('cpu',dtype=torch.bfloat16):actual=actor(ids,labels=ids)['loss']
    actual.backward(inputs=parameters)
    assert torch.equal(loss,actual) and all(torch.equal(p.grad,g) for p,g in zip(parameters,wanted))
    assert actor.counts()==counts and list(actor.state_dict())==keys
    assert all(p.grad is None for p in base.parameters())
    cache.assert_current();cache.close()
