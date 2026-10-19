import pytest
import torch
from test_model import tiny


@pytest.mark.parametrize('with_image',[False,True])
def test_cached_chunk_and_single_token_logits_match_full_forward(with_image):
    torch.manual_seed(123);model=tiny()
    ids=torch.randint(0,260,(2,15));pixels=torch.randn(2,3,16,16) if with_image else None
    with torch.no_grad():
        full=model(ids,pixels)['logits'];state=model.prefill(ids[:,:8],pixels)
        torch.testing.assert_close(state['logits'][:,0],full[:,7],rtol=2e-5,atol=2e-6)
        state=model.decode(ids[:,8:12],state['cache'])
        torch.testing.assert_close(state['logits'],full[:,8:12],rtol=2e-5,atol=2e-6)
        for n in range(12,15):
            state=model.decode(ids[:,n:n+1],state['cache'])
            torch.testing.assert_close(state['logits'][:,0],full[:,n],rtol=2e-5,atol=2e-6)
    assert state['cache'].text_length==15
    assert state['cache'].image_tokens==(model.config.image_tokens if with_image else 0)
    assert all(pair[0].shape[1]==model.config.num_kv_heads for pair in state['cache'].layers)
    assert all(not t.requires_grad for pair in state['cache'].layers for t in pair)


@pytest.mark.parametrize('with_image',[False,True])
def test_cached_greedy_generation_matches_recomputation_and_encodes_image_once(with_image):
    torch.manual_seed(71);model=tiny();ids=torch.randint(0,260,(1,12))
    pixels=torch.randn(1,3,16,16) if with_image else None;calls=[]
    handle=model.vision.register_forward_hook(lambda *args:calls.append(1))
    a=model.generate(ids,pixels,max_new_tokens=8,use_cache=True)
    assert len(calls)==int(with_image)
    calls.clear();b=model.generate(ids,pixels,max_new_tokens=8,use_cache=False)
    assert len(calls)==8*int(with_image)
    torch.testing.assert_close(a,b,rtol=0,atol=0);handle.remove()


def test_cache_ownership_lengths_and_branches_are_checked():
    model=tiny();ids=torch.tensor([[1,5,8]])
    state=model.prefill(ids);cache=state['cache']
    with pytest.raises(ValueError,match='model instance'):tiny().decode(ids[:,:1],cache)
    with pytest.raises(ValueError,match='Cache batch'):model.decode(ids[:,:1].expand(2,-1),cache)
    with pytest.raises(ValueError,match='max_text_length'):model.decode(ids.expand(1,3).repeat(1,22),cache)
    first=model.decode(torch.tensor([[7]]),cache);model.decode(torch.tensor([[9]]),cache)
    repeated=model.decode(torch.tensor([[7]]),cache)
    torch.testing.assert_close(first['logits'],repeated['logits'],rtol=0,atol=0)
    assert cache.text_length==3


@pytest.mark.parametrize('with_image',[False,True])
def test_sampling_returns_the_probabilities_and_values_used_at_generation(with_image):
    from unittest.mock import patch
    torch.manual_seed(95);model=tiny();model.value_head.weight.data.normal_(std=.03)
    ids=torch.tensor([[1,4,7,9]]);pixels=torch.randn(1,3,16,16) if with_image else None
    states=[];prefill=model.prefill;decode=model.decode
    def traced_prefill(*args,**kwargs):
        out=prefill(*args,**kwargs);states.append(out);return out
    def traced_decode(*args,**kwargs):
        out=decode(*args,**kwargs);states.append(out);return out
    with patch.object(model,'prefill',side_effect=traced_prefill),patch.object(model,'decode',side_effect=traced_decode):
        result=model.generate(ids,pixels,max_new_tokens=6,temperature=.8,return_rollout=True)
    assert result['logps'].shape==(1,6) and result['response_mask'].all()
    for n,state in enumerate(states):
        token=result['tokens'][0,ids.shape[1]+n]
        expected=(state['logits'][0,-1].float()/.8).log_softmax(-1)[token]
        torch.testing.assert_close(result['logps'][0,n],expected,rtol=0,atol=0)
        torch.testing.assert_close(result['values'][0,n],state['values'][0,-1].float(),rtol=0,atol=0)
    with pytest.raises(ValueError,match='positive temperature'):
        model.generate(ids,max_new_tokens=2,return_rollout=True)
