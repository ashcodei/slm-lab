from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from slm_lab.code_target_regions_v1 import regions,weighted_CE


class ByteOffsetsTokenizer:
    def encode(self,text,**kwargs):
        ids=[];offsets=[]
        for i,char in enumerate(text):
            for byte in char.encode():ids.append(byte+4);offsets.append((i,i+1))
        return SimpleNamespace(ids=ids,offsets=offsets)
    def decode(self,ids,**kwargs):return bytes(i-4 for i in ids).decode()


def test_UTF8_AST_columns_do_not_move_docstrings_or_executable_string_literals():
    source='def f(π): "é"; return "é"\n';actual=regions(source,ByteOffsetsTokenizer())
    encoded=ByteOffsetsTokenizer().encode(source)
    first=source.index('é');second=source.index('é',first+1)
    assert all(actual['positive_target_weights'][i]==1 for i,span in enumerate(encoded.offsets) if span==(first,first+1))
    assert all(actual['positive_target_weights'][i]==4 for i,span in enumerate(encoded.offsets) if span==(second,second+1))
    assert actual['positive_target_weights'][-1]==1 and min(actual['positive_target_weights'])>=1


def test_CRLF_comments_and_full_original_response_are_preserved_without_target_clipping():
    source='def f(x):\r\n    """Public examples."""\r\n    # keep this notice\r\n    return x + 1\r\n'
    tokenizer=ByteOffsetsTokenizer();actual=regions(source,tokenizer)
    assert tokenizer.decode(actual['whole_original_response_EOS_ids'][:-1])==source
    assert actual['response_EOS_targets']==len(source.encode())+1
    assert actual['weighted_target_denominator']>actual['response_EOS_targets']


def test_actual_shifted_weighted_CE_gradients_equal_explicit_full_target_logprob_objective():
    logits=torch.linspace(-2,2,2*6*12).reshape(2,6,12).requires_grad_()
    labels=torch.tensor([[-100,-100,4,5,6,2],[-100,7,8,9,2,-100]])
    weights=torch.tensor([[0.,0.,4.,1.,3.,1.],[0.,4.,2.,4.,1.,0.]])
    numerator,denominator=weighted_CE(logits,labels,weights);(numerator/denominator).backward();actual=logits.grad.clone()
    fresh=logits.detach().clone().requires_grad_();logp=F.log_softmax(fresh[:,:-1],-1);target=labels[:,1:];mask=target!=-100
    expected=-(logp.gather(-1,target.clamp_min(0).unsqueeze(-1)).squeeze(-1)*weights[:,1:]*mask).sum()/weights[:,1:].sum()
    expected.backward();assert torch.allclose(numerator/denominator,expected) and torch.allclose(actual,fresh.grad,atol=1e-7)
    changed=weights.clone();changed[0,2]=0
    with pytest.raises(ValueError):weighted_CE(logits,labels,changed)
    changed=weights.clone();changed[0,0]=1
    with pytest.raises(ValueError):weighted_CE(logits,labels,changed)
