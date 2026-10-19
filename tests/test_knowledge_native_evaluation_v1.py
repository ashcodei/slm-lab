from types import SimpleNamespace

import pytest
import torch

from slm_lab.knowledge_native_evaluation_v1 import choice_scores


class Tokenizer:
    def encode(self,text):
        return SimpleNamespace(ids={'A':[0],'B':[1],' A':[2,0],' B':[2,1]}[text])


class Model:
    def __init__(self):self.calls=[]
    def decode(self,ids,cache):
        assert cache=='same_original_prompt'
        self.calls.append(ids.item())
        return {'logits':torch.tensor([[[3.,0.,-2.]]]),'cache':'new_candidate_prefix'}


def test_native_and_spaced_candidates_use_their_actual_tokens_and_independent_prompt_cache():
    initial={'logits':torch.tensor([[[-1.,1.,4.]]]),'cache':'same_original_prompt'}
    model=Model()
    direct=choice_scores(model,Tokenizer(),initial,2,'cpu','')
    spaced=choice_scores(model,Tokenizer(),initial,2,'cpu',' ')
    assert direct[1]>direct[0] and spaced[0]>spaced[1]
    assert model.calls==[2,2] and initial['cache']=='same_original_prompt'
    expected=initial['logits'][0,-1].log_softmax(-1)[2]+torch.tensor([3.,0.,-2.]).log_softmax(-1)[0]
    assert spaced[0]==pytest.approx(float(expected))


def test_an_empty_candidate_cannot_receive_a_spurious_zero_logprob_advantage():
    empty=SimpleNamespace(encode=lambda text:SimpleNamespace(ids=[]))
    with pytest.raises(ValueError,match='actual tokens'):
        choice_scores(Model(),empty,{'logits':torch.zeros(1,1,3)},1,'cpu','')
