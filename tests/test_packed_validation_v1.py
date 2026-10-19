import math

import pytest
import torch

from slm_lab.packed_validation_v1 import evaluate


def test_original_valid_next_token_loss_is_complete_and_cannot_return_train_sources():
    torch.set_num_threads(1)
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.zeros(10))
        def forward(self,input_ids,labels,attention_mask):
            assert torch.equal(input_ids,labels) and bool(attention_mask.all())
            logits=self.bias.expand(input_ids.shape[0],input_ids.shape[1]-1,10)
            return dict(loss=torch.nn.functional.cross_entropy(logits.reshape(-1,10),labels[:,1:].reshape(-1)))
    class Packed:
        families=['fixture'];manifest=dict(maximum_length=4096,packed_manifest_sha256='a'*64)
        split='valid'
        def size(self,family,split):assert split=='valid';return 3
        def row(self,family,position,split):
            assert split=='valid';return dict(tokens=[1,3,4,2][:2+position],split=self.split,source_split=self.split)
    model=Model();packed=Packed();before=model.bias.detach().clone();RNG=torch.get_rng_state().clone()
    report=evaluate(model,packed)
    assert report['evaluated_windows']==3 and report['complete_causal_targets']==6
    assert report['token_mean_NLL']==pytest.approx(math.log(10))
    assert model.bias.grad is None and torch.equal(model.bias,before) and torch.equal(torch.get_rng_state(),RNG)
    packed.split='train'
    with pytest.raises(ValueError,match='held source'):evaluate(model,packed)
