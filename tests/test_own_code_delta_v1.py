import copy

import pytest
import torch

from slm_lab.model import ModelConfig,TinyMultimodalLM
from slm_lab.own_code_delta_v1 import OwnCodeDelta,factorize,matrix_names,norm_names
from slm_lab.point_grounding_head_v1 import PointGroundingHead


def own_models():
    torch.set_num_threads(1);torch.manual_seed(4161)
    config=ModelConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=64,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,
        image_size=16,patch_size=4,pool_size=2)
    general=TinyMultimodalLM(config);target=copy.deepcopy(general)
    with torch.no_grad():
        for name,p in target.named_parameters():
            if not name.startswith(('vision.','value_head.')):p.add_(.01*torch.randn_like(p))
    return general,target


def components(general,target,rank):
    before=general.state_dict();after=target.state_dict()
    return dict(matrices={n:factorize(after[n]-before[n],rank) for n in matrix_names(general.config.num_layers)},
        embedding=after['embedding.weight'].clone(),norms={n:after[n].clone() for n in norm_names(general.config.num_layers)})


def test_full_rank_own_delta_matches_target_and_code_learning_preserves_general_and_point_paths():
    general,target=own_models();original={n:t.clone() for n,t in general.state_dict().items()}
    # Match the explicit frozen-general inference contract on both sides.
    # CPU kernels can change their final rounding when requires_grad changes,
    # even with torch.no_grad and byte-identical weights.
    for parameter in general.parameters():parameter.requires_grad_(False)
    ids=torch.tensor([[1,4,7,2],[1,6,5,2]]);pixels=torch.randn(2,3,16,16)
    general.eval();target.eval()
    with torch.no_grad():
        before=general(ids,pixels)['logits'].clone();tokens=general.generate(ids,pixels,max_new_tokens=5,eos_token_id=2)
    candidate=OwnCodeDelta(general,components(general,target,16),reserved_parameters=256);candidate.eval()
    assert torch.equal(before,candidate(ids,pixels,route='general')['logits'])
    assert torch.allclose(target(ids,pixels)['logits'],candidate(ids,pixels,route='code')['logits'],atol=2e-6,rtol=2e-5)
    code=candidate.route('code');state=code.prefill(ids,pixels)
    assert torch.allclose(state['logits'],code(ids,pixels)['logits'][:,-1:],atol=2e-6,rtol=2e-5)
    next_ids=torch.tensor([[8],[9]]);decoded=code.decode(next_ids,state['cache'])
    assert torch.allclose(decoded['logits'],code(torch.cat([ids,next_ids],1),pixels)['logits'][:,-1:],atol=2e-6,rtol=2e-5)
    head=PointGroundingHead(general,rank=8)
    for parameter in head.parameters():parameter.requires_grad_(False)
    head.eval();features=head.query_features(ids).clone();attention=head(ids,pixels)['logits'].clone()
    code.train();code.gradient_checkpointing=True
    optimizer=torch.optim.AdamW(candidate.code_parameters(),lr=.003,foreach=False)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True);loss=code(ids,pixels,labels=ids)['loss'];loss.backward()
        assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in candidate.parts.parameters())
        assert all(p.grad is None for p in general.parameters());optimizer.step()
    assert all(torch.equal(t,general.state_dict()[n]) for n,t in original.items())
    candidate.eval();head.eval()
    assert torch.equal(features,head.query_features(ids)) and torch.equal(attention,head(ids,pixels)['logits'])
    assert torch.equal(tokens,candidate.generate(ids,pixels,max_new_tokens=5,eos_token_id=2,route='general'))
    assert candidate.counts()['total_unique']==sum(p.numel() for p in code.parameters())
    assert candidate.counts()['total_unique']==sum(p.numel() for p in general.parameters())+sum(p.numel() for p in candidate.parts.parameters())


def test_malformed_or_over_budget_state_rejects_and_caches_cannot_cross_routes_or_weight_updates():
    general,target=own_models();parts=components(general,target,2)
    original_flags=[p.requires_grad for p in general.parameters()]
    broken=copy.deepcopy(parts);broken['matrices'][matrix_names(2)[0]]['down'][0,0]=float('nan')
    with pytest.raises(ValueError,match='finite shape-matched'):OwnCodeDelta(general,broken,reserved_parameters=0)
    assert original_flags==[p.requires_grad for p in general.parameters()]
    with pytest.raises(ValueError,match='under1B'):OwnCodeDelta(general,parts,parameter_limit=1,reserved_parameters=0)
    candidate=OwnCodeDelta(general,parts,reserved_parameters=0);candidate.eval();ids=torch.tensor([[1,4,2]])
    g=general.prefill(ids)['cache'];c=candidate.code.prefill(ids)['cache']
    with pytest.raises(ValueError):candidate.code.decode(torch.tensor([[5]]),g)
    with pytest.raises(ValueError):general.decode(torch.tensor([[5]]),c)
    with torch.no_grad():next(candidate.parts.parameters()).add_(.001)
    with pytest.raises(ValueError,match='unchanged actual parameters'):candidate.code.decode(torch.tensor([[5]]),c)
    with pytest.raises(ValueError,match='Explicit general or code route'):candidate.route('automatic')
