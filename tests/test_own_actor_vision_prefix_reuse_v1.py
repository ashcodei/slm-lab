from contextlib import nullcontext
import copy

import pytest
import torch

from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_actor_vision_alignment_v1 import OwnActorVisionAlignment,pair_positives
from slm_lab.own_actor_vision_prefix_reuse_v1 import OwnActorVisionPrefixReuse,native_from_visual


def actor():
    torch.set_num_threads(1)
    c = ModelConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=64,vision_hidden_size=32,vision_intermediate_size=64,vision_layers=1,vision_heads=4,
        image_size=32,patch_size=16)
    return OwnDenseActor(OwnDenseAssistant(new_backbone(c,989117),reward_hidden=8,word_temporal=4,
        pointing_rank=8,router_hidden=8,detector_hidden=4,word_expansion=1),rank=2,seed=989217)


def data():
    ids = torch.tensor([[1,5,6,7,8,2],[1,9,10,11,2,0],[1,12,13,14,15,2]])
    mask = ids != 0; labels = ids.clone();labels[:,:2] = -100;labels[~mask] = -100
    generator = torch.Generator().manual_seed(989317)
    pixels = torch.randn(3,3,64,64,generator=generator)
    positive = pair_positives(['a','a','b'],['cat','generic','generic'])
    return dict(input_ids=ids,attention_mask=mask,labels=labels,pixel_values=pixels),ids,mask,positive


@pytest.mark.parametrize('bf16',[False,True])
@pytest.mark.parametrize('checkpointing',[False,True])
def test_reuse_preserves_actual_objectives_every_selected_gradient_and_one_Adam_update(bf16,checkpointing):
    original = actor(); reused = copy.deepcopy(original)
    views = [OwnActorVisionAlignment(original),OwnActorVisionPrefixReuse(reused)]
    batch,ids,mask,positive = data(); results = []; gradients = []; calls = []; optimizers = []
    sources = [{n:p.detach().clone() for n,p in v.actor.named_parameters()} for v in views]
    amp = (lambda:torch.autocast('cpu',dtype=torch.bfloat16)) if bf16 else nullcontext
    for view in views:
        view.train();view.native.gradient_checkpointing = checkpointing
        view.native.vision.train()  # Exercise image and decoder checkpoint recomputation.
        seen = []
        handle = view.native.vision.register_forward_hook(lambda *args:seen.append(1))
        selected = [p for _,p in view.named_selected()]
        optimizer = torch.optim.AdamW(selected,lr=1e-3,weight_decay=.01,foreach=False)
        with amp(): objective = view.objective(batch,ids,mask,positive)
        calls.append(len(seen));handle.remove()
        view.backward(objective)
        results.append({k:float(v.detach()) if isinstance(v,torch.Tensor) else v for k,v in objective.items()})
        gradients.append({n:p.grad.clone() for n,p in view.named_selected()})
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in selected)
        chosen = {id(p) for p in selected}
        assert all(p.grad is None for p in view.parameters() if id(p) not in chosen)
        torch.nn.utils.clip_grad_norm_(selected,1.,error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True);optimizers.append(optimizer)
        assert all(torch.equal(sources[len(results)-1][n],p) for n,p in view.actor.named_parameters()
                   if not n.startswith(('base.network.vision.','base.heads.pointing.')))
    assert calls == [2,1]
    assert results[0] == results[1]
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name],gradients[1][name],rtol=.03 if bf16 else 3e-5,atol=2e-4 if bf16 else 2e-7)
    for (name,left),(other,right) in zip(views[0].named_selected(),views[1].named_selected()):
        assert name == other
        torch.testing.assert_close(left,right,rtol=.02 if bf16 else 3e-5,atol=3e-5 if bf16 else 2e-7)
    assert len(optimizers[0].state) == len(views[0].named_selected()) == len(optimizers[1].state)
    assert all(float(state['step']) == 1 for opt in optimizers for state in opt.state.values())
    assert reused.counts() == original.counts()


def test_explicit_prefix_matches_native_hidden_values_logits_and_masks_and_remains_causal():
    view = OwnActorVisionPrefixReuse(actor()).eval();batch,ids,mask,_ = data()
    with torch.no_grad():
        visual = view.native.vision(batch['pixel_values'])
        actual = native_from_visual(view.native,ids,visual,attention_mask=mask,labels=batch['labels'],return_hidden=True)
        expected = view.native(**batch,return_hidden=True)
        assert actual.keys() == expected.keys()
        for key in actual: assert torch.equal(actual[key],expected[key])
        future = ids.clone();future[:,3] = 23
        changed = native_from_visual(view.native,future,visual,attention_mask=mask)['logits']
        assert torch.equal(changed[:,:3],actual['logits'][:,:3])
        invalid_padded_ids = ids.clone();invalid_padded_ids[1,-1] = 31
        padded = native_from_visual(view.native,invalid_padded_ids,visual,attention_mask=mask)['logits']
        assert torch.equal(padded[1,:-1],actual['logits'][1,:-1])


def test_detached_prefix_missing_EOS_wrong_geometry_and_incomplete_captions_are_rejected():
    view = OwnActorVisionPrefixReuse(actor()).eval();batch,ids,mask,positive = data()
    visual = view.native.vision(batch['pixel_values'])
    with pytest.raises(ValueError,match='autograd'): native_from_visual(view.native,ids,visual.detach())
    with pytest.raises(ValueError): native_from_visual(view.native,ids,visual[:1])
    broken = {**batch,'labels':batch['labels'].clone()};broken['labels'][0,-1] = -100
    with pytest.raises(ValueError,match='EOS'): view.objective(broken,ids,mask,positive)
    broken_mask = mask.clone();broken_mask[0,1] = False
    with pytest.raises(ValueError,match='right-padded'): view.objective(batch,ids,broken_mask,positive)
    with pytest.raises(ValueError): view.objective(batch,ids,mask,torch.ones_like(positive))


def test_prefix_view_introduces_no_parameters_and_has_no_persistent_feature_cache():
    model = actor(); initial = {id(p) for p in model.parameters()}
    view = OwnActorVisionPrefixReuse(model);batch,ids,mask,positive = data()
    before = set(view.state_dict());first = view.objective(batch,ids,mask,positive)
    view.backward(first);model.zero_grad(set_to_none=True)
    changed = {**batch,'pixel_values':batch['pixel_values']+.5}
    second = view.objective(changed,ids,mask,positive)
    assert not torch.equal(first['loss'],second['loss'])
    assert set(view.state_dict()) == before
    assert {id(p) for p in view.parameters()} == initial
    view.assert_shared()
