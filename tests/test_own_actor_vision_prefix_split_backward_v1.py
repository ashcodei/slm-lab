from contextlib import nullcontext
import copy
import io

import pytest
import torch

from test_own_actor_vision_prefix_reuse_v1 import actor,data
from slm_lab.own_actor_vision_alignment_v1 import OwnActorVisionAlignment
from slm_lab.own_actor_vision_prefix_split_backward_v1 import OwnActorVisionPrefixSplitBackward


@pytest.mark.parametrize('bf16',[False,True])
@pytest.mark.parametrize('checkpointing',[False,True])
def test_separate_backwards_keep_original_strict_update_tolerances_without_combining_prefix_gradients(bf16,checkpointing):
    baseline = actor();candidate = copy.deepcopy(baseline)
    views = [OwnActorVisionAlignment(baseline),OwnActorVisionPrefixSplitBackward(candidate)]
    batch,ids,mask,positive = data(); results = []; gradients = []; calls = []
    originals = [{n:p.detach().clone() for n,p in v.actor.named_parameters()} for v in views]
    amp = (lambda:torch.autocast('cpu',dtype=torch.bfloat16)) if bf16 else nullcontext
    for view in views:
        view.train();view.native.gradient_checkpointing = checkpointing;view.native.vision.train()
        seen = [];handle = view.native.vision.register_forward_hook(lambda *args:seen.append(1))
        selected = [p for _,p in view.named_selected()]; selected_ids = {id(p) for p in selected}
        opt = torch.optim.AdamW(selected,lr=1e-3,weight_decay=.01,foreach=False)
        with amp(): objective = view.objective(batch,ids,mask,positive)
        handle.remove();calls.append(len(seen));view.backward(objective)
        gradients.append({n:p.grad.clone() for n,p in view.named_selected()})
        results.append({k:float(objective[k].detach()) for k in ['loss','caption_CE','matching_loss']})
        assert all(p.grad is None for p in view.parameters() if id(p) not in selected_ids)
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in selected)
        torch.nn.utils.clip_grad_norm_(selected,1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True)
        assert all(torch.equal(originals[len(results)-1][n],p) for n,p in view.actor.named_parameters()
                   if not n.startswith(('base.network.vision.','base.heads.pointing.')))
        assert len(opt.state) == len(selected) and all(float(s['step']) == 1 for s in opt.state.values())
    assert calls == [2,1] and results[0] == results[1]
    for name in gradients[0]:
        a,b = gradients[0][name],gradients[1][name]
        torch.testing.assert_close(a,b,rtol=.03 if bf16 else 3e-5,atol=2e-4 if bf16 else 2e-7)
        assert not bool((a*b < 0).any())
        if not bf16: assert torch.equal(a,b)
    for (name,a),(other,b) in zip(views[0].named_selected(),views[1].named_selected()):
        assert name == other
        torch.testing.assert_close(a,b,rtol=.02 if bf16 else 3e-5,atol=3e-5 if bf16 else 2e-7)
        if not bf16: assert torch.equal(a,b)


@pytest.mark.parametrize('bf16',[False,True])
def test_split_backward_actual_next_update_recovers_exactly_with_Adam_and_changed_pixels(bf16):
    model = actor();view = OwnActorVisionPrefixSplitBackward(model).train()
    view.native.gradient_checkpointing = True;view.native.vision.train()
    opt = torch.optim.AdamW([p for _,p in view.named_selected()],lr=1e-3,weight_decay=.01,foreach=False)
    batch,ids,mask,positive = data()
    amp = (lambda:torch.autocast('cpu',dtype=torch.bfloat16)) if bf16 else nullcontext
    def step(current,optimizer,pixels):
        inputs = {**batch,'pixel_values':pixels}
        with amp(): objective = current.objective(inputs,ids,mask,positive)
        current.backward(objective)
        torch.nn.utils.clip_grad_norm_([p for _,p in current.named_selected()],1.,error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        return float(objective['loss'].detach())
    step(view,opt,batch['pixel_values'])
    stream = io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),stream)
    stream.seek(0);saved = torch.load(stream,map_location='cpu',weights_only=True)
    fresh_model = actor();fresh_model.load_state_dict(saved['model'],strict=True)
    fresh = OwnActorVisionPrefixSplitBackward(fresh_model).train()
    fresh.native.gradient_checkpointing = True;fresh.native.vision.train()
    resumed = torch.optim.AdamW([p for _,p in fresh.named_selected()],lr=1e-3,weight_decay=.01,foreach=False)
    resumed.load_state_dict(saved['optimizer'])
    changed = batch['pixel_values']*.8+.17
    assert step(view,opt,changed) == step(fresh,resumed,changed)
    for name,p in model.state_dict().items(): assert torch.equal(p,fresh_model.state_dict()[name])
    for a,b in zip(opt.state.values(),resumed.state.values()):
        assert a.keys() == b.keys()
        for name in a: assert torch.equal(a[name],b[name])
    fresh.assert_shared()


def test_split_backward_rejects_stale_gradients_and_missing_original_coefficient():
    view = OwnActorVisionPrefixSplitBackward(actor());batch,ids,mask,positive = data()
    objective = view.objective(batch,ids,mask,positive)
    broken = {k:v for k,v in objective.items() if k != 'matching_weight'}
    with pytest.raises(ValueError,match='coefficient'):view.backward(broken)
    p = view.named_selected()[0][1];p.grad = torch.zeros_like(p)
    with pytest.raises(ValueError,match='start clear'):view.backward(objective)
