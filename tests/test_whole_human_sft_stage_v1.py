import copy
import math

import pytest
import torch

from slm_lab.own_aligned_assistant_v2 import OwnAlignedAssistantV2
from slm_lab.whole_human_sft_stage_v1 import WholeHumanSFTStage, make_optimizer
from test_own_aligned_assistant_v1 import fixture, original_signature
from test_own_alignment_checkpoint_v1 import assert_nested_equal


class Source:
    family,role,purpose='dialogue','train','optimization'
    def __init__(self):
        self.ids=[[1,4,2],[1,5,6,7,2],[1,8,9,10,11,2],[1,12,2],[1,13,14,15,16,17,2]]
        self.rows=[dict(id=str(i),native=dict(response=dict(length=len(ids),target_start=2))) for i,ids in enumerate(self.ids)]
    def __len__(self):return len(self.ids)
    def batch(self,positions,field,device):
        length=max(len(self.ids[p]) for p in positions)
        ids=torch.zeros(len(positions),length,dtype=torch.long);labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids,dtype=torch.bool)
        for i,p in enumerate(positions):
            row=self.ids[p];ids[i,:len(row)]=torch.tensor(row);labels[i,2:len(row)]=torch.tensor(row[2:]);mask[i,:len(row)]=True
        return dict(input_ids=ids.to(device),labels=labels.to(device),attention_mask=mask.to(device))


def setup():
    model=OwnAlignedAssistantV2(fixture(),rank=2,reward_hidden=8,word_temporal=4).eval();data=Source()
    options=dict(seed=71,steps=math.ceil(len(data)/4),microbatch=1,accumulation=4,learning_rate=.001,
                 betas=[.9,.95],weight_decay=.01,gradient_clip=1.)
    optimizer=make_optimizer(model,options)
    stage=WholeHumanSFTStage(model,optimizer,data,options,dict(run_plan_sha256='a'*64,source_manifest_sha256='b'*64),'cpu')
    return stage


def test_whole_source_epoch_rounding_exact_recovery_and_unused_tensor_cache_retention(tmp_path):
    torch.use_deterministic_algorithms(True);stage=setup();model=stage.model
    old=original_signature(model.original)
    selected={id(p) for p in stage.parameters};fixed={n:p.clone() for n,p in model.named_parameters() if id(p) not in selected}
    fixed_versions=[(id(p),p._version,p.data_ptr()) for p in stage.fixed]
    with torch.no_grad():
        cache=model.original.language.code.prefill(torch.tensor([[1,4,7]]))['cache']
        cached=model.original.language.code.decode(torch.tensor([[8]]),cache)['logits'].clone()
    stage.update();snapshot=stage.state();path=tmp_path/'resume.pt';torch.save(snapshot,path)
    expected=stage.update();parts=model.extension_state();Adam=copy.deepcopy(stage.optimizer.state_dict())
    stage.resume(torch.load(path,weights_only=True));assert stage.update()==expected
    assert_nested_equal(parts,model.extension_state());assert_nested_equal(Adam,stage.optimizer.state_dict())
    assert fixed_versions==[(id(p),p._version,p.data_ptr()) for p in stage.fixed]
    assert all(torch.equal(t,dict(model.named_parameters())[n]) for n,t in fixed.items())
    assert old==original_signature(model.original)
    assert len({p for metric in stage.metrics for p in metric['source_positions']})==len(stage.data)
    assert sum(len(metric['source_positions']) for metric in stage.metrics)==8
    model.eval()
    with torch.no_grad():
        assert torch.equal(cached,model.original.language.code.decode(torch.tensor([[8]]),cache)['logits'])


def test_bad_sources_Adam_mask_cursors_or_unused_extension_reject_before_mutation():
    stage=setup();stage.update();state=stage.state();stage.update()
    expected=stage.model.extension_state();Adam=copy.deepcopy(stage.optimizer.state_dict())
    for mutate in [lambda s:s['core']['model']['sources'].update(source_manifest_sha256='c'*64),
                   lambda s:s['core']['sampler_state'].update(completed_draws=5),
                   lambda s:s['core']['optimizer']['state'][0]['exp_avg_sq'].fill_(-1),
                   lambda s:s['core']['model']['extensions']['vision_norms.0.weight'].add_(.1),
                   lambda s:s['core']['metrics'][0].update(response_EOS_targets=0)]:
        bad=copy.deepcopy(state);mutate(bad);rng=torch.get_rng_state().clone();signature=original_signature(stage.model)
        with pytest.raises(ValueError):stage.resume(bad)
        assert signature==original_signature(stage.model) and torch.equal(rng,torch.get_rng_state())
        assert_nested_equal(expected,stage.model.extension_state());assert_nested_equal(Adam,stage.optimizer.state_dict())


def test_token_mean_gradient_matches_full_original_target_sum_before_update():
    stage=setup();reference=copy.deepcopy(stage.model);positions=[p for group in stage.positions(0) for p in group]
    losses=[];counts=[]
    reference.aligned.train()
    for p in positions:
        batch=stage.data.batch([p],'response','cpu');count=int((batch['labels'][:,1:]!=-100).sum())
        losses.append(reference(**batch)['loss']*count);counts.append(count)
    loss=sum(losses)/sum(counts);loss.backward()
    params=reference.extension_parameters('language');torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    optimizer=make_optimizer(reference,stage.options);optimizer.step()
    metric=stage.update()
    assert metric['response_EOS_targets']==sum(counts)
    assert metric['token_mean_nll']==pytest.approx(float(loss.detach()),abs=1e-6)
    for a,b in zip(stage.parameters,params):assert torch.allclose(a,b,atol=1e-7,rtol=1e-6)
