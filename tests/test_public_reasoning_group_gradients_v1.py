"""Source/termination/real own-neural gradient and recovery controls.

Neural models here use a small explicit configuration to test mechanics. The
two-action fixture has deliberately initialized embeddings, never a trained
production checkpoint. No result is full-size capacity or model quality.
"""
import copy
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from slm_lab.CPU_frozen_cast_cache_v1 import CPUFrozenCastCache
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference
from slm_lab.model import ModelConfig
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_dense_actor_checkpoint_v1 import make_state,restore_state
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
from slm_lab.public_reasoning_group_gradients_v1 import group_gradients,warmup_gradients
from slm_lab.public_reasoning_train_data_v1 import PublicReasoningTrainData
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest


@pytest.fixture(scope='module')
def data():
    return PublicReasoningTrainData('data/public-reasoning-curriculum-v1',
        'reports/public-reasoning-curriculum-v1-readback-admission/complete.json')


def model(fixture=None):
    config=ModelConfig(vocab_size=32768,hidden_size=8,intermediate_size=16,num_layers=1,
        num_heads=2,num_kv_heads=1,max_text_length=4096,vision_hidden_size=8,vision_intermediate_size=16,
        vision_layers=1,vision_heads=2,image_size=16,patch_size=4,pool_size=2)
    base=OwnDenseAssistant(new_backbone(config,982113),pointing_rank=4,reward_hidden=4,
        word_temporal=4,router_hidden=4,detector_hidden=4,word_expansion=1)
    if fixture is not None:
        last_prefix,answer_token,only_EOS=fixture;network=base.network
        with torch.no_grad():
            network.embedding.weight.zero_()
            network.embedding.weight[last_prefix,0]=1.
            network.embedding.weight[answer_token,0]=-100. if only_EOS else 100.
            network.embedding.weight[answer_token,1]=1.
            network.embedding.weight[2,0]=100.;network.embedding.weight[2,1]=-1.
            for block in network.blocks:
                for name in ['q_proj','k_proj','v_proj','o_proj']:getattr(block.attn,name).weight.zero_()
                block.attn.o_proj.weight.copy_(torch.eye(8))
                for name in ['gate','up','down']:getattr(block,name).weight.zero_()
    return OwnDenseActor(base,rank=2).eval()


def optimizer(student):
    return torch.optim.AdamW([p for _,p in student.named_selected(('actor',))],lr=1e-4,
        betas=(.9,.95),weight_decay=.01,foreach=False)


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def test_every_actual_admitted_train_source_native_mask_and_positive_reward(data):
    assert len(data)==2568 and {row['split'] for row in data.rows}=={'train'}
    assert len({row['task'] for row in data.rows})==12
    total=0
    for position in data.slots:
        info=data.information(position);ids,labels=data.sequence(position);start=info['original_prompt_positions']
        assert labels==[-100]*start+ids[start:] and ids[-1]==2
        scored=data.score(position,ids[start:],horizon=info['full_training_action_budget'])
        assert scored['reward']==1. and scored['actually_stopped_on_EOS']
        assert scored['response']==data.row(position)['response']
        total+=info['response_EOS_targets']
    assert total==19080
    for bad in [-1,len(data),True]:
        with pytest.raises(ValueError):data.row(bad)


def test_wrong_answers_emitted_special_tokens_horizon_stops_and_arbitrary_text_are_not_repaired_or_executed(data,tmp_path):
    position=next(p for p in data.slots if data.row(p)['task']=='binary_matrix')
    info=data.information(position);ids,_=data.sequence(position);target=ids[info['original_prompt_positions']:]
    budget=info['full_training_action_budget']
    special=data.score(position,[0]+target,horizon=budget)
    assert special['reward']==0. and special['status']=='invalid_emitted_special_token'
    stopped=data.score(position,target[:-1],horizon=len(target)-1)
    assert stopped['canonical_text_correct'] and stopped['reward']==0. and stopped['status']=='horizon_without_EOS'
    sentinel=tmp_path/'never-executed'
    text="__import__('pathlib').Path("+repr(str(sentinel))+").touch()"
    actions=data.tokenizer.encode(text,add_special_tokens=False).ids+[2]
    assert data.score(position,actions,horizon=budget)['reward']==0. and not sentinel.exists()
    for impossible in [[2,2],target[:-1],[]]:
        with pytest.raises(ValueError):data.score(position,impossible,horizon=budget)
    row=data.row(position);original=row['original_entry']['answer']
    try:
        row['original_entry']['answer']='corrupt public target'
        with pytest.raises(ValueError):data.score(position,target,horizon=budget)
    finally:row['original_entry']['answer']=original


@pytest.mark.parametrize('precision',['CPU_FP32','CPU_BF16_cached'])
def test_real_own_neural_whole_target_warmup_exact_next_update_recovery_and_cold_outputs(data,precision):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    student=model();adam=optimizer(student);positions=sorted(data.slots,key=lambda p:data.information(p)['complete_context_tokens'])[:2]
    fixed=tensor_digest(student.base.named_parameters());critic=student.parts['critic'].weight.detach().clone()
    initial=tensor_digest(student.parts.named_parameters());cache=CPUFrozenCastCache() if precision=='CPU_BF16_cached' else None
    if cache:cache.attach(student.policy)
    amp=(lambda:torch.autocast('cpu',dtype=torch.bfloat16)) if cache else nullcontext
    def update(position):
        adam.zero_grad(set_to_none=True);result=warmup_gradients(student,data,[position],amp=amp)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in student.named_selected(('actor',)))
        torch.nn.utils.clip_grad_norm_([p for _,p in student.named_selected(('actor',))],1.,error_if_nonfinite=True)
        adam.step();adam.zero_grad(set_to_none=True);return result
    sources=dict(experiment_sha256='a'*64)
    cursor=dict(precision=precision,complete_source_order=positions,completed=1)
    first=update(positions[0])
    saved=copy.deepcopy(make_state(student,adam,groups=('actor',),sources=sources,completed=1,planned=2,cursor=cursor,metrics=[first],device='cpu'))
    wanted=update(positions[1]);wanted_parts=tensor_digest(student.parts.named_parameters())
    wanted_adam=copy.deepcopy(adam.state_dict());wanted_rng=capture_rng('cpu')
    assert wanted_parts!=initial and torch.equal(student.parts['critic'].weight,critic)
    bad=copy.deepcopy(saved);bad['rng']['torch']=torch.zeros(1,dtype=torch.uint8)
    with pytest.raises(ValueError):restore_state(student,adam,bad,groups=('actor',),expected_sources=sources,planned=2,expected_cursor=cursor,device='cpu')
    assert tensor_digest(student.parts.named_parameters())==wanted_parts and equal(adam.state_dict(),wanted_adam) and equal(capture_rng('cpu'),wanted_rng)
    restore_state(student,adam,saved,groups=('actor',),expected_sources=sources,planned=2,expected_cursor=cursor,device='cpu')
    assert update(positions[1])==wanted and tensor_digest(student.parts.named_parameters())==wanted_parts
    assert equal(adam.state_dict(),wanted_adam) and equal(capture_rng('cpu'),wanted_rng)
    cold=OwnDenseActor(student.base,**student.settings).eval();cold.import_export(student.export(sources),expected_sources=sources)
    if cache:cache.attach(cold.policy)
    student.eval();batch=data.batch(positions[0]);batch.pop('labels')
    with torch.no_grad(),amp():assert torch.equal(student(**batch)['logits'],cold(**batch)['logits'])
    assert tensor_digest(student.base.named_parameters())==fixed and all(p.grad is None for p in student.base.parameters())
    if cache:cache.close()


@pytest.mark.parametrize('only_EOS',[False,True])
def test_actual_full_vocabulary_samples_mixed_reward_policy_gradient_and_all_zero_CE_anchor(data,only_EOS):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    position=next(p for p in data.slots if data.row(p)['response']=='True')
    ids,labels=data.sequence(position);start=next(i for i,v in enumerate(labels) if v!=-100)
    assert len(ids[start:-1])==1
    student=model((ids[start-1],ids[start],only_EOS));reference=FrozenOwnActorReference(student,sources=dict(start_sha256='b'*64))
    fixed=tensor_digest(student.base.named_parameters());adam=optimizer(student)
    def gradients(profile):
        student.zero_grad(set_to_none=True);torch.manual_seed(982218)
        result=group_gradients(student,data,position,reference,profile=profile,group_size=8,CPU_mechanics_horizon=2)
        return result,[p.grad.detach().clone() for _,p in student.named_selected(('actor',))]
    control,control_grad=gradients('anchor_CE');rl,rl_grad=gradients('anchor_RL')
    assert control['trajectories']==rl['trajectories'] and control['global_rewards']==rl['global_rewards']
    assert rl['original_reference_used_regardless_of_rewards'] and rl['anchor']['response_EOS_targets']==len(ids)-start
    assert rl['actual_sampled_action_tokens']==sum(len(t['response_ids']) for t in rl['trajectories'])
    assert max(t['cached_full_max_absolute_difference'] for t in rl['trajectories'])<=.001
    if only_EOS:
        assert rl['global_rewards']==[0.]*8 and rl['global_advantages']==[0.]*8
        assert all(torch.equal(a,b) for a,b in zip(control_grad,rl_grad))
    else:
        assert set(rl['global_rewards'])=={0.,1.}
        assert any(not torch.equal(a,b) for a,b in zip(control_grad,rl_grad))
    before=tensor_digest(student.parts.named_parameters());adam.step();adam.zero_grad(set_to_none=True)
    assert tensor_digest(student.parts.named_parameters())!=before and tensor_digest(student.base.named_parameters())==fixed
    reference.assert_bound(student)
    for trajectory in rl['trajectories']:
        assert data.score(position,trajectory['response_ids'],horizon=2)==trajectory['actual_verifier']


def test_invalid_profiles_and_horizons_fail_before_sampling_or_parameter_changes(data):
    student=model();reference=FrozenOwnActorReference(student,sources=dict(start_sha256='c'*64))
    before=tensor_digest(student.parts.named_parameters());rng=capture_rng('cpu')
    for options in [dict(profile='unknown'),dict(profile='anchor_RL',group_size=1),dict(profile='anchor_RL',CPU_mechanics_horizon=1000)]:
        with pytest.raises(ValueError):group_gradients(student,data,0,reference,**options)
        assert tensor_digest(student.parts.named_parameters())==before and equal(capture_rng('cpu'),rng)


def test_later_source_failure_clears_partial_selected_gradients_before_any_commit(data,monkeypatch):
    student=model();before=tensor_digest(student.parts.named_parameters());original=data.batch
    def fail_second(position,device='cpu'):
        if position==1:raise RuntimeError('injected later source failure')
        return original(position,device)
    with monkeypatch.context() as context:
        context.setattr(data,'batch',fail_second)
        with pytest.raises(RuntimeError,match='injected later source failure'):
            warmup_gradients(student,data,[0,1],amp=nullcontext)
    assert tensor_digest(student.parts.named_parameters())==before
    assert all(parameter.grad is None for parameter in student.parameters())


def test_actual_complete_source_epoch_and_target_budgets_without_reserved_optimization(data):
    before=capture_rng('cpu');warmup=data.complete_schedule()
    assert len(warmup)==483
    for epoch in range(3):
        assert sorted(p for row in warmup[epoch*161:(epoch+1)*161] for p in row)==data.slots
    assert data.budget(warmup)==dict(updates=483,source_presentations=7704,complete_context_tokens=735570,
        response_EOS_targets=57240,full_native_generation_horizon_sum=726336)
    policy=data.complete_schedule(epochs=1,width=1)
    assert len(policy)==2568 and sorted(row[0] for row in policy)==data.slots
    assert data.budget(policy)['response_EOS_targets']==19080
    assert data.identity(warmup)==data.identity(data.complete_schedule())
    assert data.identity(warmup)!=data.identity(data.complete_schedule(seed=982114))
    assert equal(capture_rng('cpu'),before)
