import torch
from test_model import tiny
from slm_lab.objectives import response_logps,normalize_rollout_advantages
from slm_lab.reward_gate import gate


def report(correct=600,count=1000):
    return {'task':'preference','examples':count,'correct':correct,'checkpoint_sha256':'checkpoint',
            'data_sha256':'heldout-data','feedback_sources':['human'],'truncated_sequences':0,'ties':0}


def test_reward_gate_rejects_chance_small_synthetic_and_truncated_validation():
    assert gate(report())['passed']
    assert not gate(report(510))['passed']
    assert not gate(report(90,100))['passed']
    assert not gate({**report(),'feedback_sources':['synthetic']})['passed']
    assert not gate({**report(),'truncated_sequences':1})['passed']


def test_response_only_normalization_matches_dense_definition_and_gradients():
    torch.manual_seed(8);logits=torch.randn(2,9,13,requires_grad=True)
    ids=torch.randint(0,13,(2,9));labels=ids.clone();labels[0,:5]=-100;labels[1,:2]=-100
    mask=labels[:,1:]!=-100
    reference=(logits[:,:-1].log_softmax(-1).gather(-1,ids[:,1:,None]).squeeze(-1)*mask).sum(-1)
    score=response_logps(logits,ids,labels)
    torch.testing.assert_close(score,reference,rtol=1e-6,atol=1e-6)
    first=torch.autograd.grad(score.sum(),logits,retain_graph=True)[0]
    second=torch.autograd.grad(reference.sum(),logits)[0]
    torch.testing.assert_close(first,second,rtol=1e-6,atol=1e-6)
    assert not first[:,:-1][~mask].any()


def test_scalar_reward_and_text_features_do_not_materialize_vocabulary_logits():
    model=tiny();ids=torch.randint(0,260,(2,8));calls=[]
    handle=model.lm_head.register_forward_hook(lambda *args:calls.append(1))
    normal=model(ids,return_hidden=True);assert calls==[1];calls.clear()
    compact=model(ids,return_hidden=True,return_logits=False)
    assert calls==[] and 'logits' not in compact
    torch.testing.assert_close(normal['values'],compact['values'],rtol=0,atol=0)
    torch.testing.assert_close(normal['hidden'],compact['hidden'],rtol=0,atol=0);handle.remove()


def test_ppo_batch_normalization_preserves_cross_response_reward_order():
    rollouts=[{'advantages':torch.tensor([1.,2.])},{'advantages':torch.tensor([9.,10.])}]
    normalize_rollout_advantages(rollouts)
    assert (rollouts[0]['advantages']<0).all() and (rollouts[1]['advantages']>0).all()
    values=torch.cat([r['advantages'] for r in rollouts])
    torch.testing.assert_close(values.mean(),torch.tensor(0.),atol=1e-7,rtol=0)
    torch.testing.assert_close(values.std(unbiased=False),torch.tensor(1.))
