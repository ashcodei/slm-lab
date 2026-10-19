import torch
from slm_lab.objectives import dpo_loss,reward_loss,generalized_advantages,ppo_loss,response_logps


def test_dpo_rewards_correct_preference_relative_to_reference():
    good=torch.tensor([-2.0],requires_grad=True);bad=torch.tensor([-3.0],requires_grad=True)
    zero=torch.tensor([-3.0])
    loss=dpo_loss(good,bad,zero,zero);loss.backward()
    assert good.grad.item()<0<bad.grad.item()
    assert loss<torch.log(torch.tensor(2.0))


def test_equal_policy_and_reference_has_known_dpo_loss():
    a,b=torch.tensor([-4.0]),torch.tensor([-8.0])
    torch.testing.assert_close(dpo_loss(a,b,a,b),torch.log(torch.tensor(2.0)))


def test_reward_model_preference_gradient():
    a,b=torch.tensor([0.0],requires_grad=True),torch.tensor([0.0],requires_grad=True)
    reward_loss(a,b).backward();assert a.grad<0<b.grad


def test_gae_terminal_reward_returns():
    advantage,returns=generalized_advantages(torch.tensor([0.0,0.0,1.0]),torch.zeros(3),lam=1)
    torch.testing.assert_close(advantage,torch.ones(3));torch.testing.assert_close(returns,torch.ones(3))


def test_ppo_clips_large_positive_policy_update():
    old=torch.zeros(2);new=torch.tensor([2.0,2.0],requires_grad=True)
    zero=torch.zeros(2)
    loss=ppo_loss(new,old,torch.ones(2),zero,zero,zero,value_coef=0)
    torch.testing.assert_close(loss,torch.tensor(-1.2))
    loss.backward();torch.testing.assert_close(new.grad,zero)


def test_dpo_ignores_prompt_log_probabilities():
    ids=torch.tensor([[1,2,3,4]]);labels=torch.tensor([[-100,-100,3,4]])
    logits=torch.zeros(1,4,5);modified=logits.clone();modified[:,0,2]=100
    torch.testing.assert_close(response_logps(logits,ids,labels),response_logps(modified,ids,labels))


def test_contrastive_loss_repeated_captions_are_positive_and_features_receive_gradients():
    from slm_lab.objectives import contrastive_image_text_loss
    images=torch.tensor([[1.,0.],[1.,0.],[0.,1.]],requires_grad=True)
    text=images.detach().clone();ids=torch.tensor([1,1,2])
    good=contrastive_image_text_loss(images,text,ids,.1)
    bad=contrastive_image_text_loss(images,text.roll(1,0),ids,.1)
    assert good<.001 and bad>good
    bad.backward();assert images.grad is not None and images.grad.norm()>0
    identical=contrastive_image_text_loss(images.detach(),text,torch.ones(3,dtype=torch.long),.1)
    torch.testing.assert_close(identical,torch.tensor(0.),rtol=0,atol=1e-7)


def test_visual_retrieval_accepts_repeated_caption_positives():
    from slm_lab.vision_evaluation import retrieval
    features=torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
    good=retrieval(features,features,torch.tensor([1,1,2]))
    assert good['image_to_text_accuracy']==1 and good['text_to_image_accuracy']==1
    wrong=retrieval(features,features.flip(0),torch.tensor([1,1,2]))
    assert wrong['image_to_text_accuracy']<1
