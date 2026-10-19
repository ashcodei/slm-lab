import torch
import pytest
from slm_lab.model import ModelConfig,TinyMultimodalLM


def tiny():
    return TinyMultimodalLM(ModelConfig(vocab_size=260,hidden_size=32,intermediate_size=64,
        num_layers=2,num_heads=4,num_kv_heads=2,max_text_length=64,vision_hidden_size=16,
        vision_intermediate_size=32,vision_layers=1,vision_heads=2,image_size=16,
        patch_size=4,pool_size=2)).eval()


@pytest.mark.parametrize('with_image',[False,True])
def test_future_text_never_changes_past_logits(with_image):
    torch.manual_seed(1);model=tiny()
    ids=torch.randint(0,260,(1,12));altered=ids.clone();altered[:,8:]=3
    image=torch.randn(1,3,16,16) if with_image else None
    a,b=model(ids,image)['logits'],model(altered,image)['logits']
    torch.testing.assert_close(a[:,:8],b[:,:8],rtol=0,atol=0)


def test_image_changes_prediction_and_receives_gradients():
    model=tiny().train();ids=torch.randint(0,260,(1,10))
    image=torch.randn(1,3,16,16)
    first=model(ids,image,labels=ids)
    second=model(ids,torch.zeros_like(image))
    assert not torch.allclose(first['logits'],second['logits'])
    first['loss'].backward()
    assert model.vision.patch.weight.grad.norm()>0
    assert model.embedding.weight.grad.norm()>0


def test_masked_padding_does_not_change_valid_logits():
    model=tiny();ids=torch.randint(0,260,(2,12));mask=torch.ones_like(ids)
    mask[0,6:]=0;altered=ids.clone();altered[0,6:]=9
    a=model(ids,attention_mask=mask)['logits']
    b=model(altered,attention_mask=mask)['logits']
    torch.testing.assert_close(a[0,:6],b[0,:6])


def test_tied_head_and_production_parameter_limit():
    with torch.device('meta'):
        model=TinyMultimodalLM(ModelConfig())
    assert model.lm_head.weight is model.embedding.weight
    assert model.parameter_counts()['total']<1_000_000_000


def test_checkpointed_gradients_match():
    torch.manual_seed(7);a=tiny().train();b=tiny().train();b.load_state_dict(a.state_dict())
    b.gradient_checkpointing=True
    ids=torch.randint(0,260,(1,10));pixels=torch.randn(1,3,16,16)
    a(ids,pixels,labels=ids)['loss'].backward();b(ids,pixels,labels=ids)['loss'].backward()
    for pa,pb in zip(a.parameters(),b.parameters()):
        if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad)


def test_generation_restores_training_mode():
    model=tiny().train();ids=torch.tensor([[1,5]])
    result=model.generate(ids,max_new_tokens=3)
    assert result.shape==(1,5)
    assert model.training


def test_empty_supervision_fails():
    with pytest.raises(ValueError,match='unmasked'):
        tiny()(torch.tensor([[1,2]]),labels=torch.tensor([[-100,-100]]))
