import pytest
import torch
import json
from dataclasses import asdict
from PIL import Image
from slm_lab.corpus import file_hash
from slm_lab.rendered_text import ALPHABET,pixel_hash
from slm_lab.model import TinyMultimodalLM
from slm_lab.model import ModelConfig,VisionEncoder
from slm_lab.visual_character_probe import features,fit_ridge,scores,state_digest,evaluate


def test_ridge_uses_only_training_statistics_and_separates_known_features():
    x=torch.tensor([[1.,0.],[.9,.1],[0.,1.],[.1,.9]])
    y=torch.tensor([0,0,1,1]);head=fit_ridge(x,y,2)
    assert scores(head,torch.tensor([[1.,0.],[0.,1.]])).argmax(1).tolist()==[0,1]
    mean=head['mean'].clone();scores(head,torch.tensor([[100000.,-100000.]]))
    torch.testing.assert_close(head['mean'],mean)


def test_frozen_spatial_probe_does_not_change_weights_or_leave_hooks():
    config=ModelConfig(vocab_size=260,hidden_size=32,intermediate_size=64,num_layers=1,num_heads=4,num_kv_heads=2,
                       vision_hidden_size=16,vision_intermediate_size=32,vision_layers=1,vision_heads=2,
                       image_size=16,patch_size=4,pool_size=2)
    model=VisionEncoder(config).eval().requires_grad_(False);before=state_digest(model)
    result=features(model,torch.randn(2,3,16,16))
    assert result.shape==(2,64) and not result.requires_grad
    assert state_digest(model)==before and not model.project._forward_pre_hooks
    assert all(parameter.grad is None for parameter in model.parameters())


def test_nonfinite_features_are_rejected():
    with pytest.raises(ValueError,match='Non-finite'):
        fit_ridge(torch.tensor([[float('nan'),1.],[1.,0.]]),torch.tensor([0,1]),2)


def test_probe_completes_on_a_tiny_frozen_encoder_including_image_controls(tmp_path):
    config=ModelConfig(vocab_size=260,hidden_size=32,intermediate_size=64,num_layers=1,num_heads=4,num_kv_heads=2,
                       vision_hidden_size=16,vision_intermediate_size=32,vision_layers=1,vision_heads=2,
                       image_size=16,patch_size=4,pool_size=2)
    checkpoint=tmp_path/'toy.pt';model=TinyMultimodalLM(config)
    torch.save({'config':asdict(config),'model':model.state_dict()},checkpoint);before=file_hash(checkpoint)
    data=tmp_path/'data';data.mkdir()
    # Artificial color fields exercise orchestration only; these are not glyph
    # recognition evidence or nominal-model training/evaluation examples.
    for split,offset in [('train',0),('valid',2)]:
        rows=[]
        for label,char in enumerate(ALPHABET):
            for i in range(2):
                image=Image.new('RGB',(16,16),(label*4,offset+i,label*3))
                name=f'{split}-{label}-{i}.png';image.save(data/name)
                rows.append({'id':name,'label':label,'character':char,'image':name,
                             'stored_image_sha256':file_hash(data/name),'model_pixel_sha256':pixel_hash(image,16)})
        (data/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (data/'manifest.json').write_text(json.dumps({'configuration':{'image_size':16,'valid_per_class':2},
                                                'files_sha256':{name:file_hash(data/name) for name in ['train.jsonl','valid.jsonl']},
                                                'limitations':['Tiny artificial test fixture only.']}))
    report=evaluate([('toy',checkpoint)],data,tmp_path/'output',device='cpu',random_control=False)
    assert report['train_examples']==124 and report['valid_examples']==124
    assert report['goal_achieved'] is False
    assert len(report['encoders'])==1 and file_hash(checkpoint)==before
    assert (tmp_path/'output/summary.json').exists()
