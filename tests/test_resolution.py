import json
from pathlib import Path
import pytest
import torch
from test_model import tiny
from test_multimodal_training import setup
from slm_lab.training import train,letterbox_box


def test_high_resolution_uses_same_parameters_and_backpropagates_to_position_grid():
    model=tiny().train();model.gradient_checkpointing=True
    shapes={k:v.shape for k,v in model.state_dict().items()};count=model.parameter_counts()
    ids=torch.randint(0,260,(2,8));pixels=torch.randn(2,3,32,32)
    assert model.vision(pixels).shape==(2,16,32)
    result=model(ids,pixels,labels=ids);result['loss'].backward()
    assert torch.isfinite(result['loss'])
    assert model.vision.position.grad.norm()>0 and model.vision.patch.weight.grad.norm()>0
    assert model.parameter_counts()==count
    assert {k:v.shape for k,v in model.state_dict().items()}==shapes
    restored=tiny();restored.load_state_dict(model.state_dict(),strict=True)


def test_high_resolution_cached_decoding_and_causality_match_full_forward():
    model=tiny();ids=torch.randint(0,260,(2,13));pixels=torch.randn(2,3,32,32)
    with torch.no_grad():
        full=model(ids,pixels)['logits'];changed=ids.clone();changed[:,9:]=2
        torch.testing.assert_close(full[:,:9],model(changed,pixels)['logits'][:,:9],rtol=0,atol=0)
        first=model.prefill(ids[:,:8],pixels)
        assert first['cache'].image_tokens==16 and first['cache'].stored_tokens==24
        torch.testing.assert_close(first['logits'][:,0],full[:,7],rtol=2e-5,atol=2e-6)
        decoded=model.decode(ids[:,8:],first['cache'])
        torch.testing.assert_close(decoded['logits'],full[:,8:],rtol=2e-5,atol=2e-6)


@pytest.mark.parametrize('shape',[(1,3,8,8),(1,3,33,33),(1,3,48,48),(1,3,16,32),(1,4,32,32)])
def test_unsupported_image_shapes_fail_before_patch_attention(shape):
    with pytest.raises(ValueError):tiny().vision(torch.zeros(shape))


def test_image_resolution_is_recorded_and_cannot_change_during_resume(tmp_path):
    original,args=setup(tmp_path);args.image_size=32;args.checkpoint_every=1;args.keep_checkpoints=2
    rows=[{'prompt':'Describe it.','response':'A red square.','image':'image.png'}]*4
    (tmp_path/'data.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));train(args)
    report=json.loads((tmp_path/'trained/training.json').read_text())
    assert report['image_size']==32 and report['visual_tokens']==16
    args.checkpoint=str(tmp_path/'trained/step-00000002.pt');args.resume=True
    args.output=str(tmp_path/'resumed');train(args)
    full=torch.load(tmp_path/'trained/model.pt',weights_only=True)['model']
    resumed=torch.load(tmp_path/'resumed/model.pt',weights_only=True)['model']
    for key,value in full.items():torch.testing.assert_close(value,resumed[key],rtol=0,atol=0)
    args.image_size=16;args.output=str(tmp_path/'bad-resolution')
    with pytest.raises(ValueError,match='same image resolution'):train(args)


def test_reading_boxes_follow_the_actual_letterboxed_image():
    from PIL import Image,ImageDraw,ImageOps
    image=Image.new('RGB',(200,100),'white');ImageDraw.Draw(image).rectangle((20,20,60,40),fill='blue')
    original=[.1,.2,.3,.4]
    for size in [224,448]:
        box=letterbox_box(original,image.size,size)
        assert box==[.1,.35,.3,.45]
        pixels=ImageOps.pad(image,(size,size),method=Image.Resampling.BICUBIC,color=(128,128,128))
        center=(round((box[0]+box[2])*size/2),round((box[1]+box[3])*size/2))
        assert pixels.getpixel(center)==(0,0,255)
    assert letterbox_box([.25,.25,.75,.75],(100,200),224)==[.375,.25,.625,.75]
