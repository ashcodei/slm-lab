import copy
import torch
from torch.nn import functional as F

from slm_lab.deterministic_positions_v1 import resize,install,check_operator
from test_own_dense_growth_v1 import model as make_model


def test_full_size_original_photo_and_rectangular_word_position_forward_and_gradients_match_cpu_bicubic():
    torch.use_deterministic_algorithms(True)
    rows=check_operator('cpu')
    assert len(rows)==2 and all(r['finite_actual_position_gradient'] for r in rows)
    x=torch.linspace(-1,1,2*7*5*9).reshape(2,7,5,9).requires_grad_()
    y=resize(x,(11,4));oracle=F.interpolate(x,size=(11,4),mode='bicubic',align_corners=False)
    assert torch.allclose(y,oracle,atol=3e-6,rtol=3e-6)


def test_installed_native_vision_and_word_paths_preserve_every_parameter_and_match_original_derivatives():
    old,_=make_model();new=install(copy.deepcopy(old));counts=old.counts()
    assert new.counts()==counts and new.state_dict().keys()==old.state_dict().keys()
    photo=torch.linspace(-1,1,3*32*32).reshape(1,3,32,32)
    word=torch.linspace(-.8,.8,3*16*32).reshape(1,3,16,32)
    a=old.network.vision(photo).sum()+old.word_logits(word).sum()
    b=new.network.vision(photo).sum()+new.word_logits(word).sum()
    assert torch.allclose(a,b,atol=2e-5,rtol=2e-5)
    a.backward(inputs=[old.network.vision.position]);b.backward(inputs=[new.network.vision.position])
    assert torch.allclose(old.network.vision.position.grad,new.network.vision.position.grad,atol=3e-5,rtol=3e-5)
    assert new.network.vision.position.grad.abs().sum()>0


def test_resize_stays_FP32_under_autocast_and_base_resolution_and_trained_cold_export_are_exact():
    x=torch.randn(1,7,5,9);expected=resize(x,(11,4))
    with torch.autocast('cpu',dtype=torch.bfloat16):actual=resize(x,(11,4))
    assert actual.dtype==torch.float32 and torch.equal(expected,actual) and resize(x,(5,9)) is x
    m,_=make_model();m=install(m);cold,_=make_model();cold=install(cold)
    sources=dict(test='a'*64);payload=m.export(sources);cold.import_export(payload,expected_sources=sources)
    photo=torch.linspace(-1,1,3*32*32).reshape(1,3,32,32)
    with torch.no_grad():assert torch.equal(m.network.vision(photo),cold.network.vision(photo))
