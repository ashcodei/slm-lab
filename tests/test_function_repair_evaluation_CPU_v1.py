import pytest
import torch

from slm_lab import code_quality,evaluation_provenance
from slm_lab.function_repair_evaluation_CPU_v1 import native_actions,legacy_runtime
from test_own_dense_actor_v1 import actor


def test_observer_retains_actual_text_and_positional_image_outputs_and_restores_generator():
    torch.set_num_threads(1);model=actor().policy.eval();ids=torch.tensor([[1,4,7]])
    pixels=torch.linspace(-1,1,3*16*16).reshape(1,3,16,16);events=[];original=model.generate
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        for image in (None,pixels):
            expected=original(ids,image,max_new_tokens=3,eos_token_id=2,use_cache=True)
            with native_actions(model,events.append):actual=model.generate(ids,image,max_new_tokens=3,eos_token_id=2,use_cache=True)
            assert torch.equal(actual,expected) and events[-1]['original_input_token_ids']==ids[0].tolist()
            assert events[-1]['actual_generated_token_ids']==actual[0,ids.shape[1]:].tolist()
            assert (events[-1]['input_pixels_sha256'] is None)==(image is None)
    assert model.generate==original and not model._repair_observer_active_v1
    with pytest.raises(ValueError,match='BF16'):
        with torch.no_grad(),native_actions(model,events.append):model.generate(ids,max_new_tokens=1)
    assert model.generate==original


def test_loader_and_sealer_bind_actual_CPU_BF16_and_preserve_source_checks_and_thread_setting(tmp_path):
    torch.set_num_threads(1);model=actor().policy.eval();checkpoint=tmp_path/'actor.pt';checkpoint.write_bytes(b'unit-own-model')
    old_loader=code_quality.load_checkpoint;old_seal=evaluation_provenance.seal;old_threads=torch.set_num_threads
    with legacy_runtime(model,checkpoint,{'unit':True}):
        assert code_quality.load_checkpoint(checkpoint,'cpu',False)==(model,{'unit':True})
        with pytest.raises(ValueError):code_quality.load_checkpoint(checkpoint,'cuda',False)
        torch.set_num_threads(2);assert torch.get_num_threads()==1
        report={'decoding':'legacy assumed FP32'};proof=evaluation_provenance.capture()
        code_quality.seal(report,proof,device='cpu')
        assert report['inference_precision']=='bf16' and report['decoding']=='greedy, cached CPU BF16'
        bad={**proof,'python':'changed-runtime'}
        with pytest.raises(ValueError):evaluation_provenance.seal({},bad,device='cpu')
    assert code_quality.load_checkpoint==old_loader and evaluation_provenance.seal==old_seal and torch.set_num_threads==old_threads
