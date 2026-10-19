"""Real tiny own text/image/word outputs retain their actual intended precision."""
import torch

from slm_lab.deterministic_positions_v1 import install
from slm_lab.public_native_cold_inference_v1 import probe
from test_own_dense_growth_v1 import model as make_model


def test_actual_bf16_text_photo_and_intentionally_fp32_word_readout():
    model,_=make_model();install(model);model.eval();ids=torch.tensor([[1,4,7,2]])
    pixels=torch.linspace(-1,1,3*32*32).reshape(1,3,32,32);word=torch.linspace(-1,1,3*8*32).reshape(1,3,8,32)
    rows=probe(model,dict(text=dict(input_ids=ids),photo=dict(input_ids=ids,pixel_values=pixels),word=dict(word_pixels=word)))
    assert [r['actual_dtype'] for r in rows]==['torch.bfloat16','torch.bfloat16','torch.float32']
    assert all(r['outer_CPU_autocast_enabled'] and r['finite'] for r in rows)
    assert rows[-1]['original_word_recurrent_readout_FP32_preserved']
    assert not torch.cuda.is_initialized()
