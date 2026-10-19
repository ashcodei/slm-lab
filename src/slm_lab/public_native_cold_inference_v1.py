"""Actual CPU AMP forwards with the original FP32 recurrent word readout."""
import time

import torch


def probe(model,inputs):
    if model.training or set(inputs)!={'text','photo','word'} or any(p.device.type!='cpu' for p in model.parameters()):
        raise ValueError('One complete CPU eval model and all three unchanged native input kinds required')
    rows=[]
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        for name in ['text','photo','word']:
            start=time.monotonic();output=model.word_logits(inputs[name]['word_pixels']) if name=='word' else model(**inputs[name])['logits']
            expected=torch.float32 if name=='word' else torch.bfloat16
            finite=bool(torch.isfinite(output).all())
            if not finite or output.dtype!=expected:
                raise ValueError('Actual '+name+' cold output: finite='+str(finite)+', dtype='+str(output.dtype)+', required='+str(expected))
            rows.append(dict(kind=name,actual_output_shape=list(output.shape),actual_dtype=str(output.dtype),
                outer_CPU_autocast_enabled=torch.is_autocast_enabled('cpu'),
                original_word_recurrent_readout_FP32_preserved=name=='word',seconds=time.monotonic()-start,finite=True))
            del output
    return rows
