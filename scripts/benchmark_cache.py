import json
from pathlib import Path
import time
import torch
from slm_lab.training import load_checkpoint

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
model,meta=load_checkpoint(ROOT/'checkpoints/full-first-pretrain/model.pt','cuda',training_state=False)
model.eval();torch.manual_seed(74)
rows=[]
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    for with_image in [False,True]:
        ids=torch.randint(4,model.config.vocab_size,(1,256),device='cuda')
        pixels=torch.rand(1,3,224,224,device='cuda')*2-1 if with_image else None
        full=model(ids,pixels)['logits'][:,-1];state=model.prefill(ids,pixels)
        prefill_difference=(full-state['logits'][:,0]).abs().max().item()
        continuation=torch.randint(4,model.config.vocab_size,(1,4),device='cuda')
        decoded=model.decode(continuation,state['cache'])['logits']
        reference=model(torch.cat((ids,continuation),1),pixels)['logits'][:,-4:]
        difference=(decoded-reference).abs().max().item()
        # Check the cache's mathematical behavior at full precision. BF16
        # prefill/decode kernel shapes produce different accumulated rounding.
        with torch.autocast('cuda',enabled=False):
            precise_state=model.prefill(ids,pixels)
            precise_decoded=model.decode(continuation,precise_state['cache'])['logits']
            precise_reference=model(torch.cat((ids,continuation),1),pixels)['logits'][:,-4:]
            precise_difference=(precise_decoded-precise_reference).abs().max().item()
            torch.testing.assert_close(precise_decoded,precise_reference,rtol=2e-4,atol=2e-4)
        outputs={};times={}
        for cached in [False,True]:
            model.generate(ids,pixels,max_new_tokens=4,use_cache=cached)
            durations=[]
            for _ in range(2):
                torch.cuda.synchronize();started=time.monotonic()
                outputs[cached]=model.generate(ids,pixels,max_new_tokens=32,use_cache=cached)
                torch.cuda.synchronize();durations.append(time.monotonic()-started)
            times[cached]=sum(durations)/len(durations)
        rows.append({'with_image':with_image,'prompt_tokens':256,'new_tokens':32,
            'uncached_seconds':times[False],'cached_seconds':times[True],
            'speedup':times[False]/times[True],
            'greedy_sequences_match':torch.equal(outputs[False],outputs[True]),
            'prefill_max_absolute_logit_difference':prefill_difference,
            'decode_max_absolute_logit_difference':difference,'fp32_max_absolute_logit_difference':precise_difference,'prefill_cache_bytes':state['cache'].bytes})
report={'gpu':torch.cuda.get_device_name(),'dtype':'bf16','concurrent_training':True,'results':rows,
    'scope':'Full-size functional/timing check during concurrent training; not a dedicated serving benchmark.',
    'numerics':'Cache retains grouped KV heads and global rotary offsets; BF16 kernel shapes can change rounded logits.'}
(ROOT/'reports/l40-cache-benchmark.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
