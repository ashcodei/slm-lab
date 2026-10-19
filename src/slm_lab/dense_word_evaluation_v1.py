"""Complete held reading controls for the one-dense-model policy family.

This reuses the immutable source geometry and literal scorers of the earlier
word experiment. The old type guard and implementation remain unchanged.
"""
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .diverse_images import atomic_json
from .own_dense_actor_v1 import OwnDenseActor
from .own_dense_growth_v1 import OwnDenseAssistant
from .shared_word_data_v1 import SharedWordData,image_arrays
from .shared_word_evaluation_v1 import CONDITIONS,ctc_records,literal_scores,summarize,trim_response,wrong_permutation

FORMAT='own-single-dense-model-all-held-word-image-controls-v1'


@torch.no_grad()
def evaluate(model,data,tokenizer,output,*,batch_size=16,generation_budget=64,amp=None,checked=lambda:None,provenance=None):
    if (type(model) not in (OwnDenseAssistant,OwnDenseActor) or type(data) is not SharedWordData
            or (data.role,data.purpose)!=('valid','evaluation') or type(batch_size) is not int or batch_size<1
            or type(generation_budget) is not int or generation_budget<1):
        raise ValueError('Exact one-dense-model family, every original held crop and explicit evaluation options required')
    base=model if type(model) is OwnDenseAssistant else model.base
    model.assert_inventory() if type(model) is OwnDenseAssistant else model.assert_shared()
    output=Path(output);journal=output.with_suffix('.jsonl')
    if output.exists() or journal.exists():raise ValueError('Preserve each original complete or partial dense-word delivery')
    model.eval();amp=amp or nullcontext;device=next(model.parameters()).device
    wrong=wrong_permutation(data.rows);records=[]
    blank_word,blank_native=image_arrays(Image.new('RGB',(448,64),(128,128,128)))
    with journal.open('x') as stream:
        for start in range(0,len(data),batch_size):
            checked();positions=list(range(start,min(start+batch_size,len(data))))
            proper=data.batch(positions,device);donor=data.batch([wrong[p] for p in positions],device)
            prefixes=[data.tokens[data.rows[p]['native']['offset']:data.rows[p]['native']['offset']+data.rows[p]['native']['target_start']].astype(np.int64).tolist() for p in positions]
            if any(prefix!=prefixes[0] for prefix in prefixes):raise ValueError('Identical complete unpadded native word prompts required for batching')
            prompt=torch.tensor(prefixes,dtype=torch.long,device=device)
            local=[dict(source_position=p,id=data.rows[p]['id'],group_id=data.rows[p]['group_id'],family=data.rows[p]['byte_family'],
                expected=data.rows[p]['response'],wrong_image_id=data.rows[wrong[p]]['id'],source_image_sha256=data.rows[p]['stored_image_sha256'],
                native_prompt_ids=prefixes[i],conditions={}) for i,p in enumerate(positions)]
            for condition in CONDITIONS:
                checked()
                if condition=='blank':
                    word=torch.full_like(proper['word_pixels'],float(blank_word[0,0,0]))
                    native=torch.full_like(proper['pixel_values'],float(blank_native[0,0,0]))
                else:
                    source=proper if condition=='proper' else donor;word,native=source['word_pixels'],source['pixel_values']
                with amp():
                    ctc=ctc_records(base.word_logits(word),proper['texts'])
                    logits=model(proper['input_ids'],native,attention_mask=proper['attention_mask'])['logits']
                    delivered=model.generate(prompt,native,max_new_tokens=generation_budget,temperature=0.,eos_token_id=2)
                labels=proper['labels'][:,1:];mask=labels!=-100
                losses=F.cross_entropy(logits[:,:-1].float().reshape(-1,model.config.vocab_size),labels.reshape(-1),ignore_index=-100,reduction='none').reshape(labels.shape)
                if not bool(torch.isfinite(losses[mask]).all()):raise ValueError('Every full original dense-word response target needs a finite neural likelihood')
                for i,row in enumerate(local):
                    ids,stop=trim_response(delivered[i,prompt.shape[1]:].tolist(),generation_budget)
                    generated=tokenizer.decode(ids,skip_special_tokens=True)
                    row['conditions'][condition]=dict(CTC=ctc[i],native=dict(generated=generated,generated_ids=ids,stop=stop,
                        target_ids=labels[i][mask[i]].tolist(),per_target_NLL=losses[i][mask[i]].cpu().tolist(),**literal_scores(row['expected'],generated)))
                del logits,delivered,losses
            for row in local:stream.write(json.dumps(row,ensure_ascii=False)+'\n');records.append(row)
            stream.flush();print('complete_dense_held_words',len(records),'/',len(data),flush=True)
    if len(records)!=len(data) or {r['id'] for r in records}!={r['id'] for r in data.rows}:
        raise ValueError('Every original held source exactly once required')
    report=dict(format=FORMAT,actual_counts=model.counts(),overall=summarize(records),
        by_family={f:summarize([r for r in records if r['family']==f]) for f in sorted({r['family'] for r in records})},
        generation_budget=generation_budget,greedy_native_generation_without_cleanup=True,literal_case_whitespace_and_unicode_scoring=True,
        fixed_blank_RGB=[128,128,128],wrong_control_within_family_one_to_one_different_photo_pixels_and_transcript=True,
        complete_held_source_words=True,held_source_gradients=False,word_crops_not_full_photo_detection=True,
        provenance=provenance or {},results=records,candidate_promoted=False,goal_achieved=False)
    atomic_json(output,report);return report
