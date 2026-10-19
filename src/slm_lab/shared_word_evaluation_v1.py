"""Whole held-word reading with fixed proper, blank and wrong-image controls.

Both native greedy text and byte-CTC paths are retained without response
cleanup. Every held crop is evaluated; no ground-truth crop is described as
full-photo text detection. Controls preserve target prompts and transcripts.
"""
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .diverse_images import atomic_json
from .ocr_evaluation import distance
from .own_aligned_assistant_v1 import decode_bytes
from .own_aligned_assistant_v3 import OwnAlignedAssistantV3
from .shared_word_data_v1 import SharedWordData, image_arrays, native_bytes

CONDITIONS = ('proper','blank','wrong')


def wrong_permutation(rows, seed=948231):
    """One-to-one within-family donors, different source group/pixels/text."""
    mapping = [None]*len(rows)
    for family in sorted({r['byte_family'] for r in rows}):
        positions = [i for i,r in enumerate(rows) if r['byte_family']==family]
        random.Random(seed+len(positions)).shuffle(positions)
        def different(a,b):
            return (a['group_id']!=b['group_id'] and a['response']!=b['response']
                    and a['source_RGB_pixels_sha256']!=b['source_RGB_pixels_sha256'])
        shift = next((s for s in range(1,len(positions)) if all(
            different(rows[p],rows[positions[(i+s)%len(positions)]]) for i,p in enumerate(positions))),None)
        if shift is None: raise ValueError('Complete within-family wrong-photo/target/pixel derangement required')
        for i,p in enumerate(positions): mapping[p] = positions[(i+shift)%len(positions)]
    if sorted(mapping)!=list(range(len(rows))): raise ValueError('Each held crop must be a wrong-image donor exactly once')
    return mapping


def literal_scores(expected, generated):
    if not expected: raise ValueError('Complete nonempty original word transcript required')
    return dict(exact=expected==generated,character_edits=distance(expected,generated),reference_characters=len(expected),
                word_edits=distance(expected.split(),generated.split()),reference_words=len(expected.split()))


def ctc_records(logits, texts):
    if logits.ndim!=3 or logits.shape[-1]!=257 or logits.shape[0]!=len(texts):
        raise ValueError('Whole byte-CTC batch and corresponding original transcripts required')
    logits = logits.float().cpu(); targets = [native_bytes(t) for t in texts]
    if any(required>logits.shape[1] for _,required in targets): raise ValueError('Complete repeated-byte alignment must fit the actual frames')
    lengths = torch.tensor([len(raw) for raw,_ in targets],dtype=torch.long)
    loss = F.ctc_loss(logits.log_softmax(-1).transpose(0,1),torch.tensor([b for raw,_ in targets for b in raw]),
        torch.full_like(lengths,logits.shape[1]),lengths,blank=256,reduction='none',zero_infinity=False)/lengths
    if not bool(torch.isfinite(loss).all()): raise ValueError('Every unreplaced full held CTC loss must be finite')
    result = []
    for path,text,nll in zip(logits.argmax(-1).tolist(),texts,loss.tolist()):
        generated,valid = decode_bytes(path); raw = []; previous = None
        for value in path:
            if value!=256 and value!=previous: raw.append(value)
            previous = value
        result.append(dict(path=path,generated_UTF8_bytes=raw,generated=generated,valid_UTF8=valid,
            byte_normalized_CTC=nll,byte_edits=distance(list(text.encode('utf-8')),raw),
            reference_bytes=len(text.encode('utf-8')),**literal_scores(text,generated)))
    return result


def trim_response(ids, limit):
    ids = list(map(int,ids))
    if not ids or len(ids)>limit: raise ValueError('Actual nonempty native generation within its fixed budget required')
    if 2 in ids:
        end = ids.index(2)+1
        if any(value!=2 for value in ids[end:]): raise ValueError('Only finished-row EOS padding may follow the first actual EOS')
        return ids[:end], 'EOS'
    if len(ids)!=limit: raise ValueError('Non-EOS generation must reach the exact declared budget')
    return ids, 'length'


def summarize(rows):
    if not rows: raise ValueError('Complete nonempty held evaluation required')
    result = dict(examples=len(rows),original_groups=len({r['group_id'] for r in rows}),conditions={})
    for condition in CONDITIONS:
        records = [r['conditions'][condition] for r in rows]; targets = sum(len(r['native']['target_ids']) for r in records)
        item = dict(CTC_exact=sum(r['CTC']['exact'] for r in records),native_exact=sum(r['native']['exact'] for r in records),
            CTC_valid_UTF8=sum(r['CTC']['valid_UTF8'] for r in records),
            word_mean_byte_normalized_CTC=math.fsum(r['CTC']['byte_normalized_CTC'] for r in records)/len(records),
            native_response_EOS_targets=targets,native_token_mean_NLL=math.fsum(v for r in records for v in r['native']['per_target_NLL'])/targets,
            native_EOS_responses=sum(r['native']['stop']=='EOS' for r in records))
        for head in ('CTC','native'):
            item[head+'_character_error_rate']=sum(r[head]['character_edits'] for r in records)/sum(r[head]['reference_characters'] for r in records)
            item[head+'_word_error_rate']=sum(r[head]['word_edits'] for r in records)/sum(r[head]['reference_words'] for r in records)
        item['CTC_byte_error_rate']=sum(r['CTC']['byte_edits'] for r in records)/sum(r['CTC']['reference_bytes'] for r in records)
        result['conditions'][condition]=item
    return result


@torch.no_grad()
def evaluate(model, data, tokenizer, output, *, batch_size=16, generation_budget=64, amp=None, checked=lambda:None, provenance=None):
    if (type(model) is not OwnAlignedAssistantV3 or type(data) is not SharedWordData
            or (data.role,data.purpose)!=('valid','evaluation') or type(batch_size) is not int or batch_size<1
            or type(generation_budget) is not int or generation_budget<1):
        raise ValueError('Exact shared model, all original held crops and explicit finite evaluation options required')
    output = Path(output); journal = output.with_suffix('.jsonl')
    if output.exists() or journal.exists(): raise ValueError('Preserve each original complete/partial held-word delivery')
    model.eval(); amp = amp or nullcontext; device = next(model.parameters()).device
    wrong = wrong_permutation(data.rows); records = []
    blank_word,blank_native = image_arrays(Image.new('RGB',(448,64),(128,128,128)))
    with journal.open('x') as stream:
        for start in range(0,len(data),batch_size):
            checked(); positions = list(range(start,min(start+batch_size,len(data))))
            proper = data.batch(positions,device); donor = data.batch([wrong[p] for p in positions],device)
            prefixes = [data.tokens[data.rows[p]['native']['offset']:data.rows[p]['native']['offset']+data.rows[p]['native']['target_start']].astype(np.int64).tolist() for p in positions]
            if any(prefix!=prefixes[0] for prefix in prefixes): raise ValueError('Batched native word generation requires identical complete unpadded original prompts')
            prompt = torch.tensor(prefixes,dtype=torch.long,device=device)
            local = [dict(source_position=p,id=data.rows[p]['id'],group_id=data.rows[p]['group_id'],
                family=data.rows[p]['byte_family'],expected=data.rows[p]['response'],wrong_image_id=data.rows[wrong[p]]['id'],
                source_image_sha256=data.rows[p]['stored_image_sha256'],native_prompt_ids=prefixes[i],conditions={}) for i,p in enumerate(positions)]
            for condition in CONDITIONS:
                checked()
                if condition=='blank':
                    word = torch.full_like(proper['word_pixels'],float(blank_word[0,0,0]))
                    native = torch.full_like(proper['pixel_values'],float(blank_native[0,0,0]))
                else:
                    source = proper if condition=='proper' else donor
                    word,native = source['word_pixels'],source['pixel_values']
                with amp():
                    ctc = ctc_records(model.word_logits(word),proper['texts'])
                    logits = model.aligned(proper['input_ids'],native,attention_mask=proper['attention_mask'])['logits']
                    delivered = model.generate(prompt,native,max_new_tokens=generation_budget,temperature=0.,eos_token_id=2)
                labels = proper['labels'][:,1:]; mask = labels!=-100
                losses = F.cross_entropy(logits[:,:-1].float().reshape(-1,model.config.vocab_size),labels.reshape(-1),ignore_index=-100,reduction='none').reshape(labels.shape)
                if not bool(torch.isfinite(losses[mask]).all()): raise ValueError('Every complete held native target likelihood must be finite')
                for i,row in enumerate(local):
                    ids,stop = trim_response(delivered[i,prompt.shape[1]:].tolist(),generation_budget)
                    generated = tokenizer.decode(ids,skip_special_tokens=True)
                    row['conditions'][condition]=dict(CTC=ctc[i],native=dict(generated=generated,generated_ids=ids,stop=stop,
                        target_ids=labels[i][mask[i]].tolist(),per_target_NLL=losses[i][mask[i]].cpu().tolist(),**literal_scores(row['expected'],generated)))
                del logits,delivered,losses
            for row in local:
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');records.append(row)
            stream.flush();print('complete_held_words',len(records),'/',len(data),flush=True)
    if len(records)!=len(data) or {r['id'] for r in records}!={r['id'] for r in data.rows}: raise ValueError('Every original held crop exactly once required')
    report = dict(overall=summarize(records),by_family={f:summarize([r for r in records if r['family']==f]) for f in sorted({r['family'] for r in records})},
        generation_budget=generation_budget,greedy_native_generation_without_cleanup=True,literal_case_whitespace_and_unicode_scoring=True,
        fixed_blank_RGB=[128,128,128],wrong_control_within_family_one_to_one_different_photo_pixels_and_transcript=True,
        complete_held_source_words=True,held_source_gradients=False,word_crops_not_full_photo_detection=True,
        provenance=provenance or {},results=records,candidate_promoted=False,goal_achieved=False)
    atomic_json(output,report);return report
