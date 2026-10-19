"""All held crops in fixed global proper/blank/wrong controls on four replicas.

Only scheduling differs from the original dense-word evaluator. The source
population and global one-to-one wrong-image mapping remain unchanged.
"""
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .diverse_images import atomic_json
from .corpus import file_hash
from .shared_word_evidence_v1 import readback_held
from .own_dense_actor_v1 import OwnDenseActor
from .own_dense_growth_v1 import OwnDenseAssistant
from .shared_word_data_v1 import SharedWordData,image_arrays
from .shared_word_evaluation_v1 import CONDITIONS,ctc_records,literal_scores,summarize,trim_response,wrong_permutation

FORMAT='own-standalone-complete-held-word-shard-v1'


def positions(data,rank,world):
    if (type(data) is not SharedWordData or (data.role,data.purpose)!=('valid','evaluation')
            or type(world) is not int or not 1<=world<=len(data)
            or type(rank) is not int or not 0<=rank<world):
        raise ValueError('The complete original held word population and exact nonempty replica allocation are required')
    return list(range(rank,len(data),world))


@torch.no_grad()
def evaluate_shard(model,data,tokenizer,output,*,rank,world,batch_size=16,generation_budget=64,amp=None,checked=lambda:None,provenance=None):
    if (type(model) not in (OwnDenseAssistant,OwnDenseActor) or type(data) is not SharedWordData
            or (data.role,data.purpose)!=('valid','evaluation') or type(batch_size) is not int or batch_size<1
            or type(generation_budget) is not int or generation_budget<1):
        raise ValueError('Exact one-dense-model family, every original held crop and explicit evaluation options required')
    selected=positions(data,rank,world)
    base=model if type(model) is OwnDenseAssistant else model.base
    model.assert_inventory() if type(model) is OwnDenseAssistant else model.assert_shared()
    output=Path(output);journal=output.with_suffix('.jsonl')
    if output.exists() or journal.exists():raise ValueError('Preserve each original complete or partial dense-word delivery')
    model.eval();amp=amp or nullcontext;device=next(model.parameters()).device
    wrong=wrong_permutation(data.rows);records=[]
    blank_word,blank_native=image_arrays(Image.new('RGB',(448,64),(128,128,128)))
    with journal.open('x') as stream:
        for start in range(0,len(selected),batch_size):
            checked();source_positions=selected[start:start+batch_size]
            proper=data.batch(source_positions,device);donor=data.batch([wrong[p] for p in source_positions],device)
            prefixes=[data.tokens[data.rows[p]['native']['offset']:data.rows[p]['native']['offset']+data.rows[p]['native']['target_start']].astype(np.int64).tolist() for p in source_positions]
            if any(prefix!=prefixes[0] for prefix in prefixes):raise ValueError('Identical complete unpadded native word prompts required for batching')
            prompt=torch.tensor(prefixes,dtype=torch.long,device=device)
            local=[dict(source_position=p,id=data.rows[p]['id'],group_id=data.rows[p]['group_id'],family=data.rows[p]['byte_family'],
                expected=data.rows[p]['response'],wrong_image_id=data.rows[wrong[p]]['id'],source_image_sha256=data.rows[p]['stored_image_sha256'],
                native_prompt_ids=prefixes[i],conditions={}) for i,p in enumerate(source_positions)]
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
            stream.flush();print('complete_dense_held_words',len(records),'/',len(selected),flush=True)
    if len(records)!=len(selected) or [r['source_position'] for r in records]!=selected:
        raise ValueError('Every original held source exactly once required')
    report=dict(format=FORMAT,rank=rank,world=world,original_held_positions=selected,
        raw_journal=str(journal.resolve()),raw_journal_sha256=file_hash(journal),implementation_sha256=file_hash(__file__),
        actual_counts=model.counts(),overall=summarize(records),
        by_family={f:summarize([r for r in records if r['family']==f]) for f in sorted({r['family'] for r in records})},
        generation_budget=generation_budget,greedy_native_generation_without_cleanup=True,literal_case_whitespace_and_unicode_scoring=True,
        fixed_blank_RGB=[128,128,128],wrong_control_within_family_one_to_one_different_photo_pixels_and_transcript=True,
        complete_original_selected_shard=True,held_source_gradients=False,word_crops_not_full_photo_detection=True,
        provenance=provenance or {},results=records,candidate_promoted=False,goal_achieved=False)
    atomic_json(output,report);return report


def reconcile(data,tokenizer,paths,output,*,generation_budget=64,ctc_frames=140):
    paths=[Path(p) for p in paths];output=Path(output)
    if not paths or output.exists() or output.with_suffix('.jsonl').exists():raise ValueError('Fresh complete held-word destination required')
    world=len(paths);wrong=wrong_permutation(data.rows);seen=set();rows={};common=None;inventory={}
    keys=['actual_counts','generation_budget','greedy_native_generation_without_cleanup','literal_case_whitespace_and_unicode_scoring',
          'fixed_blank_RGB','wrong_control_within_family_one_to_one_different_photo_pixels_and_transcript','provenance']
    for path in paths:
        shard=json.loads(path.read_text());rank=shard.get('rank');selected=positions(data,rank,world);journal=Path(shard['raw_journal'])
        current={k:shard[k] for k in keys}
        if common is None:common=current
        if (shard.get('format')!=FORMAT or shard.get('world')!=world or rank in seen or current!=common
                or shard.get('original_held_positions')!=selected or shard.get('generation_budget')!=generation_budget
                or shard.get('held_source_gradients') is not False or shard.get('complete_original_selected_shard') is not True
                or shard.get('implementation_sha256')!=file_hash(__file__) or file_hash(journal)!=shard['raw_journal_sha256']
                or shard['results']!=[json.loads(line) for line in journal.open()]
                or [r['source_position'] for r in shard['results']]!=selected):
            raise ValueError('Every exact original held-word shard, native protocol and raw delivery must match')
        for row in shard['results']:
            p=row['source_position']
            if row['wrong_image_id']!=data.rows[wrong[p]]['id'] or set(row['conditions'])!=set(CONDITIONS):
                raise ValueError('The original full-population wrong-image mapping and all three conditions must be preserved')
            rows[p]=row
        if (shard['overall']!=summarize(shard['results']) or shard['by_family']!=
                {f:summarize([r for r in shard['results'] if r['family']==f]) for f in sorted({r['family'] for r in shard['results']})}):
            raise ValueError('Every source-shard word summary must reconstruct')
        seen.add(rank);inventory[str(path.resolve())]=file_hash(path);inventory[str(journal)]=file_hash(journal)
    if seen!=set(range(world)) or set(rows)!=set(range(len(data))):raise ValueError('Every original held crop must occur exactly once')
    ordered=[rows[p] for p in range(len(data))]
    report=dict(format=FORMAT,**common,overall=summarize(ordered),
        by_family={f:summarize([r for r in ordered if r['family']==f]) for f in sorted({r['family'] for r in ordered})},
        results=ordered,complete_held_source_words=True,held_source_gradients=False,word_crops_not_full_photo_detection=True,
        all_original_global_wrong_donors_preserved=True,source_files_sha256=inventory,
        neural_likelihoods_not_independently_replayed_by_reconciliation=True,candidate_promoted=False,goal_achieved=False)
    with output.with_suffix('.jsonl').open('x') as stream:
        for row in ordered:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    atomic_json(output,report)
    readback_held(data,tokenizer,output,generation_budget,ctc_frames=ctc_frames)
    return report
