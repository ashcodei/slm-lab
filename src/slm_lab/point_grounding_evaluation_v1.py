"""Actual own-head evaluation with retained predictions and matched controls."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from .corpus import file_hash
from .diverse_images import atomic_json
from .localized_pointer_evidence_v1 import image_layout
from .point_grounding_data_v1 import PointerData
from .point_grounding_head_v1 import PointGroundingHead
from .point_grounding_protocol_v1 import control_pairs, prior_logits, score, summarize


def state_digest(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(json.dumps([name,list(value.shape),str(value.dtype)],separators=(',',':')).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def evaluate(head, data, position_prior, output, *, batch_size=4):
    if type(data) is not PointerData or data.role!='validation' or data.purpose!='evaluation':
        raise ValueError('Actual independently checked validation evaluation reader required')
    if type(head) is not PointGroundingHead or type(batch_size) is not int or batch_size<1:
        raise ValueError('Own exact grounding head and explicit positive evaluation batch required')
    output=Path(output)
    if output.exists():
        raise ValueError('Preserve previous actual attention diagnostic evidence')
    pairs=control_pairs(data.rows,data.photos)
    by_id={r['id']:r for r in data.rows}; pairing={r['id']:r for r in pairs}
    # Validate the supplied train-only prior before creating output evidence.
    prior_logits('training_position',[True]*196,position_prior)
    before=state_digest(head); mode=head.training; device=next(head.parameters()).device
    manifest_sha=file_hash(data.root/'manifest.json')
    output.mkdir(parents=True)
    (output/'control-pairs.jsonl').write_text(''.join(json.dumps(p,separators=(',',':'))+'\n' for p in pairs))
    atomic_json(output/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        actual_model_state_sha256=before,candidate_manifest_sha256=manifest_sha,
        all_original_validation_queries=len(data.rows),available_original_validation_photos=len(data.photos),
        queried_original_validation_photos=len({r['image_id'] for r in data.rows}),
        position_prior=position_prior,control_pairs_sha256=file_hash(output/'control-pairs.jsonl'),
        batch_size=batch_size,device=str(device),parameter_precision=str(next(head.parameters()).dtype),
        inference_autocast_enabled=torch.is_autocast_enabled(device.type),
        inference_autocast_dtype=str(torch.get_autocast_dtype(device.type)),
        blank_image_RGB=[128,128,128],original_target_mask_used_for_every_control=True,
        labels_describe_approximate_human_attention=True,model_updates=0,goal_achieved=False))
    blank_sha=hashlib.sha256(bytes([128,128,128])*(448*448)).hexdigest()
    journal=output/'observations.jsonl'
    try:
        head.eval()
        with journal.open('x') as sink:
            def retain(condition, rows, logits, targets, valid, query_rows, actual_images):
                logits=logits.detach().float().cpu().tolist() if isinstance(logits,torch.Tensor) else logits
                counts=targets.detach().float().cpu().tolist(); masks=valid.cpu().tolist()
                for row,values,target,mask,query,image_id in zip(rows,logits,counts,masks,query_rows,actual_images):
                    photo_sha=blank_sha if image_id is None else data.photos[image_id]['model_pixel_sha256_448']
                    result=dict(id=row['id'],image_id=row['image_id'],condition=condition,
                        actual_query_id=query['id'],actual_query_input_ids=query['query_input_ids'],
                        actual_image_id=image_id,actual_448px_RGB_sha256=photo_sha,
                        original_attention_grid_histogram=row['attention_grid_histogram'],
                        original_target_valid_cells=mask,logits=values,scores=score(values,target,mask))
                    sink.write(json.dumps(result,separators=(',',':'))+'\n')
            for first in range(0,len(data),batch_size):
                positions=list(range(first,min(first+batch_size,len(data))))
                rows=[data.rows[p] for p in positions]; batch=data.batch(positions,device)
                inputs={k:v for k,v in batch.items() if k!='targets'}
                ids=[r['image_id'] for r in rows]
                retain('proper',rows,head(**inputs)['logits'],batch['targets'],batch['valid_cells'],rows,ids)
                blank=dict(inputs,pixel_values=torch.full_like(batch['pixel_values'],128/127.5-1))
                retain('blank_image',rows,head(**blank)['logits'],batch['targets'],batch['valid_cells'],rows,[None]*len(rows))
                wrong_ids=[pairing[r['id']]['wrong_image_id'] for r in rows]
                wrong=dict(inputs,pixel_values=torch.stack([data.pixels(key) for key in wrong_ids]).to(device))
                retain('wrong_image',rows,head(**wrong)['logits'],batch['targets'],batch['valid_cells'],rows,wrong_ids)
                for kind in ['uniform','center','training_position']:
                    logits=[]
                    for row,valid in zip(rows,batch['valid_cells'].cpu().tolist()):
                        (left,top,width,height),_=image_layout(data.photos[row['image_id']]['stored_dimensions'])
                        center=int((top+height/2)//32)*14+int((left+width/2)//32)
                        logits.append(prior_logits(kind,valid,position_prior,center))
                    retain(kind,rows,logits,batch['targets'],batch['valid_cells'],rows,ids)
                chosen=[i for i,r in enumerate(rows) if pairing[r['id']]['wrong_query_id'] is not None]
                if chosen:
                    swapped=[by_id[pairing[rows[i]['id']]['wrong_query_id']] for i in chosen]
                    length=max(len(r['query_input_ids']) for r in swapped)
                    query_ids=torch.zeros(len(chosen),length,dtype=torch.long,device=device)
                    mask=torch.zeros_like(query_ids,dtype=torch.bool)
                    for i,row in enumerate(swapped):
                        n=len(row['query_input_ids']);query_ids[i,:n]=torch.tensor(row['query_input_ids'],device=device);mask[i,:n]=True
                    selected=torch.tensor(chosen,dtype=torch.long,device=device)
                    swap=dict(input_ids=query_ids,attention_mask=mask,
                        pixel_values=batch['pixel_values'].index_select(0,selected),valid_cells=batch['valid_cells'].index_select(0,selected))
                    retain('wrong_query',[rows[i] for i in chosen],head(**swap)['logits'],batch['targets'].index_select(0,selected),
                        swap['valid_cells'],swapped,[rows[i]['image_id'] for i in chosen])
                sink.flush()
    finally:
        head.train(mode)
    after=state_digest(head)
    if before!=after or file_hash(data.root/'manifest.json')!=manifest_sha:
        raise ValueError('Read-only diagnostics must preserve actual model and source data')
    if any(file_hash(data.root/name)!=sha for name,sha in data.manifest['files_sha256'].items()):
        raise ValueError('Every original query/photo source file must remain unchanged')
    retained=[]
    with journal.open() as source:
        for line in source:
            row=json.loads(line); original=by_id[row['id']];target=[0.]*196
            for cell,count in original['attention_grid_histogram']:
                target[cell]=count
            if score(row['logits'],target,row['original_target_valid_cells'])!=row['scores']:
                raise ValueError('Every retained original target/prediction score must reconstruct')
            retained.append(row)
    result=summarize(retained)
    result.update(completed_at=datetime.now(timezone.utc).isoformat(),actual_model_state_sha256=before,
        every_retained_actual_prediction_and_original_attention_score_recounted=True,
        actual_model_and_original_training_mode_unchanged=True,
        observations_sha256=file_hash(journal),candidate_manifest_sha256=manifest_sha,
        public_development_validation_not_certified_independent_final_test=True,
        model_updates=0,automatic_promotion=False,goal_achieved=False)
    atomic_json(output/'complete.json',result)
    return result
