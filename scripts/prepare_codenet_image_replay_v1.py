"""Index later finalized image sources without copying images or clipping original targets."""
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
from tokenizers import Tokenizer
import torch

from prepare_codenet_native_contract_pack_v2 import occupied_writer_reservation
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_image_replay_v1 import FAMILIES,ARTIFACTS,image_role,canonical_image,ImageReplayCorpus
from slm_lab.codenet_replay_v1 import INDEX_DTYPE,FullReplayEncoder,ReplayContextDoesNotFit
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig
from slm_lab.textvqa_admission import admit as admit_reading
from slm_lab.own_ocr_question_admission import admit as admit_context

ROOT=Path(__file__).resolve().parents[1]
FOLDERS=('expanded-everyday-photos-v1','expanded-scene-words-v1','visual-pretraining-v1',
         'clevr-reasoning-v1','textvqa-human-reading-v1','own-ocr-human-questions-v1')


def pixel_digest(tensor):
    if tensor.shape!=(3,448,448) or tensor.dtype!=torch.float32:raise ValueError('Actual448px float32 image input required')
    rgb=np.rint((tensor.permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Keep all GPUs dedicated to the current curriculum')
    torch.set_num_threads(1)
    output=ROOT/'data/codenet-image-replay-v1';reports=ROOT/'reports/codenet-image-replay-v1'
    if output.exists() or reports.exists():raise ValueError('Preserve every previous image replay preparation')
    occupied=occupied_writer_reservation();reserve=26*1024**3-occupied['credited_bytes'];maximum=256*1024**2
    if shutil.disk_usage(ROOT).free<reserve+maximum:raise ValueError('Remaining writer reserve plus bounded image replay artifacts required')
    reading,reading_receipt=admit_reading(ROOT);context,context_receipt=admit_context(ROOT)
    if len(reading)!=5858 or len(context)!=5858:raise ValueError('All original admitted training question variants required')
    manifests=[json.loads((ROOT/'data'/folder/'manifest.json').read_text()) for folder in FOLDERS]
    files=[Path(__file__),ROOT/'src/slm_lab/codenet_image_replay_v1.py',ROOT/'src/slm_lab/codenet_replay_v1.py',
        ROOT/'src/slm_lab/codenet_checkpoint_v1.py',ROOT/'src/slm_lab/training.py',ROOT/'src/slm_lab/model.py',
        ROOT/'src/slm_lab/textvqa_admission.py',ROOT/'src/slm_lab/own_ocr_question_admission.py',
        ROOT/'data/codenet-durable-pack-v2/tokenizer.json',ROOT/'configs/model.json',
        ROOT/'reports/extended-curriculum-v1/plan.json',ROOT/'data/extended-curriculum-v1/joint.train.jsonl',
        ROOT/'reports/clevr-original-input-audit-v1.json',ROOT/'scripts/audit_clevr_inputs.py',
        ROOT/'reports/textvqa-original-input-audit-v1.json',ROOT/'reports/textvqa-runtime-audit-refresh-v2.json']
    for folder,manifest in zip(FOLDERS,manifests):
        directory=ROOT/'data'/folder;files.append(directory/'manifest.json')
        files.extend(directory/name for name in manifest['files_sha256'])
    for index,module in [(0,'expanded_photos'),(1,'expanded_scene_words'),(2,'visual_pretraining_data')]:
        path=ROOT/'src/slm_lab'/(module+'.py')
        if path.exists():
            if file_hash(path)!=manifests[index]['configuration']['implementation_sha256']:
                raise ValueError('Original finalized image preparation implementation changed')
            files.append(path)
    for m in manifests[:2]:
        config=m['configuration']
        for k in ['inputs_sha256','protected_inputs','protected_manifests']:
            for p,h in config.get(k,{}).items():
                if file_hash(p)!=h:raise ValueError('Original protected image source inventory changed')
                files.append(Path(p))
    for item in manifests[2]['configuration']['sources']:
        if file_hash(item['path'])!=item['sha256']:raise ValueError('Original visual-pretraining source inventory changed')
        files.append(Path(item['path']))
    clevr=manifests[3];clevr_audit=json.loads((ROOT/'reports/clevr-original-input-audit-v1.json').read_text())
    if (clevr_audit['passed'] is not True or clevr_audit['manifest_sha256']!=file_hash(ROOT/'data/clevr-reasoning-v1/manifest.json')
            or clevr_audit['audit_source_sha256']!=file_hash(ROOT/'scripts/audit_clevr_inputs.py')
            or clevr_audit['files_sha256']!=clevr['files_sha256'] or clevr_audit['splits']!=clevr['splits']):
        raise ValueError('Completed original visual-reasoning input audit required')
    for p,h in clevr_audit['source_files_sha256'].items():
        path=ROOT/p
        if file_hash(path)!=h:raise ValueError('Original visual reasoning audited source changed')
        files.append(path)
    bound={str(p):file_hash(p) for p in files}
    for folder,m in zip(FOLDERS,manifests):
        for name,h in m['files_sha256'].items():
            if bound[str(ROOT/'data'/folder/name)]!=h:raise ValueError('Original finalized image input changed')
    previous_source=ROOT/'data/extended-curriculum-v1/joint.train.jsonl'
    joint=next(s for s in json.loads((ROOT/'reports/extended-curriculum-v1/plan.json').read_text())['stages'] if s['name']=='joint')
    if bound[str(previous_source)]!=joint['data_sha256']:raise ValueError('Exact original main-joint input required')
    prior_rows=[json.loads(s) for s in previous_source.open()];prior_groups={r['group_id'] for r in prior_rows if r.get('image')}
    prior_ids={r['id'] for r in prior_rows}
    protected_paths=[ROOT/'data'/folder/name for folder,name in [
        ('alignment-curriculum-v1','images.valid.jsonl'),('everyday-images-v1','valid.jsonl'),
        ('scene-text-images-v1','valid.jsonl'),('visual-pretraining-v1','valid.jsonl'),
        ('clevr-reasoning-v1','valid.jsonl'),('textvqa-human-reading-v1','valid.jsonl')]]
    protected_groups=set();protected_pixels=set()
    for path in protected_paths:
        bound[str(path)]=file_hash(path)
        for line in path.open():
            row=json.loads(line);protected_groups.add(row['group_id'])
            if row.get('model_pixel_sha256_448'):protected_pixels.add(row['model_pixel_sha256_448'])
    output.mkdir();reports.mkdir()
    def status(message,**fields):
        atomic_json(reports/'status.json',dict(status=message,updated_at=datetime.now(timezone.utc).isoformat(),
            new_model_training_started=False,goal_achieved=False,**fields))
    status('materializing exact admitted original own-context rows; no image copies')
    view=output/'own-context-original-rows.jsonl';all_decoded=hashlib.sha256();retained=0
    with gzip.open(ROOT/'data/own-ocr-human-questions-v1/train.jsonl.gz','rb') as source,view.open('xb') as target:
        for raw in source:
            all_decoded.update(raw);row=json.loads(raw)
            if row['own_predicted_context_supplied']:
                if row!=context[retained]:raise ValueError('Original admitted own-context row order or contents changed')
                target.write(raw);retained+=1
    if retained!=5858 or all_decoded.hexdigest()!=manifests[5]['decoded_JSONL_sha256']['train.jsonl']:
        raise ValueError('Every original decoded own-context question required')
    token_path=ROOT/'data/codenet-durable-pack-v2/tokenizer.json'
    encoder=FullReplayEncoder(Tokenizer.from_file(str(token_path)),ModelConfig.load(ROOT/'configs/model.json'),ROOT,448)
    sources=[]
    for i,(name,folder,m) in enumerate(zip(FAMILIES,FOLDERS,manifests)):
        original_key='train.jsonl.gz' if i==5 else 'train.jsonl';original_path=ROOT/'data'/folder/original_key
        path=view if i==5 else original_path
        sources.append(dict(family=name,path=str(path),sha256=file_hash(path),original_manifest=str(ROOT/'data'/folder/'manifest.json'),
            original_manifest_sha256=bound[str(ROOT/'data'/folder/'manifest.json')],original_file_key=original_key,
            original_file_path=str(original_path),original_file_sha256=bound[str(original_path)],
            selection='commons groups absent from exact original joint image groups' if i==2 else 'admitted own-context variants only' if i==5 else 'all original train-file rows'))
    bound[str(view)]=file_hash(view);guard=FileBindings(bound)
    atomic_json(reports/'plan.json',dict(recorded_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,
        original_families=list(FAMILIES),sources=sources,maximum_artifact_bytes=maximum,occupied_verified_writer=occupied,
        remaining_writer_reserve_bytes=reserve,all_images_reused_without_copying=True,complete_masks_no_clipping=True,
        final_starting_checkpoint_match_verified=False,goal_achieved=False))
    bound[str(reports/'plan.json')]=file_hash(reports/'plan.json');guard=FileBindings(bound)
    receipt=dict(passed=True,held_roles_excluded=True,source_manifest_sha256={s['family']:s['original_manifest_sha256'] for s in sources},
        original_reading=reading_receipt,original_own_context=context_receipt,original_visual_reasoning_audit_sha256=file_hash(ROOT/'reports/clevr-original-input-audit-v1.json'),
        scope='Finalized original inputs and unchanged existing source admissions; independent complete replay readback still required.',goal_achieved=False)
    atomic_json(output/'source-admission.json',receipt)
    entries=[];images={};counts=Counter();contexts=Counter();targets=Counter();review_counts=Counter()
    selected_counts=Counter();seen=set();all_images=0;reading_position=0;photo_groups=set()
    with (output/'records.jsonl').open('x') as records,(output/'review.jsonl').open('x') as reviews:
        for family_index,source in enumerate(sources):
            name=source['family'];path=Path(source['path']);source_lines=0
            with path.open('rb') as stream:
                while raw:=stream.readline():
                    source_lines+=1;offset=stream.tell()-len(raw);row=json.loads(raw)
                    if name=='additional_commons' and (row.get('visual_source')!='commons' or row['group_id'] in prior_groups):continue
                    image_role(row,name)
                    if (row['id'] in seen or row['id'] in prior_ids or row['group_id'] in protected_groups
                            or row['model_pixel_sha256_448'] in protected_pixels):
                        raise ValueError('Repeated original example or protected held group/pixels in later-image replay')
                    if name=='additional_photo_captions':
                        if row['group_id'] in prior_groups:raise ValueError('Additional photo overlaps earlier admitted training parents')
                        photo_groups.add(row['group_id'])
                    if name=='additional_real_words' and row['group_id'] not in prior_groups:
                        raise ValueError('Additional word crop needs its originally admitted training parent')
                    if name in {'human_photo_reading','own_context_photo_reading'} and row['group_id'] not in prior_groups|photo_groups:
                        raise ValueError('Human question requires its originally admitted full training photo')
                    if name=='human_photo_reading':
                        if row!=reading[reading_position]:raise ValueError('Every original admitted plain human question required')
                        reading_position+=1
                    image=canonical_image(row,path,ROOT);image_row={**row,'image':str(image)}
                    if str(image) not in images:
                        if file_hash(image)!=row['stored_image_sha256']:raise ValueError('Original stored training image bytes changed')
                        actual=pixel_digest(encoder.pixels(image_row))
                        images[str(image)]=dict(path=str(image),sha256=row['stored_image_sha256'],
                            model_pixel_sha256_448=actual,state=ImageReplayCorpus._state(image))
                    checked=images[str(image)]
                    if (checked['sha256']!=row['stored_image_sha256'] or checked['model_pixel_sha256_448']!=row['model_pixel_sha256_448']
                            or checked['state']!=ImageReplayCorpus._state(image)):
                        raise ValueError('Actual448px original image inputs differ from golden hashes')
                    seen.add(row['id']);selected_counts[name]+=1;all_images+=1
                    metadata=dict(id=row['id'],group_id=row['group_id'],family=name,source_line=source_lines,
                        source_offset=offset,source_bytes=len(raw),source_row_sha256=hashlib.sha256(raw).hexdigest())
                    try:ids,labels=encoder.sequence(image_row,'response')
                    except ReplayContextDoesNotFit as error:
                        review_counts[name]+=1
                        reviews.write(json.dumps({**metadata,'qualification':'original_whole_context_exceeds_window',
                            'prompt_tokens':error.prompt_tokens,'response_tokens':error.response_tokens,
                            'complete_original_context_tokens':error.prompt_tokens+error.response_tokens+2})+'\n')
                    else:
                        count=sum(v!=-100 for v in labels)
                        if any(v in {encoder.bos,encoder.eos,encoder.pad} for v in ids[1:-1]):
                            review_counts[name]+=1
                            reviews.write(json.dumps({**metadata,'qualification':'literal_reserved_model_ids_in_original_source',
                                'complete_original_context_tokens':len(ids),'complete_original_target_tokens':count})+'\n')
                        else:
                            if ('target_tokens' in row and row['target_tokens']!=count
                                    or labels!=[-100]*(len(ids)-count)+ids[-count:]):
                                raise ValueError('Complete original image answer/EOS masks changed')
                            digest=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).digest()
                            entries.append((offset,len(raw),family_index,len(ids),count,np.void(hashlib.sha256(raw).digest()),np.void(digest)))
                            counts[name]+=1;contexts[name]+=len(ids);targets[name]+=count
                            records.write(json.dumps({**metadata,'complete_context_tokens':len(ids),'complete_target_tokens':count,
                                'complete_input_ids_sha256':digest.hex()})+'\n')
                    if all_images%1000==0:
                        guard.check();records.flush();reviews.flush()
                        if (sum(p.stat().st_size for p in output.iterdir() if p.is_file())>maximum
                                or shutil.disk_usage(ROOT).free<512*1024**2):raise ValueError('Bounded image replay storage exceeded')
                        status('indexing original complete later-image contexts and actual448px inputs',source_rows=all_images,
                            complete_fit_rows=len(entries),review_rows=sum(review_counts.values()),family=name,unique_images=len(images))
                        print('original_later_image_rows',all_images,'family',name,'fit',len(entries),'unique_pixels',len(images),flush=True)
            source.update(original_source_lines=source_lines,selected_rows=selected_counts[name],review_rows=review_counts[name])
    expected=[78255,23847,987,40000,5858,5858]
    if ([selected_counts[name] for name in FAMILIES]!=expected or len(seen)!=154805 or reading_position!=5858
            or any(counts[name]==0 for name in FAMILIES) or encoder.truncated):
        raise ValueError('Every original later-image family/row, with complete fit or explicit review, required')
    np.save(output/'index.npy',np.asarray(entries,dtype=INDEX_DTYPE),allow_pickle=False)
    with (output/'image-files.jsonl').open('x') as stream:
        for image,record in sorted(images.items()):
            if file_hash(image)!=record['sha256'] or ImageReplayCorpus._state(image)!=record['state']:
                raise ValueError('Original image changed during full preparation')
            stream.write(json.dumps({k:v for k,v in record.items() if k!='state'})+'\n')
    guard.check(full=True)
    size=sum((output/n).stat().st_size for n in ARTIFACTS)
    if size>maximum:raise ValueError('Final complete image replay artifacts exceed reserved output')
    manifest=dict(format='slm-complete-later-image-replay-v1',completed_at=datetime.now(timezone.utc).isoformat(),
        project_root=str(ROOT),tokenizer_sha256=bound[str(token_path)],image_size=448,sources=sources,
        family_names=list(FAMILIES),family_counts=dict(counts),original_family_counts=dict(selected_counts),
        source_rows=len(seen),complete_fit_rows=len(entries),review_rows=sum(review_counts.values()),
        family_complete_token_counts={name:dict(context_tokens=contexts[name],target_tokens=targets[name]) for name in FAMILIES},
        original_golden448px_image_exposures_verified=all_images,unique_original_image_files=len(images),
        protected_held_group_overlap=0,protected_recorded448px_pixel_overlap=0,encoder_truncated_sequences=encoder.truncated,
        artifact_files_sha256={n:file_hash(output/n) for n in sorted(ARTIFACTS)},stored_artifact_bytes=size,
        preparation_plan_sha256=bound[str(reports/'plan.json')],
        new_image_bytes_copied=0,final_starting_checkpoint_match_verified=False,all_later_queued_streams_covered=False,
        independent_original_replay_readback_pending=True,new_model_training_started=False,goal_achieved=False,
        scope='Exact finalized later train-image sources and admitted plain/own-context variants; original image bytes reused. Paired-photo collection, systems/interface rehearsal and final starting-policy matching remain separate. Source/pixel coverage is not retained capability.')
    atomic_json(output/'manifest.json',manifest)
    corpus=ImageReplayCorpus(output)
    if sum(corpus.size(name) for name in corpus.families)!=len(entries):raise ValueError('Actual complete image replay reader population differs')
    status('complete later-image replay prepared; independent full readback and expanded coverage pending',
        source_rows=len(seen),complete_fit_rows=len(entries),review_rows=sum(review_counts.values()),unique_images=len(images),stored_bytes=size)
    print('complete_later_image_replay',len(seen),len(entries),sum(review_counts.values()),len(images),flush=True)


if __name__=='__main__':main()
