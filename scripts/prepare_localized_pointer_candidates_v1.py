"""Join human attention spans to unchanged, licensed existing scene photos.

Candidate only. Original file roles, creator metadata, orientation, pixels and
trace indices remain explicit. No new labels enter the running curriculum.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.request import Request,urlopen

from PIL import Image,__version__ as PILLOW_VERSION
from tokenizers import Tokenizer

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json,pixel_hash
from slm_lab.expanded_photos import difference_hash
from slm_lab.localized_pointer_v1 import spans,histogram,jpeg_header
from slm_lab.vqa_pairs import known_photo_aliases

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/localized-pointer-source-v1'
OUTPUT=ROOT/'data/localized-pointer-candidate-v1'
PARENTS=[('train','expanded-everyday-photos-v1/train.jsonl'),('train','everyday-images-v1/train.jsonl'),
    ('validation','everyday-images-v1/valid.jsonl')]
TOKEN=ROOT/'checkpoints/dialogue-retention-v1/systems-variety-code/tokenizer.json'


def photo_header(key,parent,role):
    cache=ROOT/'data/image-source-cache-v1/source-images'/('train' if role=='train' else 'validation')/(key+'.jpg')
    if cache.exists() and file_hash(cache)==parent['download_sha256']:
        raw=cache.open('rb').read(131072);origin=dict(mode='exact_original_cached_download',cache=str(cache),
            original_download_sha256=parent['download_sha256'])
    else:
        subset='train' if role=='train' else 'validation'
        url=f'https://s3.amazonaws.com/open-images-dataset/{subset}/{key}.jpg'
        etag=parent.get('download_etag')
        if not etag:raise ValueError('Original downloaded object tag required for a bounded header recheck')
        request=Request(url,headers={'User-Agent':'slm-lab-public-research/1.0','Range':'bytes=0-131071','If-Match':etag})
        with urlopen(request,timeout=25) as response:
            if response.status not in {200,206} or response.headers.get('ETag')!=etag:
                raise ValueError('Original CVDF header object identity changed')
            raw=response.read(131073)
            if len(raw)>131072:raise ValueError('Header range exceeded its byte budget')
            range_value=response.headers.get('Content-Range')
            if response.status==206:
                parsed=re.fullmatch(r'bytes 0-(\d+)/(\d+)',range_value or '')
                if not parsed or int(parsed[1])+1!=len(raw):raise ValueError('Incomplete original header range')
            origin=dict(mode='original_publisher_range_header',url=url,etag=etag,content_range=range_value,
                complete_original_download_bytes_rehashed=False)
    dimensions=jpeg_header(raw)
    return raw,dimensions,origin


def check_photo(key,role,parent,metadata):
    try:
        if (metadata['License'].rstrip('/')!='https://creativecommons.org/licenses/by/2.0'
                or not metadata.get('Rotation') or float(metadata['Rotation'])!=0.
                or not metadata.get('Author') or parent['author']!=metadata['Author']):
            raise ValueError('Explicit identity rotation and original author/CC-BY metadata required')
        raw,original_dimensions,origin=photo_header(key,parent,role)
        path=Path(parent['_image_path'])
        if file_hash(path)!=parent['stored_image_sha256']:raise ValueError('Frozen original stored photo changed')
        with Image.open(path) as image:
            size=image.size
            if abs(size[0]*original_dimensions[1]-size[1]*original_dimensions[0])>max(original_dimensions):
                raise ValueError('Stored photo must preserve the original upright scene aspect ratio')
            hashes={str(n):pixel_hash(image,n) for n in [224,448]};fingerprint=difference_hash(image)
        if (hashes['224']!=parent['model_pixel_sha256']
                or (parent.get('model_pixel_sha256_448') and hashes['448']!=parent['model_pixel_sha256_448'])):
            raise ValueError('Every actual current input photo pixel hash must match its frozen source')
        header=OUTPUT/'headers'/(key+'.jpg.header');header.write_bytes(raw)
        return dict(image_id=key,split=role,image=str(path),stored_image_sha256=parent['stored_image_sha256'],
            stored_dimensions=list(size),original_CVDF_dimensions=list(original_dimensions),
            source_header=str(header.relative_to(OUTPUT)),source_header_sha256=hashlib.sha256(raw).hexdigest(),
            source_header_provenance=origin,original_EXIF_orientation=1,publisher_rotation_degrees=0.,
            model_pixel_sha256=hashes['224'],model_pixel_sha256_448=hashes['448'],difference_hash64=str(fingerprint),
            group_id=parent['group_id'],source_photo_aliases=sorted(known_photo_aliases(parent)),
            author=parent['author'],author_profile_url=parent['author_profile_url'],source_url=parent['source_url'],
            source_title=parent['source_title'],image_license=parent['license'],image_license_url=parent['license_url'],
            original_parent_records=parent['_parent_records'])
    except (ValueError,OSError,KeyError) as error:
        return dict(image_id=key,split=role,qualified_out=True,reason=str(error)[:350])


def main():
    if OUTPUT.exists():raise ValueError('Preserve original candidate/partial evidence; no implicit restart')
    collection=json.loads((SOURCE/'collection.json').read_text())
    if '2 passed' not in (ROOT/'reports/localized-pointer-v1-tests.log').read_text():
        raise ValueError('Actual source span, geometry, invalid timing and EXIF checks required')
    for name,receipt in collection['sources'].items():
        if file_hash(SOURCE/name)!=receipt['sha256']:raise ValueError('Complete original annotation source changed')
    parent_maps={r:{} for r in ['train','validation']};files={}
    for role,name in PARENTS:
        path=ROOT/'data'/name;manifest=path.parent/'manifest.json';data=json.loads(manifest.read_text())
        if file_hash(path)!=data['files_sha256'][path.name]:raise ValueError('Every frozen parent caption source required')
        files[str(path)]=file_hash(path);files[str(manifest)]=file_hash(manifest)
        for row in map(json.loads,path.read_text().splitlines()):
            key=row['source_image_id']
            if row.get('task')!='caption' or row.get('source')!='Open Images / CVDF' or key in parent_maps[role]:continue
            expected='train' if role=='train' else 'validation'
            if row['source_subset']!=expected:raise ValueError('Frozen parent role differs from original publisher role')
            image=(path.parent/row['image']).resolve()
            if not image.is_relative_to(ROOT/'data'):raise ValueError('Only original retained project photos may be read')
            parent_maps[role][key]={**row,'_image_path':str(image),'_parent_records':str(path)}
    if set(parent_maps['train'])&set(parent_maps['validation']):raise ValueError('Original parent image IDs cross roles')
    blocked=set()
    for role in ['train','valid']:
        path=ROOT/'data/vqa-source-cache-v1'/('candidate-pairs.'+role+'.jsonl');files[str(path)]=file_hash(path)
        for pair in map(json.loads,path.read_text().splitlines()):
            for side in ['first','second']:blocked.update(known_photo_aliases(pair[side]['image']))
    held_aliases=set().union(*(known_photo_aliases(row) for row in parent_maps['validation'].values()))
    held_pixels={224:set(),448:set()};held_fingerprints=[]
    for parent in parent_maps['validation'].values():
        if file_hash(parent['_image_path'])!=parent['stored_image_sha256']:raise ValueError('Frozen protected validation photo changed')
        with Image.open(parent['_image_path']) as image:
            for n in [224,448]:held_pixels[n].add(pixel_hash(image,n))
            held_fingerprints.append(difference_hash(image))
    metadata={role:{} for role in parent_maps}
    for role in parent_maps:
        path=ROOT/'data/image-source-cache-v1'/('openimages-'+role+'.csv')
        receipt=json.loads(path.with_suffix(path.suffix+'.source.json').read_text())
        if file_hash(path)!=receipt['sha256']:raise ValueError('Original creator/license/orientation CSV changed')
        files[str(path)]=receipt['sha256']
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row['ImageID'] in parent_maps[role]:metadata[role][row['ImageID']]=row
    OUTPUT.mkdir();(OUTPUT/'headers').mkdir();counts=Counter();matches=[]
    for role,name in [('train','train-shard00000.jsonl'),('validation','validation.jsonl')]:
        path=SOURCE/name
        with path.open('rb') as stream:
            while True:
                offset=stream.tell();raw=stream.readline()
                if not raw:break
                counts[role+':original_annotation_rows']+=1
                if len(raw)>2*1024**2:counts[role+':oversized_original_record_review']+=1;continue
                row=json.loads(raw);key=row['image_id'];parent=parent_maps[role].get(key)
                if parent is None:continue
                aliases=known_photo_aliases(parent)
                if aliases&blocked or (role=='train' and aliases&held_aliases):
                    counts[role+':protected_source_photo_alias']+=1;continue
                if key not in metadata[role]:counts[role+':missing_original_metadata']+=1;continue
                try:queries=spans(row)
                except (ValueError,KeyError,TypeError) as error:
                    counts[role+':original_transcript_or_trace_review']+=1;continue
                if not queries:counts[role+':no_complete_stable_content_span']+=1;continue
                counts[role+':matched_original_annotation_rows']+=1
                matches.append(dict(role=role,image_id=key,source=str(path),offset=offset,bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(),annotator_id=row['annotator_id'],queries=len(queries)))
    unique={(m['role'],m['image_id']) for m in matches}
    atomic_json(OUTPUT/'preparation-plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_collection_sha256=file_hash(SOURCE/'collection.json'),parent_files_sha256=files,
        source_annotation_roles_defined_by_original_file_not_dataset_id=True,
        unique_matched_original_images=len(unique),original_annotation_join_counts=dict(counts),
        source_implementation_sha256=file_hash(__file__),span_implementation_sha256=file_hash(ROOT/'src/slm_lab/localized_pointer_v1.py'),
        original_validation_never_optimization_input=True,image_size=448,grid_side=14,
        maximum_header_workers=2,annotation_license='CC-BY-4.0',pillow=PILLOW_VERSION,
        original_photo_pixels_reused=True,model_updates=0,training_admitted=False,goal_achieved=False))
    photos={};qualifications=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(check_photo,key,role,parent_maps[role][key],metadata[role][key]) for role,key in sorted(unique)]
        for future in as_completed(futures):
            result=future.result()
            if not result.get('qualified_out') and result['split']=='train':
                if (result['model_pixel_sha256'] in held_pixels[224] or result['model_pixel_sha256_448'] in held_pixels[448]
                        or any((int(result['difference_hash64'])^h).bit_count()<=4 for h in held_fingerprints)):
                    result=dict(image_id=result['image_id'],split='train',qualified_out=True,reason='Protected validation exact/perceptual photo input')
            if result.get('qualified_out'):qualifications.append(result)
            else:photos[(result['split'],result['image_id'])]=result
            if (len(photos)+len(qualifications))%100==0:
                atomic_json(OUTPUT/'progress.json',dict(observed_at=datetime.now(timezone.utc).isoformat(),
                    photo_headers_checked=len(photos)+len(qualifications),planned=len(unique),eligible=len(photos),
                    qualifications=len(qualifications),training_admitted=False,model_updates=0,goal_achieved=False))
                print('original_photo_orientation_pixels_checked',len(photos)+len(qualifications),'/',len(unique),flush=True)
    (OUTPUT/'photos.jsonl').write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for _,row in sorted(photos.items())))
    (OUTPUT/'photo-qualifications.jsonl').write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in qualifications))
    tokenizer=Tokenizer.from_file(str(TOKEN));row_counts=Counter();source_points=Counter();reviewed=Counter()
    handles={role:(OUTPUT/(role+'.jsonl')).open('x') for role in parent_maps};source_handles={}
    try:
        for match in matches:
            role=match['role'];photo=photos.get((role,match['image_id']))
            if photo is None:continue
            if match['source'] not in source_handles:source_handles[match['source']]=Path(match['source']).open('rb')
            stream=source_handles[match['source']];stream.seek(match['offset']);raw=stream.read(match['bytes'])
            if hashlib.sha256(raw).hexdigest()!=match['sha256']:raise ValueError('Original timed source row changed')
            record=json.loads(raw)
            for span in spans(record):
                query_ids=[tokenizer.token_to_id('<bos>')]+tokenizer.encode(span['query']).ids+[tokenizer.token_to_id('<eos>')]
                if len(query_ids)>34 or any(t in span['query'] for t in ['<pad>','<bos>','<eos>']):
                    reviewed[role+':whole_query_budget_or_reserved_marker']+=1;continue
                target=histogram(record,span,photo['stored_dimensions'])
                identity='localized-pointer:'+role+':'+match['image_id']+':'+str(record['annotator_id'])+':'+str(span['timed_utterance_index'])
                row=dict(id=identity,split=role,group_id=photo['group_id'],image_id=match['image_id'],
                    annotator_id=record['annotator_id'],source_dataset_id=record['dataset_id'],
                    query=span['query'],query_input_ids=query_ids,
                    timed_utterance_index=span['timed_utterance_index'],start_time=span['start_time'],end_time=span['end_time'],
                    source_point_refs=span['source_point_refs'],source_point_count=span['source_point_count'],
                    attention_grid_histogram=target,image_size=448,grid_side=14,
                    source_record={k:match[k] for k in ['source','offset','bytes','sha256']},
                    supervision='Original human transcript interval and in-frame stable mouse attention samples',
                    object_box_or_OCR_label=False,external_model_teacher_used=False,human_preferences=False,
                    source_photo_license='CC-BY-2.0',annotation_license='CC-BY-4.0',optimization_forbidden=role!='train')
                handles[role].write(json.dumps(row,separators=(',',':'),ensure_ascii=False)+'\n')
                row_counts[role]+=1;source_points[role]+=span['source_point_count']
    finally:
        for handle in [*handles.values(),*source_handles.values()]:handle.close()
    names=['train.jsonl','validation.jsonl','photos.jsonl','photo-qualifications.jsonl','preparation-plan.json']
    result=dict(completed_at=datetime.now(timezone.utc).isoformat(),format='own-original-human-pointer-candidate-v1',
        original_annotation_join_counts=dict(counts),photo_headers_checked=len(unique),eligible_original_photos=len(photos),
        eligible_photos_by_split=dict(Counter(role for role,_ in photos)),photo_qualification_counts=dict(Counter(r['reason'] for r in qualifications)),
        complete_query_rows=dict(row_counts),original_attention_point_samples=dict(source_points),whole_query_reviews=dict(reviewed),
        tokenizer_sha256=file_hash(TOKEN),source_collection_sha256=file_hash(SOURCE/'collection.json'),
        files_sha256={name:file_hash(OUTPUT/name) for name in names},
        image_pixels_and_author_credits_reused_from_unchanged_frozen_corpora=True,
        validation_rows_never_optimization_input=True,known_original_VQA_photo_aliases_excluded=True,
        approximate_annotator_attention_not_object_box_or_OCR_truth=True,
        semantic_or_historical_corpus_independence_certified=False,
        training_admitted=False,new_model_training_started=False,model_updates=0,goal_achieved=False,
        remaining=['Independently reconstruct every raw source interval/point, orientation/header/pixel and own-BPE query/grid target.',
            'Implement/test actual own-model grounding objective, null/wrong-image controls, full-model capacity and fixed matched protocol before any training.'])
    atomic_json(OUTPUT/'manifest.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
