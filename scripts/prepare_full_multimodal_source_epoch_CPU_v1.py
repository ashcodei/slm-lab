"""Inventory all admitted image/word views and rebuild their complete native masks.

No neural model is loaded. Existing train roles and every source occurrence are
preserved. This is source/schedule preparation only: physical pixel revalidation,
distributed image/CTC admission and an actual training controller remain pending.
"""
from collections import Counter
from contextlib import ExitStack
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
from tokenizers import Tokenizer

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.codenet_image_replay_v1 import FAMILIES as IMAGE_FAMILIES,image_role,canonical_image
from slm_lab.codenet_replay_v1 import FullReplayEncoder,INDEX_DTYPE,family as replay_family,training_role
from slm_lab.fast_joint_data_v1 import CompleteBucketSchedule
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.model import ModelConfig
from slm_lab.shared_word_data_v1 import native_bytes

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/full-multimodal-source-epoch-CPU-v1'
IMAGES=ROOT/'data/codenet-image-replay-v1'
REPLAY=ROOT/'data/codenet-replay-v1'
WORDS=ROOT/'data/shared-byte-words-v1'
CONFIG=ROOT/'data/public-reasoning-curriculum-v1/model-config.json'
TOKENIZER=ROOT/'data/tokenizer-corpus/tokenizer.json'
MAIN=ROOT/'reports/public-pretraining-v1/plan.json'
RESERVE=40*1024**3
MAX_NEW=512*1024**2
WIDTH=64
WORLD=4
SEED=987317
EXPECTED={'images:additional_photo_captions':78255,'images:additional_real_words':23847,
    'images:additional_commons':987,'images:original_visual_reasoning':40000,'images:human_photo_reading':5858,
    'images:own_context_photo_reading':5858,'replay:generic_visual':21024,'replay:image_reading':35213,'words':91585}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def atomic(path,value):
    pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists()
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or shutil.disk_usage(ROOT).free<RESERVE+MAX_NEW):
        raise ValueError('Fresh low-priority single-thread CPU source preparation above the original reserve required')
    image_manifest=read(IMAGES/'manifest.json');replay_manifest=read(REPLAY/'manifest.json');word_manifest=read(WORDS/'manifest.json')
    word_admission=read(WORDS/'admission.json');image_admission=read(IMAGES/'source-admission.json')
    if (not image_admission['passed'] or not image_admission['held_roles_excluded']
            or word_admission['training_admitted'] is not True or word_admission['admitted_role']!='train'
            or word_admission['manifest_sha256']!=sha(WORDS/'manifest.json')
            or word_admission['independent_readback_sha256']!=sha(WORDS/'independent-source-readback.json')):
        raise ValueError('Original complete training-role admission receipts required')
    files={}
    for folder,manifest in [(IMAGES,image_manifest),(REPLAY,replay_manifest)]:
        files[str(folder/'manifest.json')]=sha(folder/'manifest.json')
        files.update({str(folder/name):digest for name,digest in manifest['artifact_files_sha256'].items()})
    for item in image_manifest['sources']:
        files[str(Path(item['path']))]=item['sha256'];files[str(Path(item['original_manifest']))]=item['original_manifest_sha256']
        files[str(Path(item['original_file_path']))]=item['original_file_sha256']
    files[str(Path(replay_manifest['original_training_source']))]=replay_manifest['original_training_source_sha256']
    files[str(Path(replay_manifest['original_frozen_plan']))]=replay_manifest['original_frozen_plan_sha256']
    files.update({str(WORDS/name):word_manifest['files_sha256'][name] for name in ['train.jsonl','train.tokens.u16']})
    paths=[WORDS/'manifest.json',WORDS/'admission.json',WORDS/'independent-source-readback.json',CONFIG,TOKENIZER,MAIN,
        ROOT/'reports/own-actor-vision-replay-full-CPU-v1/complete.json',
        ROOT/'reports/own-actor-vision-replay-full-CPU-v1-readback/complete.json',*python_sources(ROOT,[Path(__file__)])]
    files.update({str(p):sha(p) for p in paths})
    if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every original metadata/source/native-token byte must be exact')
    if {image_manifest['tokenizer_sha256'],replay_manifest['tokenizer_sha256'],word_manifest['tokenizer_sha256']}!={sha(TOKENIZER)}:
        raise ValueError('Identical original own BPE for all source views required')
    metadata={p:(Path(p).stat().st_dev,Path(p).stat().st_ino,Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in files}
    def fixed():
        if any((Path(p).stat().st_dev,Path(p).stat().st_ino,Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=m for p,m in metadata.items()):
            raise ValueError('An original source changed during metadata/native reconstruction')
    def checked():wait_for_reserve(ROOT,minimum_bytes=RESERVE,guard=fixed)
    me=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]);OUT.mkdir()
    atomic(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=me,source_files_sha256=files,
        expected_complete_original_source_views=EXPECTED,total_original_train_views=302627,
        source_role='train',purpose='source_preparation_only',image_side=448,word_geometry=[3,64,448],
        proposed_complete_source_epochs=1,proposed_global_width=WIDTH,proposed_world_size=WORLD,seed=SEED,
        minimum_free_bytes=RESERVE,maximum_new_artifact_bytes=MAX_NEW,
        image_hash_fields_retained_as_original_metadata_without_claiming_fresh_pixel_byte_validation=True,
        no_current_training_or_human_RL_or_nine_endpoint_change=True,training_admitted=False,
        no_model_loaded=True,new_production_optimizer_updates=0,goal_achieved=False))
    try:
        image_index=np.load(IMAGES/'index.npy',mmap_mode='r',allow_pickle=False)
        replay_index=np.load(REPLAY/'index.npy',mmap_mode='r',allow_pickle=False)
        if image_index.dtype!=INDEX_DTYPE or replay_index.dtype!=INDEX_DTYPE:raise ValueError('Original native source index schema required')
        tokenizer=Tokenizer.from_file(str(TOKENIZER));config=ModelConfig(**read(CONFIG))
        encoder=FullReplayEncoder(tokenizer,config,ROOT,448)
        lengths={name:[] for name in EXPECTED};targets={name:[] for name in EXPECTED}
        pixel_keys={name:[] for name in EXPECTED};response_keys={name:[] for name in EXPECTED}
        groups={name:set() for name in EXPECTED};images={name:set() for name in EXPECTED}
        sources={name:Counter() for name in EXPECTED};tasks={name:Counter() for name in EXPECTED}
        duplicate_views=Counter();word_bytes=0;word_cursor=0;count=0
        with ExitStack() as stack:
            destination=stack.enter_context((OUT/'all302627-original-native-views.jsonl').open('x'))
            source_streams={}
            def consume(name,position,row,reference,expected_input=None,expected_length=None,expected_targets=None,word_tokens=None):
                nonlocal word_bytes,word_cursor,count
                training_role(row)
                image=Path(row['image']).resolve(strict=True)
                if not image.is_relative_to(ROOT):raise ValueError('Original source image must remain in this project')
                row={**row,'image':str(image)}
                ids,labels=encoder.sequence(row,'response');start=next(i for i,v in enumerate(labels) if v!=-100)
                digest=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()
                if (not 3<=len(ids)<=4096 or ids[0]!=1 or ids[-1]!=2 or any(v<4 for v in ids[1:-1])
                        or labels[:start]!=[-100]*start or labels[start:]!=ids[start:]
                        or expected_input is not None and digest!=expected_input
                        or expected_length is not None and len(ids)!=expected_length
                        or expected_targets is not None and len(ids)-start!=expected_targets):
                    raise ValueError('Every whole original native input, mask and genuine EOS must reconstruct')
                byte_count=None
                if name=='words':
                    raw,steps=native_bytes(row['response']);item=row['native']
                    if (row['split']!='train' or row['native_byte_targets']!=raw or row['minimum_CTC_steps']!=steps
                            or item['offset']!=word_cursor or item['length']!=len(ids) or item['target_start']!=start
                            or word_tokens[word_cursor:word_cursor+len(ids)].tolist()!=ids):
                        raise ValueError('Complete original word transcript, byte alignment and every packed native token required')
                    word_cursor+=len(ids);word_bytes+=len(raw);byte_count=len(raw)
                lengths[name].append(len(ids));targets[name].append(len(ids)-start)
                pixel=row.get('model_pixel_sha256_448');caption_hash=hashlib.sha256(row['response'].encode()).hexdigest()
                pixel_keys[name].append(pixel);response_keys[name].append(caption_hash)
                images[name].add(str(image));groups[name].add(str(row.get('group_id') or 'image:'+str(image)))
                sources[name][str(row.get('source',''))]+=1;tasks[name][str(row.get('task',''))]+=1
                duplicate_views[(str(image),digest)]+=1
                record=dict(family=name,original_local_position=position,original_source_reference=reference,
                    source_id=row.get('id'),source_group=row.get('group_id'),source_role='train',image_path=str(image),
                    original_stored_image_sha256=row.get('stored_image_sha256'),original_RGB448_metadata_sha256=pixel,
                    word_pixel_float_CHW_metadata_sha256=row.get('word_input_pixels_sha256'),
                    native_word_pixel_float_CHW_metadata_sha256=row.get('native_input_pixels_sha256'),
                    input_ids_sha256=digest,complete_context_tokens=len(ids),response_EOS_targets=len(ids)-start,target_start=start,
                    response_utf8_sha256=caption_hash,whole_CTC_byte_targets=byte_count,
                    source_annotation_origin=row.get('annotation_origin',row.get('label_origin')),
                    fresh_pixel_bytes_not_certified_by_this_metadata_record=True,training_admitted=False)
                text=json.dumps(record,ensure_ascii=True)+'\n'
                if destination.tell()+len(text.encode())>MAX_NEW:raise ValueError('Source metadata artifact ceiling exceeded')
                destination.write(text);count+=1
                if count%1024==0:
                    checked();destination.flush();atomic(OUT/'progress.json',dict(owner=me,phase='rebuilding all original native image/word views',
                        complete_original_views=count,total_original_views=302627,family=name,training_admitted=False,new_production_optimizer_updates=0))
            for corpus,index,manifest,names in [(IMAGES,image_index,image_manifest,list(IMAGE_FAMILIES)),
                    (REPLAY,replay_index,replay_manifest,['generic_visual','image_reading'])]:
                for family in names:
                    full=('images:' if corpus==IMAGES else 'replay:')+family
                    number=manifest['family_names'].index(family);positions=np.flatnonzero(index['family']==number)
                    if len(positions)!=EXPECTED[full]:raise ValueError('Every original family population must remain')
                    original=next(s['path'] for s in manifest['sources'] if s['family']==family) if corpus==IMAGES else manifest['original_training_source']
                    if original not in source_streams:source_streams[original]=stack.enter_context(Path(original).open('rb'))
                    stream=source_streams[original]
                    for local,position in enumerate(positions):
                        item=index[position];stream.seek(int(item['offset']));raw=stream.read(int(item['bytes']))
                        if hashlib.sha256(raw).digest()!=item['row_sha256'].tobytes():raise ValueError('Whole original source row bytes differ')
                        row=json.loads(raw)
                        if corpus==IMAGES:image_role(row,family);image=canonical_image(row,original,ROOT)
                        else:
                            if replay_family(row)!=family:raise ValueError('Original replay family/role differs')
                            image=Path(row['image']);image=(image if image.is_absolute() else Path(original).parent/image).resolve(strict=True)
                        consume(full,local,{**row,'image':str(image)},dict(path=original,offset=int(item['offset']),bytes=int(item['bytes']),row_sha256=hashlib.sha256(raw).hexdigest()),
                            expected_input=item['input_sha256'].tobytes().hex(),expected_length=int(item['context_tokens']),expected_targets=int(item['target_tokens']))
            word_tokens=np.memmap(WORDS/'train.tokens.u16',dtype='<u2',mode='r')
            with (WORDS/'train.jsonl').open() as stream:
                for position,line in enumerate(stream):
                    consume('words',position,json.loads(line),dict(path=str(WORDS/'train.jsonl'),JSONL_position=position,
                        original_line_sha256=hashlib.sha256(line.encode()).hexdigest()),word_tokens=word_tokens)
            if word_cursor!=len(word_tokens):raise ValueError('Every original packed word token must be consumed once')
            destination.flush();os.fsync(destination.fileno())
        if {n:len(v) for n,v in lengths.items()}!=EXPECTED or count!=302627:raise ValueError('All nine complete native source families required')
        schedule=CompleteBucketSchedule(lengths,{n:WIDTH for n in lengths},SEED)
        entries=[];seen={n:[] for n in lengths};no_negatives=[];empty_rank_steps=[]
        for step,(name,positions) in enumerate(schedule.entries):
            seen[name].extend(positions);row=dict(step=step+1,family=name,original_local_positions=positions,
                complete_context_tokens=sum(lengths[name][p] for p in positions),response_EOS_targets=sum(targets[name][p] for p in positions),
                proposed_rank_source_counts=[len(positions[r::WORLD]) for r in range(WORLD)])
            if min(row['proposed_rank_source_counts'])==0:empty_rank_steps.append(step+1)
            if name=='images:additional_photo_captions':
                keys=[pixel_keys[name][p] for p in positions];caps=[response_keys[name][p] for p in positions]
                if any(not isinstance(k,str) or len(k)!=64 for k in keys):raise ValueError('Original photograph pixel identities required for matching metadata')
                observed=set(zip(keys,caps));has_negatives=any((a,b) not in observed for a in keys for b in caps)
                row['global_matching_has_observed_negatives']=has_negatives
                if not has_negatives:no_negatives.append(step+1)
            entries.append(row)
        if any(sorted(seen[n])!=list(range(len(lengths[n]))) for n in lengths):raise ValueError('Every source occurrence must appear exactly once without tail dropping')
        atomic(OUT/'complete-single-epoch-schedule.json',dict(seed=SEED,global_width=WIDTH,world_size=WORLD,entries=entries,
            source_roles='all original train',training_admitted=False))
        checked()
        if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every fixed original metadata/native source byte must remain exact')
        budgets={n:dict(original_views=len(v),unique_stored_image_paths=len(images[n]),original_group_keys=len(groups[n]),
            complete_context_tokens=sum(v),response_EOS_targets=sum(targets[n]),minimum_context_tokens=min(v),maximum_context_tokens=max(v),
            source_counts=dict(sources[n]),task_counts=dict(tasks[n])) for n,v in lengths.items()}
        main_plan=read(MAIN);selected=main_plan['selected_source_budget'];main_image={n:selected[n] for n in selected if n in EXPECTED}
        atomic(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),owner=me,source_files_sha256=files,
            all_original_train_views=302627,actual_family_budgets=budgets,actual_whole_CTC_UTF8_byte_targets=word_bytes,
            exact_single_epoch_updates=len(entries),global_width=WIDTH,world_size=WORLD,
            all_original_positions_once=True,partial_final_batches_preserved=True,steps_with_empty_rank_tails=empty_rank_steps,
            photo_batches_without_observed_matching_negatives=no_negatives,
            future_trainer_must_handle_empty_rank_tails_and_degenerate_matching_without_source_dropping=True,
            duplicate_physical_image_and_native_input_views=sum(c-1 for c in duplicate_views.values()),
            duplicate_views_across_loss_recipes_retained_not_counted_as_unique_photos=True,
            distinct_stored_image_paths_across_all_families=len(set().union(*images.values())),
            original_main_selected_image_budgets=main_image,
            original_main_planned_image_presentations=sum(v['presentations'] for v in main_image.values()),
            original_main_planned_image_optimizer_updates=sum(v['updates'] for v in main_image.values()),
            all_original_native_BOS_response_EOS_masks_and_word_bytes_rebuilt=True,
            original_roles_unchanged=True,stored_pixel_hash_protocols_kept_separate=True,
            pixels_not_redecoded_or_all_physical_image_bytes_rehashed=True,
            full_pixel_revalidation_and_GPU_gradient_recovery_capacity_and_quality_pending=True,
            source_metadata_rows_sha256=sha(OUT/'all302627-original-native-views.jsonl'),
            proposed_schedule_sha256=sha(OUT/'complete-single-epoch-schedule.json'),
            new_neural_parameters=0,neural_forwards=0,new_production_optimizer_updates=0,
            original_training_human_RL_and_nine_endpoint_pipeline_unmodified=True,training_admitted=False,goal_achieved=False))
        print(json.dumps(dict(complete=True,all_original_views=count,exact_single_epoch_updates=len(entries),
            photo_batches_without_negatives=len(no_negatives),empty_rank_tail_steps=len(empty_rank_steps),training_admitted=False)),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json',dict(error=repr(error),automatic_retry=False,training_admitted=False,goal_achieved=False));raise


if __name__=='__main__':main()
