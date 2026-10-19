"""Fresh source/index/token readback of every proposed multimodal train view.

This independently checks coverage and matching via original text associations.
Pixel arrays and distributed NN capacity remain unverified by this reader.
"""
from collections import Counter
from contextlib import ExitStack
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.codenet_image_replay_v1 import image_role
from slm_lab.codenet_replay_v1 import family as replay_family,training_role
from slm_lab.own_actor_vision_alignment_v1 import pair_positives
from prepare_full_multimodal_source_epoch_CPU_v1 import sha,read,atomic,ROOT,EXPECTED,WIDTH,WORLD,WORDS,IMAGES,REPLAY,TOKENIZER

SOURCE=ROOT/'reports/full-multimodal-source-epoch-CPU-v1'
OUT=ROOT/'reports/full-multimodal-source-epoch-v1-readback'
RESERVE=40*1024**3


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists()
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh bounded one-thread low-priority CPU native-source readback required')
    proof=read(SOURCE/'complete.json');schedule=read(SOURCE/'complete-single-epoch-schedule.json')
    if (not proof.get('complete') or proof['training_admitted'] or proof['all_original_train_views']!=302627
            or proof['exact_single_epoch_updates']!=4733 or schedule['global_width']!=64 or schedule['world_size']!=4):
        raise ValueError('The complete original unadmitted nine-family source inventory required')
    files=dict(proof['source_files_sha256'])
    for p in [Path(__file__),SOURCE/'complete.json',SOURCE/'all302627-original-native-views.jsonl',SOURCE/'complete-single-epoch-schedule.json',
            ROOT/'src/slm_lab/own_actor_vision_alignment_v1.py']:files[str(p)]=sha(p)
    if (sha(SOURCE/'all302627-original-native-views.jsonl')!=proof['source_metadata_rows_sha256']
            or sha(SOURCE/'complete-single-epoch-schedule.json')!=proof['proposed_schedule_sha256']
            or any(sha(p)!=digest for p,digest in files.items())):
        raise ValueError('Every source/index/packed token and prepared byte must remain exact')
    metadata={p:(Path(p).stat().st_dev,Path(p).stat().st_ino,Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in files}
    def guard():
        if any((Path(p).stat().st_dev,Path(p).stat().st_ino,Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=m for p,m in metadata.items()):
            raise ValueError('A frozen original source changed during fresh readback')
    def checked():wait_for_reserve(ROOT,minimum_bytes=RESERVE,guard=guard)
    OUT.mkdir();atomic(OUT/'plan.json',dict(source_files_sha256=files,all_original_native_views=302627,
        source_role='train',pixel_bytes_and_neural_capacity_not_revalidated=True,training_admitted=False,goal_achieved=False))
    try:
        catalogs={}
        for prefix,folder in [('images',IMAGES),('replay',REPLAY)]:
            manifest=read(folder/'manifest.json');index=np.load(folder/'index.npy',mmap_mode='r',allow_pickle=False)
            for name in manifest['family_names']:
                full=prefix+':'+name
                if full not in EXPECTED:continue
                original=next(r['path'] for r in manifest['sources'] if r['family']==name) if prefix=='images' else manifest['original_training_source']
                catalogs[full]=(index[np.flatnonzero(index['family']==manifest['family_names'].index(name))],original)
        tokenizer=Tokenizer.from_file(str(TOKENIZER));word_tokens=np.memmap(WORDS/'train.tokens.u16',dtype='<u2',mode='r')
        counts=Counter();contexts=Counter();targets=Counter();all_images=set();duplicate=Counter()
        lengths={n:[] for n in EXPECTED};target_lists={n:[] for n in EXPECTED};photo_keys=[];photo_captions=[]
        total=0;word_cursor=0;word_bytes=0
        with ExitStack() as stack:
            word_stream=stack.enter_context((WORDS/'train.jsonl').open('rb'));streams={}
            source=stack.enter_context((SOURCE/'all302627-original-native-views.jsonl').open())
            for line in source:
                item=json.loads(line);name=item['family'];position=item['original_local_position']
                if name not in EXPECTED or position!=counts[name] or item['source_role']!='train' or item['training_admitted']:
                    raise ValueError('Every original family/local position must remain train-only and in order')
                if name=='words':
                    raw=word_stream.readline();row=json.loads(raw)
                    if item['original_source_reference']!=dict(path=str(WORDS/'train.jsonl'),JSONL_position=position,
                            original_line_sha256=hashlib.sha256(raw).hexdigest()):raise ValueError('Whole original word row bytes required')
                    native=row['native'];seq=word_tokens[native['offset']:native['offset']+native['length']]
                    if native['offset']!=word_cursor or native['length']!=item['complete_context_tokens'] or native['target_start']!=item['target_start']:
                        raise ValueError('All packed word positions and masks must be contiguous and exact')
                    digest=hashlib.sha256(seq.tobytes()).hexdigest();word_cursor+=len(seq)
                    expected_prompt=row['prompt'] if row['prompt'].rstrip().endswith('Assistant:') else 'User: '+row['prompt']+'\nAssistant:'
                    if (row['task']!='word_ocr' or int(seq[0])!=1 or int(seq[-1])!=2 or np.any(seq[1:-1]<4)
                            or tokenizer.decode(seq[1:native['target_start']].tolist(),skip_special_tokens=False)!=expected_prompt
                            or tokenizer.decode(seq[native['target_start']:-1].tolist(),skip_special_tokens=False)!=row['response']):
                        raise ValueError('Inverse native word BPE decoding must reproduce every full prompt and original transcript')
                    raw_bytes=list(row['response'].encode());required=len(raw_bytes)+sum(a==b for a,b in zip(raw_bytes,raw_bytes[1:]))
                    if (raw_bytes!=row['native_byte_targets'] or not 0<len(raw_bytes)<=64 or required!=row['minimum_CTC_steps']
                            or required>140 or item['whole_CTC_byte_targets']!=len(raw_bytes)):
                        raise ValueError('All original UTF8 bytes and repeated-byte CTC alignment must independently reconstruct')
                    word_bytes+=len(raw_bytes);expected_context=native['length'];expected_target=native['length']-native['target_start']
                    image=Path(row['image']).resolve(strict=True)
                else:
                    index,original=catalogs[name];entry=index[position]
                    if original not in streams:streams[original]=stack.enter_context(Path(original).open('rb'))
                    stream=streams[original];stream.seek(int(entry['offset']));raw=stream.read(int(entry['bytes']));row=json.loads(raw)
                    expected_reference=dict(path=original,offset=int(entry['offset']),bytes=int(entry['bytes']),row_sha256=hashlib.sha256(raw).hexdigest())
                    if item['original_source_reference']!=expected_reference or hashlib.sha256(raw).digest()!=entry['row_sha256'].tobytes():
                        raise ValueError('The whole original source row/index references must independently match')
                    family=name.split(':',1)[1]
                    if name.startswith('images:'):image_role(row,family)
                    elif replay_family(row)!=family:raise ValueError('Original replay training family changed')
                    image=Path(row['image']);image=(image if image.is_absolute() else Path(original).parent/image).resolve(strict=True)
                    digest=entry['input_sha256'].tobytes().hex();expected_context=int(entry['context_tokens']);expected_target=int(entry['target_tokens'])
                    if item['target_start']!=expected_context-expected_target:raise ValueError('Complete original response/EOS mask start required')
                training_role(row)
                if (str(image)!=item['image_path'] or row.get('id')!=item['source_id'] or row.get('group_id')!=item['source_group']
                        or digest!=item['input_ids_sha256'] or expected_context!=item['complete_context_tokens'] or expected_target!=item['response_EOS_targets']
                        or hashlib.sha256(row['response'].encode()).hexdigest()!=item['response_utf8_sha256']
                        or row.get('stored_image_sha256')!=item['original_stored_image_sha256']
                        or row.get('model_pixel_sha256_448')!=item['original_RGB448_metadata_sha256']
                        or row.get('word_input_pixels_sha256')!=item['word_pixel_float_CHW_metadata_sha256']
                        or row.get('native_input_pixels_sha256')!=item['native_word_pixel_float_CHW_metadata_sha256']
                        or not item['fresh_pixel_bytes_not_certified_by_this_metadata_record']):
                    raise ValueError('Original IDs/groups, native hashes, counts and separate pixel protocol metadata must match')
                counts[name]+=1;contexts[name]+=expected_context;targets[name]+=expected_target
                lengths[name].append(expected_context);target_lists[name].append(expected_target)
                all_images.add(str(image));duplicate[(str(image),digest)]+=1
                if name=='images:additional_photo_captions':photo_keys.append(row['model_pixel_sha256_448']);photo_captions.append(row['response'])
                total+=1
                if total%4096==0:checked();atomic(OUT/'progress.json',dict(original_views_rechecked=total,total=302627,training_admitted=False))
            if word_stream.readline():raise ValueError('No original word source tail may be omitted')
        if dict(counts)!=EXPECTED or total!=302627 or word_cursor!=len(word_tokens) or word_bytes!=proof['actual_whole_CTC_UTF8_byte_targets']:
            raise ValueError('Every source view and word byte must independently reconstruct')
        for name in EXPECTED:
            budget=proof['actual_family_budgets'][name]
            if contexts[name]!=budget['complete_context_tokens'] or targets[name]!=budget['response_EOS_targets']:
                raise ValueError('All original family native budgets must reconstruct')
        seen={name:Counter() for name in EXPECTED};empty=[];no_negative=[];per_family=Counter()
        for step,row in enumerate(schedule['entries'],1):
            name=row['family'];positions=row['original_local_positions']
            if (row['step']!=step or not 1<=len(positions)<=WIDTH or len(set(positions))!=len(positions)
                    or any(type(p) is not int or not 0<=p<EXPECTED[name] for p in positions)
                    or row['complete_context_tokens']!=sum(lengths[name][p] for p in positions)
                    or row['response_EOS_targets']!=sum(target_lists[name][p] for p in positions)
                    or row['proposed_rank_source_counts']!=[len(positions[r::WORLD]) for r in range(WORLD)]):
                raise ValueError('All whole global batches, native budgets and four-rank tail counts must reconstruct')
            seen[name].update(positions);per_family[name]+=1
            if min(row['proposed_rank_source_counts'])==0:empty.append(step)
            if name=='images:additional_photo_captions':
                positive=pair_positives([photo_keys[p] for p in positions],[photo_captions[p] for p in positions])
                actual=not bool(positive.all())
                if actual!=row['global_matching_has_observed_negatives']:raise ValueError('Canonical exact-text associations must independently reproduce global negative availability')
                if not actual:no_negative.append(step)
        if (len(schedule['entries'])!=4733 or any(seen[n]!=Counter(range(EXPECTED[n])) for n in EXPECTED)
                or empty!=proof['steps_with_empty_rank_tails'] or no_negative!=proof['photo_batches_without_observed_matching_negatives']
                or len(all_images)!=proof['distinct_stored_image_paths_across_all_families']
                or sum(c-1 for c in duplicate.values())!=proof['duplicate_physical_image_and_native_input_views']):
            raise ValueError('All source occurrences, duplicates, tail and matching cases must independently reconstruct')
        checked()
        if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every original source/native/index byte must remain exact')
        atomic(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            actual_original_views_rechecked=total,actual_family_views=dict(counts),actual_epoch_updates=4733,actual_family_updates=dict(per_family),
            actual_all_original_positions_once=True,actual_all_native_hashes_and_masks_match_original_committed_sources=True,
            all91585word_native_sequences_inverse_decoded_without_clipping=True,actual_complete_CTC_UTF8_bytes=word_bytes,
            canonical_exact_text_photo_positive_masks_rechecked=per_family['images:additional_photo_captions'],
            photo_batches_without_observed_negatives=no_negative,actual_empty_rank_tail_steps=empty,
            actual_distinct_stored_image_paths=len(all_images),actual_duplicate_native_views=sum(c-1 for c in duplicate.values()),
            full_pixel_bytes_geometry_and_GPU_capacity_and_training_and_quality_pending=True,
            source_roles_preserved_and_no_held_sources_loaded=True,training_admitted=False,
            new_neural_parameters=0,neural_forwards=0,new_production_optimizer_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=True,actual_all_original_views=total,actual_epoch_updates=4733,
            photo_positive_masks_rechecked=per_family['images:additional_photo_captions'],training_admitted=False)),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json',dict(error=repr(error),automatic_retry=False,training_admitted=False,goal_achieved=False));raise


if __name__=='__main__':main()
