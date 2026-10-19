"""Index every original visual/word view, applying four admitted corrections.

The full original metadata is referenced by byte offset/hash, not rewritten.
Every original source occurrence and complete original negative is preserved.
No neural model, GPU context or existing source/recipe is changed.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import resource

import numpy as np

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.fast_joint_data_v1 import CompleteBucketSchedule
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.multimodal_source_index_format_v1 import FORMAT,DTYPE

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'reports/full-multimodal-source-epoch-CPU-v1'
ORIGINAL = SOURCE/'all302627-original-native-views.jsonl'
CORRECTIONS = ROOT/'data/public-document-visible-corrections-admitted-v1'
OUT = ROOT/'reports/full-multimodal-corrected-source-index-CPU-v1'
WORDS = ROOT/'data/shared-byte-words-v1/train.jsonl'
COUNT,WIDTH,WORLD,SEED = 302627,64,4,987317
RESERVE,MAX_OUTPUT = 40*1024**3,128*1024**2
WORKING,OTHER_HEADROOM = 4*1024**3,64*1024**3


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):value.update(block)
    return value.hexdigest()


def state(path):
    value = Path(path).stat();return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)


def atomic(path,value):
    pending = path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh complete-source low-priority single-thread CUDA-hidden index preparation required')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0))
    original = read(SOURCE/'complete.json');reader = read(ROOT/'reports/full-multimodal-source-epoch-v1-readback/complete.json')
    admission = read(CORRECTIONS/'admission.json');qualified = read(admission['independent_readback'])
    data_file = CORRECTIONS/'all4-complete-corrections-and-preferences.jsonl'
    if (not original.get('complete') or not reader.get('complete') or original['training_admitted'] or reader['training_admitted']
            or original['all_original_train_views'] != COUNT or original['exact_single_epoch_updates'] != 4733
            or not admission['training_admitted'] or admission['admitted_role'] != 'train'
            or admission['independent_readback_sha256'] != sha(admission['independent_readback'])
            or admission['data_file_sha256'] != sha(data_file) or qualified['exact_admitted_original_views'] != 4):
        raise ValueError('Complete original source/mask readback and exactly four independently admitted source corrections required')
    names = list(original['actual_family_budgets']);expected = {n:original['actual_family_budgets'][n]['original_views'] for n in names}
    files = {**original['source_files_sha256'],**reader['source_files_sha256'],**qualified['source_files_sha256']}
    paths = [SOURCE/'complete.json',ROOT/'reports/full-multimodal-source-epoch-v1-readback/complete.json',ORIGINAL,
        SOURCE/'complete-single-epoch-schedule.json',CORRECTIONS/'admission.json',Path(admission['independent_readback']),data_file,
        *python_sources(ROOT,[Path(__file__)])]
    files.update({str(p):sha(p) for p in paths})
    if files[str(ORIGINAL)] != original['source_metadata_rows_sha256']:raise ValueError('Original full metadata bytes must remain exact')
    states = {p:state(p) for p in files};waits = Counter()
    def guard():
        if os.sched_getscheduler(0) != os.SCHED_IDLE or any(state(p) != value for p,value in states.items()):
            raise ValueError('Only this new idle data worker and unchanged original/corrective sources allowed')
    def checked():
        result = wait_for_reserve(ROOT,minimum_bytes=RESERVE,guard=guard)
        waits['seconds'] += result['waited_seconds'];waits['failed_space_observations'] += result['failed_space_observations']
        memory = {k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
        active = memory['anon']+memory['shmem']+memory['kernel'];limit = int(Path('/sys/fs/cgroup/memory.max').read_text())
        resident = next(int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
        if limit-active < OTHER_HEADROOM or resident > WORKING:raise ValueError('Original data-worker memory and other-process headroom must remain available')
    checked();OUT.mkdir();owner = dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic(OUT/'plan.json',dict(format=FORMAT,at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        exact_original_native_views=COUNT,family_names=names,all_original_family_views=expected,
        dtype=DTYPE.descr,actual_index_bytes=COUNT*DTYPE.itemsize,maximum_output_bytes=MAX_OUTPUT,
        minimum_free_bytes=RESERVE,CPU_working_allowance=WORKING,other_process_memory_headroom=OTHER_HEADROOM,
        parent_address_space_ceiling=16*1024**3,scheduler='SCHED_IDLE',global_width=WIDTH,world_size=WORLD,seed=SEED,
        exactly_four_qualified_positive_replacements_with_whole_original_negatives=True,
        no_existing_source_or_recipe_or_neural_worker_change=True,training_admitted=False,goal_achieved=False))
    try:
        resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
        if any(sha(p) != digest for p,digest in files.items()):raise ValueError('Every original/corrective/native/runtime byte must match before index preparation')
        corrections = [json.loads(line) for line in data_file.open()]
        if len(corrections) != 4 or any(not row['training_admitted'] or row['source_role'] != 'train' for row in corrections):
            raise ValueError('All four exact admitted train correction records required')
        lookup = {(row['original_prepared_native_view']['family'],row['original_prepared_native_view']['original_local_position']):(i,row) for i,row in enumerate(corrections)}
        if len(lookup) != 4:raise ValueError('Each admitted correction must map to one distinct original family occurrence')
        index_path = OUT/'complete-native-source-index.npy';index = np.lib.format.open_memmap(index_path,mode='w+',dtype=DTYPE,shape=(COUNT,))
        lengths = {n:[] for n in names};targets = {n:[] for n in names};keys = {n:[] for n in names};texts = {n:[] for n in names}
        images = set();duplicate = Counter();counts = Counter();changed = [];word_bytes = words = total = 0
        with ORIGINAL.open('rb') as stream,WORDS.open('rb') as word_stream:
            while True:
                offset = stream.tell();line = stream.readline()
                if not line:break
                if not line.endswith(b'\n') or total >= COUNT:raise ValueError('Only the exact complete original native metadata stream allowed')
                row = json.loads(line);name = row['family'];position = row['original_local_position']
                if name not in expected or position != counts[name] or row['source_role'] != 'train' or row['training_admitted']:
                    raise ValueError('Every original family/local position and source role must remain exact')
                ref = row['original_source_reference']
                if name == 'words':
                    source_offset = word_stream.tell();raw = word_stream.readline();source_bytes = len(raw);source_sha = hashlib.sha256(raw).hexdigest()
                    if (ref != dict(path=str(WORDS),JSONL_position=position,original_line_sha256=source_sha)
                            or row['whole_CTC_byte_targets'] is None):raise ValueError('Every exact original word source byte and complete byte target count required')
                    word_bytes += row['whole_CTC_byte_targets'];words += 1
                else:source_offset,source_bytes,source_sha = ref['offset'],ref['bytes'],ref['row_sha256']
                length,target_start,target,input_sha = row['complete_context_tokens'],row['target_start'],row['response_EOS_targets'],row['input_ids_sha256']
                correction = -1
                if (name,position) in lookup:
                    correction,item = lookup[(name,position)]
                    if item['original_prepared_native_view'] != row or item['complete_original_negative_native']['input_ids_sha256'] != input_sha:
                        raise ValueError('The whole original occurrence/negative must exactly match its qualified correction')
                    chosen = item['positive_native'];length,target_start,target,input_sha = len(chosen['input_ids']),chosen['target_start'],chosen['response_EOS_targets'],chosen['input_ids_sha256']
                    changed.append(dict(original_global_position=total,family=name,original_local_position=position,source_id=row['source_id'],correction_index=correction,
                        old_context_tokens=row['complete_context_tokens'],new_context_tokens=length,
                        old_response_EOS_targets=row['response_EOS_targets'],new_response_EOS_targets=target,
                        whole_original_negative_targets=item['complete_original_negative_native']['response_EOS_targets']))
                if not 1 <= target_start < length <= 4096 or target != length-target_start or not 1 <= source_bytes < 2**32:
                    raise ValueError('Every complete native prompt/response/EOS mask and original source byte range must remain valid')
                index[total] = (offset,len(line),bytes.fromhex(hashlib.sha256(line).hexdigest()),source_offset,source_bytes,bytes.fromhex(source_sha),
                    names.index(name),position,length,target_start,target,bytes.fromhex(input_sha),correction)
                lengths[name].append(length);targets[name].append(target);keys[name].append(row['original_RGB448_metadata_sha256'])
                response_sha = hashlib.sha256(corrections[correction]['corrected_response'].encode()).hexdigest() if correction >= 0 else row['response_utf8_sha256']
                texts[name].append(response_sha);images.add(row['image_path']);duplicate[(row['image_path'],input_sha)] += 1
                counts[name] += 1;total += 1
                if total%4096 == 0:
                    checked();index.flush();atomic(OUT/'progress.json',dict(owner=owner,original_views_indexed=total,total=COUNT,
                        actual_corrections_indexed=len(changed),actual_complete_words=words,training_admitted=False))
            if word_stream.readline():raise ValueError('No original word source tail may be omitted')
        index.flush();del index
        if (dict(counts) != expected or total != COUNT or len(changed) != 4 or words != 91585 or word_bytes != 644063
                or sum(sum(v) for v in targets.values()) != 4669243 or sum(sum(v) for v in lengths.values()) != 17681556
                or len(images) != 195121):
            raise ValueError('All original source occurrences/images/words and exactly the qualified60target correction delta required')
        schedule = CompleteBucketSchedule(lengths,{n:WIDTH for n in names},SEED);old_schedule = read(SOURCE/'complete-single-epoch-schedule.json')['entries']
        entries = [];seen = {n:[] for n in names};empty = [];photo_batches = 0;modified_batches = 0
        for step,(name,positions) in enumerate(schedule.entries):
            old = old_schedule[step]
            if name != old['family']:raise ValueError('Every original family interleave/update population must remain unchanged')
            if positions != old['original_local_positions']:
                if name != 'replay:image_reading':raise ValueError('Only corrected document length buckets may change source order')
                modified_batches += 1
            seen[name].extend(positions)
            item = dict(step=step+1,family=name,original_local_positions=positions,
                complete_context_tokens=sum(lengths[name][p] for p in positions),response_EOS_targets=sum(targets[name][p] for p in positions),
                proposed_rank_source_counts=[len(positions[r::WORLD]) for r in range(WORLD)])
            if min(item['proposed_rank_source_counts']) == 0:empty.append(step+1)
            if name == 'images:additional_photo_captions':
                pixel = [keys[name][p] for p in positions];caption = [texts[name][p] for p in positions];observed = set(zip(pixel,caption))
                has_negatives = any((a,b) not in observed for a in pixel for b in caption)
                if not has_negatives or not old['global_matching_has_observed_negatives']:raise ValueError('Every original photo batch must retain observed full-batch matching negatives')
                item['global_matching_has_observed_negatives'] = has_negatives;photo_batches += 1
            entries.append(item)
        if (len(entries) != 4733 or photo_batches != 1223 or empty != [3205]
                or any(sorted(seen[n]) != list(range(expected[n])) for n in names)):
            raise ValueError('All original4733updates,1223photo batches and complete singleton word tail must remain')
        schedule_path = OUT/'complete-corrected-single-epoch-schedule.json'
        atomic(schedule_path,dict(seed=SEED,global_width=WIDTH,world_size=WORLD,entries=entries,all_original_source_occurrences_once=True,training_admitted=False))
        checked()
        if sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) > MAX_OUTPUT:raise ValueError('The declared compact-index artifact ceiling was exceeded')
        if any(sha(p) != digest for p,digest in files.items()):raise ValueError('Every original/qualified source byte must remain unchanged at completion')
        budgets = {n:dict(original_views=len(lengths[n]),complete_context_tokens=sum(lengths[n]),response_EOS_targets=sum(targets[n]),
            minimum_context_tokens=min(lengths[n]),maximum_context_tokens=max(lengths[n])) for n in names}
        atomic(OUT/'complete.json',dict(complete=True,format=FORMAT,at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
            family_names=names,all_original_train_views=COUNT,actual_family_budgets=budgets,actual_positive_correction_records=changed,
            actual_all_original_positions_once=True,actual_changed_native_entries=4,actual_unchanged_native_entries=COUNT-4,
            actual_context_tokens=17681556,actual_response_EOS_targets=4669243,actual_original_negative_targets_retained=80,
            actual_word_source_rows=words,actual_whole_CTC_UTF8_byte_targets=word_bytes,actual_distinct_stored_image_paths=len(images),
            actual_duplicate_physical_image_and_native_input_views=sum(c-1 for c in duplicate.values()),
            actual_complete_epoch_updates=len(entries),actual_photo_matching_batches_with_negatives=photo_batches,
            actual_empty_rank_tail_steps=empty,actual_length_bucket_batches_with_changed_positions=modified_batches,
            actual_index_bytes=index_path.stat().st_size,index_sha256=sha(index_path),schedule_sha256=sha(schedule_path),
            original_source_metadata_referenced_not_rewritten=True,original_roles_and_physical_images_and_all_old_negative_targets_preserved=True,
            full_index_readback_and_full_pixel_and_GPU_capacity_recovery_training_quality_pending=True,
            existing_main_and_human_RL_and_nine_endpoint_scopes_unchanged=True,training_admitted=False,
            neural_forwards=0,new_production_optimizer_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=True,all_original_views=total,actual_corrections=4,actual_epoch_updates=len(entries),
            actual_response_EOS_targets=4669243,actual_index_bytes=index_path.stat().st_size,training_admitted=False)),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json',dict(owner=owner,error=repr(error),automatic_retry=False,training_admitted=False,goal_achieved=False));raise


if __name__ == '__main__':main()
