"""Fresh complete compact-index/source-reference/schedule readback.

Checks all303k metadata references and raw source byte ranges, every packed
word sequence/byte target, and the four qualified current positive branches.
It does not re-tokenize all image captions or re-decode every physical image.
"""
from collections import Counter
from contextlib import ExitStack
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource

import numpy as np

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.multimodal_source_index_format_v1 import FORMAT,DTYPE

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'reports/full-multimodal-corrected-source-index-CPU-v1'
OUT = ROOT/'reports/full-multimodal-corrected-source-index-v1-readback'
ORIGINAL = ROOT/'reports/full-multimodal-source-epoch-CPU-v1/all302627-original-native-views.jsonl'
WORDS = ROOT/'data/shared-byte-words-v1'
CORRECTIONS = ROOT/'data/public-document-visible-corrections-admitted-v1'
COUNT,WIDTH,WORLD,SEED = 302627,64,4,987317


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
        raise ValueError('Fresh bounded single-thread low-priority CPU complete-index readback required')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0));resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    proof = read(SOURCE/'complete.json');admission = read(CORRECTIONS/'admission.json')
    if (not proof.get('complete') or proof['format'] != FORMAT or proof['training_admitted']
            or proof['all_original_train_views'] != COUNT or proof['actual_changed_native_entries'] != 4
            or not admission['training_admitted']):raise ValueError('Complete source-only corrected index and separately admitted exact four corrections required')
    files = dict(proof['source_files_sha256'])
    index_path = SOURCE/'complete-native-source-index.npy';schedule_path = SOURCE/'complete-corrected-single-epoch-schedule.json'
    paths = [SOURCE/'complete.json',index_path,schedule_path,*python_sources(ROOT,[Path(__file__)])]
    files.update({str(p):sha(p) for p in paths})
    if files[str(index_path)] != proof['index_sha256'] or files[str(schedule_path)] != proof['schedule_sha256']:
        raise ValueError('The full committed compact-index and schedule bytes must be exact')
    if any(sha(p) != digest for p,digest in files.items()):raise ValueError('Every original/admitted source and new index byte must match')
    states = {p:state(p) for p in files}
    def guard():
        if os.sched_getscheduler(0) != os.SCHED_IDLE or any(state(p) != value for p,value in states.items()):
            raise ValueError('Readback priority and all immutable original/current source bytes must remain unchanged')
    def checked():
        wait_for_reserve(ROOT,minimum_bytes=40*1024**3,guard=guard)
        values = {k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
        used = values['anon']+values['shmem']+values['kernel'];limit = int(Path('/sys/fs/cgroup/memory.max').read_text())
        resident = next(int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
        if limit-used < 64*1024**3 or resident > 4*1024**3:raise ValueError('Original bounded data-worker memory and other-process headroom required')
    checked();OUT.mkdir();owner = dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic(OUT/'plan.json',dict(owner=owner,source_files_sha256=files,original_source_occurrences=COUNT,
        actual_source_ranges_and_all_word_sequences_rechecked=True,all_image_captions_not_retokenized_and_pixels_not_redecoded=True,
        schedule_reconstructed_with_separate_stdlib_implementation=True,scheduler='SCHED_IDLE',training_admitted=False,goal_achieved=False))
    try:
        index = np.load(index_path,mmap_mode='r',allow_pickle=False)
        if index.dtype != DTYPE or index.shape != (COUNT,):raise ValueError('Every original compact source entry and exact schema required')
        names = proof['family_names'];expected = {n:proof['actual_family_budgets'][n]['original_views'] for n in names}
        corrections = [json.loads(line) for line in (CORRECTIONS/'all4-complete-corrections-and-preferences.jsonl').open()]
        if len(corrections) != 4 or any(not row['training_admitted'] or row['source_role'] != 'train' for row in corrections):raise ValueError('Every exact original train correction required')
        correction_map = {(r['original_prepared_native_view']['family'],r['original_prepared_native_view']['original_local_position']):i for i,r in enumerate(corrections)}
        word_tokens = np.memmap(WORDS/'train.tokens.u16',dtype='<u2',mode='r')
        lengths = {n:[] for n in names};targets = {n:[] for n in names};keys = [];caption_hashes = []
        counts = Counter();images = set();duplicates = Counter();changed = [];raw_bytes = 0;word_cursor = word_bytes = total = 0
        with ExitStack() as stack:
            metadata = stack.enter_context(ORIGINAL.open('rb'));streams = {}
            for number,entry in enumerate(index):
                offset = metadata.tell();line = metadata.readline();row = json.loads(line);name = row['family'];position = row['original_local_position']
                if (not line.endswith(b'\n') or int(entry['metadata_offset']) != offset or int(entry['metadata_bytes']) != len(line)
                        or entry['metadata_sha256'].tobytes().hex() != hashlib.sha256(line).hexdigest()
                        or names[int(entry['family'])] != name or int(entry['original_local_position']) != position
                        or position != counts[name] or row['source_role'] != 'train' or row['training_admitted']):
                    raise ValueError('Each exact complete original metadata row/range/hash/family/local position must remain train-only')
                reference = row['original_source_reference'];source_path = Path(reference['path'])
                if str(source_path) not in streams:streams[str(source_path)] = stack.enter_context(source_path.open('rb'))
                stream = streams[str(source_path)];stream.seek(int(entry['source_offset']));raw = stream.read(int(entry['source_bytes']))
                digest = hashlib.sha256(raw).hexdigest();raw_bytes += len(raw)
                if digest != entry['source_sha256'].tobytes().hex():raise ValueError('Every compact raw-source byte range must freshly reproduce its complete original SHA')
                if name == 'words':
                    if (reference['JSONL_position'] != position or reference['original_line_sha256'] != digest
                            or source_path != WORDS/'train.jsonl'):raise ValueError('Every whole original word JSONL position/range must reproduce')
                    word = json.loads(raw);native = word['native'];values = list(word['response'].encode())
                    required = len(values)+sum(a == b for a,b in zip(values,values[1:]))
                    seq = word_tokens[native['offset']:native['offset']+native['length']]
                    if (word['split'] != 'train' or word['task'] != 'word_ocr' or native['offset'] != word_cursor
                            or not 1 <= len(values) <= 64 or required > 140 or word['native_byte_targets'] != values
                            or word['minimum_CTC_steps'] != required or row['whole_CTC_byte_targets'] != len(values)
                            or native['length'] != row['complete_context_tokens'] or native['target_start'] != row['target_start']
                            or len(seq) != native['length'] or int(seq[0]) != 1 or int(seq[-1]) != 2 or np.any(seq[1:-1] < 4)
                            or hashlib.sha256(seq.tobytes()).hexdigest() != row['input_ids_sha256']):
                        raise ValueError('All packed original word sequences, native prompt/response boundaries and complete UTF8/CTC bytes required')
                    word_cursor += len(seq);word_bytes += len(values)
                elif (int(entry['source_offset']) != reference['offset'] or int(entry['source_bytes']) != reference['bytes'] or digest != reference['row_sha256']):
                    raise ValueError('Every original image-source reference must remain exact')
                correction = correction_map.get((name,position),-1)
                wanted_length,wanted_start,wanted_targets,wanted_hash = row['complete_context_tokens'],row['target_start'],row['response_EOS_targets'],row['input_ids_sha256']
                if correction >= 0:
                    item = corrections[correction];chosen = item['positive_native'];negative = item['complete_original_negative_native']
                    if item['original_prepared_native_view'] != row or negative['input_ids_sha256'] != row['input_ids_sha256']:
                        raise ValueError('The qualified current positive must retain its complete exact original occurrence/negative')
                    ids = chosen['input_ids'];start = chosen['target_start']
                    if (chosen['labels'] != [-100]*start+ids[start:] or ids[0] != 1 or ids[-1] != 2
                            or hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest() != chosen['input_ids_sha256']):
                        raise ValueError('Every corrected whole native target/EOS and mask must remain exact')
                    wanted_length,wanted_start,wanted_targets,wanted_hash = len(ids),start,chosen['response_EOS_targets'],chosen['input_ids_sha256'];changed.append(item['source_id'])
                if (int(entry['correction_index']) != correction or int(entry['context_tokens']) != wanted_length
                        or int(entry['target_start']) != wanted_start or int(entry['response_EOS_targets']) != wanted_targets
                        or entry['input_ids_sha256'].tobytes().hex() != wanted_hash or wanted_targets != wanted_length-wanted_start):
                    raise ValueError('Only the exact four admitted replacements may alter current native values or masks')
                lengths[name].append(wanted_length);targets[name].append(wanted_targets);counts[name] += 1
                images.add(row['image_path']);duplicates[(row['image_path'],wanted_hash)] += 1
                if name == 'images:additional_photo_captions':keys.append(row['original_RGB448_metadata_sha256']);caption_hashes.append(row['response_utf8_sha256'])
                total += 1
                if total%4096 == 0:checked();atomic(OUT/'progress.json',dict(owner=owner,all_original_entries_rechecked=total,total=COUNT,
                    current_positive_replacements_rechecked=len(changed),word_UTF8_bytes_rechecked=word_bytes,training_admitted=False))
            if metadata.readline():raise ValueError('No original metadata tail may be omitted')
        if (dict(counts) != expected or total != COUNT or len(changed) != 4 or len(set(changed)) != 4
                or word_cursor != len(word_tokens) or word_bytes != 644063 or len(images) != 195121
                or sum(sum(v) for v in lengths.values()) != 17681556 or sum(sum(v) for v in targets.values()) != 4669243
                or sum(c-1 for c in duplicates.values()) != proof['actual_duplicate_physical_image_and_native_input_views']):
            raise ValueError('Every original occurrence, word/native/byte target, physical path and exact60target correction delta required')
        entries = []
        for number,name in enumerate(sorted(names)):
            rng = random.Random(SEED+number*1009);positions = list(range(expected[name]));rng.shuffle(positions);batches = []
            for start in range(0,len(positions),WIDTH*32):
                bucket = sorted(positions[start:start+WIDTH*32],key=lambda p:(lengths[name][p],p))
                batches.extend([bucket[i:i+WIDTH] for i in range(0,len(bucket),WIDTH)])
            rng.shuffle(batches)
            for occurrence,batch in enumerate(batches):entries.append((Fraction(2*occurrence+1,2*len(batches)),number,name,batch))
        entries.sort();schedule = read(schedule_path);seen = {n:Counter() for n in names};empty = [];photos = 0
        if schedule['seed'] != SEED or schedule['global_width'] != WIDTH or schedule['world_size'] != WORLD or len(schedule['entries']) != 4733:
            raise ValueError('Exact complete one-pass four-rank schedule required')
        for step,(_,_,name,positions) in enumerate(entries,1):
            row = schedule['entries'][step-1]
            rank_counts = [len(positions[r::WORLD]) for r in range(WORLD)]
            if (row['step'] != step or row['family'] != name or row['original_local_positions'] != positions
                    or row['complete_context_tokens'] != sum(lengths[name][p] for p in positions)
                    or row['response_EOS_targets'] != sum(targets[name][p] for p in positions) or row['proposed_rank_source_counts'] != rank_counts):
                raise ValueError('Every independently reconstructed complete source batch, target budget and rank tail must match')
            seen[name].update(positions)
            if not min(rank_counts):empty.append(step)
            if name == 'images:additional_photo_captions':
                observed = {(keys[p],caption_hashes[p]) for p in positions}
                if not row['global_matching_has_observed_negatives'] or not any((keys[a],caption_hashes[b]) not in observed for a in positions for b in positions):
                    raise ValueError('Every unchanged photo matching batch must retain observed negatives')
                photos += 1
        if (len(entries) != 4733 or photos != 1223 or empty != [3205]
                or any(seen[n] != Counter(range(expected[n])) for n in names)):
            raise ValueError('All303k occurrences must appear once, including the singleton word tail')
        checked()
        # Sync the already immutable newly produced data index before recording
        # successful readback. This changes no source bytes or source metadata.
        with index_path.open('rb') as stream:os.fsync(stream.fileno())
        if any(sha(p) != digest for p,digest in files.items()):raise ValueError('Every original/qualified/current byte must remain unchanged')
        atomic(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
            actual_all_original_entries=COUNT,actual_family_views=dict(counts),actual_changed_source_ids=changed,
            actual_current_positive_native_entries_changed=4,actual_original_native_entries_unchanged=COUNT-4,
            actual_all_complete_raw_source_ranges_hashed=True,actual_original_source_bytes_read=raw_bytes,
            actual_all91585packed_word_sequences_and_complete_CTC_bytes_rechecked=True,actual_whole_word_UTF8_bytes=word_bytes,
            actual_context_tokens=17681556,actual_response_EOS_targets=4669243,
            all80original_negative_targets_preserved_in_admitted_correction_data=True,
            actual_all_original_occurrences_once=True,actual_epoch_updates=4733,actual_photo_matching_batches=photos,actual_empty_rank_tail_steps=empty,
            actual_distinct_stored_image_paths=len(images),actual_native_duplicate_views=sum(c-1 for c in duplicates.values()),
            actual_index_fsynced_without_changing_bytes=True,
            all_image_captions_not_retokenized_and_physical_pixels_not_redecoded_by_this_reader=True,
            full_physical_pixel_and_full_size_GPU_capacity_recovery_training_quality_pending=True,
            existing_original_training_and_human_RL_and_nine_endpoint_scopes_unmodified=True,
            training_admitted=False,neural_forwards=0,new_production_optimizer_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=True,original_entries_rechecked=COUNT,current_corrections_rechecked=4,
            actual_epoch_updates=4733,actual_response_EOS_targets=4669243,training_admitted=False)),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json',dict(owner=owner,error=repr(error),automatic_retry=False,training_admitted=False,goal_achieved=False));raise


if __name__ == '__main__':main()
