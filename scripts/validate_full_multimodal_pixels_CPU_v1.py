"""Read every stored image behind the complete nine-family source epoch.

This is a CPU-only integrity/geometry audit, with no neural forwards or source
edits. Separate original hash protocols remain separate. Passing this audit
does not establish semantic label quality or admit a new training run.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.multimodal_pixel_protocols_v1 import file_state, merge_expectations, observe_pixels, mismatches

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'reports/full-multimodal-source-epoch-CPU-v1'
READBACK = ROOT/'reports/full-multimodal-source-epoch-v1-readback/complete.json'
OUT = ROOT/'reports/full-multimodal-pixel-validation-CPU-v1'
TESTS = ROOT/'reports/multimodal-pixel-protocols-v1-tests.log'
WORDS = ROOT/'data/shared-byte-words-v1/train.jsonl'
ROWS = SOURCE/'all302627-original-native-views.jsonl'
VIEWS, PATHS, WORD_VIEWS = 302627, 195121, 91585
RESERVE, MAX_OUTPUT = 40*1024**3, 1024**3
WORKING, OTHER_HEADROOM = 8*1024**3, 64*1024**3


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic(path, value):
    pending = path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def memory():
    values = {k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
    return values['anon']+values['shmem']+values['kernel'], int(Path('/sys/fs/cgroup/memory.max').read_text())


def resolved_image(value):
    path = Path(value).resolve(strict=True)
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('Every original image must be an existing file inside the project')
    return str(path)


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19 or OUT.exists()
            or os.environ.get('TOKENIZERS_PARALLELISM') != 'false'
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or '5 passed' not in TESTS.read_text() or shutil.disk_usage(ROOT).free < RESERVE+MAX_OUTPUT):
        raise ValueError('Fresh tested one-thread CUDA-hidden low-priority source audit with the full reserve required')
    # This changes only this newly launched nonneural audit worker. Existing
    # training, evaluation and queue processes are never signalled or modified.
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    if os.sched_getscheduler(0) != os.SCHED_IDLE:
        raise ValueError('The new source-only worker must retain idle scheduling priority')
    active, limit = memory()
    if limit-active < WORKING+OTHER_HEADROOM:
        raise ValueError('The original memory headroom must be available before a new source audit')
    proof, reader = read(SOURCE/'complete.json'), read(READBACK)
    if (not proof.get('complete') or not reader.get('complete') or proof['training_admitted'] or reader['training_admitted']
            or proof['all_original_train_views'] != VIEWS or reader['actual_original_views_rechecked'] != VIEWS
            or proof['distinct_stored_image_paths_across_all_families'] != PATHS or proof['exact_single_epoch_updates'] != 4733):
        raise ValueError('The complete original metadata/native/schedule readback must precede physical pixel validation')
    files = dict(reader['source_files_sha256'])
    files.update({str(p):sha(p) for p in [SOURCE/'complete.json', READBACK, TESTS,
        *python_sources(ROOT, [Path(__file__), ROOT/'tests/test_multimodal_pixel_protocols_v1.py'])]})
    if files[str(ROWS)] != proof['source_metadata_rows_sha256'] or str(WORDS) not in files:
        raise ValueError('Prepared metadata and every original whole word row must be source-bound')
    metadata = {p:file_state(p) for p in files}
    waits = Counter()
    def guard():
        if os.sched_getscheduler(0) != os.SCHED_IDLE or any(file_state(p) != state for p,state in metadata.items()):
            raise ValueError('Audit scheduling and every immutable original source must remain unchanged')
    def checked():
        result = wait_for_reserve(ROOT, minimum_bytes=RESERVE, guard=guard)
        waits['seconds'] += result['waited_seconds']
        waits['failed_space_observations'] += result['failed_space_observations']
        current, maximum = memory()
        resident = next(int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
        if maximum-current < OTHER_HEADROOM or resident > WORKING:
            raise ValueError('The declared source-audit allowance and other-process memory headroom must remain available')
    me = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    OUT.mkdir()
    atomic(OUT/'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=me, source_files_sha256=files,
        expected_original_train_views=VIEWS, expected_distinct_physical_paths=PATHS, expected_complete_word_views=WORD_VIEWS,
        scheduler='SCHED_IDLE',nice=os.nice(0),cpu_threads=1,CUDA_hidden=True,
        each_hash_protocol_checked_against_its_own_original_metadata=True,
        source_RGB_and_padded_RGB_and_normalized_float_and_word_float_protocols_distinct=True,
        source_and_end_file_states_checked=True,source_constancy_reported_separately_from_pixel_integrity=True,
        minimum_free_bytes=RESERVE,maximum_output_bytes=MAX_OUTPUT,CPU_working_allowance=WORKING,
        other_process_memory_headroom=OTHER_HEADROOM,parent_address_space_ceiling=24*1024**3,
        active_memory_before=active,cgroup_memory_limit=limit,bounded_disk_wait_maximum_seconds=1800,
        every_current_training_and_evaluation_and_nine_endpoint_job_unchanged=True,
        pixels_and_labels_never_written=True,semantic_label_quality_not_certified=True,training_admitted=False,
        neural_forwards=0,new_production_optimizer_updates=0,goal_achieved=False))
    try:
        resource.setrlimit(resource.RLIMIT_AS,(24*1024**3,24*1024**3))
        checked()
        if any(sha(p) != digest for p,digest in files.items()):
            raise ValueError('All original metadata/runtime bytes must match before any physical observations')
        catalog = {}; family_views = Counter(); words = 0
        with ROWS.open() as stream:
            for number,line in enumerate(stream,1):
                row = json.loads(line); family = row['family']
                if row['source_role'] != 'train' or row['training_admitted'] or row['original_local_position'] != family_views[family]:
                    raise ValueError('Every original train-only view must retain its full ordered family position')
                path = resolved_image(row['image_path'])
                entry = catalog.setdefault(path, dict(expected={}, families=Counter(), word_views=0))
                merge_expectations(entry['expected'], dict(stored_image_sha256=row['original_stored_image_sha256'],
                    native_RGB448_sha256=row['original_RGB448_metadata_sha256'],
                    word_float_CHW64x448_sha256=row['word_pixel_float_CHW_metadata_sha256'],
                    native_float_CHW448_sha256=row['native_word_pixel_float_CHW_metadata_sha256']))
                entry['families'][family] += 1; family_views[family] += 1
                if family == 'words':
                    entry['word_views'] += 1
                if number%4096 == 0:
                    checked()
                    atomic(OUT/'progress.json', dict(owner=me,phase='merging separate original pixel protocols',
                        original_views=number,distinct_paths=len(catalog),training_admitted=False))
        if sum(family_views.values()) != VIEWS or len(catalog) != PATHS or dict(family_views) != reader['actual_family_views']:
            raise ValueError('All 302627 original views and 195121 physical paths must be included')
        with WORDS.open() as stream:
            for line in stream:
                row = json.loads(line); path = resolved_image(row['image'])
                if row['split'] != 'train' or row['task'] != 'word_ocr' or path not in catalog or not catalog[path]['word_views'] or not row['response']:
                    raise ValueError('Every whole original word crop must already belong to the complete train-only epoch')
                merge_expectations(catalog[path]['expected'], dict(stored_image_sha256=row['stored_image_sha256'],
                    source_RGB_sha256=row['source_RGB_pixels_sha256'],source_dimensions=row['source_dimensions'],
                    native_RGB448_sha256=row.get('model_pixel_sha256_448'),
                    native_float_CHW448_sha256=row['native_input_pixels_sha256'],
                    word_float_CHW64x448_sha256=row['word_input_pixels_sha256']))
                words += 1
                if words%4096 == 0: checked()
        if words != WORD_VIEWS or sum(e['word_views'] for e in catalog.values()) != WORD_VIEWS:
            raise ValueError('Every original word view and its complete raw RGB and both normalized inputs required')
        for folder in ['codenet-image-replay-v1','codenet-replay-v1']:
            inventory = ROOT/'data'/folder/'image-files.jsonl'
            if str(inventory) not in files: raise ValueError('Every original image-file inventory must be source-bound')
            with inventory.open() as stream:
                for line in stream:
                    row = json.loads(line); path = resolved_image(row['path'])
                    if path not in catalog: raise ValueError('No inventory image may silently expand or replace the complete source epoch')
                    merge_expectations(catalog[path]['expected'], dict(stored_image_sha256=row['sha256'],
                        native_RGB448_sha256=row.get('model_pixel_sha256_448')))
        if any(not {'stored_image_sha256','native_RGB448_sha256'} <= set(e['expected']) for e in catalog.values()):
            raise ValueError('Every physical image requires original stored-file and canonical RGB448 expectations')
        checked();counts = Counter(); protocols = Counter(); checked_views = Counter(); states = {}
        rows_path = OUT/'all195121-physical-pixel-observations.jsonl'
        with rows_path.open('x') as stream:
            for index,(path,entry) in enumerate(sorted(catalog.items()),1):
                if index%128 == 1: checked()
                record = dict(original_physical_position=index-1,path=path,source_families=dict(entry['families']),
                    original_view_occurrences=sum(entry['families'].values()),word_view_occurrences=entry['word_views'],
                    expected_original_protocols=entry['expected'],source_role='train',training_admitted=False)
                try:
                    actual = observe_pixels(path,word=bool(entry['word_views']))
                    different = mismatches(entry['expected'],actual)
                    record.update(actual=actual,mismatched_original_protocols=different,pixel_integrity_passed=not different)
                    states[path] = actual['source_file_state']
                    counts['physical_paths_observed'] += 1
                    counts['pixel_integrity_passes' if not different else 'pixel_integrity_failures'] += 1
                    if entry['word_views']: counts['physical_word_paths_observed'] += 1
                    if actual['source_is_spatially_constant_RGB']:
                        counts['spatially_constant_raw_RGB_sources'] += 1
                        counts['word_views_on_spatially_constant_raw_sources'] += entry['word_views']
                    protocols.update(entry['expected'].keys())
                except Exception as error:
                    record.update(pixel_integrity_passed=False,physical_observation_error=repr(error))
                    counts['physical_observation_errors'] += 1
                checked_views.update(entry['families']); counts['original_physical_positions'] += 1
                encoded = json.dumps(record,ensure_ascii=True)+'\n'
                if stream.tell()+len(encoded.encode()) > MAX_OUTPUT:
                    raise ValueError('The original complete physical observation artifact exceeded its storage allowance')
                stream.write(encoded)
                if index%128 == 0:
                    stream.flush()
                    atomic(OUT/'progress.json', dict(owner=me,phase='actual stored image decoding and separate model-input protocol checks',
                        actual_counts=dict(counts),total_physical_paths=PATHS,actual_views_covered=dict(checked_views),
                        actual_protocol_checks=dict(protocols),bounded_disk_waits=dict(waits),training_admitted=False))
            stream.flush();os.fsync(stream.fileno())
        changed = 0
        with (OUT/'end-file-state-differences.jsonl').open('x') as stream:
            for index,(path,state) in enumerate(states.items(),1):
                try: current = file_state(path)
                except OSError as error: current = dict(error=repr(error))
                if current != state:
                    stream.write(json.dumps(dict(path=path,observed_state=state,end_state=current))+'\n');changed += 1
                if index%4096 == 0: checked()
            stream.flush();os.fsync(stream.fileno())
        checked()
        if any(sha(p) != digest for p,digest in files.items()):
            raise ValueError('Every immutable original metadata/runtime byte must match at the end')
        if counts['original_physical_positions'] != PATHS or checked_views != family_views:
            raise ValueError('The complete physical source population must have one disposition each')
        passed = counts['pixel_integrity_passes'] == PATHS and not changed
        atomic(OUT/'complete.json', dict(complete=True,passed=passed,at=datetime.now(timezone.utc).isoformat(),owner=me,
            source_files_sha256=files,actual_counts=dict(counts),actual_original_views_covered=dict(checked_views),
            actual_original_protocol_checks=dict(protocols),actual_physical_states_rechecked=len(states),
            actual_physical_files_changed_after_observation=changed,observations_sha256=sha(rows_path),
            end_file_state_differences_sha256=sha(OUT/'end-file-state-differences.jsonl'),bounded_disk_waits=dict(waits),
            source_pixel_constancy_requires_separate_review=True,semantic_label_quality_not_certified=True,
            independent_readback_and_distributed_full_size_capacity_and_training_and_quality_pending=True,
            every_current_training_and_evaluation_and_nine_endpoint_job_unchanged=True,training_admitted=False,
            new_neural_parameters=0,neural_forwards=0,new_production_optimizer_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=True,passed=passed,actual_counts=dict(counts),training_admitted=False)),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json', dict(owner=me,error=repr(error),automatic_retry=False,training_admitted=False,goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
