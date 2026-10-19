"""Freshly read every physical pixel source after its exact producer ends.

Rebuild all original metadata expectations independently, then decode every
195121physical source again, covering all302627native views and91585whole
words. Preserve every failed observation; do not admit training or modify
labels, pixels, active jobs, queues or benchmark populations.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.independent_multimodal_pixels_v1 import file_state, merge, observe, differences
from slm_lab.local_source_closure_v2 import python_sources

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/full-multimodal-source-epoch-CPU-v1'
SOURCE_READER = ROOT / 'reports/full-multimodal-source-epoch-v1-readback/complete.json'
PRODUCER = ROOT / 'reports/full-multimodal-pixel-validation-CPU-v1'
OUT = ROOT / 'reports/full-multimodal-pixel-readback-CPU-v1'
TESTS = ROOT / 'reports/independent-multimodal-pixels-v1-tests.log'
PARENT = dict(pid=51248, ticks='53113593')
VIEWS, PATHS, WORDS = 302627, 195121, 91585
RESERVE, WORKING, HEADROOM = 40 * 1024**3, 8 * 1024**3, 64 * 1024**3


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def records(path):
    with Path(path).open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                raise ValueError('Every immutable JSONL source must contain only complete LF records')
            yield json.loads(line)


def image_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Every original physical path must remain inside the project')
    return str(path)


def catalog(proof, checked):
    result = {}
    families = Counter()
    for number, row in enumerate(records(SOURCE / 'all302627-original-native-views.jsonl'), 1):
        name = row['family']
        if (row['source_role'] != 'train' or row['training_admitted'] is not False
                or row['original_local_position'] != families[name]):
            raise ValueError('Every original train-only native view must retain its whole ordered source position')
        entry = result.setdefault(image_path(row['image_path']), dict(expected={}, families=Counter(), words=0))
        merge(entry['expected'], dict(stored_image_sha256=row['original_stored_image_sha256'],
            native_RGB448_sha256=row['original_RGB448_metadata_sha256'],
            native_float_CHW448_sha256=row['native_word_pixel_float_CHW_metadata_sha256'],
            word_float_CHW64x448_sha256=row['word_pixel_float_CHW_metadata_sha256']))
        entry['families'][name] += 1
        entry['words'] += int(name == 'words')
        families[name] += 1
        if number % 4096 == 0:
            checked()
    if sum(families.values()) != VIEWS or len(result) != PATHS or dict(families) != proof['actual_family_views']:
        raise ValueError('All302627views,all195121paths,and all nine original families are required')
    words = 0
    for row in records(ROOT / 'data/shared-byte-words-v1/train.jsonl'):
        path = image_path(row['image'])
        if (row['split'] != 'train' or row['task'] != 'word_ocr' or not row['response']
                or path not in result or not result[path]['words']):
            raise ValueError('Every original whole word must already belong to the full train-only population')
        merge(result[path]['expected'], dict(stored_image_sha256=row['stored_image_sha256'],
            source_RGB_sha256=row['source_RGB_pixels_sha256'], source_dimensions=row['source_dimensions'],
            native_RGB448_sha256=row.get('model_pixel_sha256_448'),
            native_float_CHW448_sha256=row['native_input_pixels_sha256'],
            word_float_CHW64x448_sha256=row['word_input_pixels_sha256']))
        words += 1
        if words % 4096 == 0:
            checked()
    if words != WORDS or sum(x['words'] for x in result.values()) != WORDS:
        raise ValueError('All91585whole-word occurrences and both input geometries are required')
    for folder in ['codenet-image-replay-v1', 'codenet-replay-v1']:
        for row in records(ROOT / 'data' / folder / 'image-files.jsonl'):
            path = image_path(row['path'])
            if path not in result:
                raise ValueError('Original image inventories cannot expand or replace the full source population')
            merge(result[path]['expected'], dict(stored_image_sha256=row['sha256'],
                native_RGB448_sha256=row.get('model_pixel_sha256_448')))
    if any(not {'stored_image_sha256', 'native_RGB448_sha256'} <= set(x['expected']) for x in result.values()):
        raise ValueError('Every original physical source needs its own complete file and nativeRGBprotocol')
    return result, families


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or '6 passed' not in TESTS.read_text()
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh tested one-thread CUDA-hidden nice19 independent pixel readback required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    producer_plan = read(PRODUCER / 'plan.json')
    if {k: producer_plan['owner'][k] for k in ['pid', 'ticks']} != PARENT:
        raise ValueError('Only the exact currently running full physical pixel producer may supply this readback')
    files = {str(p): sha(p) for p in [PRODUCER / 'plan.json', SOURCE / 'complete.json', SOURCE_READER, TESTS,
        *python_sources(ROOT, [Path(__file__), ROOT / 'tests/test_independent_multimodal_pixels_v1.py'])]}
    states = {p: file_state(p) for p in files}
    me = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    OUT.mkdir()
    atomic(OUT / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=me,
        exact_full_producer=PARENT, source_files_sha256=files, all_original_native_views=VIEWS,
        all_original_physical_paths=PATHS, all_original_whole_word_views=WORDS,
        actual_original_metadata_expectations_reconstructed_before_redecode=True,
        producer_observation_and_canonical_word_array_helpers_not_used_by_fresh_decoder=True,
        every_physical_source_and_every_original_finite_protocol_retained=True,
        complete_source_file_identity_rechecked_after_all_observations=True,
        no_live_producer_journal_or_completion_is_source_bound=True,
        scheduler='SCHED_IDLE', nice=os.nice(0), CPU_threads=1, CUDA_hidden=True,
        CPU_working_allowance=WORKING, other_worker_headroom=HEADROOM,
        minimum_free_bytes=RESERVE, maximum_output_bytes=1024**3,
        original_neural_workers_and_human_RL_and_nine_final_endpoints_unchanged=True,
        semantic_label_quality_not_certified=True, source_role='train', training_admitted=False,
        models_constructed=0, neural_forwards=0, new_production_optimizer_updates=0, goal_achieved=False))
    waits = Counter()

    def guard():
        if os.sched_getscheduler(0) != os.SCHED_IDLE or any(file_state(p) != state for p, state in states.items()):
            raise ValueError('Idle scheduling and all immutable original source identities must remain exact')

    def checked():
        result = wait_for_reserve(ROOT, minimum_bytes=RESERVE, guard=guard)
        waits['seconds'] += result['waited_seconds']
        waits['failed_space_observations'] += result['failed_space_observations']
        stat = {k: int(v) for k, v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
        resident = stat['anon'] + stat['shmem'] + stat['kernel']
        own = next(int(line.split()[1]) * 1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
        if int(Path('/sys/fs/cgroup/memory.max').read_text()) - resident < HEADROOM or own > WORKING:
            raise ValueError('Original other-worker headroom and bounded source-only working set required')

    try:
        while not terminal(PARENT):
            guard()
            atomic(OUT / 'progress.json', dict(owner=me, phase='waiting for exact full pixel producer terminal and complete',
                exact_parent_live=True, neural_forwards=0, training_admitted=False))
            time.sleep(30)
        if (PRODUCER / 'failure.json').exists() or not (PRODUCER / 'complete.json').exists():
            raise ValueError('The exact original producer ended without its complete physical source population')
        producer = read(PRODUCER / 'complete.json')
        source = read(SOURCE_READER)
        old_rows = PRODUCER / 'all195121-physical-pixel-observations.jsonl'
        if (not producer['complete'] or producer['actual_counts']['original_physical_positions'] != PATHS
                or producer['actual_original_views_covered'] != source['actual_family_views']
                or producer['observations_sha256'] != sha(old_rows)):
            raise ValueError('Every original full physical disposition and source-family occurrence required')
        for path, digest in producer['source_files_sha256'].items():
            if path in files and files[path] != digest:
                raise ValueError('Conflicting immutable source identity between original producer and reader')
            files[path] = digest
        files.update({str(p): sha(p) for p in [PRODUCER / 'complete.json', old_rows,
            PRODUCER / 'end-file-state-differences.jsonl']})
        if any(sha(p) != digest for p, digest in files.items()):
            raise ValueError('Every original metadata, runtime and terminal producer byte must match')
        states.update({p: file_state(p) for p in files})
        atomic(OUT / 'terminal-source-bindings.json', dict(owner=me, source_files_sha256=files,
            exact_full_producer_terminal=True, live_output_never_bound=True))
        resource.setrlimit(resource.RLIMIT_AS, (24 * 1024**3, 24 * 1024**3))
        checked()
        entries, families = catalog(source, checked)
        old_stream = iter(records(old_rows))
        counts = Counter()
        covered = Counter()
        final_states = {}
        output = OUT / 'all195121-independent-fresh-pixel-observations.jsonl'
        with output.open('x') as stream:
            for position, (path, entry) in enumerate(sorted(entries.items())):
                if position % 128 == 0:
                    checked()
                original = next(old_stream)
                if (original['original_physical_position'] != position or original['path'] != path
                        or original['source_families'] != dict(entry['families'])
                        or original['original_view_occurrences'] != sum(entry['families'].values())
                        or original['word_view_occurrences'] != entry['words']
                        or original['expected_original_protocols'] != entry['expected']
                        or original['source_role'] != 'train' or original['training_admitted'] is not False):
                    raise ValueError('Every original physical metadata/order/protocol must reconstruct before fresh decoding')
                if 'actual' in original:
                    old_mismatch = differences(entry['expected'], original['actual'])
                    if (original['mismatched_original_protocols'] != old_mismatch
                            or original['pixel_integrity_passed'] != (not old_mismatch)):
                        raise ValueError('Every original pixel-integrity verdict must follow its actual protocol values')
                elif original['pixel_integrity_passed'] or not original.get('physical_observation_error'):
                    raise ValueError('A missing physical observation cannot be certified by the original producer')
                row = dict(original_physical_position=position, path=path, source_families=dict(entry['families']),
                    original_view_occurrences=sum(entry['families'].values()), word_view_occurrences=entry['words'],
                    expected_original_protocols=entry['expected'], original_integrity_passed=original['pixel_integrity_passed'],
                    source_role='train', training_admitted=False)
                try:
                    current = observe(path, word=bool(entry['words']))
                    mismatch = differences(entry['expected'], current)
                    old_differences = [k for k in sorted(set(current) | set(original.get('actual', {})))
                        if current.get(k) != original.get('actual', {}).get(k)]
                    row.update(actual_fresh=current, mismatched_original_protocols=mismatch,
                        different_actual_fields_from_producer=old_differences,
                        fresh_integrity_passed=not mismatch and not old_differences)
                    final_states[path] = current['source_file_state']
                    counts['fresh_integrity_passes' if row['fresh_integrity_passed'] else 'fresh_integrity_failures'] += 1
                    if current['source_is_spatially_constant_RGB']:
                        counts['spatially_constant_raw_RGB_sources'] += 1
                    counts['fresh_word_paths'] += int(bool(entry['words']))
                except Exception as error:
                    row.update(fresh_integrity_passed=False, fresh_physical_observation_error=repr(error))
                    counts['fresh_physical_observation_errors'] += 1
                counts['original_physical_positions'] += 1
                counts['original_word_views'] += entry['words']
                covered.update(entry['families'])
                encoded = json.dumps(row, ensure_ascii=True) + '\n'
                if stream.tell() + len(encoded.encode()) > 1024**3:
                    raise ValueError('Every original observation must fit the declared full-readback output bound')
                stream.write(encoded)
                if (position + 1) % 128 == 0:
                    stream.flush()
                    atomic(OUT / 'progress.json', dict(owner=me, phase='all physical sources freshly decoded against rebuilt metadata',
                        actual_counts=dict(counts), actual_views_covered=dict(covered), total_original_paths=PATHS,
                        bounded_disk_waits=dict(waits), training_admitted=False))
            os.fsync(stream.fileno())
        if next(old_stream, None) is not None:
            raise ValueError('No extra original physical disposition may be silently ignored')
        changed = 0
        with (OUT / 'fresh-end-file-state-differences.jsonl').open('x') as stream:
            for position, (path, state) in enumerate(final_states.items()):
                try:
                    current = file_state(path)
                except OSError as error:
                    current = dict(error=repr(error))
                if current != state:
                    stream.write(json.dumps(dict(path=path, observed=state, final=current)) + '\n')
                    changed += 1
                if position % 4096 == 0:
                    checked()
            os.fsync(stream.fileno())
        checked()
        if (counts['original_physical_positions'] != PATHS or counts['original_word_views'] != WORDS
                or covered != families or sum(covered.values()) != VIEWS
                or any(sha(p) != digest for p, digest in files.items())):
            raise ValueError('Every original full source occurrence and immutable metadata byte must survive readback')
        passed = bool(producer['passed'] and counts['fresh_integrity_passes'] == PATHS and changed == 0)
        atomic(OUT / 'complete.json', dict(complete=True, passed=passed, at=datetime.now(timezone.utc).isoformat(), owner=me,
            source_files_sha256=files, actual_original_physical_paths=PATHS, actual_original_native_views=VIEWS,
            actual_original_whole_word_views=WORDS, actual_counts=dict(counts), actual_family_views=dict(covered),
            all_original_protocol_expectations_independently_rebuilt=True,
            all195121physical_sources_freshly_observed=True, fresh_observations_sha256=sha(output),
            actual_fresh_source_states_rechecked=len(final_states), actual_end_file_state_differences=changed,
            every_original_and_fresh_failure_preserved=True, bounded_disk_waits=dict(waits),
            source_label_semantics_not_certified=True, source_role='train', training_admitted=False,
            complete_source_admission_full_size_GPU_capacity_training_and_quality_pending=True,
            models_constructed=0, neural_forwards=0, new_production_optimizer_updates=0,
            all_existing_training_evaluation_and_nine_final_endpoint_scopes_unchanged=True, goal_achieved=False))
        print(json.dumps(dict(report=str(OUT / 'complete.json'), passed=passed, paths=PATHS, views=VIEWS, training_admitted=False)))
    except BaseException as error:
        atomic(OUT / 'failure.json', dict(owner=me, error=repr(error), automatic_retry=False,
            source_role='train', training_admitted=False, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
