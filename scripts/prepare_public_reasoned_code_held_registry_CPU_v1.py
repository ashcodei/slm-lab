"""Prepare held-only question text for screening the new public code candidate.

Publisher held answers/tests stay in immutable raw files and never become
training targets. Only complete question text enters the overlap registry.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
METADATA = Path('/tmp/slm-public-reasoned-code-question-metadata-v1')
QUESTIONS = Path('/tmp/slm-public-reasoned-code-train-questions-v1')
OUT = Path('/tmp/slm-public-reasoned-code-held-registry-v1')
HELD = (
    ('deepmind/code_contests', 'code_contests', '802411c3010cb00d1b05bad57ca77365a3c699d6', 'test', 165),
    ('deepmind/code_contests', 'code_contests', '802411c3010cb00d1b05bad57ca77365a3c699d6', 'valid', 117),
    ('BAAI/TACO', 'taco', 'd593ed0a2becbbc952230bb89be09189bf1056dc', 'test', 1000),
)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    with (OUT / name).open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))):
        raise ValueError('Fresh CUDA-hidden nice19 single-thread held-only source preparation required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import pyarrow as pa
    import pyarrow.parquet as pq
    from slm_lab.code_contests_curriculum_candidates_v1 import question_hash
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    base_path = ROOT / 'reports/CodeContests-native-held-text-v2/unchanged-held-text-registry.json'
    base_receipt = ROOT / 'reports/CodeContests-native-held-text-v2/complete.json'
    cache_receipt = QUESTIONS / 'complete.json'
    cache = json.loads(cache_receipt.read_text())
    assert cache['complete'] and cache['complete_original_train_questions_preserved']
    question_file = Path(cache['questions_file'])
    assert sha(question_file) == cache['questions_file_sha256']
    prior = json.loads(base_receipt.read_text())
    assert prior['complete'] and prior['source_files_sha256'][str(base_path)] == sha(base_path)
    profiles_path = ROOT / 'reports/code-contests-complete-curriculum-candidate-v1/all-original-problem-source-dispositions.json'
    exclusions_path = ROOT / 'reports/CodeContests-native-held-text-v2/additional-component-exclusions.json'
    inputs = {str(path): sha(path) for path in [Path(__file__).resolve(), base_path, base_receipt,
        cache_receipt, question_file, profiles_path, exclusions_path,
        ROOT / 'src/slm_lab/code_contests_curriculum_candidates_v1.py']}
    plans = []
    for repo, dataset, revision, role, count in HELD:
        slug = repo.replace('/', '--')
        for suffix in ('-info.json', '-tree.json', '-README.md'):
            path = METADATA / (slug + suffix)
            inputs[str(path)] = sha(path)
        info = json.loads((METADATA / (slug + '-info.json')).read_text())
        assert info['id'] == repo and info['sha'] == revision and info['lastModified'] < '2025-05-01'
        tree = json.loads((METADATA / (slug + '-tree.json')).read_text())
        prefix = ('data/' if dataset == 'code_contests' else 'ALL/') + role + '-'
        selected = [row for row in tree if row['type'] == 'file' and row['path'].startswith(prefix)
            and row['path'].endswith('.parquet')]
        assert len(selected) == 1 and selected[0]['size'] == selected[0]['lfs']['size']
        plans.append(dict(repo=repo, dataset=dataset, revision=revision, role=role, rows=count, original_file=selected[0]))
    download_bytes = sum(plan['original_file']['size'] for plan in plans)
    assert download_bytes == 360690905 and download_bytes < 512 * 1024**2
    stat = dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    nonfile = sum(int(stat[k]) for k in ('anon', 'shmem', 'kernel'))
    if (shutil.disk_usage(OUT.parent).free < 42 * 1024**3
            or int(Path('/sys/fs/cgroup/memory.max').read_text()) - nonfile < 50 * 1024**3):
        raise ValueError('2GiB own plus48GiB other-job memory and40GiB scratch reserve required')
    OUT.mkdir()
    (OUT / 'original-held').mkdir()
    owner = dict(pid=os.getpid(), birth=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    write('owner-plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=inputs,
        original_held_sources=plans, complete_original_download_bytes=download_bytes,
        role='question-only exclusion registry; no held answer, test or target admitted to training',
        native_internal_exclusions_retain_original_component_roles=True,
        full_question_text_not_truncated=True, CPU_threads=1, nice=19, scheduler='SCHED_IDLE',
        data_admitted_to_training=False, model_updates=0, original_workers_unchanged=True))
    try:
        registry = json.loads(base_path.read_text())
        assert len(registry) == 32953 and len({r['audit_id'] for r in registry}) == len(registry)
        profiles = json.loads(profiles_path.read_text())
        by_name_question = defaultdict(list)
        for row in profiles:
            by_name_question[(row['original_name'], row['complete_original_question_normalized_sha256'])].append(row)
        exclusions = json.loads(exclusions_path.read_text())['component_reasons']
        counts = Counter(base_held_stems=len(registry))
        matches_seen = set()
        for line in question_file.open():
            row = json.loads(line)
            if row['dataset'] != 'code_contests':
                continue
            matches = by_name_question.get((row['source_metadata']['name'], question_hash(row['question'])), [])
            assert len(matches) == 1
            profile = matches[0]
            assert profile['id'] not in matches_seen
            matches_seen.add(profile['id'])
            if profile.get('split') != 'train' or profile.get('group_id') in exclusions:
                registry.append(dict(audit_id='CodeContests-own-excluded:' + profile['id'],
                    benchmark='CodeContests-own-excluded', question=row['question']))
                counts['CodeContests_original_internal_or_component_exclusions'] += 1
        assert len(matches_seen) == 13328
        downloads = []
        for plan in plans:
            entry = plan['original_file']
            url = f'https://huggingface.co/datasets/{plan["repo"]}/resolve/{plan["revision"]}/{entry["path"]}'
            destination = OUT / 'original-held' / plan['dataset'] / entry['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            received = 0
            started = time.monotonic()
            request = urllib.request.Request(url + '?download=true&slm_held_questions=' + str(time.time_ns()))
            with urllib.request.urlopen(request, timeout=30) as remote, destination.open('xb') as local:
                while block := remote.read(1024**2):
                    received += len(block)
                    if received > entry['size'] or shutil.disk_usage(OUT).free < 40 * 1024**3:
                        raise ValueError('Exact held source extent and original scratch floor required')
                    local.write(block)
                    digest.update(block)
            assert received == entry['size'] and digest.hexdigest() == entry['lfs']['oid']
            fields = ['description'] if plan['dataset'] == 'code_contests' else ['question']
            local_rows = 0
            for batch in pq.ParquetFile(destination).iter_batches(columns=fields, batch_size=32, use_threads=False):
                for row in batch.to_pylist():
                    text = row[fields[0]]
                    assert isinstance(text, str) and text.strip() and '\0' not in text
                    registry.append(dict(audit_id='publisher-held:' + plan['dataset'] + ':' + plan['role'] + ':' + str(local_rows),
                        benchmark='publisher-held:' + plan['dataset'] + ':' + plan['role'], question=text))
                    local_rows += 1
            assert local_rows == plan['rows']
            counts[plan['dataset'] + '_' + plan['role'] + '_question_stems'] += local_rows
            downloads.append(dict(dataset=plan['dataset'], role=plan['role'], revision=plan['revision'],
                original_path=entry['path'], local_path=str(destination), bytes=received, sha256=digest.hexdigest(),
                complete_question_rows=local_rows, seconds=time.monotonic()-started))
            print(json.dumps(dict(phase='held_only_questions_complete', dataset=plan['dataset'], role=plan['role'],
                rows=local_rows, bytes=received)), flush=True)
        assert len({row['audit_id'] for row in registry}) == len(registry)
        assert all(sha(path) == value for path, value in inputs.items())
        assert all(sha(row['local_path']) == row['sha256'] for row in downloads)
        write('held-question-registry.json', registry)
        write('original-held-downloads.json', downloads)
        write('complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=inputs, counts=dict(counts), total_held_question_stems=len(registry),
            base_registry_preserved_exactly=True, all13328_original_train_identities_uniquely_checked=True,
            held_registry_sha256=sha(OUT / 'held-question-registry.json'),
            original_held_downloads_sha256=sha(OUT / 'original-held-downloads.json'),
            complete_original_held_files=3, complete_original_held_bytes=download_bytes,
            only_question_columns_decoded_from_new_held_sources=True,
            no_held_answer_or_test_field_decoded_as_training_supervision=True,
            actual_candidate_overlap_screening_and_execution_qualification_still_required=True,
            data_admitted_to_training=False, original_model_workers_unchanged=True,
            model_updates=0, model_quality_gain_established=False, goal_achieved=False))
        print(json.dumps(dict(complete=True, total_stems=len(registry), counts=dict(counts))), flush=True)
    except BaseException as error:
        write('failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            partial_outputs_retained=True, data_admitted_to_training=False, original_workers_unchanged=True))
        raise


if __name__ == '__main__':
    main()
