"""Retain pinned upstream train files and their complete ordered questions.

This is source preparation only. It neither executes source code nor admits
examples to a curriculum. Original test/validation files are never requested.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
METADATA = Path('/tmp/slm-public-reasoned-code-question-metadata-v1')
OUT = Path('/tmp/slm-public-reasoned-code-train-questions-v1')
SOURCES = (
    ('deepmind/code_contests', 'code_contests', '802411c3010cb00d1b05bad57ca77365a3c699d6', 39, 13328),
    ('codeparrot/apps', 'apps', '21e74ddf8de1a21436da12e3e653065c5213e9d1', 1, 5000),
    ('BAAI/TACO', 'taco', 'd593ed0a2becbbc952230bb89be09189bf1056dc', 9, 25443),
)
GIB = 1024**3


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    with (OUT / name).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def is_train(path, dataset):
    if dataset == 'code_contests':
        return path.startswith('data/train-') and path.endswith('.parquet')
    if dataset == 'taco':
        return path.startswith('ALL/train-') and path.endswith('.parquet')
    return path == 'train.jsonl'


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))):
        raise ValueError('Fresh CUDA-hidden nice19 one-thread source preparation required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    inputs = {str(Path(__file__).resolve()): sha(__file__)}
    plans = []
    for repo, dataset, revision, file_count, row_count in SOURCES:
        slug = repo.replace('/', '--')
        paths = [METADATA / (slug + suffix) for suffix in ('-info.json', '-tree.json', '-README.md')]
        if dataset != 'code_contests':
            paths.append(METADATA / (slug + '-' + repo.rsplit('/', 1)[-1] + '.py'))
        inputs.update({str(path): sha(path) for path in paths})
        info = json.loads(paths[0].read_text())
        assert info['id'] == repo and info['sha'] == revision
        # All three upstream repository revisions predate NVIDIA's May 2025
        # release, avoiding a newer source-index ordering masquerading as it.
        assert info['lastModified'] < '2025-05-01'
        tree = json.loads(paths[1].read_text())
        files = sorted([row for row in tree if row['type'] == 'file' and is_train(row['path'], dataset)], key=lambda r: r['path'])
        assert len(files) == file_count
        assert all(row['size'] == row['lfs']['size'] for row in files)
        plans.append(dict(repo=repo, dataset=dataset, revision=revision, files=files, required_complete_rows=row_count))
    total_bytes = sum(row['size'] for plan in plans for row in plan['files'])
    assert total_bytes == 9790914839 and total_bytes < 10 * GIB
    stat = dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    nonfile = sum(int(stat[k]) for k in ('anon', 'shmem', 'kernel'))
    if (shutil.disk_usage(OUT.parent).free < 52 * GIB
            or int(Path('/sys/fs/cgroup/memory.max').read_text()) - nonfile < 50 * GIB):
        raise ValueError('12GiB scratch allowance,40GiB scratch floor,2GiB own plus48GiB other-job memory required')
    OUT.mkdir()
    (OUT / 'original-train').mkdir()
    owner = dict(pid=os.getpid(), birth=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    write('owner-plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), sources=plans,
        source_files_sha256=inputs, complete_original_download_bytes=total_bytes,
        scratch_allowance_bytes=12 * GIB, scratch_floor_bytes=40 * GIB,
        working_memory_allowance_bytes=2 * GIB, other_job_nonfile_memory_reserve_bytes=48 * GIB,
        pyarrow_version=pa.__version__, Python=sys.version, CPU_threads=1, nice=19, scheduler='SCHED_IDLE',
        ordering='publisher train paths sorted, original file rows unchanged; APPS unfiltered JSONL order',
        upstream_loaders_retained_as_text_never_executed=True, original_test_or_validation_files_requested=False,
        data_admitted_to_training=False, model_updates=0, goal_achieved=False))
    downloads = []
    counts = Counter()
    question_file = OUT / 'complete-train-questions.jsonl'
    try:
        with question_file.open('x') as questions:
            for plan in plans:
                global_index = 0
                for entry in plan['files']:
                    dataset = plan['dataset']
                    destination = OUT / 'original-train' / dataset / entry['path']
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    url = f'https://huggingface.co/datasets/{plan["repo"]}/resolve/{plan["revision"]}/{entry["path"]}'
                    digest = hashlib.sha256()
                    received = 0
                    started = time.monotonic()
                    request = urllib.request.Request(url + '?download=true&slm_questions=' + str(time.time_ns()))
                    with urllib.request.urlopen(request, timeout=30) as remote, destination.open('xb') as local:
                        while block := remote.read(1024**2):
                            received += len(block)
                            if received > entry['size'] or shutil.disk_usage(OUT).free < 40 * GIB:
                                raise ValueError('Exact file extent and original scratch floor required')
                            local.write(block)
                            digest.update(block)
                    assert received == entry['size'] and digest.hexdigest() == entry['lfs']['oid']
                    receipt = dict(dataset=dataset, revision=plan['revision'], publisher_path=entry['path'],
                        local_path=str(destination), bytes=received, sha256=digest.hexdigest(), seconds=time.monotonic()-started)
                    downloads.append(receipt)
                    if dataset == 'apps':
                        def read_apps():
                            with destination.open('rb') as source:
                                for raw in source:
                                    if len(raw) > 128 * 1024**2:
                                        raise ValueError('Unexpected whole JSONL record beyond128MiB allowance')
                                    yield json.loads(raw)
                        rows = read_apps()
                    else:
                        parquet = pq.ParquetFile(destination)
                        columns = ['description', 'name', 'source'] if dataset == 'code_contests' else [
                            'question', 'starter_code', 'name', 'source', 'url']
                        def read_parquet():
                            for batch in parquet.iter_batches(batch_size=32, columns=columns, use_threads=False):
                                yield from batch.to_pylist()
                        rows = read_parquet()
                    local_index = 0
                    for row in rows:
                        question = row['description'] if dataset == 'code_contests' else row['question']
                        assert isinstance(question, str)
                        record = dict(dataset=dataset, split='train', index=global_index,
                            original_file=entry['path'], original_file_row=local_index,
                            original_file_sha256=digest.hexdigest(), question=question,
                            question_utf8_sha256=hashlib.sha256(question.encode()).hexdigest(),
                            source_metadata={k: row[k] for k in ('name', 'source', 'url', 'difficulty', 'id') if k in row},
                            starter_code=row.get('starter_code', ''))
                        questions.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
                        counts[dataset + '_rows'] += 1
                        counts[dataset + '_empty_questions'] += not bool(question.strip())
                        counts[dataset + '_question_utf8_bytes'] += len(question.encode())
                        global_index += 1
                        local_index += 1
                    if dataset != 'apps':
                        assert local_index == parquet.metadata.num_rows
                    receipt['whole_original_rows'] = local_index
                    print(json.dumps(dict(phase='complete_original_train_file', files=len(downloads), dataset=dataset,
                        rows=global_index, bytes=received)), flush=True)
                assert global_index == plan['required_complete_rows']
                questions.flush()
                os.fsync(questions.fileno())
        assert all(sha(path) == expected for path, expected in inputs.items())
        assert all(sha(row['local_path']) == row['sha256'] for row in downloads)
        write('downloads-complete.json', downloads)
        write('complete.json', dict(complete=True, at=datetime.now(timezone.utc).isoformat(), owner=owner,
            source_files_sha256=inputs, original_files=len(downloads), complete_original_bytes=sum(r['bytes'] for r in downloads),
            questions_file=str(question_file), questions_file_sha256=sha(question_file), counts=dict(counts),
            complete_original_train_questions_preserved=True, original_questions_not_clipped=True,
            original_test_and_validation_files_not_requested=True, source_code_not_executed=True,
            upstream_terms_and_attribution_retained_in_bound_cards=True,
            candidate_question_join_context_fit_held_overlap_and_execution_qualification_still_required=True,
            data_admitted_to_training=False, no_model_weights_or_teacher_API_calls=True,
            model_updates=0, capability_gain_established=False, goal_achieved=False))
        print(json.dumps(dict(complete=True, counts=dict(counts))), flush=True)
    except BaseException as error:
        write('failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            partial_outputs_retained=True, data_admitted_to_training=False, original_workers_not_changed=True))
        raise


if __name__ == '__main__':
    main()
