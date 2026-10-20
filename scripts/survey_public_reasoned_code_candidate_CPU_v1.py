"""Stage and inspect one pinned public shard; never admit it to training."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/tmp/slm-public-reasoned-code-candidate-v1')
TOKENIZER = ROOT / 'data/tokenizer-corpus/tokenizer.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    with (OUT / name).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    if (not OUT.is_dir() or (OUT / 'owner-plan.json').exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or os.nice(0) != 19 or any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))):
        raise ValueError('Fresh existing pinned metadata, CUDA-hidden nice19 one-thread survey required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import pyarrow as pa
    import pyarrow.parquet as pq
    import tokenizers
    from tokenizers import Tokenizer

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    info = json.loads((OUT / 'dataset-info.json').read_text())
    assert info['id'] == 'nvidia/OpenCodeReasoning-2'
    revision = info['sha']
    assert len(revision) == 40 and all(c in '0123456789abcdef' for c in revision)
    tree = json.loads((OUT / 'file-tree.json').read_text())
    candidates = sorted([row for row in tree if row.get('path', '').startswith('train/python/')
        and row['path'].endswith('.parquet')], key=lambda row: row['path'])
    selected = candidates[0]
    assert selected['path'] == 'train/python/train-00000-of-00070.parquet'
    assert selected['size'] == selected['lfs']['size'] < 512 * 1024**2
    source_url = f'https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2/resolve/{revision}/{selected["path"]}'
    files = {str(path): sha(path) for path in [Path(__file__).resolve(), TOKENIZER,
        OUT / 'dataset-info.json', OUT / 'file-tree.json', OUT / 'README.publisher.md']}
    raw_tokenizer = json.loads(TOKENIZER.read_text())
    assert raw_tokenizer.get('truncation') is None and raw_tokenizer.get('padding') is None
    stat = dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    live = sum(int(stat[k]) for k in ('anon', 'shmem', 'kernel'))
    if (int(Path('/sys/fs/cgroup/memory.max').read_text()) - live < 50 * 1024**3
            or shutil.disk_usage(OUT).free < 42 * 1024**3):
        raise ValueError('2GiB survey allowance,48GiB other-job memory and40GiB scratch floor required')
    owner = dict(pid=os.getpid(), birth=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    write('owner-plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        dataset=info['id'], revision=revision, selected_original_file=selected, original_url=source_url,
        selection='first whole Python shard in publisher path order, before inspecting examples',
        maximum_download_bytes=512 * 1024**2, private_working_allowance_bytes=2 * 1024**3,
        CPU_threads=1, scheduler='SCHED_IDLE', nice=19, pyarrow_version=pa.__version__, tokenizers_version=tokenizers.__version__,
        Python=sys.version, native_context_limit=4096, all_rows_and_full_text_fields_preserved=True,
        data_admitted_to_training=False, no_model_weights_or_teacher_API_requested=True,
        original_workers_and_curricula_not_changed=True, goal_achieved=False))
    try:
        print(json.dumps(dict(phase='download_pinned_complete_shard', expected_bytes=selected['size'])), flush=True)
        raw_path = OUT / 'original-python-00000.parquet'
        begin = time.monotonic()
        digest = hashlib.sha256()
        count = 0
        request = urllib.request.Request(source_url + '?download=true&slm_survey=' + str(time.time_ns()))
        with urllib.request.urlopen(request, timeout=30) as remote, raw_path.open('xb') as local:
            while block := remote.read(1024**2):
                count += len(block)
                if count > selected['size'] or shutil.disk_usage(OUT).free < 40 * 1024**3:
                    raise ValueError('Exact input size and original writer scratch floor required')
                local.write(block)
                digest.update(block)
        assert count == selected['size'] and digest.hexdigest() == selected['lfs']['oid']
        files[str(raw_path)] = digest.hexdigest()
        write('download-complete.json', dict(complete=True, bytes=count, sha256=digest.hexdigest(), seconds=time.monotonic()-begin,
            original_url=source_url, revision=revision, data_admitted_to_training=False))
        parquet = pq.ParquetFile(raw_path)
        columns = ['id', 'question_id', 'question', 'r1_generation', 'qwq_critique', 'solution',
            'judgement', 'pass_rate', 'dataset', 'split', 'source', 'license', 'index']
        assert set(columns) <= set(parquet.schema_arrow.names)
        tokenizer = Tokenizer.from_file(str(TOKENIZER))
        totals = Counter()
        sources = Counter()
        roles = Counter()
        licences = Counter()
        judgements = Counter()
        ids, questions, original_problem_keys = set(), set(), set()
        for batch in parquet.iter_batches(batch_size=32, columns=columns, use_threads=False):
            for row in batch.to_pylist():
                totals['rows'] += 1
                assert isinstance(row['id'], str) and row['id'] not in ids
                ids.add(row['id'])
                questions.add(row['question_id'])
                original_problem_keys.add((row['dataset'], row['split'], row['index']))
                sources[str(row['source'])] += 1
                roles[str(row['dataset']) + '/' + str(row['split'])] += 1
                licences[str(row['license'])] += 1
                judgements[str(row['judgement'])] += 1
                missing = not isinstance(row['question'], str) or row['question'].strip() in {'', '-'}
                totals['question_requires_original_source_reconstruction'] += missing
                try:
                    rate = float(row['pass_rate'])
                except (ValueError, TypeError):
                    rate = None
                positive = rate == 1.
                totals['reported_pass_rate_one'] += positive
                totals['reported_unknown_or_unvalidated_pass_rate'] += rate is None or not math.isfinite(rate) or rate < 0
                totals['upstream_role_not_train'] += row['split'] != 'train'
                solution = row['solution'] if isinstance(row['solution'], str) else ''
                nonempty = bool(solution.strip())
                totals['nonempty_solutions'] += nonempty
                parseable = False
                if nonempty:
                    try:
                        ast.parse(solution)
                    except (SyntaxError, ValueError):
                        pass
                    else:
                        parseable = True
                totals['solutions_parseable_current_Python'] += parseable
                for field in ('solution', 'r1_generation', 'qwq_critique'):
                    value = row[field] if isinstance(row[field], str) else ''
                    length = len(tokenizer.encode(value).ids) if value else 0
                    totals[field + '_complete_tokens'] += length
                    totals[field + '_max_complete_tokens'] = max(totals[field + '_max_complete_tokens'], length)
                    totals[field + '_nonempty_targets_at_most4094_tokens_without_question'] += 0 < length <= 4094
                solution_length = len(tokenizer.encode(solution).ids) if nonempty else 0
                totals['reported_positive_parseable_code_target_within4094_upstream_train_prompt_unresolved'] += (
                    positive and parseable and 0 < solution_length <= 4094 and row['split'] == 'train')
            print(json.dumps(dict(phase='whole_shard_schema_and_token_survey', rows=totals['rows'], original_rows=parquet.metadata.num_rows)), flush=True)
        assert totals['rows'] == parquet.metadata.num_rows == len(ids)
        assert all(sha(path) == expected for path, expected in files.items())
        write('survey-complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, selected_original_file=selected['path'], revision=revision,
            whole_shard_rows=totals['rows'], unique_question_ids=len(questions),
            unique_original_dataset_split_index_keys=len(original_problem_keys), counts=dict(totals),
            upstream_dataset_split_counts=dict(roles), platform_counts=dict(sources), license_counts=dict(licences),
            publisher_judgement_counts=dict(judgements), publisher_correctness_metadata_not_independent_execution=True,
            code_parseability_not_functional_correctness=True, target_lengths_exclude_unreconstructed_question=True,
            original_questions_and_held_role_identity_checks_still_required=True,
            complete_source_question_plus_target_context_fit_not_established=True,
            current_benchmarks_not_used_for_gradient_or_source_selection=True,
            data_admitted_to_training=False, no_new_model_training_or_teacher_API_calls=True,
            no_model_capability_gain_established=True, all_ten_final_requirements_incomplete=True, goal_achieved=False))
        print(json.dumps(dict(complete=True, rows=totals['rows'], questions=len(questions), counts=dict(totals))), flush=True)
    except BaseException as error:
        write('failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            partial_outputs_retained=True, data_admitted_to_training=False, original_workers_not_changed=True, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
