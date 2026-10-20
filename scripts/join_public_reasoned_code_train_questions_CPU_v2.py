"""Join complete upstream-train questions and measure two lossless SFT views.

No curriculum admission: full held-text/program screening and independent
execution qualification remain separate requirements. Upstream test rows stay
quarantined and their missing questions are not fetched.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = Path('/tmp/slm-public-reasoned-code-candidate-v1')
QUESTIONS = Path('/tmp/slm-public-reasoned-code-train-questions-v1')
OUT = Path('/tmp/slm-public-reasoned-code-question-join-v2')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    with (OUT / name).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))):
        raise ValueError('Fresh CUDA-hidden nice19 single-thread complete-source join required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    from slm_lab.code_contests_curriculum_candidates_v1 import question_hash
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    survey = json.loads((CANDIDATE / 'survey-complete.json').read_text())
    cache = json.loads((QUESTIONS / 'complete.json').read_text())
    assert survey['complete'] and survey['whole_shard_rows'] == 20000
    assert cache['complete'] and cache['original_files'] == 49
    assert cache['counts']['code_contests_rows'] == 13328
    assert cache['counts']['apps_rows'] == 5000 and cache['counts']['taco_rows'] == 25443
    assert not survey['data_admitted_to_training'] and not cache['data_admitted_to_training']
    question_file = Path(cache['questions_file'])
    assert sha(question_file) == cache['questions_file_sha256']
    records_path = ROOT / 'data/code-contests-native-train-records-v1/all-original-problem-records.jsonl'
    profiles_path = ROOT / 'reports/code-contests-complete-curriculum-candidate-v1/all-original-problem-source-dispositions.json'
    exclusions_path = ROOT / 'reports/CodeContests-native-held-text-v2/additional-component-exclusions.json'
    apps_roles_path = ROOT / 'data/public-apps-complete-pack-v2/original-rows.jsonl.gz'
    tokenizer_path = ROOT / 'data/tokenizer-corpus/tokenizer.json'
    inputs = {str(path): sha(path) for path in [Path(__file__).resolve(), CANDIDATE / 'survey-complete.json',
        QUESTIONS / 'complete.json', QUESTIONS / 'downloads-complete.json', question_file, records_path,
        profiles_path, exclusions_path, apps_roles_path, tokenizer_path,
        ROOT / 'src/slm_lab/code_contests_curriculum_candidates_v1.py',
        ROOT / 'scripts/join_public_reasoned_code_train_questions_CPU_v1.py',
        Path('/tmp/slm-public-reasoned-code-question-join-v1.log')]}
    inputs.update(survey['source_files_sha256'])
    inputs.update(cache['source_files_sha256'])
    assert all(sha(path) == value for path, value in inputs.items())
    assert json.loads(tokenizer_path.read_text()).get('truncation') is None
    stat = dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    nonfile = sum(int(stat[k]) for k in ('anon', 'shmem', 'kernel'))
    if (shutil.disk_usage(OUT.parent).free < 42 * 1024**3
            or int(Path('/sys/fs/cgroup/memory.max').read_text()) - nonfile < 50 * 1024**3):
        raise ValueError('2GiB own and48GiB other-job memory headroom plus40GiB scratch floor required')
    original = [json.loads(line) for line in records_path.open()]
    profiles = {row['id']: row for row in json.loads(profiles_path.read_text())}
    own_by_name_and_question = defaultdict(list)
    for row in original:
        profile = profiles[row['id']]
        own_by_name_and_question[(row['original_name'],
            profile['complete_original_question_normalized_sha256'])].append(row)
    exclusions = json.loads(exclusions_path.read_text())['component_reasons']
    excluded_hashes = defaultdict(list)
    for row in profiles.values():
        if row.get('split') != 'train' or row.get('group_id') in exclusions:
            excluded_hashes[row['complete_original_question_normalized_sha256']].append(row['id'])
    apps_validation = set()
    with gzip.open(apps_roles_path, 'rt') as source:
        for line in source:
            row = json.loads(line)
            if row['split'] == 'validation':
                apps_validation.add(int(row['source_problem_id']))
    questions = {}
    cc_order_checks = 0
    matched_own_identities = set()
    for line in question_file.open():
        row = json.loads(line)
        key = (row['dataset'], row['split'], row['index'])
        assert key not in questions and row['split'] == 'train'
        assert hashlib.sha256(row['question'].encode()).hexdigest() == row['question_utf8_sha256']
        if row['dataset'] == 'code_contests':
            # Keep NVIDIA's pinned HF source index. The separate original
            # publisher cache has a different record order, so identify its
            # protected source role by a unique whole-question/name match.
            matches = own_by_name_and_question.get((row['source_metadata']['name'],
                question_hash(row['question'])), [])
            assert len(matches) == 1
            own = matches[0]
            assert own['id'] not in matched_own_identities
            matched_own_identities.add(own['id'])
            profile = profiles[own['id']]
            assert row['source_metadata']['name'] == own['original_name']
            assert question_hash(row['question']) == profile['complete_original_question_normalized_sha256']
            row['own_original_problem_id'] = own['id']
            row['known_held_source_component'] = profile.get('split') != 'train' or profile.get('group_id') in exclusions
            cc_order_checks += 1
        questions[key] = row
    assert len(questions) == 43771 and cc_order_checks == 13328 and len(matched_own_identities) == 13328
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    OUT.mkdir()
    owner = dict(pid=os.getpid(), birth=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    write('owner-plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=inputs,
        complete_candidate_rows=20000, complete_source_questions=43771,
        code_contests_unique_original_identity_matches=cc_order_checks,
        code_contests_question_check='unique full normalized question and original name, all13328own identities; not protobuf byte equality',
        pinned_HF_source_row_indices_preserved_without_assuming_original_publisher_record_order=True,
        failed_v1_source_and_log_preserved=True,
        views=['entire publisher code-only solution', 'entire publisher R1 response including its reasoning'],
        complete_native_context_limit=4096, CPU_threads=1, nice=19, scheduler='SCHED_IDLE',
        full_held_text_program_screening_not_yet_done=True, data_admitted_to_training=False,
        original_workers_and_curricula_unchanged=True, model_updates=0, goal_achieved=False))
    counts = Counter()
    try:
        with (OUT / 'every-candidate-disposition.jsonl').open('x') as ledger:
            for batch in pq.ParquetFile(CANDIDATE / 'original-python-00000.parquet').iter_batches(
                    columns=['id', 'question_id', 'dataset', 'split', 'index', 'solution', 'r1_generation', 'pass_rate'],
                    batch_size=16, use_threads=False):
                for row in batch.to_pylist():
                    counts['original_rows'] += 1
                    key = (row['dataset'], row['split'], int(row['index']))
                    output = dict(position=counts['original_rows']-1, id=row['id'], question_id=row['question_id'],
                        dataset=key[0], split=key[1], index=key[2], data_admitted_to_training=False)
                    if row['split'] != 'train':
                        counts['upstream_non_train_quarantined'] += 1
                        output['disposition'] = 'upstream_non_train_quarantined_without_fetching_question'
                    else:
                        question = questions[key]
                        counts['whole_train_question_joined'] += 1
                        text = question['question']
                        starter = question.get('starter_code', '')
                        assert isinstance(starter, str)
                        suffix = '\n\nStarter code:\n' + starter if starter.strip() else ''
                        code_prompt = 'Write a Python solution to the following problem. Follow its specified interface. Return only the complete code.\n\n' + text + suffix
                        reasoning_prompt = 'Solve the following programming problem in Python. Explain your reasoning and provide the complete solution.\n\n' + text + suffix
                        known_excluded = bool(question.get('known_held_source_component'))
                        cross_source = excluded_hashes.get(question_hash(text), [])
                        apps_held = key[0] == 'apps' and int(question['source_metadata']['id']) in apps_validation
                        flagged = known_excluded or bool(cross_source) or apps_held
                        counts['known_held_identity_or_exact_normalized_question_flagged'] += flagged
                        output.update(disposition='joined_pending_complete_screening_and_execution',
                            question_utf8_sha256=question['question_utf8_sha256'],
                            source_file=question['original_file'], source_file_row=question['original_file_row'],
                            known_held_source_component=known_excluded, APPS_internal_validation=apps_held,
                            known_excluded_CodeContests_normalized_question_matches=cross_source,
                            publisher_reported_pass_rate=row['pass_rate'], views={})
                        for label, field, prompt in [('code_only', 'solution', code_prompt), ('reasoning', 'r1_generation', reasoning_prompt)]:
                            target = row[field]
                            assert isinstance(target, str)
                            prefix = [1] + tokenizer.encode('User: ' + prompt + '\nAssistant:', add_special_tokens=False).ids
                            response = tokenizer.encode(target, add_special_tokens=False).ids
                            reserved = any(value < 4 for value in prefix[1:] + response)
                            context = len(prefix) + len(response) + 1
                            fits = bool(response) and not reserved and context <= 4096
                            output['views'][label] = dict(prompt=prompt, prompt_tokens_including_BOS=len(prefix),
                                entire_target_utf8_sha256=hashlib.sha256(target.encode()).hexdigest(),
                                entire_response_tokens_including_EOS=len(response)+1, entire_context_tokens=context,
                                contains_reserved_native_source_token=reserved, complete_context_fits_4096=fits)
                            counts[label + '_complete_context_fits_4096'] += fits
                            counts[label + '_maximum_whole_context_tokens'] = max(counts[label + '_maximum_whole_context_tokens'], context)
                            counts[label + '_fits_and_no_current_identity_flag_publisher_reports_pass_one'] += (
                                fits and not flagged and row['pass_rate'] == 1.)
                    ledger.write(json.dumps(output, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
                print(json.dumps(dict(phase='complete_question_and_target_fit', rows=counts['original_rows'], total=20000)), flush=True)
        assert counts['original_rows'] == 20000 and counts['whole_train_question_joined'] == 17424
        assert counts['upstream_non_train_quarantined'] == 2576
        assert all(sha(path) == value for path, value in inputs.items())
        write('complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=inputs, counts=dict(counts),
            ledger_sha256=sha(OUT / 'every-candidate-disposition.jsonl'),
            all_complete_upstream_train_questions_joined=True, upstream_non_train_roles_never_admitted=True,
            both_view_targets_preserved_whole_never_truncated=True, code_only_field_is_publisher_supplied_not_cropped_reasoning=True,
            publisher_pass_metadata_not_independent_execution=True,
            full_held_text_program_and_cross_source_role_screening_still_required=True,
            source_ancestry_and_semantic_independence_not_certified=True,
            independent_execution_qualification_and_reader_still_required=True,
            data_admitted_to_training=False, no_new_model_training=True, model_capability_gain_established=False,
            all_ten_final_requirements_incomplete=True, goal_achieved=False))
        print(json.dumps(dict(complete=True, counts=dict(counts))), flush=True)
    except BaseException as error:
        write('failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            partial_outputs_retained=True, data_admitted_to_training=False, original_workers_not_changed=True))
        raise


if __name__ == '__main__':
    main()
