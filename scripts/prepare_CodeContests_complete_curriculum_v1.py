"""Prepare complete native code candidates with known held-source quarantine.

Visits all publisher positives, selects at most three fitting unique original
programs per problem/language, and propagates known held matches across linked
problem components. Raw publisher records remain available without selection.
This is preparation, not execution reward admission or a new training run.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from tokenizers import Tokenizer

from slm_lab.code_contests_curriculum_candidates_v1 import (normalized_program_hash,
    original_problem_keys, question_hash, select_complete_solutions, url_key)
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CodePackWriter
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'data/code-contests-native-train-records-v1'
CACHE = ROOT / 'data/code-contests-original-train-cache-v2'
OUT = ROOT / 'data/code-contests-complete-curriculum-candidate-v1'
REPORTS = ROOT / 'reports/code-contests-complete-curriculum-candidate-v1'
READBACK = ROOT / 'reports/code-contests-all-original-records-v1-readback/complete.json'
APPS = ROOT / 'data/public-apps-original-inventory-v1'
APPS_TEST = ROOT / 'data/public-apps-publisher-test-evaluation-v1'
TOKENIZER = ROOT / 'data/verified-code-views-v1/tokenizer.json'


def emit(stream, row):
    stream.write((json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode())


def original_records(rows, proto):
    stream = None; previous = None; end = 0
    try:
        for row in rows:
            path = row['extracted_record_file']
            if path != previous:
                if stream:
                    if stream.read(1): raise ValueError('Unindexed original shard bytes')
                    stream.close()
                stream = open(path, 'rb'); previous = path; end = 0
            if row['extracted_record_offset'] != end: raise ValueError('Contiguous whole original records required')
            raw = stream.read(row['extracted_record_bytes'])
            if hashlib.sha256(raw).hexdigest() != row['actual_original_record_sha256']:
                raise ValueError('Original record payload differs')
            problem = proto.ContestProblem()
            if problem.ParseFromString(raw) != len(raw): raise ValueError('Complete original schema parse required')
            end += len(raw)
            yield row, problem
        if stream and stream.read(1): raise ValueError('Unindexed final original shard bytes')
    finally:
        if stream: stream.close()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists() or REPORTS.exists():
        raise ValueError('Fresh CPU-hidden lossless candidate preparation required')
    if shutil.disk_usage(ROOT).free < 36 * 1024**3:
        raise ValueError('Fund bounded candidate above original writer reserve')
    manifest_path = ORIGINAL / 'manifest.json'; manifest = json.loads(manifest_path.read_text())
    readback = json.loads(READBACK.read_text())
    if not readback.get('every_original_record_and_extracted_byte_compared') or readback['actual_counts'] != manifest['actual_counts']:
        raise ValueError('Every complete original publisher record must already reconcile')
    schema_path = ROOT / 'data/code-contests-protobuf-schema-v1/manifest.json'
    schema = json.loads(schema_path.read_text())
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    if (file_hash(TOKENIZER) != '0072bb6de61da87bd9ba5f4e6739b594cc3e60c62c75a59d25d499629a70088f'
            or tokenizer.get_vocab_size() != 32768 or any(tokenizer.token_to_id(k) != v for k, v in [('<bos>', 1), ('<eos>', 2)])):
        raise ValueError('Exact existing own scratch BPE required')
    sys.path.insert(0, schema['isolated_reader_site_packages']); sys.path.insert(0, str(schema_path.parent))
    import contest_problem_pb2 as proto
    extra = [manifest_path, READBACK, schema_path, TOKENIZER, ROOT / 'configs/model.json',
        APPS / 'test-problems.json', APPS / 'train-problems.json',
        ROOT / 'data/public-apps-complete-pack-v2/manifest.json',
        ROOT / 'data/public-apps-complete-pack-v2/original-rows.jsonl.gz',
        ROOT / 'data/codenet-internal-splits-v1/problem-roles.json',
        ROOT / 'data/full-held-question-pool-v2/question-registry.jsonl',
        ROOT / 'data/verified-code-expanded-v1/reserved-mbpp.jsonl',
        ROOT / 'tests/test_code_contests_curriculum_candidates_v1.py',
        ROOT / 'reports/code-contests-curriculum-candidates-v1-tests.log',
        CACHE / 'publisher-README.md', CACHE / 'publisher-LICENSE']
    extra += python_sources(ROOT, [Path(__file__), ROOT / 'tests/test_code_contests_curriculum_candidates_v1.py'])
    files = {str(p): file_hash(p) for p in extra}; files.update(manifest['files_sha256'])
    protected_keys = defaultdict(list); protected_questions = defaultdict(list); protected_programs = defaultdict(list)
    prior_training_keys = set()
    test_inventory = json.loads((APPS / 'test-problems.json').read_text())
    train_inventory = json.loads((APPS / 'train-problems.json').read_text())
    validation = set()
    with gzip.open(ROOT / 'data/public-apps-complete-pack-v2/original-rows.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if row['split'] == 'validation': validation.add(row['source_problem_id'])
    for split, values in [('test', test_inventory), ('validation', {i: train_inventory[i] for i in validation})]:
        for identity, row in values.items():
            label = 'APPS:' + split + ':' + identity
            key = url_key(row['metadata'].get('url'))
            if key: protected_keys[key].append(label)
            folder = (APPS_TEST / 'original-test' if split == 'test' else APPS / 'original-train') / identity
            question = folder / 'question.txt'; expected = row['files']['question.txt']['sha256']
            if file_hash(question) != expected: raise ValueError('Every original held APPS problem text must match')
            files[str(question)] = expected
            protected_questions[question_hash(question.read_bytes().decode())].append(label)
    for identity, row in train_inventory.items():
        key = url_key(row['metadata'].get('url'))
        if key and identity not in validation: prior_training_keys.add(key)
    roles = json.loads((ROOT / 'data/codenet-internal-splits-v1/problem-roles.json').read_text())
    for identity, row in roles.items():
        key = ('CODENET', identity)
        if row['split'] == 'train': prior_training_keys.add(key)
        else: protected_keys[key].append('CodeNet:' + row['split'] + ':' + identity)
    for line in (ROOT / 'data/full-held-question-pool-v2/question-registry.jsonl').open():
        row = json.loads(line)
        protected_questions[question_hash(row['question'])].append(row['audit_id'])
    for line in (ROOT / 'data/verified-code-expanded-v1/reserved-mbpp.jsonl').open():
        row = json.loads(line); protected_programs[normalized_program_hash(row['code'])].append('reserved-MBPP:' + str(row['task_id']))
    guard = FileBindings(files)
    OUT.mkdir(); REPORTS.mkdir(); started = time.monotonic()
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    atomic_json(REPORTS / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner,
        source_files_sha256=files, all_original_problems=manifest['actual_counts']['problem_records'],
        all_publisher_positives_visited=manifest['actual_counts']['publisher_correct_solutions'],
        maximum_original_program_presentations_per_problem_and_language=3,
        selection='First publisher-order unique unmodified positive programs fitting full4096 native context',
        role_assignment='Known protected sources quarantined across exact-question/platform connected components; known prior train stays train; remaining component SHA256 modulo20==0 validation',
        named_APPS_original_test_problems=len(test_inventory), named_APPS_internal_validation_problems=len(validation),
        CodeNet_held_original_problem_IDs=sum(r['split'] != 'train' for r in roles.values()),
        all_original_data_retained_in_lossless_raw_source=True,
        original_private_and_generated_tests_not_in_model_input=True, incorrect_solutions_not_positive_targets=True,
        cross_platform_unresolved_URL_semantic_perceptual_and_all_prior_ancestry_not_certified=True,
        publisher_correctness_labels_not_fresh_execution=True,
        dataset_admitted_to_training=False, model_updates=0, goal_achieved=False))
    rows = [json.loads(line) for line in (ORIGINAL / 'all-original-problem-records.jsonl').open()]
    profiles = {}; counts = Counter(); selection_counts = Counter(); parent = {r['id']: r['id'] for r in rows}; seen = {}
    def find(identity):
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]; identity = parent[identity]
        return identity
    def link(a, b):
        a, b = find(a), find(b); parent[max(a,b)] = min(a,b)
    staged = REPORTS / 'all-selected-original-presentation-candidates.jsonl.gz'
    writer = None
    try:
        with gzip.open(staged, 'xb', compresslevel=1) as stream:
            for position, (row, problem) in enumerate(original_records(rows, proto)):
                identity = row['id']; keys = original_problem_keys(row); qhash = question_hash(problem.description)
                blockers = []
                for key in keys: blockers += protected_keys.get(key, [])
                blockers += protected_questions.get(qhash, [])
                selected, dispositions, program_matches = select_complete_solutions(problem, tokenizer, protected_programs, 3)
                for match in program_matches: blockers += match['protected_sources']
                profile = dict(id=identity, original_name=problem.name, original_source=row['original_source'],
                    known_platform_identity_keys=keys, complete_original_question_normalized_sha256=qhash,
                    direct_named_protected_source_matches=sorted(set(blockers)), protected_program_matches=program_matches,
                    known_previous_training_source=any(key in prior_training_keys for key in keys),
                    publisher_correct_solutions=row['publisher_correct_solutions'], candidate_presentations=len(selected),
                    original_solution_dispositions=dispositions,
                    all_original_public_private_generated_tests_preserved_in_raw_source=True)
                profiles[identity] = profile; selection_counts.update(dispositions)
                for key in [('question', qhash), *[('platform', *k) for k in keys]]:
                    if key in seen: link(identity, seen[key])
                    else: seen[key] = identity
                for chosen in selected:
                    emit(stream, dict(original_problem_id=identity, **chosen))
                counts['original_problems_visited'] += 1
                if (position + 1) % 128 == 0:
                    if staged.stat().st_size > 2 * 1024**3 or shutil.disk_usage(ROOT).free < 30 * 1024**3:
                        raise ValueError('Candidate staging storage reserve exceeded')
                    atomic_json(REPORTS / 'progress.json', dict(owner=owner, actual_original_problems=position+1,
                        total_original_problems=len(rows), all_original_solution_dispositions=dict(selection_counts),
                        elapsed_seconds=time.monotonic()-started, model_updates=0))
                    if (position+1) % 1024 == 0: print(json.dumps(dict(problems=position+1, targets=selection_counts['selected_complete_original_programs'])), flush=True)
        if (len(profiles) != len(rows) or selection_counts['original_publisher_correct_solutions_visited'] != manifest['actual_counts']['publisher_correct_solutions']):
            raise ValueError('Every original problem and positive must have a disposition')
        groups = defaultdict(list)
        for identity in profiles: groups[find(identity)].append(identity)
        assignments = {}; role_counts = Counter(); quarantined = Counter()
        for group, members in groups.items():
            protected = sorted({v for i in members for v in profiles[i]['direct_named_protected_source_matches']})
            prior_train = any(profiles[i]['known_previous_training_source'] for i in members)
            role = 'quarantine' if protected else ('train' if prior_train else ('validation' if int(hashlib.sha256(('own-CodeContests-role-v1:'+group).encode()).hexdigest()[:8], 16) % 20 == 0 else 'train'))
            for identity in members:
                assignments[identity] = dict(group_id=group, split=role, component_protected_source_matches=protected,
                    known_prior_training_source_in_component=prior_train)
                role_counts[role] += 1
                if protected: quarantined.update(protected)
        atomic_json(REPORTS / 'all-original-problem-source-dispositions.json',
                    [{**profiles[r['id']], **assignments[r['id']]} for r in rows])
        shutil.copy2(TOKENIZER, OUT / 'tokenizer.json')
        shutil.copy2(ROOT / 'configs/model.json', OUT / 'model-config.json')
        shutil.copy2(CACHE / 'publisher-README.md', OUT / 'publisher-README.md')
        shutil.copy2(CACHE / 'publisher-LICENSE', OUT / 'DATASET-LICENSE.txt')
        writer = CodePackWriter(OUT, maximum_bytes=2*1024**3, minimum_free_gib=26)
        language_counts = defaultdict(Counter); dropped = 0
        with gzip.open(staged, 'rt') as stream:
            for line in stream:
                candidate = json.loads(line); source_id = candidate.pop('original_problem_id'); assignment = assignments[source_id]
                if assignment['split'] == 'quarantine': dropped += 1; continue
                ids, labels = candidate.pop('ids'), candidate.pop('labels')
                row = dict(**candidate, id=source_id+':solution:'+str(candidate['original_solution_position']),
                    source_problem_id=source_id, **assignment, publisher_split='train',
                    original_source=profiles[source_id]['original_source'],
                    original_source_problem_name=profiles[source_id]['original_name'],
                    source='google-deepmind/code_contests:original-public-GCS-train',
                    publisher_correctness_label=True, fresh_local_execution_pending=True,
                    original_unchanged_program_bytes=True, full_original_question_not_clipped=True)
                writer.add(row, ids, labels); language_counts[assignment['split']][row['language']] += 1
        provenance = dict(original_lossless_manifest_sha256=file_hash(manifest_path), original_all_record_readback_sha256=file_hash(READBACK),
            original_source_split='train', actual_original_problems=len(rows),
            actual_original_positive_solutions_visited=selection_counts['original_publisher_correct_solutions_visited'],
            source_problem_dispositions=dict(role_counts), original_solution_dispositions=dict(selection_counts),
            quarantined_selected_presentations=dropped, complete_selected_presentations_by_role_and_language={k:dict(v) for k,v in language_counts.items()},
            known_held_source_quarantine_propagated_through_problem_components=True,
            source_semantic_independence_and_execution_reward_admission_pending=True,
            original_private_test_answers_not_in_model_prompts=True,
            original_publisher_terms_file=str(CACHE / 'publisher-README.md'),
            code_license='Apache-2.0 according to publisher README; retain original third-party terms caveats',
            non_code_license='CC-BY-4.0 according to publisher README; retain attribution and third-party terms caveats')
        result = writer.finish(provenance); writer = None
        guard.check(full=True)
        receipts = [REPORTS / 'plan.json', REPORTS / 'all-original-problem-source-dispositions.json', staged, OUT / 'manifest.json']
        atomic_json(REPORTS / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), complete=True,
            actual_original_problem_dispositions=dict(role_counts), actual_original_solution_dispositions=dict(selection_counts),
            actual_native_pack_counts=result['counts'], actual_native_pack_languages={k:dict(v) for k,v in language_counts.items()},
            all_original_train_problem_and_positive_solution_positions_visited=True,
            quarantined_selected_presentations=dropped,
            all_original_data_retained_in_lossless_raw_source=True,
            complete_original_source_program_and_native_target_masks_preserved=True,
            publisher_correctness_not_new_local_execution=True,
            actual_complete_pack_readback_and_local_execution_and_further_overlap_review_pending=True,
            source_files_sha256={str(p):file_hash(p) for p in receipts},
            dataset_admitted_to_training=False, new_training_started=False, model_updates=0, candidate_promoted=False, goal_achieved=False))
        print(json.dumps(dict(complete=str(REPORTS / 'complete.json'), native_counts=result['counts'],
            source_problem_dispositions=dict(role_counts), admitted_to_training=False)), flush=True)
    except BaseException as error:
        if writer is not None: writer.close()
        atomic_json(REPORTS / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
            dataset_admitted_to_training=False, model_updates=0, goal_achieved=False))
        raise


if __name__ == '__main__': main()
