"""Verify every staged whole source doctest, preserving full targets and roles.

Only confined children execute published code. No GPU, generated target, model
update, held solution, or rewritten function is used to construct this pack.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from tokenizers import Tokenizer
from slm_lab.codenet_token_pack_v1 import CodePackWriter, CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.documented_doctest_v1 import execute, implementation_sources, PRELUDE, sha, training_example
from slm_lab.verified_code import canonical_code
from slm_lab.code_tasks import TASKS
from slm_lab.wikipedia_corpus import protected_questions

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/'data/documented-doctest-stage-v1'
OUTPUT = ROOT/'data/documented-doctest-verified-v1'
REPORTS = ROOT/'reports/documented-doctest-verified-v1'


def normal(text):
    return ' '.join(text.casefold().split())


def source_inventory(candidates, archive):
    """Compare every candidate with its exact row in the original byte segment."""
    positions = {c['original_source_segment_row']: c['source'] for c in candidates}
    if len(positions) != len(candidates):
        raise ValueError('Distinct original source positions required')
    digest = hashlib.sha256(); position = 0; matched = 0
    with Path(archive['retained_anchor']).open('rb') as f:
        f.seek(archive['offset']); end = archive['offset'] + archive['original_bytes']
        while f.tell() < end:
            line = f.readline()
            if not line or f.tell() > end:
                raise ValueError('Entire original source segment must reconstruct')
            digest.update(line)
            if position in positions:
                if json.loads(line) != positions[position]:
                    raise ValueError('Candidate is not the unchanged original publisher row')
                matched += 1
            position += 1
    if matched != len(candidates) or digest.hexdigest() != archive['original_sha256']:
        raise ValueError('Full original source segment/candidate population differs')
    return dict(original_segment_rows=position, exact_candidate_rows=matched, source_segment_sha256=digest.hexdigest())


def license_inventory(candidates, source_path, expected):
    """Retain every distinct supplied notice, checking its original text hash."""
    needed = {(c['source']['license_path'], c['source']['license_text_sha256']) for c in candidates}
    found = {}; digest = hashlib.sha256()
    with gzip.open(source_path, 'rb') as f:
        for line in f:
            digest.update(line); row = json.loads(line)
            key = (row['path'], hashlib.sha256(row['text'].encode('utf-8', 'surrogatepass')).hexdigest())
            if key in needed:
                if key in found and found[key] != row:
                    raise ValueError('Conflicting original publisher license notice')
                found[key] = row
    if digest.hexdigest() != expected or set(found) != needed:
        raise ValueError('Every exact supplied license text must reconstruct')
    return [found[k] for k in sorted(found)]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Public-source verification must hide every GPU')
    if OUTPUT.exists() or REPORTS.exists():
        raise ValueError('Preserve all complete or partial verification attempts')
    if shutil.disk_usage(ROOT).free < 27*1024**3:
        raise ValueError('Preserve the persistent 26GiB reserve plus the bounded pack')
    started = time.monotonic()
    inventory_path = STAGE/'inventory.json'; candidates_path = STAGE/'candidates.jsonl'
    inventory = json.loads(inventory_path.read_text())
    archive_path = Path(inventory['source_segment_archive']); archive = json.loads(archive_path.read_text())
    licenses_path = ROOT/'data/code-search-expanded-v1/parts/python/provided-licenses.jsonl.lossless-v1.gz'
    license_meta_path = licenses_path.with_suffix('.json'); license_meta = json.loads(license_meta_path.read_text())
    protected_path = ROOT/'data/code-search-expanded-v1/protected-repositories.txt'
    prior_path = ROOT/'data/verified-code-expanded-v1/train.jsonl'
    prior_meta_path = prior_path.with_name('manifest.json'); prior_meta = json.loads(prior_meta_path.read_text())
    mbpp_root = ROOT/'data/verified-code-source-cache-v1'
    mbpp_meta_path = mbpp_root/'sources.json'; mbpp_meta = json.loads(mbpp_meta_path.read_text())
    question_path = ROOT/'data/knowledge-eval-v1/questions.jsonl'
    tokenizer_path = ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'
    config_path = ROOT/'configs/model.json'
    expected = {candidates_path: inventory['candidate_rows_sha256'], archive_path: inventory['source_segment_archive_sha256'],
        Path(archive['retained_anchor']): archive['retained_anchor_sha256'], protected_path: inventory['protected_repositories_sha256'],
        licenses_path: license_meta['archive_sha256'], prior_path: prior_meta['files_sha256']['train.jsonl'],
        tokenizer_path: '0072bb6de61da87bd9ba5f4e6739b594cc3e60c62c75a59d25d499629a70088f'}
    expected.update({mbpp_root/r['path']: r['sha256'] for r in mbpp_meta['files']})
    paths = [Path(__file__), inventory_path, license_meta_path, prior_meta_path, mbpp_meta_path, question_path, config_path,
        ROOT/'src/slm_lab/codenet_token_pack_v1.py', ROOT/'src/slm_lab/verified_code.py', ROOT/'src/slm_lab/code_quality.py',
        ROOT/'src/slm_lab/code_tasks.py', ROOT/'src/slm_lab/wikipedia_corpus.py', ROOT/'src/slm_lab/corpus.py',
        ROOT/'src/slm_lab/diverse_images.py', ROOT/'tests/test_documented_doctest_v1.py', ROOT/'reports/documented-doctest-v1-tests.log']
    bound = {str(p.resolve()): digest for p, digest in expected.items()}
    bound.update({str(p.resolve()): file_hash(p) for p in paths}); bound.update(implementation_sources())
    for path, digest in bound.items():
        if file_hash(path) != digest:
            raise ValueError('Pinned original source, target, notice, or runtime differs: '+path)
    states = {p: (Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in bound}
    def guard(full=False):
        for path, digest in bound.items():
            s = Path(path).stat()
            if (s.st_ino, s.st_size, s.st_mtime_ns) != states[path] or (full and file_hash(path) != digest):
                raise ValueError('Bound source verification inputs changed: '+path)
    if '8 passed' not in (ROOT/'reports/documented-doctest-v1-tests.log').read_text():
        raise ValueError('The whole-example, role, context, isolation and target-mask checks must pass')
    candidates = list(map(json.loads, candidates_path.open()))
    if len(candidates) != inventory['complete_source_counts']['candidate_rows'] or len({c['source']['id'] for c in candidates}) != len(candidates):
        raise ValueError('Full fixed candidate population and unique original IDs required')
    source_proof = source_inventory(candidates, archive)
    protected = set(protected_path.read_text().splitlines())
    reserved_codes = {sha(canonical_code(t[4])) for t in TASKS}; reserved_prompts = {normal(t[2]) for t in TASKS}
    for row in map(json.loads, (mbpp_root/'mbpp.jsonl').open()):
        if row['task_id'] <= 600:
            reserved_prompts.add(normal(row['text']))
            try: reserved_codes.add(sha(canonical_code(row['code'])))
            except SyntaxError: pass
    seen = {row['original_program_sha256'] for row in map(json.loads, prior_path.open())}
    matcher = protected_questions(question_path); dispositions = {}; work = []
    for index, c in enumerate(candidates):
        source = c['source']; reason = None; digest = sha(canonical_code(c['program']))
        if source.get('language') != 'Python' or source.get('original_partition') != 'train':
            raise ValueError('A non-training publisher role reached verification')
        if source['group_id'].split(':', 2)[1].casefold() in protected:
            raise ValueError('An official held repository reached verification')
        prompt = normal(source['prompt'])
        if digest in reserved_codes: reason = 'exact_protected_reference_program'
        elif any(s in prompt for s in reserved_prompts if s): reason = 'exact_protected_coding_specification'
        elif any(matcher.iter(prompt)): reason = 'exact_protected_knowledge_question'
        elif digest in seen: reason = 'prior_verified_or_earlier_candidate_canonical_duplicate'
        else: seen.add(digest)
        base = dict(candidate_index=index, source_id=source['id'], canonical_program_sha256=digest,
            program_sha256=c['program_sha256'], training_pack_included=False)
        if reason: dispositions[index] = dict(**base, status=reason, source_code_executed=False)
        else: work.append((index, c, base))
    REPORTS.mkdir(); OUTPUT.mkdir(); writer = None; counts = Counter(); accepted = []; observations = {}
    plan = dict(format='own-whole-documented-source-doctest-plan-v1', recorded_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=bound, fixed_candidate_count=len(candidates), runtime_candidate_count=len(work),
        source_reconstruction=source_proof, declared_prelude=PRELUDE, declared_prelude_sha256=sha(PRELUDE),
        workers=2, per_source_wall_seconds=5, expected_text_stays_outside_worker=True, all_original_examples_required=True,
        exact_original_formatted_target_required=True, maximum_complete_context_tokens=4096,
        source_only_protected_literal_exclusions=True, certified_semantic_independence=False,
        no_new_holdout_created_from_train=True, previous_raw_code_exposure_not_erased=True,
        machine_execution_feedback_not_human_preference=True, new_model_updates=0, goal_achieved=False)
    atomic_json(REPORTS/'plan.json', plan)
    def status(message, **extra):
        atomic_json(REPORTS/'status.json', dict(status=message, observed_at=datetime.now(timezone.utc).isoformat(),
            completed_candidates=sum(counts.values()), counts=dict(counts), admitted_to_training=extra.pop('admitted_to_training',False),
            model_updates=0, goal_achieved=False, **extra))
    def verify(item):
        index, c, base = item
        try: result = execute(c)
        except (ValueError, SyntaxError) as e:
            result = dict(status='static_context_or_contract_qualification', qualification=str(e), source_code_executed=False)
        return index, dict(base, **result)
    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path)); config = json.loads(config_path.read_text())
        if config['max_text_length'] != 4096 or config['vocab_size'] != 32768 or [tokenizer.token_to_id(t) for t in ('<pad>','<bos>','<eos>')] != [0,1,2]:
            raise ValueError('Canonical native tokenizer/window differs')
        shutil.copyfile(tokenizer_path, OUTPUT/'tokenizer.json'); shutil.copyfile(config_path, OUTPUT/'model-config.json')
        writer = CodePackWriter(OUTPUT, maximum_bytes=512*1024**2, minimum_free_gib=26)
        status('executing every eligible complete original published example in confined children')
        with (REPORTS/'observations.jsonl').open('x') as journal, ThreadPoolExecutor(max_workers=2) as pool:
            def record(index, result):
                c = candidates[index]
                if result.get('all_original_examples_passed'):
                    try: row, ids, labels = training_example(c, result, tokenizer)
                    except ValueError as e:
                        result['pack_context_qualification'] = str(e)
                    else:
                        writer.add(row, ids, labels); accepted.append(c); observations[c['source']['id']] = result
                        result['training_pack_included'] = True
                counts[result['status']] += 1
                journal.write(json.dumps(result, ensure_ascii=False)+'\n'); journal.flush()
                if sum(counts.values()) % 100 == 0:
                    guard(); writer.guard(); status('whole-source verification in progress', complete_training_pack_candidates=len(accepted))
                    print(json.dumps(dict(completed=sum(counts.values()), accepted=len(accepted), counts=dict(counts))), flush=True)
            for index, result in dispositions.items(): record(index, result)
            for index, result in pool.map(verify, work): record(index, result)
        if sum(counts.values()) != len(candidates):
            raise ValueError('Every fixed original candidate requires an explicit disposition')
        notices = license_inventory(accepted, licenses_path, license_meta['original_sha256'])
        with (OUTPUT/'publisher-license-notices.jsonl').open('x') as f:
            for notice in notices: f.write(json.dumps(notice, ensure_ascii=True)+'\n')
        (OUTPUT/'DATASET-LICENSE.txt').write_text('Original CodeSearchNet/GitHub source notices apply per record. No new relicensing is asserted.\n'
            'Full supplied notice text, original source URLs, notice paths and UTF-8/surrogatepass SHA256 hashes are retained.\n'
            'See publisher-license-notices.jsonl and each parent_source. Authorship and legal suitability are not independently verified.\n')
        (OUTPUT/'publisher-README.md').write_text('Existing publisher-train CodeSearchNet Python descriptions and formatted implementations.\n'
            'Every original example command executes without edits under the explicit stdlib prelude and existing Landlock/seccomp sandbox.\n'
            'All original output expectations are compared outside the worker. Complete existing formatted targets are unchanged.\n'
            'Declared context replaces the old implicit-surrounding-module footer. Targets exceeding 4096 tokens are qualified, never truncated.\n'
            'Exact held repositories/specifications/references and prior verified normalized functions are excluded. Semantic independence is not certified.\n'
            'This is finite source-example execution feedback, not general correctness, efficiency, fresh human preferences or model-quality evidence.\n'
            'The original raw functions may already have appeared in earlier pretraining. No held examples become training data.\n')
        guard(full=True)
        manifest = writer.finish(dict(plan_sha256=file_hash(REPORTS/'plan.json'), observations_sha256=file_hash(REPORTS/'observations.jsonl'),
            source_files_sha256=bound, declared_prelude_sha256=sha(PRELUDE), full_supplied_notices_sha256=file_hash(OUTPUT/'publisher-license-notices.jsonl'),
            finite_source_examples_only=True, previous_raw_source_exposure_retained=True))
        writer = None
        pack = CompleteCodePack(OUTPUT, 'train', purpose='audit')
        with gzip.open(OUTPUT/'original-rows.jsonl.gz', 'rt') as f: rows = list(map(json.loads, f))
        if len(rows) != len(accepted) or len(pack) != len(accepted): raise ValueError('Complete pack population differs')
        for index, (c, packed_row) in enumerate(zip(accepted, rows)):
            row, ids, labels = training_example(c, observations[c['source']['id']], tokenizer)
            if any(packed_row[k] != v for k, v in row.items()) or pack.sequence(index) != (ids, labels):
                raise ValueError('Full original source/target/token/mask readback differs')
        guard(full=True)
        result = dict(format='own-whole-documented-source-doctest-training-v1', observed_at=datetime.now(timezone.utc).isoformat(),
            dataset_manifest_sha256=file_hash(OUTPUT/'manifest.json'), admitted_to_training=True,
            plan_sha256=file_hash(REPORTS/'plan.json'), observations_sha256=file_hash(REPORTS/'observations.jsonl'),
            all_candidate_count=len(candidates), all_candidate_dispositions=dict(counts), complete_training_programs=len(pack),
            new_unique_functions_with_fewer_than_two_old_literal_calls=sum(c['old_literal_call_example_count']<2 for c in accepted),
            published_examples=sum(c['whole_original_example_count'] for c in accepted), counts=manifest['counts'],
            license_counts=dict(Counter(c['source']['license'] for c in accepted)), supplied_notice_count=len(notices),
            unchanged_original_targets_and_full_token_masks_read_back=True, exact_original_source_rows_read_back=True,
            runtime_execution_replayed_by_reader=False, model_updates=0, fresh_human_preference_labels=0,
            finite_source_examples_not_general_correctness_or_efficiency=True, certified_semantic_independence=False,
            independent_reader_pending=True, source_files_sha256=bound, seconds=time.monotonic()-started, goal_achieved=False)
        atomic_json(OUTPUT/'admission.json', result)
        status('whole-source verification and complete token-pack admission finished; independent reader pending',
            complete_training_pack_candidates=len(pack), admitted_to_training=True, admission_sha256=file_hash(OUTPUT/'admission.json'))
        print(json.dumps({k:v for k,v in result.items() if k!='source_files_sha256'}), flush=True)
    except BaseException as e:
        status('verification stopped; preserve partial output and inspect before any training use', error=str(e))
        raise
    finally:
        if writer is not None: writer.close()


if __name__ == '__main__': main()
