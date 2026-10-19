"""Retain all original APPS test questions/cases strictly for evaluation.

No original test solutions are extracted, encoded or exposed to training.
Every unsupported original remains preserved as an explicit review.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from tokenizers import Tokenizer

from slm_lab.apps_published_cases_v2 import contract, implementation_sources
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT/'data/public-apps-original-inventory-v1'
ARCHIVE = ROOT/'data/public-apps-publisher-source-v1/APPS.tar.gz'
TOKENIZER = ROOT/'checkpoints/dialogue-retention-v1/systems-variety-code/tokenizer.json'
OUTPUT = ROOT/'data/public-apps-publisher-test-evaluation-v1'
KEEP = {'question.txt', 'input_output.json', 'metadata.json', 'starter_code.py'}


def main():
    if OUTPUT.exists():
        raise ValueError('Preserve original test evidence; no implicit re-extraction')
    inventory = json.loads((INVENTORY/'inventory.json').read_text())
    expected = json.loads((INVENTORY/'test-problems.json').read_text())
    if (len(expected) != 5000 or any(inventory['publisher_train_test_overlap'].values())
            or not inventory['independent_second_archive_pass_reconciles_every_member_and_extracted_train_byte']):
        raise ValueError('Complete original role audit required')
    if file_hash(ARCHIVE) != inventory['original_archive_sha256']:
        raise ValueError('Exact original publisher archive required')
    tokenizer_sha = file_hash(TOKENIZER)
    if tokenizer_sha != '0072bb6de61da87bd9ba5f4e6739b594cc3e60c62c75a59d25d499629a70088f':
        raise ValueError('Original own scratch-trained tokenizer required')
    if shutil.disk_usage(ROOT).free < 28*1024**3:
        raise ValueError('Fund at most2GiB original test copies above26GiB reserve')
    files = implementation_sources()
    for p in [Path(__file__).resolve(), INVENTORY/'inventory.json', INVENTORY/'test-problems.json',
            INVENTORY/'train-problems.json', TOKENIZER]:
        files[str(p)] = file_hash(p)
    OUTPUT.mkdir()
    stored = 0
    copied = {}
    with tarfile.open(ARCHIVE, 'r|gz') as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if len(path.parts) != 4 or path.parts[:2] != ('APPS', 'test'):
                continue
            problem, name = path.parts[2:]
            if name not in KEEP:
                continue
            if (not member.isfile() or problem not in expected or name not in expected[problem]['files']
                    or member.size != expected[problem]['files'][name]['bytes']):
                raise ValueError('Exact original regular test member/schema required')
            target = OUTPUT/'original-test'/problem/name
            key = str(target.relative_to(OUTPUT))
            if key in copied:
                raise ValueError('Repeated original test file')
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            count = 0
            source = archive.extractfile(member)
            with source, target.open('xb') as stream:
                while True:
                    block = source.read(1024**2)
                    if not block:
                        break
                    count += len(block)
                    stored += len(block)
                    if stored > 2*1024**3:
                        raise ValueError('Original test copy exceeds sealed2GiB bound')
                    stream.write(block)
                    digest.update(block)
            if count != member.size or digest.hexdigest() != expected[problem]['files'][name]['sha256']:
                raise ValueError('Complete original test bytes differ from independently inventoried archive')
            copied[key] = digest.hexdigest()
    wanted = {f'original-test/{i}/{name}' for i,row in expected.items() for name in row['files'] if name in KEEP}
    if set(copied) != wanted:
        raise ValueError('Every original test question/case/metadata/starter file required')
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    rows = []
    counts = Counter()
    modes = Counter()
    difficulties = Counter()
    case_count = 0
    for problem in sorted(expected):
        item = expected[problem]
        folder = OUTPUT/'original-test'/problem
        question = (folder/'question.txt').read_text()
        starter = (folder/'starter_code.py').read_text() if (folder/'starter_code.py').exists() else None
        raw = (folder/'input_output.json').read_bytes()
        reasons = []
        try:
            cases = contract(raw)
        except ValueError as error:
            cases = None
            reasons.append(str(error))
        # The interface name is metadata, never a test answer in a prompt. For
        # an oversized/unsupported contract the entire source stays in review.
        name = cases['function'] if cases else None
        interface = ('Use the original required function or class interface.' if starter or name
            else 'Read standard input and write standard output as specified.')
        prompt = ('Write a Python solution to the following problem. '+interface+
            ' Return only the complete code.\n\n'+question)
        if starter:
            prompt += '\n\nOriginal starter code:\n'+starter
        prefix = [tokenizer.token_to_id('<bos>')]+tokenizer.encode('User: '+prompt+'\nAssistant:').ids
        if len(prefix)+1024 > 4096:
            reasons.append('complete_original_prompt_plus1024generation_exceeds4096')
        if any(marker in prompt for marker in ['<pad>', '<bos>', '<eos>']):
            reasons.append('literal_reserved_token_marker_requires_review')
        supported = not reasons
        for reason in reasons:
            counts[reason] += 1
        if supported:
            modes[cases['mode']] += 1
            difficulties[item['metadata']['difficulty']] += 1
            case_count += cases['total_cases']
        rows.append(dict(id='apps:publisher-test:'+problem, publisher_split='test', purpose='evaluation_only',
            problem=problem, difficulty=item['metadata']['difficulty'], source_url=item['metadata'].get('url'),
            original_question_sha256=item['files']['question.txt']['sha256'],
            original_tests_sha256=item['files']['input_output.json']['sha256'],
            original_starter_sha256=item['files'].get('starter_code.py',{}).get('sha256'),
            complete_prompt=prompt, complete_prompt_tokens=len(prefix),
            complete_prompt_input_ids_sha256=hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest(),
            mode=cases['mode'] if cases else None, function=name,
            complete_published_case_count=cases['total_cases'] if cases else None,
            supported_by_declared_local_protocol=supported, review_reasons=reasons,
            original_source_bytes_preserved=True, source_program_reference_extracted=False,
            optimization_forbidden=True, generation_budget=1024, full_context_window=4096,
            input_or_case_clipping_permitted=False))
    records = OUTPUT/'evaluation-records.jsonl'
    records.write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in rows))
    copied[records.name] = file_hash(records)
    # Original question/full-source URL separation is checked directly again;
    # this does not certify absence from any older public pretraining corpus.
    train = json.loads((INVENTORY/'train-problems.json').read_text())
    train_questions = {r['normalized_question_sha256'] for r in train.values()}
    train_urls = {r['metadata'].get('url') for r in train.values() if r['metadata'].get('url')}
    overlaps = [i for i,r in expected.items() if r['normalized_question_sha256'] in train_questions
        or (r['metadata'].get('url') and r['metadata']['url'] in train_urls)]
    if overlaps:
        raise ValueError('Original publisher test questions/URLs overlap publisher train')
    if file_hash(ARCHIVE) != inventory['original_archive_sha256']:
        raise ValueError('Original archive changed during test preparation')
    summary = dict(completed_at=datetime.now(timezone.utc).isoformat(), format='own-original-APPS-test-evaluation-contracts-v1',
        original_publisher_test_problems=len(rows), original_test_bytes_retained=stored,
        original_test_files_retained=len(copied)-1, original_test_solution_files_extracted=0,
        supported_complete_test_problems=sum(r['supported_by_declared_local_protocol'] for r in rows),
        supported_complete_published_cases=case_count, supported_by_interface=dict(modes), supported_by_difficulty=dict(difficulties),
        original_problem_reviews=sum(bool(r['review_reasons']) for r in rows), review_counts=dict(counts),
        whole_original_prompts_and_case_sets_required=True, original_source_test_bytes_clipped=0,
        publisher_train_full_question_or_URL_matches=0, earlier_public_corpus_or_semantic_independence_certified=False,
        tokenizer_sha256=tokenizer_sha, source_files_sha256=files, files_sha256=copied,
        original_archive_sha256=inventory['original_archive_sha256'], original_publisher_revision=inventory['original_publisher_revision'],
        generation_protocol=dict(decoding='greedy', max_new_tokens=1024, context_window=4096,
            precision='BF16 on CUDA; FP32 CPU results must be reported separately', original_complete_prompt_required=True,
            extraction='Complete returned code or one Python code fence; no repair, no answer/solution in prompt'),
        execution_protocol='slm_lab.apps_published_cases_v2, stdlib-only local contract; not official publisher runtime',
        selection_before_model_generations=True, model_generations=0, model_updates=0,
        training_admitted=False, optimization_forbidden=True, official_APPS_score=False, goal_achieved=False)
    atomic_json(OUTPUT/'manifest.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in {'files_sha256','source_files_sha256'}},indent=2),flush=True)


if __name__ == '__main__':
    main()
