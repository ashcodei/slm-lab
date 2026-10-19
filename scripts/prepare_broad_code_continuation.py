"""Prepare additional original library code with explicit language metadata."""
from argparse import Namespace
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import ahocorasick
from slm_lab.corpus import file_hash, pack
from slm_lab.cpp_tasks import reference
from slm_lab.code_tasks import TASKS
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.exercise_cpp_runner import extract_files
from slm_lab.mix_corpora import mix
from slm_lab.wikipedia_corpus import protected_questions

ROOT = Path(__file__).resolve().parents[1]
DISPLAY = {'.cpp': 'C++', '.cc': 'C++', '.hpp': 'C++', '.h': 'C/C++ header', '.c': 'C',
           '.rs': 'Rust', '.py': 'Python', '.sh': 'Shell', '.md': 'Documentation',
           '.rst': 'Documentation', '.txt': 'Documentation'}


def verify_packed(root, tokenizer_sha):
    root = Path(root); manifest = json.loads((root / 'corpus.json').read_text())
    if manifest['tokenizer_sha256'] != tokenizer_sha or manifest['dtype'] != '<u2': raise ValueError('Code corpus tokenizer/dtype mismatch')
    for split in manifest['splits'].values():
        if sum(row['tokens'] for row in split['shards']) != split['tokens']: raise ValueError('Code shard token totals differ')
        for row in split['shards']:
            path = root / row['path']
            if path.stat().st_size != 2 * row['tokens'] or file_hash(path) != row['sha256']: raise ValueError('Code shard bytes/hash differ')
    return manifest


def main():
    source = ROOT / 'data/reviewed-systems-code-v1'; manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['failures'] or len(manifest['sources']) != 12 or file_hash(source / 'all.jsonl') != manifest['all_sha256']:
        raise ValueError('Require the complete finalized twelve-source library collection')
    for item in manifest['sources']:
        if file_hash(source / (item['name'] + '.jsonl')) != item['sha256']: raise ValueError('Original library source JSON changed')
        if file_hash(source / 'licenses' / (item['name'] + '.txt')) != item['license_sha256']: raise ValueError('Stored publisher license changed')
        if item.get('archive_path') and file_hash(item['archive_path']) != item['archive_sha256']: raise ValueError('Stored original archive changed')
    output = ROOT / 'data/broad-code-continuation-v1'
    if output.exists(): raise ValueError('Existing preparation needs inspection, not overwrite')
    output.mkdir(); rows = list(jsonl(source / 'all.jsonl'))
    questions = ROOT / 'data/knowledge-eval-v1/questions.jsonl'; knowledge = protected_questions(questions)
    code = ahocorasick.Automaton()
    for task in TASKS:
        text = ''.join(reference(task[0]).split())
        if len(text) >= 80: code.add_word(text, 'local-cpp:' + task[0])
    for row in jsonl(ROOT / 'data/exercism-code-v1/valid.jsonl'):
        if row['language'] != 'C++': continue
        for body in extract_files(row['response']).values():
            text = ''.join(body.split())
            if len(text) >= 80: code.add_word(text, row['id'])
    code.make_automaton(); blocked = {}; seen = set(); duplicates = 0
    for row in rows:
        if next(knowledge.iter(' '.join(row['text'].casefold().split())), None) is not None:
            blocked[row['group_id']] = 'exact reserved knowledge question'
        elif next(code.iter(''.join(row['text'].split())), None) is not None:
            blocked[row['group_id']] = 'whitespace-normalized reserved C++ reference'
    prepared = output / 'new-libraries.jsonl'; languages = Counter(); groups = set(); admitted = 0
    with prepared.open('w') as target:
        for row in rows:
            if row['group_id'] in blocked: continue
            language = DISPLAY.get(row['language_extension'], row['language_extension'])
            item = {**row, 'text': 'Programming language: ' + language + '\n' + row['text'],
                    'declared_language': language, 'source_chunk_id': row['id']}
            digest = hashlib.sha256(item['text'].encode()).hexdigest()
            if digest in seen: duplicates += 1; continue
            seen.add(digest); item['id'] = digest; groups.add(row['group_id']); languages[language] += 1; admitted += 1
            target.write(json.dumps(item, ensure_ascii=False) + '\n')
    tokenizer = ROOT / 'data/tokenizer-corpus/tokenizer.json'; tokenizer_sha = file_hash(tokenizer)
    pack(Namespace(jsonl=str(prepared), output=str(output / 'new-packed'), tokenizer=str(tokenizer),
                   max_tokens=100_000_000, shard_tokens=8_000_000, dataset=None))
    new = verify_packed(output / 'new-packed', tokenizer_sha)
    if new['documents'] != admitted: raise ValueError('New library corpus hit budget or skipped source chunks')
    old_root = ROOT / 'data/packed-code-combined-v1'; old = verify_packed(old_root, tokenizer_sha)
    combined = mix([('prior-broad-code', old_root, None), ('new-systems-libraries', output / 'new-packed', None)], output / 'packed')
    verify_packed(output / 'packed', tokenizer_sha)
    report = {'prepared_at': datetime.now(timezone.utc).isoformat(), 'source_manifest_sha256': file_hash(source / 'manifest.json'),
              'source_collection': str(source), 'publisher_repositories': len(manifest['sources']), 'source_chunks': len(rows),
              'admitted_new_chunks': admitted, 'original_file_groups': len(groups), 'declared_language_chunks': dict(languages),
              'excluded_source_groups': blocked, 'exact_duplicate_chunks_removed': duplicates,
              'prepared_jsonl_sha256': file_hash(prepared), 'tokenizer_sha256': tokenizer_sha,
              'new_packed_manifest_sha256': file_hash(output / 'new-packed/corpus.json'),
              'prior_packed_manifest_sha256': file_hash(old_root / 'corpus.json'),
              'packed_manifest_sha256': file_hash(output / 'packed/corpus.json'),
              'old_train_tokens_rehearsed': old['splits']['train']['tokens'],
              'new_train_tokens': new['splits']['train']['tokens'], 'new_valid_tokens': new['splits']['valid']['tokens'],
              'combined_train_tokens': combined['splits']['train']['tokens'], 'combined_valid_tokens': combined['splits']['valid']['tokens'],
              'protected_questions_sha256': file_hash(questions), 'preparation_source_sha256': file_hash(__file__),
              'goal_achieved': False, 'trained': False,
              'limitations': ['Prior214M training-code tokens are rehearsal, not newly collected tokens.',
                              'Exact prospective question/reference filters do not rule out semantic benchmark overlap or prior contamination.',
                              'Cross-component near duplicates and project/API correctness are not exhaustively audited.',
                              'Raw library code is data only; original projects are not imported/executed; no external teacher/weights.',
                              'This adds training material; held-out generation and full capability guards still decide usefulness.']}
    atomic_json(output / 'manifest.json', report); atomic_json(ROOT / 'reports/broad-code-continuation-input-audit-v1.json', report)
    print('Broad code corpus ready', report['combined_train_tokens'], 'train tokens; new', report['new_train_tokens'], flush=True)


if __name__ == '__main__': main()
