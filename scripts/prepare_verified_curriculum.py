"""Freeze verified training rows after raw/sanitized benchmark-reference checks."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.verified_code import canonical_code
from slm_lab.code_tasks import TASKS


def prepare(source,cache,output):
    source,cache,output=map(Path,[source,cache,output]);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'manifest.json').read_text())
    for name,digest in manifest['files_sha256'].items():
        if file_hash(source/name)!=digest:raise ValueError('Verified source records changed')
    for item in manifest['inputs']['mbpp_sources']['files']:
        if file_hash(cache/item['path'])!=item['sha256']:raise ValueError('Pinned public benchmark source changed')
    reserved_codes=set();reserved_prompts=set()
    raw=list(jsonl(cache/'mbpp.jsonl'))
    sanitized=json.loads((cache/'sanitized-mbpp.json').read_text())
    for row in raw+sanitized:
        if row['task_id']>600:continue
        reserved_prompts.add(' '.join(row.get('text',row.get('prompt','')).casefold().split()))
        try:reserved_codes.add(hashlib.sha256(canonical_code(row['code']).encode()).hexdigest())
        except SyntaxError:pass
    for task in TASKS:
        reserved_codes.add(hashlib.sha256(canonical_code(task[4]).encode()).hexdigest())
        reserved_prompts.add(' '.join(task[2].casefold().split()))
    rows=list(jsonl(source/'train.jsonl'));rejected=set()
    for row in rows:
        if row['original_program_sha256'] in reserved_codes or any(
            text in ' '.join(row['prompt'].casefold().split()) for text in reserved_prompts if text):
            rejected.add(row['original_program_sha256'])
    retained=[row for row in rows if row['original_program_sha256'] not in rejected]
    if not retained:raise ValueError('No verified examples survive benchmark exclusions')
    configuration={'source_manifest_sha256':file_hash(source/'manifest.json'),
                   'raw_mbpp_sha256':file_hash(cache/'mbpp.jsonl'),'sanitized_mbpp_sha256':file_hash(cache/'sanitized-mbpp.json')}
    if (output/'manifest.json').exists():
        old=json.loads((output/'manifest.json').read_text())
        if old['configuration']!=configuration:raise ValueError('Frozen verified curriculum configuration differs')
        if file_hash(output/'train.jsonl')!=old['train_sha256']:raise ValueError('Frozen verified rows changed')
        return old
    (output/'train.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=True)+'\n' for row in retained))
    result={'configuration':configuration,'train_examples':len(retained),
            'unique_programs':len({row['original_program_sha256'] for row in retained}),
            'by_source':dict(Counter(row['source'] for row in retained)),
            'excluded_reference_programs':sorted(rejected),'source_inputs':manifest['inputs'],
            'train_sha256':file_hash(output/'train.jsonl'),
            'limitations':['Checks exclude exact normalized reference code and specification substrings within this supplement.',
                           'These checks do not audit all previous pretraining tokens or exclude semantic benchmark overlap.',
                           'Published-example verification is not proof of general correctness or optimal efficiency.']}
    atomic_json(output/'manifest.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--cache',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.source,args.cache,args.output),indent=2))


if __name__=='__main__':main()
