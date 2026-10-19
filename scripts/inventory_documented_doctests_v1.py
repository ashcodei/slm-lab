"""Inventory retained publisher-train Python functions with whole doctests.

Read the exact losslessly retained source segment; never restore or edit a
bound source. This stage parses data only and admits no new training targets.
"""
from collections import Counter
from datetime import datetime,timezone
import ast
import doctest
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'data/documented-doctest-stage-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Source-only inventory must hide every GPU')
    if OUTPUT.exists():raise ValueError('Preserve prior complete or partial source inventory')
    from slm_lab.corpus import file_hash
    from slm_lab.code_quality import instruction_code
    from slm_lab.diverse_images import atomic_json
    from slm_lab.verified_code import doctest_cases
    archive=ROOT/'data/code-search-expanded-v1/source-copy-archives-v1/python-sft-train.json';meta=json.loads(archive.read_text())
    anchor=Path(meta['retained_anchor']);protected_path=ROOT/'data/code-search-expanded-v1/protected-repositories.txt'
    protected=set(protected_path.read_text().splitlines())
    if file_hash(anchor)!=meta['retained_anchor_sha256']:raise ValueError('The entire retained immutable source anchor changed')
    OUTPUT.mkdir();started=time.monotonic();counts=Counter();licenses=Counter();digest=hashlib.sha256();examples=0;selected=[]
    with anchor.open('rb') as f,(OUTPUT/'candidates.jsonl').open('x') as out:
        f.seek(meta['offset']);end=meta['offset']+meta['original_bytes'];position=0
        while f.tell()<end:
            line=f.readline()
            if not line or f.tell()>end:raise ValueError('The complete retained original JSONL segment must reconstruct')
            digest.update(line);row=json.loads(line);counts['all_original_rows']+=1
            if row.get('language')!='Python' or row.get('original_partition')!='train':raise ValueError('Only original publisher-train Python source is eligible')
            repo=row['group_id'].split(':',2)[1].casefold()
            if repo in protected:raise ValueError('An official held repository appeared in the training source')
            if '>>>' not in row['prompt']:position+=1;continue
            counts['rows_with_doctest_markers']+=1
            try:tests=doctest.DocTestParser().get_examples(row['prompt'])
            except ValueError:counts['malformed_original_doctest_text']+=1;position+=1;continue
            if len(tests)<2:counts['fewer_than_two_whole_examples']+=1;position+=1;continue
            counts['rows_with_two_or_more_whole_examples']+=1
            try:
                code=instruction_code(row['response']);tree=ast.parse(code)
                functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
                if len(functions)!=1 or len(tree.body)!=1:raise ValueError('Not one ordinary complete function')
                node=functions[0];arguments=[a.arg for a in node.args.posonlyargs+node.args.args]
                if arguments and arguments[0] in ('self','cls'):raise ValueError('Needs missing class or receiver context')
                if any(e.options.get(doctest.SKIP,False) for e in tests):raise ValueError('Some original examples request SKIP')
            except (SyntaxError,ValueError) as error:
                counts['static_context_or_skip_qualification']+=1;position+=1;continue
            literal=len(doctest_cases(row['prompt'],node.name));counts['candidate_rows']+=1;examples+=len(tests);licenses[row['license']]+=1
            counts['candidates_with_two_literal_call_examples']+=int(literal>=2)
            candidate=dict(source=row,original_source_segment_row=position,function=node.name,program=code,doctest_text=row['prompt'],
                whole_original_example_count=len(tests),old_literal_call_example_count=literal,
                existing_formatted_source_target_sha256=hashlib.sha256(row['response'].encode()).hexdigest(),
                program_sha256=hashlib.sha256(code.encode()).hexdigest(),doctest_text_sha256=hashlib.sha256(row['prompt'].encode()).hexdigest(),
                source_code_executed=False,dataset_admitted_to_training=False)
            out.write(json.dumps(candidate,ensure_ascii=False)+'\n');selected.append(row['id']);position+=1
    if digest.hexdigest()!=meta['original_sha256']:raise ValueError('Every original source byte in the retained segment must hash exactly')
    # Audit the already-admitted APPS interface balance rather than infer it
    # from example programs or change the existing frozen prompts.
    apps=Counter()
    with gzip.open(ROOT/'data/APPS-verified-generation-v1/original-rows.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line);fn=row.get('original_fn_name');apps['original_programs']+=1;apps['function' if fn else 'stdin']+=1
            if fn:apps['required_function_name_missing_from_prompt']+=int(fn not in row['prompt'])
    atomic_json(OUTPUT/'inventory.json',dict(observed_at=datetime.now(timezone.utc).isoformat(),source_segment_archive=str(archive),
        source_segment_archive_sha256=file_hash(archive),retained_anchor=str(anchor),retained_anchor_sha256=meta['retained_anchor_sha256'],
        exact_source_segment_sha256=digest.hexdigest(),protected_repositories_sha256=file_hash(protected_path),
        complete_source_counts=dict(counts),whole_candidate_examples=examples,provided_license_counts=dict(licenses),
        candidate_rows_sha256=file_hash(OUTPUT/'candidates.jsonl'),APPS_interface_balance=dict(apps),seconds=time.monotonic()-started,
        source_code_executed=False,new_training_targets_admitted=False,original_source_and_benchmarks_unchanged=True,
        source_comments_and_code_authorship_not_independently_verified=True,model_updates=0,goal_achieved=False))
    print(json.dumps(dict(counts=counts,examples=examples,APPS_interface=apps,inventory=str(OUTPUT/'inventory.json'))),flush=True)


if __name__=='__main__':main()
