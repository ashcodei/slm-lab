"""Read every original admitted target/mask, then reexecute all source examples.

This separate process checks retained source rows, literal source exclusions,
full native framing, supplied notices and another confined run of every admitted
function. It does not train or certify general correctness/efficiency.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import doctest
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from tokenizers import Tokenizer
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.documented_doctest_v1 import PRELUDE, execute, sha
from slm_lab.code_tasks import TASKS
from slm_lab.verified_code import canonical_code
from slm_lab.wikipedia_corpus import protected_questions

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/documented-doctest-verified-v1'
REPORTS = ROOT/'reports/documented-doctest-verified-v1'
RESULT = REPORTS/'independent-complete-source-and-runtime-readback-v1.json'


def output_matches(test, raw):
    flags = sum(bit for bit, on in test.options.items() if on)
    checker = doctest.OutputChecker()
    if raw['exception_only'] is None:
        return test.exc_msg is None and checker.check_output(test.want, raw['stdout'], flags)
    if test.exc_msg is None: return False
    if checker.check_output(test.exc_msg, raw['exception_only'], flags): return True
    if flags & doctest.IGNORE_EXCEPTION_DETAIL:
        return test.exc_msg.split(':', 1)[0].split('.')[-1].strip() == raw['exception_only'].split(':', 1)[0].split('.')[-1].strip()
    return False


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('Every GPU must be hidden')
    if RESULT.exists() or (REPORTS/'independent-runtime-replay-v1.jsonl').exists():
        raise ValueError('Preserve prior independent readback attempts')
    started = time.monotonic(); plan = json.loads((REPORTS/'plan.json').read_text())
    admission = json.loads((DATA/'admission.json').read_text()); manifest = json.loads((DATA/'manifest.json').read_text())
    if (admission['format'] != 'own-whole-documented-source-doctest-training-v1' or admission['admitted_to_training'] is not True
            or admission['dataset_manifest_sha256'] != file_hash(DATA/'manifest.json')
            or admission['plan_sha256'] != file_hash(REPORTS/'plan.json')
            or admission['observations_sha256'] != file_hash(REPORTS/'observations.jsonl')):
        raise ValueError('Complete original data-admission chain required')
    bound = dict(plan['source_files_sha256'])
    additional = [Path(__file__), DATA/'admission.json', DATA/'manifest.json', REPORTS/'plan.json', REPORTS/'observations.jsonl',
        DATA/'publisher-license-notices.jsonl'] + [DATA/name for name in manifest['files_sha256']]
    bound.update({str(p.resolve()): file_hash(p) for p in additional})
    def guard():
        for p, digest in bound.items():
            if file_hash(p) != digest: raise ValueError('Original bound source/readback input changed: '+p)
    guard()
    if file_hash(DATA/'publisher-license-notices.jsonl') != manifest['provenance']['full_supplied_notices_sha256']:
        raise ValueError('Complete supplied notice bytes differ')
    candidates = list(map(json.loads, (ROOT/'data/documented-doctest-stage-v1/candidates.jsonl').open()))
    observations = list(map(json.loads, (REPORTS/'observations.jsonl').open()))
    if len(observations) != len(candidates) or sorted(o['candidate_index'] for o in observations) != list(range(len(candidates))):
        raise ValueError('Every original candidate needs exactly one observation')
    observed = {o['candidate_index']: o for o in observations}
    for index, c in enumerate(candidates):
        if observed[index]['source_id'] != c['source']['id'] or observed[index]['program_sha256'] != sha(c['program']):
            raise ValueError('Original observed candidate identities differ')
    counts = Counter(o['status'] for o in observations)
    if dict(counts) != admission['all_candidate_dispositions']: raise ValueError('Original complete disposition counts differ')
    archive = json.loads((ROOT/'data/code-search-expanded-v1/source-copy-archives-v1/python-sft-train.json').read_text())
    wanted = {c['original_source_segment_row']: c['source'] for c in candidates}; matched = 0; position = 0; digest = hashlib.sha256()
    with Path(archive['retained_anchor']).open('rb') as f:
        f.seek(archive['offset']); end = archive['offset']+archive['original_bytes']
        while f.tell() < end:
            line = f.readline()
            if not line or f.tell() > end: raise ValueError('Whole original source byte segment required')
            digest.update(line)
            if position in wanted:
                if json.loads(line) != wanted[position]: raise ValueError('Original candidate row differs')
                matched += 1
            position += 1
    if matched != len(candidates) or digest.hexdigest() != archive['original_sha256']:
        raise ValueError('Every candidate must match the complete retained original segment')
    protected = set((ROOT/'data/code-search-expanded-v1/protected-repositories.txt').read_text().splitlines())
    forbidden = {sha(canonical_code(t[4])) for t in TASKS}; prompts = {' '.join(t[2].casefold().split()) for t in TASKS}
    for row in map(json.loads, (ROOT/'data/verified-code-source-cache-v1/mbpp.jsonl').open()):
        if row['task_id'] <= 600:
            prompts.add(' '.join(row['text'].casefold().split()))
            try: forbidden.add(sha(canonical_code(row['code'])))
            except SyntaxError: pass
    prior = {r['original_program_sha256'] for r in map(json.loads, (ROOT/'data/verified-code-expanded-v1/train.jsonl').open())}
    matcher = protected_questions(ROOT/'data/knowledge-eval-v1/questions.jsonl')
    selected = [(i,c) for i,c in enumerate(candidates) if observed[i]['training_pack_included']]
    pack = CompleteCodePack(DATA, 'train', purpose='training')
    with gzip.open(DATA/'original-rows.jsonl.gz', 'rt') as f: rows = list(map(json.loads, f))
    if len(pack) != len(rows) or len(pack) != len(selected) or len(pack) != admission['complete_training_programs']:
        raise ValueError('Every admitted complete function must be present once')
    notices = {}
    for r in map(json.loads, (DATA/'publisher-license-notices.jsonl').open()):
        key = (r['path'],hashlib.sha256(r['text'].encode('utf-8','surrogatepass')).hexdigest()); notices[key] = r
    tokenizer = Tokenizer.from_file(str(DATA/'tokenizer.json')); total_context = total_target = 0; example_count = 0; used_codes = set(); maximum = 0
    footer = 'The surrounding module or class supplies its existing imports, types and helpers. Return only the implementation.'
    for index, ((candidate_index, c), row) in enumerate(zip(selected, rows)):
        source = c['source']; o = observed[candidate_index]; code_hash = sha(canonical_code(c['program']))
        if (source['original_partition'] != 'train' or source['language'] != 'Python'
                or source['group_id'].split(':',2)[1].casefold() in protected or code_hash in forbidden or code_hash in prior
                or code_hash in used_codes): raise ValueError('Held, prior verified or duplicate source leaked into training')
        normalized = ' '.join(source['prompt'].casefold().split())
        if any(s in normalized for s in prompts if s) or any(matcher.iter(normalized)):
            raise ValueError('An exact protected evaluation specification reached training')
        used_codes.add(code_hash)
        if (source['license_path'],source['license_text_sha256']) not in notices: raise ValueError('Original supplied notice missing')
        tests = doctest.DocTestParser().get_examples(source['prompt']); raw = o['raw_results']
        if (len(tests) != c['whole_original_example_count'] or len(tests) != len(raw) or len(tests)<2
                or any(t.options.get(doctest.SKIP,False) for t in tests)
                or not all(output_matches(t,r) for t,r in zip(tests,raw))
                or sum(r['entry_body_calls'] for r in raw) != o['entry_body_calls'] or o['entry_body_calls'] <= 0):
            raise ValueError('Every complete original published example must have an exercised entry point and passing raw output')
        if not source['prompt'].endswith(footer): raise ValueError('Original source formatter differs')
        prompt = ('Implement the documented Python function using only its declared surrounding context. '
            'Return the complete function and any necessary helpers.\n\nProvided standard-library context:\n```python\n'+PRELUDE+
            '```\n\nOriginal source description and signature:\n'+source['prompt'][:-len(footer)])
        prefix = [1]+tokenizer.encode('User: '+prompt+'\nAssistant:').ids; targets = tokenizer.encode(source['response']).ids+[2]
        ids = prefix+targets; labels = [-100]*len(prefix)+targets
        if (len(ids)>4096 or pack.sequence(index)!=(ids,labels) or row['parent_source']!=source or row['prompt']!=prompt
                or row['response']!=source['response'] or row['split']!='train' or row['token_pack_index']!=index
                or row['complete_input_ids_sha256']!=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()
                or row['target_tokens']!=len(targets) or row['unchanged_retained_target_sha256']!=sha(source['response'])):
            raise ValueError('Original target, complete context or response/EOS labels differ')
        total_context+=len(ids); total_target+=len(targets); maximum=max(maximum,len(ids)); example_count+=len(tests)
    if manifest['counts']['train'] != dict(examples=len(pack),context_tokens=total_context,target_tokens=total_target):
        raise ValueError('Complete context/target counts differ')
    runtime_passes=0; runtime_examples=0; failures=[]
    with (REPORTS/'independent-runtime-replay-v1.jsonl').open('x') as journal, ThreadPoolExecutor(max_workers=2) as pool:
        for (index,c), result in zip(selected, pool.map(execute, [c for _,c in selected])):
            journal.write(json.dumps(dict(candidate_index=index, **result),ensure_ascii=False)+'\n'); journal.flush()
            if result.get('all_original_examples_passed'):
                runtime_passes+=1; runtime_examples+=c['whole_original_example_count']
            else: failures.append(dict(source_id=c['source']['id'],status=result['status']))
    guard()
    result = dict(format='own-whole-documented-source-doctest-independent-readback-v1', observed_at=datetime.now(timezone.utc).isoformat(),
        complete=True, all_source_rows_and_native_tokens_read_back=True, all_original_targets_unchanged=True,
        all_selected_sources_reexecuted=True, all_reexecuted_examples_passed=not failures,
        admitted_pack_sha256=file_hash(DATA/'admission.json'), manifest_sha256=file_hash(DATA/'manifest.json'),
        exact_original_candidate_rows=matched, all_original_candidate_dispositions=len(observations), original_segment_rows=position,
        training_functions=len(pack), complete_context_tokens=total_context, response_EOS_target_tokens=total_target,
        maximum_complete_context_tokens=maximum, whole_published_examples=example_count,
        independently_reexecuted_functions=len(selected), independently_passing_functions=runtime_passes,
        independently_passing_original_examples=runtime_examples, runtime_replay_failures=failures,
        runtime_replay_sha256=file_hash(REPORTS/'independent-runtime-replay-v1.jsonl'), supplied_notices=len(notices),
        finite_source_examples_not_general_correctness_or_algorithmic_efficiency=True,
        previous_raw_source_training_exposure_not_erased=True, semantic_independence_certified=False,
        source_files_sha256=bound, seconds=time.monotonic()-started, model_updates=0, genuine_human_feedback_labels=0, goal_achieved=False)
    atomic_json(RESULT,result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_files_sha256'}),flush=True)
    if failures: raise ValueError('Some complete published examples did not reproduce; require a revised source pack before training')


if __name__=='__main__': main()
