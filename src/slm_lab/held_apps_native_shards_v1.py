"""Fixed original publisher-test APPS cohort, native generation and full-case execution.

No test solutions are extracted and no optimization API is provided. This is
a bounded diagnostic under the retained local runtime, not an official score.
"""
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch

from .apps_published_cases_v2 import contract, execute
from .code_quality import instruction_code
from .corpus import file_hash
from .diverse_images import atomic_json

FORMAT = 'own-held-original-APPS-native-shard-v1'


def select(rows, count=64):
    if type(count) is not int or count < 1:
        raise ValueError('Positive fixed original held cohort size required')
    if any(r.get('publisher_split') != 'test' or r.get('purpose') != 'evaluation_only'
           or r.get('optimization_forbidden') is not True for r in rows):
        raise ValueError('Every original source must remain held and forbidden to optimization')
    groups = defaultdict(list)
    for row in rows:
        if (row.get('supported_by_declared_local_protocol') is True
                and row['complete_prompt_tokens'] + 1024 <= 4096):
            if row['generation_budget'] != 1024 or row['full_context_window'] != 4096 or row['input_or_case_clipping_permitted']:
                raise ValueError('Unchanged original generation budget and complete case/context contract required')
            groups[(row['difficulty'], row['mode'])].append(row)
    for values in groups.values():
        values.sort(key=lambda row: hashlib.sha256(row['id'].encode()).digest())
    selected = []
    while groups and len(selected) < count:
        for key in sorted(list(groups)):
            selected.append(groups[key].pop(0))
            if not groups[key]: del groups[key]
            if len(selected) == count: break
    if len(selected) != count or len({r['id'] for r in selected}) != count:
        raise ValueError('All distinct original held source presentations required')
    return selected


@torch.no_grad()
def evaluate_shard(model, tokenizer, checkpoint, data, cohort_path, output, *, rank, world=4, device='cuda:0', checked=lambda:None):
    if type(rank) is not int or type(world) is not int or not 0 <= rank < world:
        raise ValueError('Exact held shard assignment required')
    data, cohort_path = Path(data), Path(cohort_path)
    rows = list(map(json.loads, (data/'evaluation-records.jsonl').read_text().splitlines()))
    cohort = json.loads(cohort_path.read_text()); selected = select(rows, len(cohort['selected_original_test_ids']))
    if [r['id'] for r in selected] != cohort['selected_original_test_ids']:
        raise ValueError('Exact source-preselected held cohort required')
    if cohort['manifest_sha256'] != file_hash(data/'manifest.json') or cohort['records_sha256'] != file_hash(data/'evaluation-records.jsonl'):
        raise ValueError('Original full held-source manifest and contract records must match')
    model.eval(); results = []
    for row in selected[rank::world]:
        checked()
        path = data/'original-test'/row['problem']/'input_output.json'
        if file_hash(path) != row['original_tests_sha256']:
            raise ValueError('Every complete original held test file must remain unchanged')
        cases = contract(path.read_bytes())
        prefix = [1] + tokenizer.encode('User: '+row['complete_prompt']+'\nAssistant:').ids
        digest = hashlib.sha256(torch.tensor(prefix, dtype=torch.int64).numpy().astype('<u2').tobytes()).hexdigest()
        if (digest != row['complete_prompt_input_ids_sha256'] or len(prefix) != row['complete_prompt_tokens']
                or cases['total_cases'] != row['complete_published_case_count']):
            raise ValueError('Unchanged complete native held prompt and every original case required')
        ids = torch.tensor([prefix], device=device)
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
        with amp(): generated = model.generate(ids, max_new_tokens=1024, eos_token_id=2, use_cache=True)
        tokens = generated[0, len(prefix):].tolist(); response = tokenizer.decode(tokens); code = instruction_code(response)
        try: outcome = execute(code, cases, seconds=20)
        except ValueError as error:
            outcome = dict(status='candidate_source_qualification', syntax_valid=False, all_cases_observed=False,
                all_finite_published_cases_pass=False, total_cases=cases['total_cases'], passed_cases=0, diagnostic=str(error))
        results.append(dict(id=row['id'], problem=row['problem'], difficulty=row['difficulty'], mode=row['mode'],
            original_tests_sha256=row['original_tests_sha256'], complete_prompt_input_ids_sha256=digest,
            generation_token_ids=tokens, raw_response=response, extracted_code=code,
            stopped_on_EOS=bool(tokens and tokens[-1] == 2), execution=outcome))
    atomic_json(output, dict(format=FORMAT, rank=rank, world=world, results=results,
        checkpoint_sha256=file_hash(checkpoint), cohort_sha256=file_hash(cohort_path),
        original_manifest_sha256=file_hash(data/'manifest.json'), generation_budget=1024,
        inference_device_type='cuda' if device.startswith('cuda') else 'cpu',
        inference_precision='bf16_autocast' if device.startswith('cuda') else 'float32',
        official_APPS_score=False, optimization_forbidden=True, model_updates=0, goal_achieved=False))


def reconcile(paths, cohort_path, output):
    shards = [json.loads(Path(p).read_text()) for p in paths]
    cohort = json.loads(Path(cohort_path).read_text())
    if (len(shards) != 4 or {s['rank'] for s in shards} != set(range(4))
            or any(s['format'] != FORMAT or s['world'] != 4 or s['optimization_forbidden'] is not True for s in shards)):
        raise ValueError('All exact original held native shards required')
    for key in ['checkpoint_sha256', 'cohort_sha256', 'original_manifest_sha256', 'generation_budget', 'inference_device_type', 'inference_precision']:
        if len({s[key] for s in shards}) != 1: raise ValueError('Matched held generation/source protocol required')
    if (shards[0]['cohort_sha256'] != file_hash(cohort_path) or shards[0]['generation_budget'] != 1024
            or shards[0]['original_manifest_sha256'] != cohort['manifest_sha256']):
        raise ValueError('Exact actual cohort and original held generation budget required')
    by_id = {}; ordered = cohort['selected_original_test_ids']
    for shard in shards:
        if [r['id'] for r in shard['results']] != ordered[shard['rank']::4]:
            raise ValueError('Every original held case must occupy its predetermined shard once')
        for row in shard['results']:
            if row['id'] in by_id: raise ValueError('Duplicate held original source')
            by_id[row['id']] = row
    if set(by_id) != set(ordered): raise ValueError('Missing original held source')
    rows = [by_id[k] for k in ordered]
    total = len(rows); correct = sum(r['execution']['all_finite_published_cases_pass'] for r in rows)
    atomic_json(output, dict(results=rows, tasks=total, all_finite_cases_passed=correct,
        functional_pass_at_1=correct/total, syntax_valid=sum(r['execution']['syntax_valid'] for r in rows),
        whole_case_execution_completed=sum(r['execution']['all_cases_observed'] for r in rows),
        full_case_pass_and_delivered_EOS=sum(r['execution']['all_finite_published_cases_pass'] and r['stopped_on_EOS'] for r in rows),
        **{k:shards[0][k] for k in ['checkpoint_sha256','cohort_sha256','generation_budget','inference_device_type','inference_precision']},
        source_files_sha256={str(p):file_hash(p) for p in paths}, official_APPS_score=False,
        bounded_diagnostic_not_full_standard_benchmark=True, earlier_public_or_semantic_independence_not_certified=True,
        independent_candidate_execution_readback_still_required=True, optimization_forbidden=True,
        model_updates=0, candidate_promoted=False, goal_achieved=False))
