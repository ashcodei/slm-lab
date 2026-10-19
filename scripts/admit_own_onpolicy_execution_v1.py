"""Reconcile64native own rollouts; reexecute and admit only complete wrong values."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import torch
from tokenizers import Tokenizer

import assess_joint_java_comparison_v1 as resources
from slm_lab.apps_published_cases_v2 import compare, execute, implementation_sources
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.execution_preference_v1 import validate
from slm_lab.own_onpolicy_code_data_v1 import prepare
from slm_lab.own_onpolicy_execution_v1 import example

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = ROOT / 'reports/own-onpolicy-code-64-v1'
OUTPUT = ROOT / 'data/own-onpolicy-execution-v1'
REPORTS = ROOT / 'reports/own-onpolicy-execution-v1'


def main():
    if OUTPUT.exists() or REPORTS.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Fresh GPU-hidden admission without prior output required')
    plan = json.loads((COLLECTION / 'plan.json').read_text())
    if resources.same(plan['owner']) or any(resources.same(a) for a in json.loads((COLLECTION / 'children.json').read_text())):
        raise ValueError('Original own-policy collector and all registered actors must actually end')
    if not json.loads((COLLECTION / 'restored.json').read_text())['all_registered_point_actors_terminal']:
        raise ValueError('Original collector must restore its idle waiter before admission')
    if plan['training_problems'] != 64 or plan['strengths'] != [0.]:
        raise ValueError('All64native original greedy publisher-train generations required')
    if shutil.disk_usage(ROOT).free < 26 * 1024**3 + 128 * 1024**2:
        raise ValueError('Retain original26GiB reserve plus128MiB commitment')
    files = dict(plan['source_files_sha256']); files.update(implementation_sources())
    for path in [Path(__file__).resolve(), ROOT / 'src/slm_lab/own_onpolicy_execution_v1.py',
        ROOT / 'src/slm_lab/execution_preference_v1.py', ROOT / 'tests/test_own_onpolicy_execution_v1.py',
        ROOT / 'reports/own-onpolicy-execution-v1-tests.log', *COLLECTION.glob('*.json'), *COLLECTION.glob('*.jsonl')]:
        files[str(path)] = file_hash(path)
    if '2 passed' not in (ROOT / 'reports/own-onpolicy-execution-v1-tests.log').read_text():
        raise ValueError('Actual native whole-response/counterexample/EOS/held-role checks required')
    guard = FileBindings(files)
    tokenizer = Tokenizer.from_file(plan['tokenizer'])
    rows = list(map(json.loads, (COLLECTION / 'calibration.jsonl').read_text().splitlines()))
    if rows != prepare(ROOT, tokenizer):
        raise ValueError('All64preselected complete original train questions/programs/source bytes must reconstruct')
    records = []
    for shard in range(4):
        path = COLLECTION / f'generations-{shard}.jsonl'
        done = json.loads((COLLECTION / f'complete-{shard}.json').read_text())
        if file_hash(path) != done['generations_sha256'] or not done['original_code_head_weights_and_versions_unchanged']:
            raise ValueError('All original native generation bytes and unchanged policy state required')
        part = list(map(json.loads, path.read_text().splitlines()))
        if [r['id'] for r in part] != [r['id'] for i, r in enumerate(rows) if i % 4 == shard]:
            raise ValueError('Every original disjoint train source must have exactly one own generation')
        records.extend(part)
    if len(records) != 64 or len({r['id'] for r in records}) != 64:
        raise ValueError('All64actual original generations exactly once required')
    by_id = {r['id']: r for r in records}
    OUTPUT.mkdir(); REPORTS.mkdir()
    atomic_json(REPORTS / 'plan.json', dict(created_at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        frozen_own_policy_manifest_sha256=plan['manifest_sha256'], source_programs=64,
        admission_rule='Complete native own greedy response ending in genuineEOS; unchanged human source passes all original cases; own delivered program completes all cases and has a concrete wrong value. Both programs independently reexecute. Runtime/syntax/budget-qualified negatives are not admitted.',
        expected_values_not_added_to_model_generation_prompts=True, machine_execution_not_human_preferences=True,
        model_updates=0, goal_achieved=False))
    pairs = []; reviews = []; reexecutions = []; all_statuses = Counter(); original_cases = 0
    for index, source in enumerate(rows):
        generation = by_id[source['id']]; negative = generation['execution']
        positive = json.loads((COLLECTION / f'reference-{index}.json').read_text())
        cases = source['cases']; original_cases += cases['total_cases']; all_statuses[negative['status']] += 1
        for observed in [positive, negative]:
            if observed['all_cases_observed']:
                raw = observed['raw_results']['results']
                flags = [compare(r['value'], expected, cases['mode']) for r, expected in zip(raw, cases['outputs'])]
                if (len(raw) != cases['total_cases'] or observed['case_passes'] != flags
                        or observed['passed_cases'] != sum(flags) or observed['all_finite_published_cases_pass'] != all(flags)):
                    raise ValueError('Every complete original raw finite-case value/qualification must reconstruct')
            elif observed['all_finite_published_cases_pass']:
                raise ValueError('An incomplete original result cannot count as passing')
        pair, reason = example(source, generation, positive, negative, tokenizer, policy_sha256=plan['manifest_sha256'])
        if reason:
            reviews.append(dict(id=source['id'], reason=reason, actual_execution_status=negative['status']))
            continue
        fresh_positive = execute(source['original_human_program'], cases, seconds=20)
        fresh_negative = execute(generation['code'], cases, seconds=20)
        new_pair, new_reason = example(source, generation, fresh_positive, fresh_negative, tokenizer, policy_sha256=plan['manifest_sha256'])
        if new_reason or new_pair != pair:
            reviews.append(dict(id=source['id'], reason='independent_original_case_execution_changed_or_qualified',
                original_pair=pair, fresh_positive=fresh_positive, fresh_negative=fresh_negative))
            continue
        old_values = [r['value'] for r in negative['raw_results']['results']]
        new_values = [r['value'] for r in fresh_negative['raw_results']['results']]
        if old_values != new_values:
            reviews.append(dict(id=source['id'], reason='independent_negative_raw_values_not_identical'))
            continue
        pairs.append(pair); reexecutions.append(dict(id=source['id'], positive=fresh_positive, negative=fresh_negative))
    if not pairs:
        raise ValueError('No eligible independently reproduced complete own-policy wrong-value pair; do not fabricate negatives')
    counts = validate(pairs, True)
    for name, values in [('pairs.jsonl', pairs), ('reviews.jsonl', reviews)]:
        with (OUTPUT / name).open('x') as stream:
            for row in values:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
    with (REPORTS / 'independent-reexecutions.jsonl').open('x') as stream:
        for row in reexecutions:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    guard.check(full=True)
    admission = dict(format='own-onpolicy-execution-feedback-admission-v1', admitted_at=datetime.now(timezone.utc).isoformat(),
        admitted_for_complete_program_machine_DPO=True, original_publisher_train_problems_observed=64,
        complete_original_source_cases=original_cases, observed_execution_statuses=dict(all_statuses),
        independently_reexecuted_admitted_pairs=len(pairs), reviewed_not_admitted=len(reviews),
        unchanged_original_human_response_EOS_targets=sum(n[0] for n in counts),
        actual_complete_own_response_EOS_targets=sum(n[1] for n in counts),
        pairs_sha256=file_hash(OUTPUT / 'pairs.jsonl'), reviews_sha256=file_hash(OUTPUT / 'reviews.jsonl'),
        independent_reexecutions_sha256=file_hash(REPORTS / 'independent-reexecutions.jsonl'),
        source_plan_sha256=file_hash(COLLECTION / 'plan.json'), admission_plan_sha256=file_hash(REPORTS / 'plan.json'),
        frozen_reference_policy_sha256=plan['manifest_sha256'], source_files_sha256=files,
        original_human_programs_unmodified=True, actual_own_generated_negative_responses_unmodified=True,
        all_native_prefixes_target_masks_and_genuine_EOS_reconstructed=True,
        negative_syntax_runtime_and_budget_qualifications_not_labelled_wrong_answers=True,
        expected_case_values_not_in_original_generation_prompts=True,
        source_cases_are_train_not_held_benchmark=True, semantic_or_prior_public_corpus_independence_certified=False,
        genuine_human_RLHF=False, human_preferences=False, external_model_teacher_used=False,
        finite_cases_not_algorithmic_or_efficiency_proof=True, model_updates=0, candidate_promoted=False, goal_achieved=False)
    atomic_json(OUTPUT / 'admission.json', admission)
    complete = {key: value for key, value in admission.items() if key != 'source_files_sha256'}
    complete['data_admission_sha256'] = file_hash(OUTPUT / 'admission.json')
    atomic_json(REPORTS / 'complete.json', complete)
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == '__main__':
    main()
