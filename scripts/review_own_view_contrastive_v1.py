"""Independently reexecute all48 train-calibration deliveries on original cases."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

import assess_joint_java_comparison_v1 as resources
from slm_lab.apps_published_cases_v2 import contract, execute, compare, implementation_sources
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.code_quality import instruction_code
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'reports/own-view-contrastive-v1'
REPORTS = ROOT / 'reports/own-view-contrastive-v1-readback'


def main():
    if REPORTS.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Fresh GPU-hidden CPU execution readback required; never overwrite previous results')
    plan = json.loads((ORIGINAL / 'plan.json').read_text())
    actors = json.loads((ORIGINAL / 'children.json').read_text())
    if resources.same(plan['owner']) or any(resources.same(actor) for actor in actors):
        raise ValueError('Actual original owner and all registered actors must end before independent readback')
    restored = json.loads((ORIGINAL / 'restored.json').read_text())
    if not restored['all_registered_point_actors_terminal'] or not resources.same(plan['waiter']):
        raise ValueError('The exact original waiter must have been restored')
    files = dict(plan['source_files_sha256']); files.update(implementation_sources())
    files[str(Path(__file__).resolve())] = file_hash(Path(__file__).resolve())
    for path in ORIGINAL.iterdir():
        if path.suffix in {'.json', '.jsonl'}:
            files[str(path)] = file_hash(path)
    guard = FileBindings(files); REPORTS.mkdir()
    atomic_json(REPORTS / 'plan.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=files, original_owner_and_all_registered_workers_terminal=True,
        scope='All original train prompts/targets/cases and48delivered token/code/raw-case decisions; complete independent native reexecution',
        all_original_cases_used_only_by_scorer=True, model_updates=0, goal_achieved=False))
    tokenizer = Tokenizer.from_file(plan['tokenizer'])
    rows = list(map(json.loads, (ORIGINAL / 'calibration.jsonl').read_text().splitlines()))
    if len(rows) != 16 or len({r['id'] for r in rows}) != 16:
        raise ValueError('All sixteen original train problems exactly once required')
    sources = {r['id']: r for r in rows}
    for index, row in enumerate(rows):
        if row['split'] != 'train' or row['publisher_split'] != 'train':
            raise ValueError('Never use held sources for decoder calibration')
        folder = ROOT / 'data/public-apps-original-inventory-v1/original-train' / row['original_problem']
        cases = contract((folder / 'input_output.json').read_bytes())
        if cases != row['cases']:
            raise ValueError('All unchanged original cases must reconstruct')
        question = (folder / 'question.txt').read_text()
        starter_path = folder / 'starter_code.py'
        starter = starter_path.read_text() if starter_path.exists() else None
        interface = ('Use the original required function or class interface.' if starter or cases['function']
            else 'Read standard input and write standard output as specified.')
        prompt = ('Write a Python solution to the following problem. ' + interface +
            ' Return only the complete code.\n\n' + question)
        if starter:
            prompt += '\n\nOriginal starter code:\n' + starter
        prefix = [1, *tokenizer.encode('User: ' + prompt + '\nAssistant:').ids]
        body = [*tokenizer.encode(row['original_human_program']).ids, 2]
        if (row['prompt'] != prompt or prefix != row['prefix_ids'] or len(body) != row['target_tokens']
                or len(body) > 512 or len(prefix) + 512 > 4096
                or hashlib.sha256(np.asarray(prefix + body, dtype='<u2').tobytes()).hexdigest() != row['complete_input_ids_sha256']):
            raise ValueError('Whole original training question/interface/target/EOS/own-token boundaries must reconstruct')
        reference = json.loads((ORIGINAL / f'reference-{index}.json').read_text())
        raw = reference['raw_results']['results']
        flags = [compare(r['value'], expected, cases['mode']) for r, expected in zip(raw, cases['outputs'])]
        if len(raw) != cases['total_cases'] or not all(flags) or not reference['all_finite_published_cases_pass']:
            raise ValueError('Every original human source reference must pass all original cases')
    records = []
    for shard in range(4):
        part = list(map(json.loads, (ORIGINAL / f'generations-{shard}.jsonl').read_text().splitlines()))
        expected = [(r['id'], s) for index, r in enumerate(rows) if index % 4 == shard for s in plan['strengths']]
        if [(r['id'], r['strength']) for r in part] != expected:
            raise ValueError('Every original problem/setting must appear in its original disjoint shard exactly once')
        records.extend(part)
    summaries = {}; qualifications = Counter(); interfaces = Counter(); repeated = Counter()
    with (REPORTS / 'reexecutions.jsonl').open('x') as stream:
        for record in records:
            original = sources[record['id']]; code = record['code']; response = record['response_ids']
            if (record['prefix_ids'] != original['prefix_ids'] or record['raw_response'] != tokenizer.decode(response, skip_special_tokens=True)
                    or instruction_code(record['raw_response']) != code or not 1 <= len(response) <= 512
                    or record['stopped_on_all_EOS'] != (response[-1] == 2)
                    or [r['token_ids'][0] for r in record['trace']] != response
                    or any(r['active'] != [True] for r in record['trace'])
                    or 2 in response[:-1] or record['plausibility'] != .1):
                raise ValueError('Every delivered native token, EOS, prefix, trace and unmodified extracted code must reconstruct')
            for entry in record['trace']:
                if not np.isfinite(entry['code_log_probability']).all() or (record['strength'] and not np.isfinite(entry['general_log_probability']).all()):
                    raise ValueError('Every retained actually chosen log probability must be finite')
            observed = (execute(code, original['cases'], seconds=20) if code else dict(syntax_valid=False,
                all_cases_observed=False, all_finite_published_cases_pass=False, status='empty_response'))
            old = record['execution']
            for key in ['syntax_valid', 'all_cases_observed', 'all_finite_published_cases_pass', 'status']:
                if old.get(key) != observed.get(key):
                    raise ValueError('Independent original-case execution changed its qualification/pass decision: ' + record['id'])
            if observed['all_cases_observed']:
                if old['case_passes'] != observed['case_passes'] or [x['value'] for x in old['raw_results']['results']] != [x['value'] for x in observed['raw_results']['results']]:
                    raise ValueError('Every complete originally observed raw value and finite-case decision must reproduce')
            strength = str(record['strength']); key = (strength, original['mode'])
            interfaces[key] += bool(observed['all_finite_published_cases_pass'])
            if observed['status'] == 'runtime_or_resource_qualification':
                stderr = observed.get('execution', {}).get('stderr', '').strip().splitlines()
                diagnostic = stderr[-1].split(':', 1)[0] if stderr else 'resource_or_missing_diagnostic'
                qualifications[(strength, diagnostic)] += 1
            elif observed['status'] == 'completed_published_case_mismatch':
                qualifications[(strength, 'complete_finite_case_wrong_value')] += 1
            if observed.get('syntax_valid') and original['cases']['function']:
                functions = [n.name for n in ast.walk(ast.parse(code)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                if original['cases']['function'] not in functions:
                    qualifications[(strength, 'required_function_name_absent')] += 1
            tokens = record['response_ids']; grams = [tuple(tokens[i:i+4]) for i in range(max(0, len(tokens)-3))]
            repeated[strength] += len(grams) - len(set(grams))
            stats = summaries.setdefault(strength, dict(training_problems=0, syntax=0, all_finite_cases_pass=0,
                all_cases_observed=0, EOS_completions=0, total_generated_tokens=0, execution_status=Counter()))
            stats['training_problems'] += 1; stats['syntax'] += bool(observed.get('syntax_valid'))
            stats['all_finite_cases_pass'] += bool(observed['all_finite_published_cases_pass'])
            stats['all_cases_observed'] += bool(observed['all_cases_observed'])
            stats['EOS_completions'] += bool(record['stopped_on_all_EOS']); stats['total_generated_tokens'] += len(tokens)
            stats['execution_status'][observed['status']] += 1
            stream.write(json.dumps(dict(id=record['id'], strength=record['strength'], independently_reexecuted=observed)) + '\n'); stream.flush()
    comparison = json.loads((ORIGINAL / 'comparison.json').read_text())
    if summaries != comparison['metrics'] or comparison['calibration_choice_for_later_new_sealed_held_experiment'] is not None:
        raise ValueError('All independently reconstructed metrics and the actual negative selection decision must match')
    guard.check(full=True)
    complete = dict(completed_at=datetime.now(timezone.utc).isoformat(),
        original_plan_sha256=file_hash(ORIGINAL / 'plan.json'), original_comparison_sha256=file_hash(ORIGINAL / 'comparison.json'),
        independent_reexecutions_sha256=file_hash(REPORTS / 'reexecutions.jsonl'),
        complete_original_train_questions_human_targets_own_tokens_EOS_and_cases_reconstructed=True,
        all_48_actual_model_deliveries_independently_reexecuted=True, all_original_raw_values_and_qualification_decisions_reproduced=True,
        actual_calibration_metrics=summaries, selected_setting=None,
        observed_runtime_and_interface_diagnostics={str(s): {kind: count for (setting, kind), count in qualifications.items() if setting == str(s)} for s in plan['strengths']},
        overlapping_diagnostics_not_a_disjoint_algorithmic_error_taxonomy=True,
        repeated_token_fourgrams=dict(repeated), no_held_questions_or_answers_used=True,
        runtime_qualifications_not_relabelled_algorithmic_wrong_answers=True,
        next_action='Do not advance this decoder to held assessment. Diagnose training-time problem/input/interface conditioning before a new neural intervention.',
        model_updates=0, CUDA_initialized=torch.cuda.is_initialized(), candidate_promoted=False, goal_achieved=False)
    atomic_json(REPORTS / 'complete.json', complete)
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == '__main__':
    main()
