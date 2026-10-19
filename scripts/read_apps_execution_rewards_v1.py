"""Independently replay every new no-op and read all original reference cases."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/apps-execution-rewards-v1'
REPORTS = ROOT / 'reports/apps-execution-rewards-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or (DATA / 'admission.json').exists() or (REPORTS / 'reader-plan.json').exists():
        raise ValueError('Fresh CPU-only complete independent control readback required')
    from slm_lab.apps_published_cases_v2 import contract, compare, execute
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.codenet_token_pack_v1 import CompleteCodePack
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    prepared = json.loads((REPORTS / 'prepared.json').read_text())
    manifest = json.loads((DATA / 'manifest.json').read_text())
    if prepared['manifest_sha256'] != file_hash(DATA / 'manifest.json'):
        raise ValueError('Complete prepared catalog required')
    files = dict(manifest['source_files_sha256'])
    for path in [Path(__file__), REPORTS / 'prepared.json', DATA / 'manifest.json']:
        files[str(path)] = file_hash(path)
    for name, digest in manifest['files_sha256'].items():
        files[str(DATA / name)] = digest
    guard = FileBindings(files); guard.check(full=True)
    pack = CompleteCodePack(manifest['original_pack'], 'train', 'training')
    with gzip.open(pack.folder / 'original-rows.jsonl.gz', 'rt') as f:
        originals = list(map(json.loads, f))
    catalog = list(map(json.loads, (DATA / 'catalog.jsonl').read_text().splitlines()))
    controls = list(map(json.loads, (DATA / 'controls.jsonl').read_text().splitlines()))
    if len(originals) != 3548 or len(catalog) != 3548 or len(controls) != 3548:
        raise ValueError('Every original source and complete control ledger required')
    atomic_json(REPORTS / 'reader-plan.json', dict(at=datetime.now(timezone.utc).isoformat(),
        all_original_reference_raw_values_recompared=True, original_references_already_independently_executed=True,
        all_3548_new_noops_to_reexecute=True, CPU_workers=4, source_files_sha256=files,
        actual_model_updates=0, goal_achieved=False))

    def read(item):
        position, entry, observed, original = item
        if (entry['id'] != original['id'] or entry['pack_local_position'] != position
                or entry['id'] != observed['source_id'] or original['split'] != 'train'
                or original['publisher_split'] != 'train' or entry['split'] != 'train'
                or entry['publisher_split'] != 'train'):
            raise ValueError('Every source identity, order and original train role required')
        source = ROOT / 'data/public-apps-original-inventory-v1/original-train' / original['source_problem_id'] / 'input_output.json'
        cases = contract(source.read_bytes())
        if cases['original_tests_sha256'] != original['original_tests_sha256'] or str(source) != entry['source_case_file']:
            raise ValueError('Original source-case identity differs')
        expected_noop = ('pass\n' if cases['mode'] == 'stdin' else
                         f'def {cases["function"]}(*args, **kwargs):\n    return None\n')
        if expected_noop != observed['noop_program']:
            raise ValueError('Exact declared no-op required')
        actual = execute(expected_noop, cases)
        raw_noop = observed['actual_noop_execution']
        ref = observed['retained_original_reference']
        values = []
        for result, body in [(actual, expected_noop), (raw_noop, expected_noop), (ref, original['response'])]:
            if (result['original_tests_sha256'] != cases['original_tests_sha256']
                    or result['original_program_sha256'] != hashlib.sha256(body.encode()).hexdigest()
                    or result['total_cases'] != cases['total_cases'] or not result['all_cases_observed']
                    or not result['syntax_valid'] or result['execution']['status'] != 'completed'
                    or len(result['raw_results']['results']) != cases['total_cases']):
                raise ValueError('Every complete actual program and original case required')
            flags = [compare(row['value'], expected, cases['mode'])
                     for row, expected in zip(result['raw_results']['results'], cases['outputs'])]
            if (flags != result['case_passes'] or sum(flags) != result['passed_cases']
                    or all(flags) != result['all_finite_published_cases_pass']):
                raise ValueError('Retained raw values and original verdicts differ')
            values.append(flags)
        actual_flags, retained_flags, reference_flags = values
        if (actual_flags != retained_flags or reference_flags != [True] * cases['total_cases']
                or ref['id'] != original['original_source_program_id']):
            raise ValueError('New no-op must replay and every reference case must pass')
        indices = [i for i, flag in enumerate(actual_flags) if not flag]
        admitted = observed['control']
        if (admitted['informative_case_indices'] != indices or admitted['noop_case_passes'] != actual_flags
                or admitted['informative'] != bool(indices) or entry['informative'] != bool(indices)
                or entry['informative_cases'] != len(indices) or admitted['total_cases'] != cases['total_cases']):
            raise ValueError('Entire counterfactual source disposition must reconstruct')
        ids, labels = pack.sequence(position)
        start = next(i for i, label in enumerate(labels) if label != -100)
        targets = len(ids) - start
        if (entry['prefix_positions'] != start or entry['response_EOS_targets'] != targets
                or entry['complete_context_positions'] != len(ids)
                or entry['full_training_action_budget'] != min(4096 - start, max(512, targets + 64))
                or entry['original_program_sha256'] != original['original_source_body_sha256']):
            raise ValueError('Complete native prompt, source targets and bounded horizon must reconstruct')
        return dict(source_id=entry['id'], original_cases=cases['total_cases'], informative_cases=len(indices),
            informative=bool(indices), actual_noop_execution=actual,
            all_original_reference_raw_case_values_recompared=True, original_reference_freshly_reexecuted=False)

    totals = Counter(); results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for count, result in enumerate(pool.map(read, ((i, *values) for i, values in enumerate(zip(catalog, controls, originals)))), 1):
            results.append(result); totals['programs'] += 1; totals['cases'] += result['original_cases']
            totals['informative_cases'] += result['informative_cases']; totals['informative_programs'] += result['informative']
            if count % 100 == 0:
                atomic_json(REPORTS / 'reader-progress.json', dict(at=datetime.now(timezone.utc).isoformat(), counts=dict(totals)))
    guard.check(full=True)
    if any(totals[key] != manifest['counts'][key] for key in totals):
        raise ValueError('Full reconstructed source/case populations differ')
    atomic_json(REPORTS / 'independent-reader.json', dict(at=datetime.now(timezone.utc).isoformat(),
        counts=dict(totals), all_new_noops_independently_reexecuted=True,
        every_original_reference_raw_result_recompared=True,
        every_complete_source_target_mask_and_training_budget_rederived=True,
        original_references_not_freshly_reexecuted_in_this_reader=True,
        references_inherited_from_existing_independent_complete_case_admission=True,
        control_reexecutions=results, source_files_sha256=files,
        model_quality_measured=False, actual_model_updates=0, goal_achieved=False))
    files[str(REPORTS / 'independent-reader.json')] = file_hash(REPORTS / 'independent-reader.json')
    atomic_json(DATA / 'admission.json', dict(format='own-public-train-APPS-execution-reward-admission-v1',
        at=datetime.now(timezone.utc).isoformat(), scoring_admitted_to_training=True,
        manifest_sha256=file_hash(DATA / 'manifest.json'), independent_reader_sha256=file_hash(REPORTS / 'independent-reader.json'),
        counts=dict(totals), all_complete_original_native_targets_and_prompts_unchanged=True,
        no_held_source_roles_used=True, finite_reference_and_noop_observations_not_general_correctness=True,
        original_local_runtime_and_comparator_qualifications_retained=True, human_preferences=False,
        genuine_human_RLHF=False, model_updates=0, source_files_sha256=files, goal_achieved=False))
    print(json.dumps(dict(counts=dict(totals), scoring_admitted_to_training=True, actual_model_updates=0)), flush=True)


if __name__ == '__main__':
    main()
