"""Prepare no-op-controlled execution feedback for every admitted APPS program."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/apps-execution-rewards-v1'
REPORTS = ROOT / 'reports/apps-execution-rewards-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or DATA.exists() or REPORTS.exists():
        raise ValueError('Fresh CPU-only original train preparation required')
    from slm_lab.apps_execution_reward_v1 import control, noop
    from slm_lab.apps_published_cases_v2 import contract, execute, implementation_sources
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.codenet_token_pack_v1 import CompleteCodePack
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    pack = CompleteCodePack(ROOT / 'data/APPS-verified-generation-v1', 'train', 'training')
    admission_path = pack.folder / 'admission.json'
    admission = json.loads(admission_path.read_text())
    FileBindings(admission['source_files_sha256']).check(full=True)
    if len(pack) != 3548 or '2 passed' not in (ROOT / 'reports/apps-execution-reward-v1-tests.log').read_text():
        raise ValueError('Every original admitted source and actual new reward controls required')
    with gzip.open(pack.folder / 'original-rows.jsonl.gz', 'rt') as stream:
        rows = list(map(json.loads, stream))
    if len(rows) != len(pack) or any(r['split'] != 'train' or r['publisher_split'] != 'train' for r in rows):
        raise ValueError('Every unchanged publisher-train row required')
    ref_path = ROOT / 'reports/APPS-original-complete-cases-v2/observations.jsonl'
    refs = {r['id']: r for r in map(json.loads, ref_path.read_text().splitlines())
            if r['status'] == 'all_finite_published_cases_pass'}
    if len(refs) != 3548 or set(refs) != {r['original_source_program_id'] for r in rows}:
        raise ValueError('Every admitted full-case passing reference required')
    files = dict(implementation_sources())
    for p in [Path(__file__), admission_path, ref_path, pack.folder / 'manifest.json',
              ROOT / 'src/slm_lab/apps_execution_reward_v1.py', ROOT / 'tests/test_apps_execution_reward_v1.py',
              ROOT / 'reports/apps-execution-reward-v1-tests.log']:
        files[str(p)] = file_hash(p)
    files.update({str(pack.folder / name): digest for name, digest in pack.manifest['files_sha256'].items()})
    original = ROOT / 'data/public-apps-original-inventory-v1/original-train'
    for row in rows:
        path = original / row['source_problem_id'] / 'input_output.json'
        if file_hash(path) != row['original_tests_sha256']:
            raise ValueError('Original complete source cases changed')
        files[str(path)] = row['original_tests_sha256']
    guard = FileBindings(files); guard.check(full=True)
    DATA.mkdir(); REPORTS.mkdir(); started = time.monotonic()
    atomic_json(REPORTS / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(),
        original_admitted_programs=len(rows), complete_original_cases=17990, CPU_workers=4,
        original_reference_observations_reused_with_every_raw_result_recompared=True,
        every_noop_actual_execution_required=True, reward_train_admission_pending_independent_reader=True,
        source_files_sha256=files, model_updates=0, human_preferences=False, goal_achieved=False))
    def prepare(item):
        position, row = item
        cases = contract((original / row['source_problem_id'] / 'input_output.json').read_bytes())
        ref = refs[row['original_source_program_id']]
        if (ref['admitted_role'] != 'train' or ref['publisher_role'] != 'train'
                or ref['problem'] != row['source_problem_id']):
            raise ValueError('Reference source identity and roles differ')
        empty = execute(noop(cases), cases)
        admitted = control(row, cases, row['response'], ref, empty)
        ids, labels = pack.sequence(position)
        prefix = next(i for i, value in enumerate(labels) if value != -100)
        targets = len(ids) - prefix
        budget = min(4096 - prefix, max(512, targets + 64))
        if not targets <= budget or ids[-1] != 2:
            raise ValueError('Whole original target/EOS and explicit action horizon required')
        entry = dict(id=row['id'], source_problem_id=row['source_problem_id'],
            original_source_program_id=row['original_source_program_id'], split='train', publisher_split='train',
            pack_local_position=position, interface=cases['mode'], difficulty=row['difficulty'],
            source_url=row['source_url'], original_program_sha256=row['original_source_body_sha256'],
            original_tests_sha256=row['original_tests_sha256'], source_case_file=str(original / row['source_problem_id'] / 'input_output.json'),
            complete_context_positions=len(ids), prefix_positions=prefix, response_EOS_targets=targets,
            full_training_action_budget=budget, total_cases=cases['total_cases'], informative=admitted['informative'],
            informative_cases=len(admitted['informative_case_indices']))
        observation = dict(source_id=row['id'], control=admitted, noop_program=noop(cases),
            actual_noop_execution=empty, retained_original_reference=ref)
        return entry, observation
    totals = Counter(); maximum = 0
    with (DATA / 'catalog.jsonl').open('x') as catalog, (DATA / 'controls.jsonl').open('x') as ledger:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for count, (entry, observed) in enumerate(pool.map(prepare, enumerate(rows)), 1):
                catalog.write(json.dumps(entry, separators=(',', ':')) + '\n')
                ledger.write(json.dumps(observed, separators=(',', ':')) + '\n')
                totals['programs'] += 1; totals['cases'] += entry['total_cases']
                totals['informative_cases'] += entry['informative_cases']; totals['informative_programs'] += entry['informative']
                totals['response_EOS_targets'] += entry['response_EOS_targets']; maximum = max(maximum, entry['full_training_action_budget'])
                if count % 100 == 0:
                    catalog.flush(); ledger.flush(); guard.check()
                    atomic_json(REPORTS / 'progress.json', dict(at=datetime.now(timezone.utc).isoformat(),
                        counts=dict(totals), seconds=time.monotonic() - started, model_updates=0, goal_achieved=False))
    guard.check(full=True)
    if totals['programs'] != 3548 or totals['cases'] != 17990 or totals['response_EOS_targets'] != 550701:
        raise ValueError('Complete admitted program, test and target populations required')
    manifest = dict(format='own-public-APPS-noop-controlled-execution-catalog-v1',
        created_at=datetime.now(timezone.utc).isoformat(), counts=dict(totals),
        maximum_training_action_budget=maximum, original_pack=str(pack.folder),
        original_pack_manifest_sha256=file_hash(pack.folder / 'manifest.json'),
        original_pack_admission_sha256=file_hash(admission_path),
        files_sha256={name: file_hash(DATA / name) for name in ['catalog.jsonl', 'controls.jsonl']},
        source_files_sha256=files, complete_original_program_targets_unchanged=True,
        generation_prompts_do_not_receive_expected_case_outputs=True,
        variable_train_budget_derived_from_public_reference_length_not_held_scores=True,
        original_runtime_and_publisher_comparator_qualifications_retained=True,
        finite_machine_feedback_not_human_preferences=True, genuine_human_RLHF=False,
        formal_correctness_or_efficiency_proven=False, scoring_admitted_to_training=False,
        actual_model_updates=0, goal_achieved=False)
    atomic_json(DATA / 'manifest.json', manifest)
    atomic_json(REPORTS / 'prepared.json', dict(at=datetime.now(timezone.utc).isoformat(),
        counts=dict(totals), manifest_sha256=file_hash(DATA / 'manifest.json'),
        maximum_training_action_budget=maximum, seconds=time.monotonic() - started,
        independent_reader_required=True, actual_model_updates=0, goal_achieved=False))
    print(json.dumps(dict(counts=dict(totals), maximum_training_action_budget=maximum,
        seconds=time.monotonic() - started, scoring_admitted_to_training=False)), flush=True)


if __name__ == '__main__':
    main()
