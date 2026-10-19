"""Prepare and reconcile every previously verified original APPS train target."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from tokenizers import Tokenizer

from slm_lab.apps_verified_generation_v1 import generation_example
from slm_lab.apps_published_cases_v2 import PRELUDE
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CodePackWriter, CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/public-apps-original-inventory-v1/original-train'
VIEW = ROOT / 'data/public-apps-human-solution-view-v1'
EXECUTION = ROOT / 'reports/APPS-original-complete-cases-v2'
OUTPUT = ROOT / 'data/APPS-verified-generation-v1'
REPORTS = ROOT / 'reports/APPS-verified-generation-v1'


def main():
    if OUTPUT.exists() or REPORTS.exists():
        raise ValueError('Preserve prior verified generation data; no implicit retry')
    parent_admission = json.loads((VIEW / 'admission.json').read_text())
    parent = Path(parent_admission['complete_pack'])
    audit = json.loads((EXECUTION / 'independent-readback.json').read_text())
    summary = json.loads((EXECUTION / 'summary.json').read_text())
    if (parent_admission['admitted_for_original_human_solution_CE'] is not True
            or parent_admission['only_publisher_train_sources'] is not True
            or not audit['all_selected_original_IDs_program_hashes_and_test_hashes_reconciled']
            or not audit['every_retained_raw_case_result_independently_compared_to_original_expected']
            or summary['observations_sha256'] != file_hash(EXECUTION / 'observations.jsonl')):
        raise ValueError('Original admitted source roles and complete finite source-case audit required')
    files = dict(json.loads((EXECUTION / 'plan.json').read_text())['source_files_sha256'])
    files.update(parent_admission['files_sha256'])
    names = ['scripts/prepare_apps_verified_generation_v1.py', 'src/slm_lab/apps_verified_generation_v1.py',
        'src/slm_lab/apps_published_cases_v2.py', 'src/slm_lab/codenet_token_pack_v1.py',
        'tests/test_apps_verified_generation_v1.py', 'reports/APPS-verified-generation-v1-tests.log',
        'data/public-apps-human-solution-view-v1/admission.json',
        'reports/APPS-original-complete-cases-v2/plan.json', 'reports/APPS-original-complete-cases-v2/summary.json',
        'reports/APPS-original-complete-cases-v2/observations.jsonl', 'reports/APPS-original-complete-cases-v2/independent-readback.json']
    for name in names:
        files[str(ROOT / name)] = file_hash(ROOT / name)
    if '2 passed' not in (ROOT / 'reports/APPS-verified-generation-v1-tests.log').read_text():
        raise ValueError('Actual held-role, raw-case corruption, source-byte and token-boundary checks required')
    guard = FileBindings(files)
    guard.check(full=True)
    positives = {row['id']: row for row in map(json.loads, (EXECUTION / 'observations.jsonl').read_text().splitlines())
                 if row['status'] == 'all_finite_published_cases_pass'}
    admitted = {row['id']: row for row in map(json.loads, (VIEW / 'train-records.jsonl').read_text().splitlines())}
    if len(positives) != 3548 or not positives.keys() <= admitted.keys():
        raise ValueError('All3548 complete-case original positives must have admitted train roles')
    OUTPUT.mkdir(); REPORTS.mkdir()
    for name in ['tokenizer.json', 'model-config.json', 'DATASET-LICENSE.txt', 'publisher-README.md']:
        shutil.copy2(parent / name, OUTPUT / name)
    (OUTPUT / 'declared-runtime-prelude.py').write_text(PRELUDE)
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        selected_original_programs=3548, source_selection='Every original complete-case passing program from the sealed shortest-source-per-train-problem experiment',
        source_selection_not_based_on_model_or_held_results=True, prompt='Exact existing native APPS generation prompt',
        stdlib_prelude_is_declared_execution_context_not_added_to_original_prompt=True,
        target='Unchanged complete original human Python program and EOS', no_clipping=True,
        model_generated_examples=0, external_model_teacher_used=False, genuine_RLHF_labels=False,
        model_updates=0, goal_achieved=False)
    atomic_json(REPORTS / 'plan.json', plan)
    tokenizer = Tokenizer.from_file(str(OUTPUT / 'tokenizer.json'))
    writer = CodePackWriter(OUTPUT, maximum_bytes=512 * 1024**2, minimum_free_gib=26)
    modes = Counter(); difficulties = Counter(); identities = set(); cases_count = 0
    with gzip.open(parent / 'original-rows.jsonl.gz', 'rt') as stream:
        for source in map(json.loads, stream):
            if source['id'] not in positives:
                continue
            if source['id'] in identities or source['split'] != 'train':
                raise ValueError('Each actual original admitted train program exactly once required')
            observed = positives[source['id']]
            folder = SOURCE / observed['problem']
            program = json.loads((folder / 'solutions.json').read_bytes())[observed['original_solution_index']]
            question = (folder / 'question.txt').read_bytes().decode('utf-8')
            starter = (folder / 'starter_code.py').read_bytes().decode('utf-8') if (folder / 'starter_code.py').exists() else None
            row, ids, labels = generation_example(source, question, starter, program,
                (folder / 'input_output.json').read_bytes(), observed, tokenizer)
            if admitted[source['id']]['complete_input_ids_sha256'] != source['complete_input_ids_sha256']:
                raise ValueError('Exact named-screened source token identity required')
            writer.add(row, ids, labels)
            identities.add(source['id']); modes[observed['mode']] += 1; difficulties[observed['difficulty']] += 1
            cases_count += observed['total_cases']
    if identities != positives.keys() or cases_count != 17990:
        raise ValueError('All3548 original human programs and17990 original finite cases required')
    manifest = writer.finish(dict(plan_sha256=file_hash(REPORTS / 'plan.json'),
        parent_human_CE_admission_sha256=file_hash(VIEW / 'admission.json'),
        original_execution_audit_sha256=file_hash(EXECUTION / 'independent-readback.json')))
    # Re-open and independently recover both token regions from every actual
    # stored row. This checks the on-disk writer, not only the input objects.
    packed = CompleteCodePack(OUTPUT, 'train', purpose='audit')
    seen = set(); targets = 0; contexts = 0; maximum = 0
    with gzip.open(OUTPUT / 'original-rows.jsonl.gz', 'rt') as stream:
        for position, row in enumerate(map(json.loads, stream)):
            ids, labels = packed.sequence(position)
            start = next(i for i, label in enumerate(labels) if label != -100)
            prefix = [1, *tokenizer.encode('User: ' + row['prompt'] + '\nAssistant:').ids]
            response = [*tokenizer.encode(row['response']).ids, 2]
            if (ids != prefix + response or labels != [-100] * len(prefix) + response
                    or row['original_source_program_id'] not in positives or row['split'] != 'train'
                    or hashlib.sha256(row['response'].encode()).hexdigest() != positives[row['original_source_program_id']]['original_program_sha256']
                    or len(response) != row['target_tokens'] or start != len(prefix)):
                raise ValueError('Every stored whole original context, response, mask and source hash must reconstruct')
            seen.add(row['original_source_program_id']); targets += len(response); contexts += len(ids); maximum = max(maximum, len(ids))
    if seen != positives.keys() or targets != 550701 or contexts != manifest['counts']['train']['context_tokens']:
        raise ValueError('Complete source/response target inventory differs after durable readback')
    guard.check(full=True)
    output_files = {str(OUTPUT / name): digest for name, digest in manifest['files_sha256'].items()}
    for path in [OUTPUT / 'manifest.json', OUTPUT / 'declared-runtime-prelude.py', REPORTS / 'plan.json']:
        output_files[str(path)] = file_hash(path)
    admission = dict(format='own-complete-human-code-verified-generation-admission-v1',
        admitted_at=datetime.now(timezone.utc).isoformat(), admitted_to_training=True,
        dataset_manifest_sha256=file_hash(OUTPUT / 'manifest.json'), complete_examples=len(packed),
        complete_response_EOS_targets=targets, complete_context_tokens=contexts, maximum_complete_context_tokens=maximum,
        complete_finite_original_cases=cases_count, programs_by_interface=dict(modes), programs_by_difficulty=dict(difficulties),
        all_original_questions_programs_tokens_masks_and_raw_case_results_reconciled=True,
        original_train_groups_only=True, unchanged_native_generation_prompts=True, original_program_bytes_unchanged=True,
        inherited_named_exact_held_screen=parent_admission['exact_screen_scope'],
        semantic_or_prior_public_corpus_independence_certified=False,
        runtime='Declared local APPS stdlib prelude; complete finite original cases; not official APPS runtime',
        finite_case_evidence_is_not_general_correctness_or_algorithmic_efficiency=True,
        genuine_RLHF_labels=False, human_preferences=False, machine_execution_feedback=True,
        model_generated_examples=0, external_model_teacher_used=False, model_updates=0,
        source_files_sha256={**files, **output_files}, candidate_promoted=False, goal_achieved=False)
    atomic_json(OUTPUT / 'admission.json', admission)
    loaded = CompleteCodePack(OUTPUT, 'train', purpose='training')
    try:
        CompleteCodePack(OUTPUT, 'validation', purpose='training')
    except ValueError:
        pass
    else:
        raise ValueError('Held optimization role was accepted')
    atomic_json(REPORTS / 'complete.json', dict(admission_sha256=file_hash(OUTPUT / 'admission.json'),
        complete_examples=len(loaded), complete_response_EOS_targets=targets, complete_finite_original_cases=cases_count,
        maximum_complete_context_tokens=maximum, all_actual_stored_BPE_sequences_masks_and_source_hashes_reconciled=True,
        held_optimization_role_rejected=True, training_started=False, model_updates=0, goal_achieved=False))
    print(json.dumps({key: value for key, value in admission.items() if key != 'source_files_sha256'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
