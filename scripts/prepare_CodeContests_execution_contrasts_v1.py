"""Observe whole original negatives against already verified public references.

Every one of3959declared original source positions gets a disposition. The
first publisher-order Python3 negative is fixed before local execution; no
fallback to an easier negative or shorter native context is permitted.
Creates a candidate only. Existing production queues and admissions stay fixed.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from tokenizers import Tokenizer

from slm_lab.code_contests_CE_view_v1 import CodeContestsCEView
from slm_lab.code_contests_contrast_candidates_v1 import select, original_cases, execute_negative, counterexample
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.held_text_overlap_v1 import HeldTextOverlap
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.source_jsonl_records_v1 import records

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'reports/CodeContests-full-Python-references-v1'
REFERENCE_READER = ROOT / 'reports/CodeContests-reference-outputs-v1-readback'
SCHEMA = ROOT / 'data/code-contests-protobuf-schema-v1'
OUT = ROOT / 'reports/CodeContests-execution-contrast-candidate-v1'
DATA = ROOT / 'data/CodeContests-execution-contrast-candidate-v1'


def terminal(identity):
    p = Path('/proc') / str(identity['pid']) / 'stat'
    if not p.exists():
        return True
    fields = p.read_text().split(') ', 1)[1].split()
    return fields[19] != identity['ticks'] or fields[0] in {'Z', 'X'}


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('TOKENIZERS_PARALLELISM') != 'false'
            or OUT.exists() or DATA.exists()):
        raise ValueError('Fresh CPU-hidden public execution-contrast candidate required')
    log = ROOT / 'reports/code-contests-contrast-candidates-v1-tests.log'
    if '3 passed' not in log.read_text():
        raise ValueError('Actual completed counterexample, repeat, mask, source-role and corruption controls must pass')
    reference_plan = json.loads((REFERENCE / 'plan.json').read_text())
    reader_plan = json.loads((REFERENCE_READER / 'plan.json').read_text())
    reader = json.loads((REFERENCE_READER / 'complete.json').read_text())
    reference_done = json.loads((REFERENCE / 'complete.json').read_text())
    if (not terminal(reference_plan['owner']) or not terminal(reader_plan['owner'])
            or not reader['every_declared_original_reference_output_and_complete_source_rebuilt']
            or not reader['every_retained_stdout_compared_directly_against_original_expected_tokens']
            or reference_done['actual_original_training_problems'] != 3959):
        raise ValueError('Complete original reference run and full output reader are prerequisites')
    if shutil.disk_usage(ROOT).free < 32 * 1024**3:
        raise ValueError('Preserve the persistent writer reserve')
    view = CodeContestsCEView(ROOT)
    registry_path = ROOT / 'reports/CodeContests-native-held-text-v2/unchanged-held-text-registry.json'
    source_path = ROOT / 'data/code-contests-native-train-records-v1/all-original-problem-records.jsonl'
    schema = json.loads((SCHEMA / 'manifest.json').read_text())
    files = {**reference_plan['source_files_sha256'], **reader['source_files_sha256']}
    for p in [REFERENCE / 'complete.json', REFERENCE_READER / 'complete.json', registry_path,
              SCHEMA / 'manifest.json', source_path, log, view.tokenizer_path,
              *python_sources(ROOT, [Path(__file__), ROOT / 'tests/test_code_contests_contrast_candidates_v1.py'])]:
        files[str(p)] = file_hash(p)
    files.update({str(SCHEMA / name): value for name, value in schema['files_sha256'].items()})
    guard = FileBindings(files)
    OUT.mkdir(); DATA.mkdir(); (OUT / 'source-observations').mkdir()
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    started = time.monotonic()
    atomic_json(OUT / 'plan.json', dict(
        at=datetime.now(timezone.utc).isoformat(), owner=owner, source_files_sha256=files,
        all_original_reference_positions=3959, CPU_worker_limit=4,
        first_original_Python3_negative_fixed_before_observing_execution=True,
        whole_unchanged_positive_and_negative_native_context_limit=4096,
        no_fallback_after_native_runtime_or_output_qualification=True,
        positive_requires_all_original_cases_and_complete_output_readback=True,
        negative_requires_completed_wrong_output_and_two_fresh_byte_identical_observations=True,
        negative_runtime_syntax_and_wall_failures_are_qualifications_not_algorithmic_failure_labels=True,
        original_public_private_generated_cases_qualify_preferences_but_are_not_appended_to_model_text=True,
        all_32953_unchanged_held_stems_screen_both_actual_native_sequences=True,
        entire_source_component_excluded_on_positive_exposure_signal=True,
        no_semantic_or_all_history_independence_certificate=True,
        machine_preferences_not_new_human_judgments=True,
        dataset_admitted_to_training=False, original_training_queues_unchanged=True,
        models_constructed=0, model_updates=0, goal_achieved=False,
    ))
    sys.path.insert(0, schema['isolated_reader_site_packages']); sys.path.insert(0, str(SCHEMA))
    import contest_problem_pb2 as proto
    tokenizer = Tokenizer.from_file(str(view.tokenizer_path))
    registry = json.loads(registry_path.read_text())
    if len(registry) != 32953:
        raise ValueError('All unchanged original held stems are required')
    exposure = HeldTextOverlap(registry)
    identities = json.loads((REFERENCE / 'all-original-selected-reference-identities.json').read_text())
    sources = {r['id']: r for r in records(source_path)}
    if len(identities) != 3959:
        raise ValueError('Every original selected reference position must remain')
    counts = Counter(); dispositions = {}; qualified = {}; blocked = set(); result_bindings = {}

    def next_task(position):
        identity = identities[position]; row = view.row(identity['view_position'])
        source = sources[identity['original_problem_id']]
        with open(source['extracted_record_file'], 'rb') as stream:
            stream.seek(source['extracted_record_offset']); raw = stream.read(source['extracted_record_bytes'])
        if hashlib.sha256(raw).hexdigest() != identity['original_record_sha256']:
            raise ValueError('An original complete publisher record changed')
        problem = proto.ContestProblem()
        if problem.ParseFromString(raw) != len(raw):
            raise ValueError('Every complete original protobuf field must parse')
        reference_path = REFERENCE / 'original-problem-executions' / f'{position:05d}.json'
        reference = json.loads(reference_path.read_text())
        pair, why = select(row, problem, reference, tokenizer)
        counts['original_source_positions_rebuilt'] += 1
        counts['retained_positive_reference_case_outputs_rechecked'] += reference.get('actual_execution', {}).get('actual_observed_case_count', 0)
        base = dict(original_reference_position=position, original_problem_id=identity['original_problem_id'],
                    original_view_position=identity['view_position'], original_record_sha256=identity['original_record_sha256'],
                    positive_result_sha256=file_hash(reference_path), status=why,
                    dataset_admitted_to_training=False)
        if pair is None:
            counts[why] += 1; dispositions[position] = base
            return None
        if view.sequence(identity['view_position']) != (pair['chosen_ids'], pair['chosen_labels']):
            raise ValueError('The whole positive must be the unchanged admitted CE native sequence')
        signals = {}
        for arm in ('chosen', 'rejected'):
            decoded = tokenizer.decode(pair[arm + '_ids'][1:-1], skip_special_tokens=False)
            literal, lexical = exposure.literal(decoded), exposure.lexical_copies(decoded)
            signals[arm] = dict(literal=literal, lexical=lexical)
            if literal or lexical:
                blocked.add(pair['group_id'])
        base.update(original_negative_solution_position=pair['original_negative_solution_position'],
                    original_negative_program_sha256=pair['original_negative_program_sha256'], held_exposure_signals=signals)
        counts['whole_native_first_negative_pairs_scanned'] += 1
        if any(s['literal'] or s['lexical'] for s in signals.values()):
            base['status'] = 'source_component_has_held_exposure_signal'
            counts[base['status']] += 1; dispositions[position] = base
            return None
        return base, pair, original_cases(problem)

    try:
        context = multiprocessing.get_context('spawn')
        submitted = 0; active = {}
        with ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
            while submitted < len(identities) or active:
                while submitted < len(identities) and len(active) < 4:
                    position = submitted; submitted += 1
                    task = next_task(position)
                    if task is not None:
                        base, pair, cases = task
                        future = pool.submit(execute_negative, pair['rejected'], cases)
                        active[future] = (position, base, pair, cases)
                if active:
                    done, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        position, base, pair, cases = active.pop(future)
                        actual = future.result()
                        witness, why = counterexample(pair['rejected'], cases, actual)
                        path = OUT / 'source-observations' / f'{position:05d}.json'
                        atomic_json(path, dict(**base, pair=pair, actual_negative_execution=actual,
                                              counterexample=witness, actual_disposition=why))
                        result_bindings[str(path)] = file_hash(path)
                        base.update(status=why, negative_observation_sha256=file_hash(path))
                        dispositions[position] = base; counts[why] += 1
                        counts['actual_fresh_negative_case_executions'] += actual['first']['actual_observed_case_count']
                        if actual['repeat'] is not None:
                            counts['actual_fresh_negative_case_executions'] += actual['repeat']['actual_observed_case_count']
                        if witness is not None:
                            qualified[position] = dict(pair, counterexample=witness,
                                original_reference_position=position, original_view_position=base['original_view_position'],
                                original_record_sha256=base['original_record_sha256'],
                                actual_negative_observation_file=str(path), actual_negative_observation_sha256=file_hash(path))
                if shutil.disk_usage(ROOT).free < 28 * 1024**3:
                    raise ValueError('Preserve all persistent training writers and original evidence')
                atomic_json(OUT / 'progress.json', dict(
                    owner=owner, original_positions_completed=len(dispositions), original_positions_total=len(identities),
                    original_positions_prepared=submitted, active_CPU_observers=len(active),
                    actual_counts=dict(counts), elapsed_seconds=time.monotonic() - started,
                    dataset_admitted_to_training=False, models_constructed=0, model_updates=0, goal_achieved=False,
                ))
        if set(dispositions) != set(range(len(identities))):
            raise ValueError('Every original declared source needs a complete disposition')
        with (OUT / 'all-original-source-dispositions.jsonl').open('x') as stream:
            for i in range(len(identities)):
                stream.write(json.dumps(dispositions[i], ensure_ascii=False) + '\n')
        final = [row for i, row in sorted(qualified.items()) if row['group_id'] not in blocked]
        counts['qualified_pairs_excluded_by_whole_source_component'] = len(qualified) - len(final)
        counts['complete_candidate_pairs'] = len(final)
        for arm in ('chosen', 'rejected'):
            counts[arm + '_complete_context_positions'] = sum(len(row[arm + '_ids']) for row in final)
            counts[arm + '_response_EOS_targets'] = sum(sum(v != -100 for v in row[arm + '_labels']) for row in final)
        with (DATA / 'candidate-pairs.jsonl').open('x') as stream:
            for row in final:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        for source, destination in [
            (view.tokenizer_path, DATA / 'tokenizer.json'),
            (view.pack.folder / 'DATASET-LICENSE.txt', DATA / 'DATASET-LICENSE.txt'),
            (ROOT / 'data/code-contests-original-train-cache-v2/publisher-README.md', DATA / 'publisher-README.md'),
        ]:
            shutil.copy2(source, destination)
        atomic_json(OUT / 'source-component-exclusions.json', dict(
            blocked_source_components=sorted(blocked), unchanged_held_stems=32953,
            exposure_configuration=exposure.configuration, negative_search_not_independently_repeated=True,
            semantic_and_all_history_independence_not_certified=True,
        ))
        atomic_json(OUT / 'all-actual-negative-observation-bindings.json', result_bindings)
        guard.check(full=True)
        evidence = {str(p): file_hash(p) for p in [OUT / 'plan.json', OUT / 'all-original-source-dispositions.jsonl',
                    OUT / 'source-component-exclusions.json', OUT / 'all-actual-negative-observation-bindings.json',
                    *sorted(DATA.iterdir())]}
        manifest = dict(format='own-original-public-execution-contrast-candidate-v1',
            actual_counts=dict(counts), original_source_positions=3959,
            first_original_Python3_negative_fixed_before_execution=True,
            positive_all_original_finite_cases_pass_and_negative_two_completed_output_counterexamples=True,
            original_cases_not_appended_to_model_text=True, no_native_prompt_program_or_EOS_clipped=True,
            source_files_sha256=evidence, dataset_admitted_to_training=False,
            separate_complete_source_native_mask_and_observation_readback_still_required=True,
            human_preferences=False, models_constructed=0, model_updates=0, goal_achieved=False)
        atomic_json(DATA / 'manifest.json', manifest)
        atomic_json(OUT / 'complete.json', dict(
            at=datetime.now(timezone.utc).isoformat(), complete=True,
            every_original_reference_source_dispositioned=True, actual_counts=dict(counts),
            source_files_sha256={**evidence, str(DATA / 'manifest.json'): file_hash(DATA / 'manifest.json')},
            candidate_dataset=str(DATA), dataset_admitted_to_training=False,
            original_production_queues_and_existing_source_admissions_unchanged=True,
            all_candidate_negative_outputs_locally_recompared_not_separate_independent_full_readback=True,
            new_model_quality_not_measured=True, models_constructed=0, model_updates=0, goal_achieved=False,
        ))
        print(json.dumps(dict(complete=str(OUT / 'complete.json'), actual_counts=dict(counts),
                              dataset_admitted_to_training=False)), flush=True)
    except BaseException as error:
        atomic_json(OUT / 'failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            original_source_positions_completed=len(dispositions), original_training_untouched=True,
            dataset_admitted_to_training=False, models_constructed=0, model_updates=0, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
