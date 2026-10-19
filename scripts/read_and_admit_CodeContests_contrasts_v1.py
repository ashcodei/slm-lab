"""Rebuild every source/pair/mask/raw counterexample before separate admission."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from tokenizers import Tokenizer

from slm_lab.code_contests_CE_view_v1 import CodeContestsCEView
from slm_lab.code_contests_contrast_readback_v1 import reconstruct_pair, reconstruct_witness
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.execution_contrast_data_v1 import ExecutionContrastData
from slm_lab.held_text_overlap_v1 import HeldTextOverlap
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.source_jsonl_records_v1 import records

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT / 'reports/CodeContests-execution-contrast-candidate-v1'
CANDIDATE = ROOT / 'data/CodeContests-execution-contrast-candidate-v1'
OUT = ROOT / 'reports/CodeContests-execution-contrast-v1-readback-admission'
ADMITTED = ROOT / 'data/CodeContests-execution-contrast-admitted-v1'
REFERENCE = ROOT / 'reports/CodeContests-full-Python-references-v1'
SCHEMA = ROOT / 'data/code-contests-protobuf-schema-v1'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists() or ADMITTED.exists():
        raise ValueError('A fresh separate CPU-hidden source readback and admission is required')
    tests = ROOT / 'reports/code-contests-contrast-readback-v1-tests.log'
    if '2 passed' not in tests.read_text():
        raise ValueError('Actual separate source/native/observation reconstruction controls must pass')
    dependency = read(PREPARATION / 'plan.json')
    registry_path = ROOT / 'reports/CodeContests-native-held-text-v2/unchanged-held-text-registry.json'
    source_path = ROOT / 'data/code-contests-native-train-records-v1/all-original-problem-records.jsonl'
    schema = read(SCHEMA / 'manifest.json')
    files = {**dependency['source_files_sha256']}
    for p in [PREPARATION / 'plan.json', tests, registry_path, source_path, SCHEMA / 'manifest.json',
              *python_sources(ROOT, [Path(__file__), ROOT / 'tests/test_code_contests_contrast_readback_v1.py']),
              ROOT / 'tests/test_code_contests_contrast_candidates_v1.py']:
        files[str(p)] = file_hash(p)
    guard = FileBindings(files); OUT.mkdir()
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    atomic_json(OUT / 'plan.json', dict(
        at=datetime.now(timezone.utc).isoformat(), owner=owner, source_files_sha256=files,
        exact_original_preparation_owner=dependency['owner'],
        all3959original_sources_and_every_actual_native_pair_and_counterexample_rebuilt=True,
        reader_uses_no_candidate_selector_formatter_execution_wrapper_or_counterexample_reader=True,
        actual_programs_not_reexecuted_or_OS_confinement_independently_certified=True,
        admission_scope='machine_DPO_and_chosen_anchor_CE_only', rejected_positive_CE_forbidden=True,
        machine_preferences_not_new_human_judgments=True,
        original_training_queues_and_candidate_dataset_remain_unchanged=True,
        models_constructed=0, model_updates=0, goal_achieved=False,
    ))
    counts = Counter()
    try:
        while not terminal(dependency['owner']):
            if (PREPARATION / 'failure.json').exists():
                raise ValueError('Original preparation failed; all partial evidence is preserved')
            guard.check(full=False)
            atomic_json(OUT / 'waiting.json', dict(owner=owner, exact_preparation_owner=dependency['owner'],
                status='waiting for the same original public negative-observation job to finish',
                dataset_admitted_to_training=False, models_constructed=0, model_updates=0, goal_achieved=False))
            time.sleep(15)
        if (PREPARATION / 'failure.json').exists() or not (PREPARATION / 'complete.json').is_file():
            raise ValueError('Exact original owner ended without complete preparation evidence')
        done = read(PREPARATION / 'complete.json'); manifest = read(CANDIDATE / 'manifest.json')
        if (not done['every_original_reference_source_dispositioned'] or done['dataset_admitted_to_training'] is not False
                or manifest['format'] != 'own-original-public-execution-contrast-candidate-v1'):
            raise ValueError('Every original source disposition and unchanged unadmitted candidate is required')
        completed_guard = FileBindings(done['source_files_sha256'])
        view = CodeContestsCEView(ROOT); tokenizer = Tokenizer.from_file(str(view.tokenizer_path))
        if file_hash(view.tokenizer_path) != file_hash(CANDIDATE / 'tokenizer.json'):
            raise ValueError('The same own tokenizer must frame every full pair')
        sys.path.insert(0, schema['isolated_reader_site_packages']); sys.path.insert(0, str(SCHEMA))
        import contest_problem_pb2 as proto
        identities = read(REFERENCE / 'all-original-selected-reference-identities.json')
        sources = {r['id']: r for r in records(source_path)}
        dispositions = list(records(PREPARATION / 'all-original-source-dispositions.jsonl'))
        if len(identities) != 3959 or len(dispositions) != len(identities):
            raise ValueError('All original declared sources need unchanged dispositions')
        registry = read(registry_path)
        if len(registry) != 32953:
            raise ValueError('All original held stems must remain')
        exposure = HeldTextOverlap(registry); blocked = set(); qualified = {}; seen_observations = set()
        bindings = read(PREPARATION / 'all-actual-negative-observation-bindings.json')
        with (OUT / 'all-original-source-pair-output-readbacks.jsonl').open('x') as ledger:
            for position, (identity, disposition) in enumerate(zip(identities, dispositions)):
                if (disposition['original_reference_position'] != position
                        or disposition['original_problem_id'] != identity['original_problem_id']
                        or disposition['original_view_position'] != identity['view_position']):
                    raise ValueError('Original source identity, ordering or selection differs')
                row = view.row(identity['view_position']); source = sources[identity['original_problem_id']]
                with open(source['extracted_record_file'], 'rb') as stream:
                    stream.seek(source['extracted_record_offset']); raw = stream.read(source['extracted_record_bytes'])
                if hashlib.sha256(raw).hexdigest() != identity['original_record_sha256']:
                    raise ValueError('Whole original publisher record differs')
                problem = proto.ContestProblem()
                if problem.ParseFromString(raw) != len(raw):
                    raise ValueError('Complete original protobuf record must parse')
                reference_path = REFERENCE / 'original-problem-executions' / f'{position:05d}.json'
                reference = read(reference_path)
                if disposition['positive_result_sha256'] != file_hash(reference_path):
                    raise ValueError('Original all-case reference evidence changed')
                pair, why = reconstruct_pair(row, problem, reference, tokenizer)
                counts['original_source_positions_rebuilt'] += 1
                counts['retained_positive_reference_case_outputs_rechecked'] += reference.get('actual_execution', {}).get('actual_observed_case_count', 0)
                witness = None; signals = {}
                if pair is not None:
                    if view.sequence(identity['view_position']) != (pair['chosen_ids'], pair['chosen_labels']):
                        raise ValueError('Every positive native token and complete EOS mask must match its original source')
                    counts['whole_native_first_negative_pairs_scanned'] += 1
                    for arm in ('chosen', 'rejected'):
                        decoded = tokenizer.decode(pair[arm + '_ids'][1:-1], skip_special_tokens=False)
                        signals[arm] = dict(literal=exposure.literal(decoded), lexical=exposure.lexical_copies(decoded))
                        if signals[arm]['literal'] or signals[arm]['lexical']:
                            blocked.add(pair['group_id'])
                    if disposition['held_exposure_signals'] != signals:
                        raise ValueError('Actual whole native held-source signal readback differs')
                    if any(v['literal'] or v['lexical'] for v in signals.values()):
                        why = 'source_component_has_held_exposure_signal'
                    else:
                        path = PREPARATION / 'source-observations' / f'{position:05d}.json'
                        digest = file_hash(path); seen_observations.add(str(path))
                        if digest != bindings[str(path)] or digest != disposition['negative_observation_sha256']:
                            raise ValueError('An actual fresh negative observation changed')
                        actual = read(path)
                        if actual['pair'] != pair:
                            raise ValueError('Both complete source programs, actual BPE ids, EOS masks and metadata must rebuild')
                        witness, why = reconstruct_witness(pair['rejected'], problem, actual['actual_negative_execution'])
                        if actual['counterexample'] != witness or actual['actual_disposition'] != why:
                            raise ValueError('Independent raw-output counterexample or qualification differs')
                        execution = actual['actual_negative_execution']
                        counts['actual_fresh_negative_case_executions'] += execution['first']['actual_observed_case_count']
                        if execution['repeat'] is not None:
                            counts['actual_fresh_negative_case_executions'] += execution['repeat']['actual_observed_case_count']
                        if witness is not None:
                            qualified[position] = dict(pair, counterexample=witness, original_reference_position=position,
                                original_view_position=identity['view_position'], original_record_sha256=identity['original_record_sha256'],
                                actual_negative_observation_file=str(path), actual_negative_observation_sha256=digest)
                if disposition['status'] != why:
                    raise ValueError('Every original candidate and qualification must retain its independent disposition')
                counts[why] += 1
                ledger.write(json.dumps(dict(original_reference_position=position, original_problem_id=identity['original_problem_id'],
                    disposition=why, complete_native_pair_rebuilt=pair is not None,
                    actual_counterexample_rebuilt=witness is not None, actual_native_held_signals=signals), ensure_ascii=False) + '\n')
        if set(bindings) != seen_observations:
            raise ValueError('Every original actual negative observation must be covered exactly once')
        final = [r for _, r in sorted(qualified.items()) if r['group_id'] not in blocked]
        candidates = list(records(CANDIDATE / 'candidate-pairs.jsonl'))
        if not final or final != candidates:
            raise ValueError('Every whole candidate pair and source-component exclusion must rebuild exactly')
        counts['qualified_pairs_excluded_by_whole_source_component'] = len(qualified) - len(final)
        counts['complete_candidate_pairs'] = len(final)
        for arm in ('chosen', 'rejected'):
            counts[arm + '_complete_context_positions'] = sum(len(r[arm + '_ids']) for r in final)
            counts[arm + '_response_EOS_targets'] = sum(sum(v != -100 for v in r[arm + '_labels']) for r in final)
        if dict(counts) != done['actual_counts'] or dict(counts) != manifest['actual_counts']:
            raise ValueError('Every source, target, actual execution and qualification denominator must reconcile')
        if read(PREPARATION / 'source-component-exclusions.json')['blocked_source_components'] != sorted(blocked):
            raise ValueError('Whole source-component exclusion set must remain exact')
        ADMITTED.mkdir()
        rows = [dict(row, optimization_admitted=True, original_candidate_optimization_admitted=False,
                     original_candidate_position=i) for i, row in enumerate(final)]
        with (ADMITTED / 'pairs.jsonl').open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        for name in ('tokenizer.json', 'DATASET-LICENSE.txt', 'publisher-README.md'):
            shutil.copy2(CANDIDATE / name, ADMITTED / name)
        atomic_json(ADMITTED / 'manifest.json', dict(
            format='own-original-public-execution-contrast-admitted-v1', examples=len(rows),
            pairs_sha256=file_hash(ADMITTED / 'pairs.jsonl'), tokenizer_sha256=file_hash(ADMITTED / 'tokenizer.json'),
            actual_counts=dict(counts), original_candidate_manifest_sha256=file_hash(CANDIDATE / 'manifest.json'),
            candidate_unchanged_and_unadmitted=True, all_full_native_response_EOS_masks_rebuilt=True,
            both_programs_unmodified_original_public_train_sources=True, human_preferences=False,
            training_scope='machine_DPO_and_chosen_anchor_CE_only',
            rejected_programs_must_never_be_positive_CE_targets=True,
            full_publisher_judge_and_semantic_source_independence_not_certified=True,
        ))
        guard.check(full=True); completed_guard.check(full=True)
        evidence = {str(p): file_hash(p) for p in [OUT / 'plan.json', OUT / 'all-original-source-pair-output-readbacks.jsonl',
            PREPARATION / 'complete.json', CANDIDATE / 'manifest.json', *sorted(ADMITTED.iterdir())]}
        readback_path = OUT / 'independent-source-readback.json'
        atomic_json(readback_path, dict(
            every_original_source_and_candidate_native_pair_and_raw_counterexample_rebuilt=True,
            actual_counts=dict(counts), all_original_source_positions=3959, actual_held_stems=32953,
            full_case_output_reference_readback_not_new_reference_execution=True,
            negative_execution_prefix_and_repeated_case_outputs_not_all_negative_cases_reexecuted=True,
            all_candidate_native_inputs_rescanned_no_independent_negative_search_certificate=True,
            source_files_sha256=evidence, models_constructed=0, model_updates=0, goal_achieved=False,
        ))
        atomic_json(ADMITTED / 'admission.json', dict(
            training_admitted=True, source_manifest_sha256=file_hash(ADMITTED / 'manifest.json'),
            independent_readback_file=str(readback_path), independent_readback_sha256=file_hash(readback_path),
            training_scope='machine_DPO_and_chosen_anchor_CE_only', human_preferences=False,
            rejected_programs_must_never_be_positive_CE_targets=True,
            source_files_sha256={**evidence, str(readback_path): file_hash(readback_path)},
        ))
        cold = ExecutionContrastData(ROOT, purpose='audit')
        if len(cold) != len(rows):
            raise ValueError('Every separately admitted source must cold-read')
        for i, row in enumerate(rows):
            for arm in ('chosen', 'rejected'):
                if cold.sequence(i, arm) != (row[arm + '_ids'], row[arm + '_labels']):
                    raise ValueError('Every actual admitted native source/mask must cold-read exactly')
                batch = cold.batch([i], arm, 'cpu')
                if int((batch['labels'][:, 1:] != -100).sum()) != sum(v != -100 for v in row[arm + '_labels']):
                    raise ValueError('Every admitted target and EOS must reach an actual complete CPU batch')
        atomic_json(OUT / 'complete.json', dict(
            at=datetime.now(timezone.utc).isoformat(), complete=True, actual_counts=dict(counts),
            every_original_source_and_candidate_pair_and_actual_raw_counterexample_rebuilt=True,
            every_admitted_whole_sequence_and_target_EOS_count_cold_read_and_batched_exact=True,
            dataset_admitted_to_machine_preference_training=True,
            training_scope='machine_DPO_and_chosen_anchor_CE_only',
            original_candidate_and_all_production_queues_unchanged=True,
            no_actual_new_machine_preference_production_updates=True,
            source_files_sha256={**evidence, str(readback_path): file_hash(readback_path),
                                 str(ADMITTED / 'admission.json'): file_hash(ADMITTED / 'admission.json')},
            models_constructed=0, model_updates=0, goal_achieved=False,
        ))
        print(json.dumps(dict(complete=str(OUT / 'complete.json'), actual_counts=dict(counts),
                              dataset_admitted_to_machine_preference_training=True)), flush=True)
    except BaseException as error:
        atomic_json(OUT / 'failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            actual_counts=dict(counts), original_training_untouched=True, models_constructed=0,
            model_updates=0, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
