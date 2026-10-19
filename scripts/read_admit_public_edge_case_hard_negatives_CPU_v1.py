"""Reconstruct all mining dispositions and freshly execute every admitted pair."""
from collections import Counter
from datetime import datetime, timezone
import doctest
import gzip
import json
import os
from pathlib import Path
import shutil
import traceback

from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.documented_doctest_v1 import compare
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_edge_case_hard_negatives_v1 import candidates, pair, sha
from slm_lab.public_function_contract_census_v2 import cases, digest, family, identity, observe, passes
from slm_lab.public_function_rollout_observation_v1 import observe as published


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/public-edge-case-hard-negatives-CPU-v1'
CANDIDATE = ROOT / 'data/public-edge-case-hard-negatives-candidate-v1'
OUT = ROOT / 'reports/public-edge-case-hard-negatives-v1-readback-admission'
DATA = ROOT / 'data/public-edge-case-hard-negatives-admitted-v1'
ORIGINAL = ROOT / 'data/direct-algorithm-function-contracts-admitted-v1'


def read(path):
    return json.loads(Path(path).read_text())


def public_pass(row, program, observed):
    if observed.get('actual_generated_program_sha256') != sha(program) or observed.get('original_source_id') != row['id']:
        raise ValueError('Exact original source and actual public execution program required')
    if not observed['all_original_examples_observed']:
        if observed['all_original_examples_passed']:
            raise ValueError('Incomplete public observations cannot pass')
        return False
    tests = doctest.DocTestParser().get_examples(row['source_candidate']['original_literal_docstring'])
    raw = observed['raw_results']
    if len(tests) != len(raw):
        raise ValueError('Every original public example must remain')
    checks = [compare(test, actual) for test, actual in zip(tests, raw)]
    result = all(checks) and sum(v['entry_body_calls'] for v in raw) > 0
    if checks != observed['example_passes'] or result != observed['all_original_examples_passed']:
        raise ValueError('All raw published outputs and body calls must reproduce their disposition')
    return result


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(p.exists() for p in [OUT, DATA])):
        raise ValueError('Fresh CUDA-hidden low-priority independent readback required')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan = read(SOURCE / 'plan.json')
    done = read(SOURCE / 'complete.json')
    if (not terminal(plan['owner']) or not done.get('complete') or done['owner'] != plan['owner']
            or (SOURCE / 'failure.json').exists() or done['original_sources'] != 57 or done['candidate_pairs'] < 1):
        raise ValueError('Exact terminal successful complete producer and nonempty candidates required')
    files = dict(done['source_files_sha256'])
    for path in [SOURCE / 'complete.json', *python_sources(ROOT, [Path(__file__)])]:
        files[str(path)] = file_hash(path)
    guard = FileBindings(files)
    rows = {r['id']: r for r in map(json.loads, gzip.open(ORIGINAL / 'original-rows.jsonl.gz', 'rt'))}
    fixed = read(SOURCE / 'all57-fixed-source-candidates.json')
    outcomes = [json.loads(v) for v in (SOURCE / 'all-actual-program-runs.jsonl').read_text().splitlines()]
    stored = [json.loads(v) for v in (CANDIDATE / 'pairs.jsonl').read_text().splitlines()]
    if len(rows) != 841 or len(fixed) != 57 or len(stored) != done['candidate_pairs']:
        raise ValueError('Complete original records and every declared candidate pair required')
    tokenizer = Tokenizer.from_file(str(CANDIDATE / 'tokenizer.json'))
    position = 0
    expected_pairs = []
    counts = Counter()
    for source_position, item in enumerate(fixed):
        row = rows[item['source_id']]
        rebuilt = dict(source_id=row['id'], source_identity=identity(row), family=family(row),
                       cases_sha256=digest(cases(row)), candidates=candidates(row, maximum=16))
        if rebuilt != item:
            raise ValueError('Every declared whole-source lexical candidate must reconstruct')
        original = outcomes[position]
        position += 1
        if (original['source_position'] != source_position or original['source_id'] != row['id']
                or original['kind'] != 'original_positive'
                or not original['actual_contract']['all_cases_passed']
                or not public_pass(row, row['response'], original['actual_public'])):
            raise ValueError('All57 original positives and complete public observations required')
        positive = original['actual_contract']
        selected = cases(row)
        if (positive.get('source_id') != row['id'] or positive.get('program_sha256') != sha(row['response'])
                or positive.get('cases_sha256') != digest(selected) or not positive.get('all_cases_observed')
                or len(positive.get('raw_results', [])) != len(selected)
                or [passes(case, raw) for case, raw in zip(selected, positive['raw_results'])] != [True] * len(selected)
                or positive['case_passes'] != [True] * len(selected)):
            raise ValueError('Every actual positive finite return/type/mutation must reconstruct')
        counts['actual_program_runs'] += 2
        for mutation in item['candidates']:
            value = outcomes[position]
            position += 1
            if value['kind'] != 'single_lexical_candidate' or value['source_position'] != source_position or value['mutation'] != mutation:
                raise ValueError('Every and only fixed lexical trial in its complete original order required')
            counts['actual_program_runs'] += 1
            counts['candidates_observed'] += 1
            disposition = 'published_examples_failed_or_not_fully_observed'
            if public_pass(row, mutation['program'], value['actual_public']):
                counts['published_passes'] += 1
                counts['actual_program_runs'] += 1
                actual = value['actual_contract']
                if (actual.get('source_id') != row['id'] or actual.get('program_sha256') != mutation['program_sha256']
                        or actual.get('cases_sha256') != digest(cases(row))):
                    raise ValueError('Every finite candidate must match its exact source/program/case identity')
                disposition = ('finite_contract_not_fully_observed_no_label' if not actual['all_cases_observed']
                               else 'no_counterexample_in_fixed_cases' if actual['all_cases_passed'] else 'candidate_hard_negative')
                if actual['all_cases_observed']:
                    selected = cases(row)
                    flags = [passes(case, result) for case, result in zip(selected, actual['raw_results'])]
                    if len(actual['raw_results']) != len(selected) or flags != actual['case_passes'] or all(flags) != actual['all_cases_passed']:
                        raise ValueError('Every complete finite trial return/type/mutation must reconstruct')
                if disposition == 'candidate_hard_negative':
                    try:
                        result = pair(row, mutation, original['actual_contract'], actual, value['actual_public'], tokenizer)
                    except ValueError as error:
                        if 'fit without truncation' not in str(error):
                            raise
                        disposition = 'whole_native_pair_exceeds4096_preserved_no_admission'
                    else:
                        expected_pairs.append(result)
            if disposition != value['disposition']:
                raise ValueError('Candidate dispositions must follow the original complete tests, including abstentions')
            counts[disposition] += 1
    if (position != len(outcomes) or expected_pairs != stored or dict(counts) != done['counts']
            or counts['candidates_observed'] != plan['actual_lexical_candidates']):
        raise ValueError('All original trial accounting and complete native pair bytes must reproduce')
    if shutil.disk_usage(ROOT).free < 26 * 1024**3 + 64 * 1024**2:
        raise ValueError('Complete independent artifacts must fit above the original reserve')
    OUT.mkdir()
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    atomic_json(OUT / 'plan.json', dict(owner=owner, source_files_sha256=files, all536_candidates_reconstructed=len(outcomes)-57,
                candidate_pairs=len(stored), every_pair_fresh_positive_and_negative_public_and_finite_execution=True,
                shared_bound_oracle_not_independent_oracle_implementation=True, CPU_threads=1, CPU_nice=os.nice(0),
                new_neural_updates=0, goal_achieved=False))
    cache = {}
    fresh = 0
    try:
        with (OUT / 'all-fresh-actual-program-runs.jsonl').open('x') as stream:
            for index, value in enumerate(stored):
                guard.check()
                row = rows[value['source_id']]
                if row['id'] not in cache:
                    positive = observe(row)
                    public_positive = published(row, row['response'])
                    if not positive['all_cases_passed'] or not public_positive['all_original_examples_passed']:
                        raise ValueError('Every fresh original positive must still pass complete observations')
                    cache[row['id']] = positive
                    fresh += 2
                    stream.write(json.dumps(dict(kind='original_positive', source_id=row['id'], actual_contract=positive,
                                                 actual_public=public_positive)) + '\n')
                negative = observe(row, value['rejected'], seconds=3)
                public = published(row, value['rejected'])
                actual = pair(row, value['mutation'], cache[row['id']], negative, public, tokenizer)
                fresh += 2
                if actual != value:
                    raise ValueError('Every fresh program/counterexample/native target must exactly reproduce the candidate')
                stream.write(json.dumps(dict(kind='hard_negative', pair_id=value['id'], actual_contract=negative,
                                             actual_public=public)) + '\n')
                stream.flush()
                atomic_json(OUT / 'progress.json', dict(actual_pairs_reproduced=index + 1, planned_pairs=len(stored),
                            actual_fresh_program_runs=fresh, new_neural_updates=0, goal_achieved=False))
        guard.check(full=True)
        DATA.mkdir()
        for name in ['pairs.jsonl', 'tokenizer.json', 'DATASET-LICENSE.txt', 'publisher-README.md']:
            shutil.copyfile(CANDIDATE / name, DATA / name)
        data_files = {name: file_hash(DATA / name) for name in ['pairs.jsonl', 'tokenizer.json', 'DATASET-LICENSE.txt', 'publisher-README.md']}
        atomic_json(DATA / 'manifest.json', dict(format='public-original-lexical-edge-case-machine-preferences-admitted-v1',
                    complete=True, candidate_pairs=len(stored), distinct_positive_sources=len(cache),
                    source_files_sha256=files, files_sha256=data_files,
                    native_response_EOS_targets=done['native_response_EOS_targets'],
                    actual_fresh_confined_program_runs=fresh, all_public_examples_pass_for_both_sides=True,
                    all_original_positive_finite_checks_pass_and_each_negative_has_observed_counterexample=True,
                    single_lexical_source_token_edits_with_all_other_bytes_unchanged=True,
                    machine_preference_training_admitted=True, human_preference_training_admitted=False,
                    on_policy_RL_training_admitted=False, new_model_training_started=False, goal_achieved=False))
        if torch.cuda.is_initialized():
            raise ValueError('CPU data admission must not initialize CUDA')
        atomic_json(OUT / 'complete.json', dict(complete=True, at=datetime.now(timezone.utc).isoformat(), owner=owner,
                    source_files_sha256={**files, **{str(DATA / name): file_hash(DATA / name) for name in [*data_files, 'manifest.json']},
                                         str(OUT / 'all-fresh-actual-program-runs.jsonl'): file_hash(OUT / 'all-fresh-actual-program-runs.jsonl')},
                    all57_original_source_candidate_lists_and_all536_declared_trials_reconstructed=True,
                    actual_original_trials_reconstructed=counts['candidates_observed'], actual_admitted_machine_pairs=len(stored),
                    distinct_positive_sources=len(cache), actual_fresh_confined_program_runs=fresh,
                    native_response_EOS_targets=done['native_response_EOS_targets'], actual_new_neural_updates=0,
                    original_training_and_all_full_benchmarks_unmodified=True, candidate_promoted=False, goal_achieved=False))
        print(json.dumps(dict(complete=True, admitted_pairs=len(stored), fresh_program_runs=fresh, goal_achieved=False)), flush=True)
    except BaseException as error:
        atomic_json(OUT / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), automatic_retry=False,
                    original_training_unmodified=True, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
