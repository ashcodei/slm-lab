"""Mine fixed-budget source-token bugs missed by all published examples."""
from collections import Counter
from datetime import datetime, timezone
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
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_edge_case_hard_negatives_v1 import candidates, pair
from slm_lab.public_function_contract_census_v2 import cases, digest, family, identity, observe
from slm_lab.public_function_rollout_observation_v1 import observe as published


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/public-edge-case-hard-negatives-CPU-v1'
DATA = ROOT / 'data/public-edge-case-hard-negatives-candidate-v1'
ORIGINAL = ROOT / 'data/direct-algorithm-function-contracts-admitted-v1'
ADMISSION = ROOT / 'reports/public-contract-census-v3-readback-admission/complete.json'
CENSUS = ROOT / 'reports/public-contract-census-CPU-v2/all-selected-original-public-and-contract-runs.jsonl'
MAXIMUM_BYTES = 256 * 1024**2
RESERVE = 26 * 1024**3


def read(path):
    return json.loads(Path(path).read_text())


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(path.exists() for path in [OUT, DATA])):
        raise ValueError('Fresh CUDA-hidden low-priority CPU mining required')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tests = ROOT / 'reports/public-edge-case-hard-negatives-v1-tests.log'
    if '3 passed' not in tests.read_text():
        raise ValueError('Actual lexical/native/runtime counterexample tests required')
    admission = read(ADMISSION)
    if not admission.get('complete') or admission['actual_admitted']['direct']['examples'] != 57:
        raise ValueError('Exact independently admitted finite-qualified original positives required')
    original = list(map(json.loads, gzip.open(ORIGINAL / 'original-rows.jsonl.gz', 'rt')))
    census = list(map(json.loads, CENSUS.read_text().splitlines()))
    qualified = {r['source_id'] for r in census if r['disposition'] == 'finite_contract_qualified'}
    selected = [row for row in original if row['id'] in qualified]
    if len(original) != 841 or len(selected) != len(qualified) or len(selected) != 57:
        raise ValueError('All841 original records and exactly57 original qualified train positives required')
    files = dict(admission['source_files_sha256'])
    for path in [ADMISSION, CENSUS, tests, ORIGINAL / 'original-rows.jsonl.gz', ORIGINAL / 'tokenizer.json',
                 *python_sources(ROOT, [Path(__file__), ROOT / 'tests/test_public_edge_case_hard_negatives_v1.py'])]:
        files[str(path)] = file_hash(path)
    guard = FileBindings(files)
    if shutil.disk_usage(ROOT).free < RESERVE + MAXIMUM_BYTES:
        raise ValueError('Complete bounded data artifacts must fit above the original writer reserve')
    tokenizer = Tokenizer.from_file(str(ORIGINAL / 'tokenizer.json'))
    prepared = [dict(source_id=row['id'], source_identity=identity(row), family=family(row),
                     cases_sha256=digest(cases(row)), candidates=candidates(row, maximum=16)) for row in selected]
    OUT.mkdir()
    DATA.mkdir()
    for name in ['tokenizer.json', 'DATASET-LICENSE.txt', 'publisher-README.md']:
        shutil.copyfile(ORIGINAL / name, DATA / name)
    atomic_json(OUT / 'all57-fixed-source-candidates.json', prepared)
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    atomic_json(OUT / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner,
                source_files_sha256=files, original_source_records=841, preserved_train=826, preserved_validation=15,
                exact_original_positive_sources=57, maximum_candidates_per_source=16,
                actual_lexical_candidates=sum(len(v['candidates']) for v in prepared),
                fixed_candidates_sha256=file_hash(OUT / 'all57-fixed-source-candidates.json'),
                exact_original_prompt_and_positive_programs_unchanged=True,
                every_negative_passes_every_original_public_example_but_fails_a_complete_finite_contract=True,
                timeouts_and_incomplete_observations_never_labelled_wrong=True,
                lexical_edits_preserve_all_other_source_bytes=True, single_CPU_worker=True, CPU_nice=os.nice(0),
                maximum_all_new_artifact_bytes=MAXIMUM_BYTES, minimum_persistent_free_bytes=RESERVE,
                no_existing_training_queue_or_source_modified=True, new_neural_forwards=0, new_neural_updates=0,
                human_preferences=False, on_policy_RL=False, optimization_admitted=False, goal_achieved=False))
    counts = Counter()
    pair_ids = []
    targets = Counter()
    def append(stream, value):
        payload = (json.dumps(value, ensure_ascii=False) + '\n').encode()
        total = sum(p.stat().st_size for folder in [OUT, DATA] for p in folder.iterdir() if p.is_file())
        if total + len(payload) > MAXIMUM_BYTES or shutil.disk_usage(ROOT).free - len(payload) < RESERVE:
            raise ValueError('Declared complete-artifact bound or original persistent reserve reached')
        stream.write(payload)
        stream.flush()
    try:
        with (OUT / 'all-actual-program-runs.jsonl').open('xb') as runs, (DATA / 'pairs.jsonl').open('xb') as pairs:
            for source_position, (row, fixed) in enumerate(zip(selected, prepared)):
                guard.check()
                positive = observe(row)
                public_positive = published(row, row['response'])
                append(runs, dict(source_position=source_position, kind='original_positive', source_id=row['id'],
                                  actual_contract=positive, actual_public=public_positive))
                counts['actual_program_runs'] += 2
                if not positive['all_cases_passed'] or not public_positive['all_original_examples_passed']:
                    raise ValueError('An independently qualified positive failed its fresh complete reference execution')
                for mutation in fixed['candidates']:
                    public = published(row, mutation['program'])
                    counts['actual_program_runs'] += 1
                    outcome = dict(source_position=source_position, kind='single_lexical_candidate', mutation=mutation,
                                   actual_public=public, disposition='published_examples_failed_or_not_fully_observed')
                    counts['candidates_observed'] += 1
                    if public['all_original_examples_passed']:
                        counts['published_passes'] += 1
                        semantic = observe(row, mutation['program'], seconds=3)
                        counts['actual_program_runs'] += 1
                        outcome['actual_contract'] = semantic
                        outcome['disposition'] = ('finite_contract_not_fully_observed_no_label' if not semantic['all_cases_observed']
                                                  else 'no_counterexample_in_fixed_cases' if semantic['all_cases_passed']
                                                  else 'candidate_hard_negative')
                        if outcome['disposition'] == 'candidate_hard_negative':
                            try:
                                candidate = pair(row, mutation, positive, semantic, public, tokenizer)
                            except ValueError as error:
                                if 'fit without truncation' not in str(error):
                                    raise
                                outcome['disposition'] = 'whole_native_pair_exceeds4096_preserved_no_admission'
                            else:
                                append(pairs, candidate)
                                pair_ids.append(candidate['id'])
                                targets['positive'] += candidate['complete_positive_targets']
                                targets['negative'] += candidate['complete_negative_targets']
                    counts[outcome['disposition']] += 1
                    append(runs, outcome)
                atomic_json(OUT / 'progress.json', dict(owner=owner, original_sources_complete=source_position + 1,
                            planned_original_sources=57, counts=dict(counts), candidate_pairs=len(pair_ids),
                            new_neural_updates=0, optimization_admitted=False, goal_achieved=False))
        if counts['candidates_observed'] != sum(len(v['candidates']) for v in prepared) or torch.cuda.is_initialized():
            raise ValueError('Every declared lexical candidate must finish without initializing CUDA')
        guard.check(full=True)
        artifacts = {str(p): file_hash(p) for folder in [OUT, DATA] for p in folder.iterdir() if p.is_file()}
        atomic_json(DATA / 'manifest.json', dict(format='public-original-lexical-edge-case-hard-negative-candidates-v1',
                    complete=True, source_files_sha256=files, files_sha256=artifacts, candidate_pairs=len(pair_ids),
                    native_response_EOS_targets=dict(targets), counts=dict(counts),
                    whole_original_source_positives_and_unmodified_original_task_prompts=True,
                    no_new_human_or_model_generated_labels=True, source_train_not_independent_benchmark=True,
                    candidate_only_pending_fresh_readback=True, optimization_admitted=False, goal_achieved=False))
        atomic_json(OUT / 'complete.json', dict(complete=True, at=datetime.now(timezone.utc).isoformat(), owner=owner,
                    source_files_sha256={**files, **artifacts, str(DATA / 'manifest.json'): file_hash(DATA / 'manifest.json')},
                    original_sources=57, counts=dict(counts), candidate_pairs=len(pair_ids),
                    native_response_EOS_targets=dict(targets), actual_new_neural_updates=0,
                    no_existing_training_queue_or_source_modified=True, optimization_admitted=False,
                    candidate_promoted=False, goal_achieved=False))
        print(json.dumps(dict(complete=True, counts=dict(counts), candidate_pairs=len(pair_ids), goal_achieved=False)), flush=True)
    except BaseException as error:
        atomic_json(OUT / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
                    automatic_retry=False, original_training_unmodified=True, optimization_admitted=False, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
