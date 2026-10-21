"""Test every declared operator intervention in all173verified train functions.

All original public example commands remain intact. Only complete normal-value
discrimination can propose a higher source-operator weight; runtime/context and
exception differences stay qualifications. Nothing enters live training.
"""
from collections import Counter
from datetime import datetime, timezone
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/documented-doctest-verified-v1'
OUT = ROOT/'reports/documented-operator-weights-v1'
MAXIMUM = 256*1024**2


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def guard():
    if (sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) > MAXIMUM
            or shutil.disk_usage(ROOT).free < 8*1024**3+MAXIMUM):
        raise ValueError('Keep the256MiB report budget and shared8GiB persistent writer reserve')


def write(name, value):
    guard(); p = OUT/name; q = p.with_suffix(p.suffix+'.pending')
    with q.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(q, p)


def main():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 0:
        raise ValueError('Fresh explicit CPU-only normal-priority complete-example execution is required')
    from tokenizers import Tokenizer
    from slm_lab.code_quality import instruction_code
    from slm_lab.documented_doctest_v1 import contract, sha as text_sha
    from slm_lab.documented_operator_observation_v1 import observe
    from slm_lab.execution_operator_weights_v1 import catalogue, candidate_target_weights
    from slm_lab.local_source_closure_v2 import python_sources
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise ValueError('This source-only study must never initialize CUDA')
    tests = ROOT/'reports/execution-operator-weights-v1-final-tests.log'
    if '5 passed' not in tests.read_text():
        raise ValueError('Exact-source/runtime/isolation/Unicode/gradient controls must pass')
    admission = json.loads((DATA/'admission.json').read_text()); manifest = json.loads((DATA/'manifest.json').read_text())
    reader_path = ROOT/'reports/documented-doctest-verified-v1/independent-complete-source-and-runtime-readback-v1.json'
    reader = json.loads(reader_path.read_text())
    if (not admission['admitted_to_training'] or admission['dataset_manifest_sha256'] != sha(DATA/'manifest.json')
            or not reader['all_source_rows_and_native_tokens_read_back'] or not reader['all_reexecuted_examples_passed']
            or reader['training_functions'] != 173 or reader['whole_published_examples'] != 715
            or reader['admitted_pack_sha256'] != sha(DATA/'admission.json')):
        raise ValueError('The entire source-admitted173-function/715-example corpus and independent replay are required')
    files = {str(p):sha(p) for p in [tests, reader_path, DATA/'admission.json', DATA/'manifest.json',
        DATA/'publisher-license-notices.jsonl', *python_sources(ROOT, [Path(__file__).resolve(), ROOT/'tests/test_execution_operator_weights_v1.py'])]}
    for name, digest in manifest['files_sha256'].items():
        if sha(DATA/name) != digest:
            raise ValueError('An original sealed source byte changed')
        files[str(DATA/name)] = digest
    with gzip.open(DATA/'original-rows.jsonl.gz', 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != 173 or sum(r['whole_original_example_count'] for r in rows) != 715:
        raise ValueError('Every original function and complete published example population is required')
    tokenizer = Tokenizer.from_file(str(DATA/'tokenizer.json'))
    if tokenizer.get_vocab_size() != 32768 or tokenizer.truncation is not None or tokenizer.padding is not None:
        raise ValueError('One exact native tokenizer without implicit cropping/padding is required')
    population = []
    for position, row in enumerate(rows):
        source = row['parent_source']; program = instruction_code(source['response'])
        if (row['split'] != 'train' or row['publisher_split'] != 'train' or row['token_pack_index'] != position
                or program == '' or text_sha(program) != row['executable_program_sha256']
                or row['response'] != source['response']):
            raise ValueError('Every whole original publisher-train target must reconstruct')
        candidate = dict(source=source, program=program, program_sha256=text_sha(program),
            function=ast.parse(program).body[0].name, doctest_text=source['prompt'], doctest_text_sha256=text_sha(source['prompt']),
            existing_formatted_source_target_sha256=row['unchanged_retained_target_sha256'],
            whole_original_example_count=row['whole_original_example_count'])
        contract(candidate); population.append((row, candidate, catalogue(program)))
    OUT.mkdir(); started = time.perf_counter(); counts = Counter(); examples = 0; sources_passed = 0; proposals = 0; total_targets = 0
    write('plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=dict(pid=os.getpid(),
        birth=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]), source_files_sha256=files,
        original_functions=173, original_examples=715, complete_declared_operator_mutations=sum(len(x[2]['mutations']) for x in population),
        complete_lexical_qualifications=sum(len(x[2]['lexical_qualifications']) for x in population),
        fixed_alternative_per_supported_operator=True, per_function_operator_prefix_sampling=False,
        full_original_example_sequence_per_program=True, wall_seconds_per_program=5, OS_confined_memory_bytes=512*1024**2,
        workers=1, CPU_affinity=sorted(os.sched_getaffinity(0)), normal_priority_reference_execution=True,
        source_roles_and_original_targets_unchanged=True, protected_benchmark_references_not_used=True,
        only_normal_value_discrimination_without_failed_exception_comparisons_can_propose_weight=True,
        weight=4., all_other_targets_and_genuine_EOS_weight=1., maximum_report_bytes=MAXIMUM,
        source_augmentation_not_model_rollouts_or_human_feedback=True, data_admitted_to_training=False, goal_achieved=False))
    with (OUT/'observations.jsonl').open('x') as journal:
        for position, (row, candidate, inventory) in enumerate(population):
            original = observe(candidate); selected = []; dispositions = Counter()
            journal.write(json.dumps(dict(source_position=position, original_reference=True, observation=original), allow_nan=False)+'\n')
            if original['all_original_examples_passed']:
                sources_passed += 1; examples += original['original_example_count']
            for mutation in inventory['mutations']:
                if not original['all_original_examples_passed']:
                    result = dict(status='original_reference_population_qualification', eligible_value_discrimination=False,
                                  executed=False, mutation_id=mutation['mutation_id'])
                else:
                    result = observe(candidate, mutation)
                dispositions[result['status']] += 1; counts[result['status']] += 1
                if result['eligible_value_discrimination']:
                    selected.append(mutation['mutation_id'])
                raw = json.dumps(dict(source_position=position, original_reference=False, mutation=mutation, observation=result), allow_nan=False)+'\n'
                if journal.tell()+len(raw.encode()) > MAXIMUM:
                    raise ValueError('The complete raw operator observation population exceeds its report budget')
                journal.write(raw); journal.flush(); guard()
            proposed = candidate_target_weights(row['response'], candidate['program'], inventory['mutations'], selected, tokenizer)
            if proposed['response_EOS_targets'] != row['target_tokens']:
                raise ValueError('Every original complete response/EOS target must remain present')
            total_targets += proposed['response_EOS_targets']; proposals += len(selected)
            journal.flush(); os.fsync(journal.fileno())
            write(f'source-{position:03d}.json', dict(source_position=position, original_id=row['id'], original_source_id=candidate['source']['id'],
                original_program_sha256=candidate['program_sha256'], whole_original_example_count=candidate['whole_original_example_count'],
                reference_observation=original, declared_operator_inventory=inventory, dispositions=dict(dispositions),
                proposed_complete_target_weights=proposed, unchanged_target_sha256=row['unchanged_retained_target_sha256'],
                original_source_role='train', future_independent_runtime_and_native_weight_readback_required=True,
                data_admitted_to_training=False, new_model_updates=0))
            write('progress.json', dict(complete_functions=position+1, planned_functions=173,
                passing_reference_functions=sources_passed, passing_original_reference_examples=examples,
                value_discriminating_operator_proposals=proposals, dispositions=dict(counts),
                elapsed_seconds=time.perf_counter()-started, data_admitted_to_training=False))
    if total_targets != 33517:
        raise ValueError('Every original source target and genuine EOS must survive across all173functions')
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError('An original source or execution/weighting implementation changed')
    if torch.cuda.is_initialized():
        raise ValueError('No CUDA context may be created')
    write('complete.json', dict(complete=True, original_functions=173, original_examples=715, passing_references=sources_passed,
        passing_original_reference_examples=examples, response_EOS_targets=total_targets,
        declared_operator_mutations=sum(len(x[2]['mutations']) for x in population), dispositions=dict(counts),
        value_discriminating_operator_proposals=proposals, observations_sha256=sha(OUT/'observations.jsonl'),
        source_files_sha256=files, seconds=time.perf_counter()-started, no_model_forward_or_optimization=True,
        original_existing_pack_and_live_jobs_unchanged=True, independent_readback_and_controlled_training_still_required=True,
        data_admitted_to_training=False, human_preference_labels=False, model_quality_gain_proved=False, goal_achieved=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if OUT.exists():
            write('failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), goal_achieved=False))
        raise
