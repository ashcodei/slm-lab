"""Rebuild all operator/target evidence and freshly replay every proposed span.

All173references are executed again, using every original public command.
Every producer mutation is re-compared arithmetically; every positive proposed
mutation is freshly reexecuted. This is source-data verification, not a model
score, general semantic proof or permission to alter a live training corpus.
"""
from collections import Counter
from datetime import datetime, timezone
import ast
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/documented-doctest-verified-v1'
PRODUCER = ROOT/'reports/documented-operator-weights-v1'
OUT = ROOT/'reports/documented-operator-weights-readback-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    p = OUT/name; q = p.with_suffix(p.suffix+'.pending')
    with q.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(q, p)


def live(actor):
    try:
        fields = Path('/proc', str(actor['pid']), 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[0] not in ['Z', 'X'] and fields[19] == actor['birth']


def main():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 0:
        raise ValueError('A fresh explicitly CPU-only normal-priority independent runtime reader is required')
    from tokenizers import Tokenizer
    import numpy as np
    import torch
    from slm_lab.code_quality import instruction_code
    from slm_lab.documented_doctest_v1 import contract, compare, sha as text_sha
    from slm_lab.documented_operator_observation_v1 import observe
    from slm_lab.execution_operator_weights_v1 import catalogue, derivative
    from slm_lab.local_source_closure_v2 import python_sources
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    plan = json.loads((PRODUCER/'plan.json').read_text()); owner = plan['owner']; OUT.mkdir(); started = time.perf_counter()
    write('plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=dict(pid=os.getpid(),
        birth=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]), original_producer_owner=owner,
        original_functions=173, all_declared_mutations_recompared=True, all_positive_mutations_and_all_references_reexecuted=True,
        every_original_command_required=True, no_neural_model_or_optimizer=True, data_admitted_to_training=False))
    while not (PRODUCER/'complete.json').exists():
        if (PRODUCER/'failure.json').exists() or not live(owner):
            raise ValueError('The exact source producer ended before its complete declared population closed')
        write('progress.json', dict(phase='wait_exact_live_source_producer', exact_producer_live=True)); time.sleep(5)
    read = lambda path: json.loads(Path(path).read_text())
    complete = read(PRODUCER/'complete.json'); files = dict(complete['source_files_sha256'])
    for p in [PRODUCER/'plan.json', PRODUCER/'complete.json', PRODUCER/'observations.jsonl', *python_sources(ROOT, [Path(__file__).resolve()])]:
        files[str(p)] = sha(p)
    if (not complete['complete'] or complete['original_functions'] != 173 or complete['response_EOS_targets'] != 33517
            or sha(PRODUCER/'observations.jsonl') != complete['observations_sha256']):
        raise ValueError('The complete unmodified producer population is required')
    for name, digest in files.items():
        if sha(name) != digest:
            raise ValueError('An original source or execution/weighting implementation changed')
    with gzip.open(DATA/'original-rows.jsonl.gz', 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    index = np.load(DATA/'index.npy', allow_pickle=False, mmap_mode='r'); tokens = np.memmap(DATA/'input-ids.bin', dtype='<u2', mode='r')
    tokenizer = Tokenizer.from_file(str(DATA/'tokenizer.json')); counts = Counter(); selected_count = 0; targets = 0; references = 0; original_examples = 0
    def arithmetic(observed, tests, *, mutant, reference_passed=True):
        if observed['status'] == 'original_reference_population_qualification':
            if reference_passed or observed['eligible_value_discrimination']:
                raise ValueError('An unqualified reference cannot supply an operator weight')
            return False
        if not observed['all_original_examples_observed']:
            if observed['eligible_value_discrimination'] or observed['status'] not in ['runtime_or_resource_population_qualification', 'incomplete_raw_output_qualification']:
                raise ValueError('Incomplete runtime populations cannot certify a useful intervention')
            return False
        raw = observed['raw_results']; checks = [compare(t, r) for t, r in zip(tests, raw)]
        if len(raw) != len(tests):
            raise ValueError('Every whole original example must remain in the observation')
        calls = sum(r['entry_body_calls'] for r in raw)
        normal = [i for i, (t, r, ok) in enumerate(zip(tests, raw, checks)) if not ok and t.exc_msg is None and r['exception_only'] is None]
        failed_exceptions = [i for i, (r, ok) in enumerate(zip(raw, checks)) if not ok and r['exception_only'] is not None]
        unexpected = [i for i, (t, r) in enumerate(zip(tests, raw)) if t.exc_msg is None and r['exception_only'] is not None]
        passed = all(checks) and calls > 0; eligible = mutant and reference_passed and calls > 0 and bool(normal) and not failed_exceptions
        status = ('original_examples_do_not_exercise_entrypoint' if not calls else
                  ('not_distinguished_by_original_examples' if mutant else 'reference_all_original_examples_passed') if passed else
                  'operator_value_discriminated_by_original_examples' if eligible else 'completed_exception_or_context_difference_qualification')
        if (observed['example_passes'] != checks or observed['passed_examples'] != sum(checks)
                or observed['entry_body_calls'] != calls or observed['normal_output_mismatch_example_positions'] != normal
                or observed['unexpected_exception_example_positions'] != unexpected
                or observed['failed_exception_comparison_positions'] != failed_exceptions
                or observed['all_original_examples_passed'] != passed or observed['eligible_value_discrimination'] != eligible
                or observed['status'] != status):
            raise ValueError('Every recorded original-output comparison and qualification must reconstruct')
        return eligible
    with (PRODUCER/'observations.jsonl').open() as ledger, (OUT/'fresh-replay.jsonl').open('x') as replay:
        for position, row in enumerate(rows):
            path = PRODUCER/f'source-{position:03d}.json'; files[str(path)] = sha(path); source = read(path)
            parent = row['parent_source']; program = instruction_code(parent['response'])
            candidate = dict(source=parent, program=program, program_sha256=text_sha(program), function=ast.parse(program).body[0].name,
                doctest_text=parent['prompt'], doctest_text_sha256=text_sha(parent['prompt']), whole_original_example_count=row['whole_original_example_count'],
                existing_formatted_source_target_sha256=row['unchanged_retained_target_sha256'])
            tests = contract(candidate); inventory = catalogue(program)
            if row['split'] != 'train' or parent['original_partition'] != 'train' or source['declared_operator_inventory'] != inventory:
                raise ValueError('Every unchanged train-only source and full operator population must reconstruct')
            reference_event = json.loads(next(ledger))
            if reference_event['source_position'] != position or reference_event['original_reference'] is not True or reference_event['observation'] != source['reference_observation']:
                raise ValueError('The complete original reference disposition must occur once')
            recorded_reference = source['reference_observation']; arithmetic(recorded_reference, tests, mutant=False)
            fresh_reference = observe(candidate)
            replay.write(json.dumps(dict(source_position=position, original_reference=True, observation=fresh_reference), allow_nan=False)+'\n')
            references += int(fresh_reference['all_original_examples_passed'])
            original_examples += len(tests) if fresh_reference['all_original_examples_passed'] else 0
            selected = []; dispositions = Counter()
            for mutation in inventory['mutations']:
                event = json.loads(next(ledger)); observed = event['observation']
                if (event['source_position'] != position or event['original_reference'] is not False or event['mutation'] != mutation):
                    raise ValueError('Every original declared operator intervention must occur once in source order')
                eligible = arithmetic(observed, tests, mutant=True, reference_passed=recorded_reference['all_original_examples_passed'])
                counts[observed['status']] += 1; dispositions[observed['status']] += 1
                if 'executed_program_sha256' in observed and observed['executed_program_sha256'] != text_sha(derivative(program, mutation)):
                    raise ValueError('The exact single-operator derivative bytes must reconstruct')
                if eligible:
                    if not fresh_reference['all_original_examples_passed']:
                        raise ValueError('A newly failing reference cannot support a proposed operator weight')
                    fresh = observe(candidate, mutation)
                    replay.write(json.dumps(dict(source_position=position, mutation=mutation, observation=fresh), allow_nan=False)+'\n')
                    if (not fresh['eligible_value_discrimination'] or fresh['example_passes'] != observed['example_passes']
                            or fresh['normal_output_mismatch_example_positions'] != observed['normal_output_mismatch_example_positions']):
                        raise ValueError('Every positive proposed span must independently reproduce its complete example discrimination')
                    selected.append(mutation['mutation_id']); selected_count += 1
            proposed = source['proposed_complete_target_weights']; response = row['response']; encoded = tokenizer.encode(response, add_special_tokens=False)
            if response.count(program) != 1 or proposed['selected_mutation_ids'] != selected or source['dispositions'] != dict(dispositions):
                raise ValueError('Every proposed span must follow only the complete original runtime evidence')
            mask = bytearray(len(response)); shift = response.index(program); by_id = {m['mutation_id']:m for m in inventory['mutations']}
            for key in selected:
                m = by_id[key]; a, b = shift+m['program_char_start'], shift+m['program_char_end']; mask[a:b] = b'\1'*(b-a)
            weights = [1+3*sum(mask[a:b])/(b-a) if b > a else 1 for a, b in encoded.offsets]+[1.]
            ids = [1, *tokenizer.encode('User: '+row['prompt']+'\nAssistant:', add_special_tokens=False).ids, *encoded.ids, 2]
            entry = index[position]; packed = tokens[int(entry['offset']):int(entry['offset'])+int(entry['length'])].astype(np.int64).tolist()
            if (packed != ids or int(entry['role']) != 0 or packed[int(entry['target_start']):] != [*encoded.ids, 2]
                    or proposed['response_EOS_ids'] != [*encoded.ids, 2] or proposed['positive_target_weights'] != weights
                    or proposed['response_EOS_targets'] != row['target_tokens'] or proposed['weighted_denominator'] != math.fsum(weights)
                    or weights[-1] != 1 or min(weights) < 1):
                raise ValueError('Every full original input/target/EOS and independently rebuilt positive weight must match')
            targets += len(weights); replay.flush(); os.fsync(replay.fileno())
            write('progress.json', dict(complete_functions=position+1, planned_functions=173, passing_fresh_reference_functions=references,
                independently_reexecuted_positive_mutations=selected_count, seconds=time.perf_counter()-started))
        if ledger.read():
            raise ValueError('No original operator or source disposition may be silently omitted')
    if (len(rows) != 173 or targets != 33517 or selected_count != complete['value_discriminating_operator_proposals']
            or dict(counts) != complete['dispositions'] or sum(counts.values()) != plan['complete_declared_operator_mutations']):
        raise ValueError('The complete original population and all proposed targets must reconcile')
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError('A producer source or closed observation changed during independent readback')
    if torch.cuda.is_initialized():
        raise ValueError('No GPU context may be created')
    write('complete.json', dict(complete=True, original_functions=173, original_examples=715,
        passing_fresh_reference_functions=references, passing_fresh_reference_examples=original_examples,
        all_declared_mutations_recompared=sum(counts.values()), independent_positive_mutations_reexecuted=selected_count,
        all_original_inputs_targets_masks_EOS_and_weights_rebuilt=True, original_response_EOS_targets=targets,
        source_files_sha256=files, fresh_runtime_replay_sha256=sha(OUT/'fresh-replay.jsonl'),
        public_example_output_discrimination_not_general_semantics_or_model_quality=True,
        controlled_training_comparison_and_new_admission_still_required=True, no_model_or_optimizer_forward=True,
        data_admitted_to_training=False, fresh_human_feedback_labels=False, seconds=time.perf_counter()-started, goal_achieved=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if OUT.exists():
            write('failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), goal_achieved=False))
        raise
