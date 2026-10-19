"""Compare all48actual full-size baseline code deliveries across observers.

Only closed code-worker outputs are bound. The recovery parent may continue
its separate broad suites. No model, GPU context or candidate execution.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from tokenizers import Tokenizer

from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.native_code_action_readback_v1 import validate

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'reports/public-pretraining-v1/initial'
NEW = ROOT / 'reports/public-pretraining-assessment-recovery-v1/initial'
OUT = ROOT / 'reports/native-observer-full-size-equivalence-CPU-v1'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    result = []
    with path.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                raise ValueError('Every closed action journal needs complete LF records')
            result.append(json.loads(line))
    return result


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh CUDA-hidden one-thread nice19 source-only comparison required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    children = read(NEW.parent / 'children.json')
    workers = [c for c in children if c.get('subject', '').startswith('assess-initial-code-')]
    if len(workers) != 4 or any(not terminal(c) for c in workers):
        raise ValueError('All four actual recovery baseline code workers must be terminal before source binding')
    old_children = read(ROOT / 'reports/public-pretraining-recovery-v1/children.json')
    if any(not terminal(c) for c in old_children):
        raise ValueError('All failed original runtime workers must be actually terminal')
    token_path = ROOT / 'checkpoints/public-pretraining-v1/initial/tokenizer.json'
    if sha(token_path) != '0072bb6de61da87bd9ba5f4e6739b594cc3e60c62c75a59d25d499629a70088f':
        raise ValueError('Original complete own tokenizer required')
    tokenizer = Tokenizer.from_file(str(token_path))
    files = {str(token_path): sha(token_path)}
    rows = []
    for rank in range(4):
        paths = [folder / ('code-actual-native-actions-' + str(rank) + '.jsonl') for folder in [OLD, NEW]]
        old_events, new_events = [lines(p) for p in paths]
        files.update({str(p): sha(p) for p in paths})
        if len(old_events) != 12 or len(new_events) != 12:
            raise ValueError('All original two-frame baseline events required')
        for number, frame in enumerate(['completion', 'instruction']):
            paths = [folder / ('code-' + frame + '-shard-' + str(rank) + '.raw.json') for folder in [OLD, NEW]]
            before, after = [read(p) for p in paths]
            files.update({str(p): sha(p) for p in paths})
            fields = ['checkpoint_sha256', 'generation_budget', 'decoding', 'prompt_mode', 'task_count',
                'suite_sha256', 'tokenizer_sha256', 'inference_device_type', 'inference_precision', 'evaluation_provenance']
            if (any(before[k] != after[k] for k in fields) or before['task_count'] != 6
                    or before['checkpoint_sha256'] != 'ca55969e121a5761f5074f98cb6164bf7b36fdac03f4696b4cd8e307443bc66a'):
                raise ValueError('Both actual full-size baseline profiles must match')
            for i, (a, b) in enumerate(zip(before['results'], after['results'])):
                position = number * 6 + i
                original, recovered = old_events[position], new_events[position]
                validate(a, frame, original, tokenizer, expected_case_position=position)
                validate(b, frame, recovered, tokenizer, expected_case_position=position)
                same_delivery = all(a[k] == b[k] for k in ['language', 'task', 'raw_response', 'code'])
                same_actions = all(original[k] == recovered[k] for k in ['original_input_token_ids',
                    'actual_generated_token_ids', 'complete_input_ids_sha256', 'requested_native_action_budget',
                    'actual_native_action_count', 'actual_stopped_on_EOS'])
                same_verdict = all(a['candidate'].get(k) == b['candidate'].get(k) for k in ['syntax_valid', 'functional_passed'])
                rows.append(dict(rank=rank, frame=frame, language=a['language'], task=a['task'],
                    whole_original_delivery_bytes_identical=same_delivery, all_actual_native_actions_identical=same_actions,
                    original_native_syntax_functional_verdicts_identical=same_verdict,
                    actual_generated_tokens=len(recovered['actual_generated_token_ids'])))
    if len(rows) != 48:
        raise ValueError('Every complete original baseline case is required')
    paths = [Path(__file__), ROOT / 'src/slm_lab/native_code_action_readback_v1.py',
        ROOT / 'src/slm_lab/native_greedy_observer_v1.py', ROOT / 'src/slm_lab/native_greedy_observer_v2.py',
        ROOT / 'reports/native-greedy-observer-v2-tests.log', ROOT / 'reports/native-code-action-readback-v1-tests.log']
    files.update({str(p): sha(p) for p in paths})
    OUT.mkdir()
    proof = dict(complete=True, passed=all(all(row[k] for k in ['whole_original_delivery_bytes_identical',
        'all_actual_native_actions_identical', 'original_native_syntax_functional_verdicts_identical']) for row in rows),
        at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        actual_full_size_baseline_cases=48, actual_dense_model_parameters=984399690, observations=rows,
        actual_code_workers_terminal_before_binding=True, changing_runtime_child_registry_not_source_bound=True,
        inference_only_observer_change_has_not_changed_any_actual_baseline_code_delivery=True,
        timing_measurements_not_asserted_identical=True, fresh_all96program_execution_still_pending=True,
        no_model_constructed=True, no_code_execution_or_generation=True, model_updates=0, goal_achieved=False)
    (OUT / 'complete.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(dict(report=str(OUT / 'complete.json'), passed=proof['passed'], actual_baseline_cases=48)))


if __name__ == '__main__':
    main()
