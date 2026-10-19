"""Run the documented full native CLI with ancestor artifact reads denied.

One unrepaired128-action CPU FP32 instruction delivery checks the actual
standalone inference interface. It is not a coding benchmark or model gain.
The Python audit hook records/rejects protected artifact reads and network
connections; it is a reproducibility check, not an operating-system sandbox.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(source, output):
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.reserved_atomic_metadata_v1 import write_json
    source, output = Path(source).resolve(), Path(output).absolute()
    if (output.exists() or output.resolve() != output or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('A fresh CUDA-hidden one-thread low-priority CPU native interface check is required')
    plan, done = read(source / 'plan.json'), read(source / 'complete.json')
    if (not terminal(plan['owner']) or (source / 'failure.json').exists() or done.get('complete') is not True
            or done.get('actual_full_size_CPU_initializer') is not True
            or done.get('actual_counts', {}).get('total_unique') != 984399690
            or done.get('all_native_values_and_actual_text_photo_word_cold_outputs_exact') is not True):
        raise ValueError('The actual complete full native CPU export must precede direct CLI delivery')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0)); output.mkdir()
    files = {str(p): sha(p) for p in python_sources(ROOT, [Path(__file__)])}
    files[str(source / 'complete.json')] = sha(source / 'complete.json')
    def guard():
        if os.sched_getscheduler(0) != os.SCHED_IDLE or os.nice(0) != 19 or any(sha(p) != h for p, h in files.items()):
            raise ValueError('Every original complete source/runtime byte and idle CPU interface context must remain')
    def write(name, value):
        return write_json(output / name, value, root=output, minimum_free_bytes=40 * 1024**3,
            maximum_metadata_bytes=32 * 1024**2, guard=guard)
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    prompt = 'Write a Python function add(a, b) that returns the sum of a and b.'
    write('plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        exact_complete_CPU_initializer_owner=plan['owner'], checkpoint=done['checkpoint'], checkpoint_sha256=done['checkpoint_sha256'],
        complete_prompt=prompt, requested_native_actions=128, precision='fp32', actual_threads=1,
        scope='one direct documented native CLI delivery only; not an accuracy or broad benchmark',
        no_original_benchmark_population_or_generation_budget_changed=True, GPU_contexts=0, optimizer_updates=0, goal_achieved=False))
    write('expected-sources.json', done['sources'])
    protected = [ROOT / name for name in ['checkpoints', 'data', 'reports']]
    attempted, artifact_opens = [], []
    package_inputs = {Path(done['checkpoint']).resolve(), Path(done['checkpoint']).parent.resolve() / 'tokenizer.json'}
    def audit(event, args):
        if event == 'socket.connect':
            attempted.append(dict(event=event)); raise ValueError('Standalone native inference cannot contact a network model or service')
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)): return
        path = Path(os.fsdecode(args[0])).resolve()
        if any(path.is_relative_to(folder) for folder in protected):
            attempted.append(dict(event=event, path=str(path)))
            raise ValueError('Standalone native inference attempted a protected ancestor/data/report read: ' + str(path))
        if path in package_inputs: artifact_opens.append(str(path))
    try:
        sys.addaudithook(audit)
        from slm_lab.own_native_query_v1 import main as query
        result = query(['--checkpoint', done['checkpoint'], '--expected-sha256', done['checkpoint_sha256'],
            '--expected-tokenizer-sha256', done['tokenizer_sha256'], '--expected-sources', str(output / 'expected-sources.json'),
            '--prompt', prompt, '--max-new-tokens', '128', '--precision', 'fp32', '--threads', '1', '--json'])
        import torch
        if (attempted or result['actual_registered_parameters'] != 984399690 or result['external_model_or_teacher_used']
                or result['checkpoint_sha256'] != done['checkpoint_sha256'] or torch.cuda.is_initialized()
                or not result['source_map_verified_against_external_input']
                or not result['generated_text_returned_without_repair']
                or not set(artifact_opens) == {str(p) for p in package_inputs}):
            raise ValueError('The actual full native interface must use only its packaged weights/tokenizer and preserve every original action')
        write('complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(), actual_native_delivery=result,
            actual_allowed_package_file_opens=artifact_opens, actual_denied_ancestor_or_network_accesses=attempted,
            exact_source_receipt_sha256=sha(source / 'complete.json'), actual_original_human_updates=5362,
            new_joint_training_updates=0, inference_package_needs_no_training_ancestor=True,
            original_full_raw_and_tool_free_benchmarks_still_required=True, coding_quality_gain_not_established=True,
            GPU_contexts=0, optimizer_updates=0, goal_achieved=False))
    except BaseException as exc:
        write('failure.json', dict(owner=owner, error=repr(exc), traceback=traceback.format_exc(), attempted_protected_reads=attempted,
            original_outputs_and_all_native_actions_preserved=True, automatic_retry=False, goal_achieved=False)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/tmp/slm-completed-joint-initializer-CPU-v3'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/slm-standalone-native-delivery-CPU-v1'))
    args = parser.parse_args(); main(args.source, args.output)
