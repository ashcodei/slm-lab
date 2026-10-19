"""CPU-only completion of the original endpoint with its original domain rows.

The original v1 reader omitted the required knowledge-question argument. Its
source/failure remain unchanged. This process-local adapter supplies those
bound original questions and keeps every original outcome check in force.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

import assess_joint_java_comparison_v1 as actors
from slm_lab import own_code_training_readback_v1 as reader
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig
from slm_lab.own_code_training_stage_v1 import PreparedCodeStream

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/own-code-generation-repair-v1'
OUTPUT = ROOT / 'checkpoints/own-code-generation-repair-v1'
AUDIT = ROOT / 'reports/own-code-generation-repair-endpoint-v2'
QUESTIONS = ROOT / 'data/knowledge-eval-v1/questions.jsonl'
FOLDERS = dict(generation=ROOT / 'data/APPS-verified-generation-v1', repair=ROOT / 'data/APPS-execution-repairs-v1',
               variety=ROOT / 'data/codenet-durable-pack-v2')


def bound_summary(original_summary, questions):
    if not questions or len({row['id'] for row in questions}) != len(questions):
        raise ValueError('Complete unique original domain-question inventory required')
    def summarize(report, name):
        return original_summary(report, name, questions if name == 'knowledge.json' else None)
    return summarize


def main():
    if AUDIT.exists() or (REPORTS / 'independent-comparison.json').exists():
        raise ValueError('Preserve previous endpoint output; no implicit overwrite')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU endpoint audit must keep GPU visibility disabled')
    torch.set_num_threads(2)
    plan = json.loads((REPORTS / 'plan.json').read_text())
    if actors.same(plan['owner']) or any(actors.same(a) for a in json.loads((REPORTS / 'children.json').read_text())):
        raise ValueError('Every original code owner/registered actor must actually be terminal')
    if json.loads((REPORTS / 'failure.json').read_text())['error'] != 'Every actual original domain question required':
        raise ValueError('This reviewer addresses only the observed missing original knowledge rows')
    files = dict(plan['source_files_sha256'])
    for path in [Path(__file__).resolve(), ROOT / 'tests/test_own_code_endpoint_v2.py',
            ROOT / 'reports/own-code-endpoint-v2-tests.log', REPORTS / 'plan.json', REPORTS / 'failure.json',
            REPORTS / 'restored.json', REPORTS / 'training-complete.json', OUTPUT / 'model.pt', OUTPUT / 'resume.pt']:
        files[str(path)] = file_hash(path)
    for phase in ['baseline', 'candidate']:
        for path in sorted((REPORTS / phase).glob('*.json')):
            files[str(path)] = file_hash(path)
    if '1 passed' not in (ROOT / 'reports/own-code-endpoint-v2-tests.log').read_text():
        raise ValueError('Actual original-domain recount and corrupt/missing-question rejection check required')
    guard = FileBindings(files); provenance = FileBindings(plan['provenance_files_sha256'])
    questions = list(map(json.loads, QUESTIONS.read_text().splitlines()))
    if len(questions) != 559:
        raise ValueError('All original559domain questions required')
    AUDIT.mkdir()
    atomic_json(AUDIT / 'plan.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=files, original_run_plan_sha256=file_hash(REPORTS / 'plan.json'),
        original_reader_source_unchanged=True, change='Supply bound original question rows to the existing native domain summary',
        original_training_and_all_GPU_assessments_actually_complete=True, model_or_optimizer_updates=0, CUDA_initialized=False,
        original_failure_preserved=True, goal_achieved=False))
    snapshot = torch.load(OUTPUT / 'resume.pt', map_location='cpu', weights_only=True, mmap=True)
    for rng in snapshot['worker_rng_states']:
        random.Random().setstate(rng['python']); torch.Generator(device='cpu').set_state(rng['torch'])
        n = rng['numpy']
        np.random.RandomState(0).set_state((n['kind'], np.asarray(n['keys'], dtype=np.uint32), n['position'], n['has_gauss'], n['cached_gaussian']))
        if rng['cuda'].dtype != torch.uint8 or rng['cuda'].ndim != 1 or not rng['cuda'].numel():
            raise ValueError('Every actual saved worker CUDA RNG byte payload must be retained')
    config = ModelConfig(**snapshot['config'])
    token = Path(plan['base_checkpoint']).parent / 'tokenizer.json'
    streams = {name: PreparedCodeStream(name, folder, token, config) for name, folder in FOLDERS.items()}
    original = reader.summarize
    reader.summarize = bound_summary(original, questions)
    try:
        result = reader.reconcile(ROOT, REPORTS, OUTPUT, streams)
    finally:
        reader.summarize = original
    guard.check(full=True); provenance.check(full=True)
    if torch.cuda.is_initialized():
        raise ValueError('Endpoint reconciliation must never initialize CUDA')
    atomic_json(AUDIT / 'complete.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        original_run_plan_sha256=file_hash(REPORTS / 'plan.json'), endpoint_sha256=file_hash(REPORTS / 'independent-comparison.json'),
        exact_original_reader_and_training_runtime_sources_still_match_run_plan=True,
        all_four_CPU_numpy_python_RNG_schemas_checked_and_CUDA_bytes_retained=True,
        complete_original_domain_questions_supplied_and_all_original_summary_checks_preserved=True,
        native_comparisons=result['native_comparisons'], public_subset=result['public_subset'],
        original_failure_preserved=True, model_or_optimizer_updates=0, CUDA_initialized=False,
        candidate_promoted=False, goal_achieved=False))
    print(json.dumps(dict(endpoint=str(REPORTS / 'independent-comparison.json'),
                         actual_updates=result['actual_steps'], public_subset=result['public_subset'],
                         model_updates=0, CUDA_initialized=False), indent=2), flush=True)


if __name__ == '__main__':
    main()
