"""Read actual complete optimizer clocks to quantify visual update exposure.

No model, forward, optimizer, GPU context or new training. Read the final
immutable private snapshot with mmap, reconstruct all4447actual journal
groups, and compare every445leaf's saved Adam clock before drawing a limited
training-exposure diagnosis. Do not infer a causal quality result.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/public-pretraining-modality-clock-readback-CPU-v1'
PUBLIC = ROOT / 'reports/public-pretraining-v1'
RUNTIME = ROOT / 'reports/public-pretraining-recovery-v1'
STATE = ROOT / 'checkpoints/public-pretraining-v1/resume.pt'
EPOCH = ROOT / 'reports/public-pretraining-complete-epoch-value-readback-CPU-v2/complete.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def file_state(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def read(path):
    return json.loads(Path(path).read_text())


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh one-thread CUDA-hidden nice19 nonneural clock readback required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    torch.set_num_threads(1)
    runtime = read(RUNTIME / 'plan.json')
    if not terminal(runtime['owner']) or any(not terminal(c) for c in read(RUNTIME / 'children.json')):
        raise ValueError('Every exact original state-producing worker must already be terminal')
    epoch = read(EPOCH)
    plan = read(PUBLIC / 'plan.json')
    if not epoch['passed'] or epoch['actual_complete_updates'] != 4447:
        raise ValueError('The independently read complete original epoch must precede this diagnostic')
    before = file_state(STATE)
    payload = torch.load(STATE, weights_only=True, mmap=True, map_location='cpu')
    core = payload['core']
    if (payload['format'] != 'own-complete-public-pretraining-all-Adam-and-worker-RNG-v1'
            or payload['schedule_identity_sha256'] != plan['source_schedule_sha256']
            or core['completed'] != 4447 or core['world_size'] != 4 or len(core['rngs']) != 4
            or core['run_plan_sha256'] != sha(PUBLIC / 'plan.json')):
        raise ValueError('Exact actual complete source-bound private Adam snapshot required')
    journal = []
    with (RUNTIME / 'training-metrics.jsonl').open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                raise ValueError('Only complete closed LF metric records may be read')
            journal.append(json.loads(line))
    if len(journal) != 4447 or core['metrics'] != journal:
        raise ValueError('Every saved private metric must equal the independently read actual closed journal')
    clocks = Counter()
    families = Counter()
    learning = {g: [] for g in ['language', 'vision', 'word']}
    updates = {g: [] for g in learning}
    presentations = Counter()
    for i, row in enumerate(journal, 1):
        groups = row['actual_active_groups']
        expected = ['vision', 'word'] if row['family'] == 'words' else ['vision'] if row['family'].startswith(('images:', 'replay:image_reading')) else ['language']
        if row['step'] != i or groups != expected:
            raise ValueError('Every actual modality group must match its original complete source family')
        families[row['family']] += 1
        clocks.update(groups)
        for group in groups:
            rate = row['learning_rates'][group]
            if type(rate) not in [int, float] or not math.isfinite(rate) or rate <= 0:
                raise ValueError('Every actually used modality learning rate must be finite and positive')
            learning[group].append(rate)
            updates[group].append(i)
            presentations[group] += row['global_examples']
    if dict(clocks) != dict(language=4426, vision=21, word=5):
        raise ValueError('The complete actual exposure partition must reconstruct exactly')
    adam = core['optimizer']
    if [g['name'] for g in adam['param_groups']] != ['language', 'vision', 'word']:
        raise ValueError('Exact original optimizer partition required')
    observed = []
    groups = {}
    for group in adam['param_groups']:
        name = group['name']
        values = 0
        for key in group['params']:
            state = adam['state'][key]
            step = state['step']
            if (set(state) != {'step', 'exp_avg', 'exp_avg_sq'} or step.device.type != 'cpu'
                    or step.numel() != 1 or float(step) != clocks[name]
                    or state['exp_avg'].shape != state['exp_avg_sq'].shape):
                raise ValueError('Every actual saved Adam leaf must retain its true group-specific update clock')
            values += state['exp_avg'].numel()
            observed.append(dict(group=name, original_state_key=key, actual_Adam_step=int(step),
                moment_shape=list(state['exp_avg'].shape), optimized_parameter_values=state['exp_avg'].numel()))
        groups[name] = dict(actual_Adam_leaves=len(group['params']), actual_clock=clocks[name],
            optimized_parameter_values=values, actual_source_presentations=presentations[name],
            actual_global_steps=updates[name], used_learning_rate_min=min(learning[name]),
            used_learning_rate_max=max(learning[name]), sum_of_used_learning_rates=sum(learning[name]))
    if (len(observed) != 445 or set(adam['state']) != {r['original_state_key'] for r in observed}
            or [groups[n]['actual_Adam_leaves'] for n in ['language', 'vision', 'word']] != [317, 112, 16]):
        raise ValueError('All445actual optimized leaves and no invented optimizer entries required')
    del payload, core, adam
    digest = sha(STATE)
    if before != file_state(STATE) or torch.cuda.is_initialized():
        raise ValueError('Immutable private snapshot and no CUDA context required')
    paths = [STATE, EPOCH, PUBLIC / 'plan.json', RUNTIME / 'training-metrics.jsonl',
        ROOT / 'src/slm_lab/public_pretraining_update_v1.py', ROOT / 'src/slm_lab/public_pretraining_checkpoint_v1.py', Path(__file__)]
    files = {str(p): digest if p == STATE else sha(p) for p in paths}
    OUT.mkdir()
    rows = OUT / 'all445-actual-saved-Adam-clocks.jsonl'
    with rows.open('x') as stream:
        for row in observed:
            stream.write(json.dumps(row) + '\n')
        os.fsync(stream.fileno())
    proof = dict(complete=True, passed=True, at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=files, actual_complete_updates=4447, actual_groups=groups,
        actual_global_family_update_counts=dict(families), actual_Adam_leaves_checked=445,
        Adam_clock_observations_sha256=sha(rows),
        visual_update_exposure_is_measured_not_inferred_from_changed_values=True,
        visual_quality_regression_cause_not_proved_by_clock_counts=True,
        inference='Very sparse visual/word updates and continued text-decoder changes are a plausible contributor to the observed image regressions; a matched training comparison is still required.',
        future_full_visual_epoch_planned_updates=4733, future_whole_word_updates=1432,
        future_full_visual_epoch_has_not_been_trained=True, existing_neural_jobs_unchanged=True,
        no_model_constructed=True, neural_forwards=0, new_production_optimizer_updates=0, goal_achieved=False)
    (OUT / 'complete.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(dict(report=str(OUT / 'complete.json'), actual_clocks=dict(clocks), leaves=445, models_constructed=0)))


if __name__ == '__main__':
    main()
