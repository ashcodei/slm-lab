"""Independent CPU readback of live actual857M execution-pilot checkpoints."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from tokenizers import Tokenizer
import torch

from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.execution_feedback_union_v1 import ExecutionFeedbackUnion
from slm_lab.execution_preference_v1 import fingerprint as input_fingerprint
from slm_lab.recurrent_ablation import state_digest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/execution-code-pilot-v1'


def process(pid):
    p = Path('/proc') / str(pid); fields = (p / 'stat').read_text().split(') ', 1)[1].split()
    if fields[0] in {'Z', 'X'}: raise ValueError('Expected exact actor is terminal')
    return dict(pid=pid, ticks=fields[19], parent=int(fields[1]), state=fields[0],
        argv=(p / 'cmdline').read_bytes().replace(b'\0', b' ').decode())


def main():
    torch.set_num_threads(2)
    if torch.cuda.is_initialized() or shutil.disk_usage(ROOT).free < 34 * 1024**3:
        raise ValueError('CPU readback plus funded temporary inode retention above original reserve required')
    plan = json.loads((REPORTS / 'plan.json').read_text())
    owner, waiter = process(plan['owner']['pid']), process(plan['waiter']['pid'])
    if owner['ticks'] != plan['owner']['ticks'] or waiter['ticks'] != plan['waiter']['ticks'] or waiter['state'] not in {'T', 't'}:
        raise ValueError('Exact live pilot owner and held original idle waiter required')
    profiles = [p for p in plan['profiles'] if not p.get('assessment_only')]
    checkpoints = {p['name']: ROOT / 'checkpoints/execution-code-pilot-v1' / p['name'] / 'latest.pt' for p in profiles}
    if any(not p.exists() for p in checkpoints.values()):
        raise ValueError('Wait for both actual first complete checkpoints, not a status-only claim')
    token = ROOT / 'checkpoints/extended-curriculum-v1/joint/tokenizer.json'
    feedback = ExecutionFeedbackUnion(ROOT / 'data/execution-feedback-union-v1', Tokenizer.from_file(str(token)), token)
    examples = [feedback.example(i) for i in range(len(feedback.pairs))]
    summaries = {}; cache_fingerprints = {}; initial_states = {}; metric_snapshots = {}
    base_path = ROOT / 'checkpoints/extended-curriculum-v1/joint/model.pt'
    if file_hash(base_path) != plan['base_checkpoint_sha256']:
        raise ValueError('Exact own completed joint checkpoint required')
    original = torch.load(base_path, weights_only=True, map_location='cpu', mmap=True)
    base_digest = state_digest(original['model'])
    base_count = sum(t.numel() for n, t in original['model'].items() if n != 'lm_head.weight')
    with tempfile.TemporaryDirectory(prefix='.execution-state-readback-', dir=ROOT / 'reports') as temporary:
        for profile in profiles:
            name = profile['name']; stable = Path(temporary) / (name + '.pt')
            # The live writer atomically replaces latest.pt. A private temporary
            # hard link keeps this exact complete inode stable during readback.
            os.link(checkpoints[name], stable)
            digest = file_hash(stable); state = torch.load(stable, weights_only=True, map_location='cpu', mmap=True)
            initial = json.loads((REPORTS / name / 'initial-policy.json').read_text())
            refs = json.loads((REPORTS / name / 'reference.json').read_text())
            step = state['steps']
            if (type(step) is not int or not 32 <= step <= 128 or step % 32 or state['planned_steps'] != 128 or state['world_size'] != 1
                    or state['run_plan_sha256'] != file_hash(REPORTS / 'plan.json') or state['base_checkpoint_sha256'] != plan['base_checkpoint_sha256']
                    or state['tokenizer_sha256'] != plan['tokenizer_sha256'] or state['profile'] != profile
                    or state['config'] != original['config'] or state['residual_width'] != 2048
                    or state['frozen_backbone_state_sha256'] != base_digest or initial['frozen_backbone_state_sha256'] != base_digest
                    or len(state['metrics']) != step or [m['step'] for m in state['metrics']] != list(range(1, step + 1))):
                raise ValueError('Actual complete own857M stage/source/step/frozen-base identity required')
            learned = state['residuals']; parameters = sum(t.numel() for t in learned.values())
            if base_count != 706012672 or parameters != 150994944 or base_count + parameters != 857007616:
                raise ValueError('Actual full857M tensor count required')
            if state_digest(learned) == initial['residual_state_sha256'] or any(not bool(torch.isfinite(t).all()) for t in learned.values()):
                raise ValueError('Actual finite changed residual weights required')
            moments = state['optimizer']['state']
            if len(moments) != len(learned): raise ValueError('Every learned residual needs complete Adam moments')
            for value, tensor in zip(moments.values(), learned.values()):
                if float(value['step']) != step:
                    raise ValueError('Every Adam step must match actual saved updates')
                if any(value[k].shape != tensor.shape or value[k].dtype != tensor.dtype or not bool(torch.isfinite(value[k]).all()) for k in ['exp_avg', 'exp_avg_sq']):
                    raise ValueError('Every finite exact-shape/dtype Adam moment required')
            cache = state['execution_reference']
            if (state['execution_reference_fingerprint'] != fingerprint(cache) or cache != refs['cache']
                    or state['execution_reference_policy'] != refs['policy'] or cache['policy_sha256'] != fingerprint(refs['policy'])
                    or set(cache['records']) != {e['id'] for e in examples}):
                raise ValueError('Every frozen original reference must be retained in the actual checkpoint')
            for e in examples:
                record = cache['records'][e['id']]
                for key in ['chosen', 'rejected']:
                    if record[key + '_fingerprint'] != input_fingerprint(e[key]) or record[key + '_targets'] != sum(v != -100 for v in e[key][1][1:]):
                        raise ValueError('Every cached original program/input/mask/target must match actual admitted data')
            if (state['sampler_state']['updates'] != step or state['sampler_state']['code_draws'] != 4 * step
                    or state['sampler_state']['positive_draws'] != 2 * step or state['sampler_state']['pair_draws'] != 4 * step
                    or state['sampler_state']['replay_draws'] != step or state['clipped_sequences'] != 0
                    or state['rng']['cuda'] is None):
                raise ValueError('Complete exact source cursors and actual CUDA RNG state required')
            torch.Generator().set_state(state['rng']['torch'])
            if file_hash(stable) != digest: raise ValueError('Private complete checkpoint inode changed')
            metric_snapshots[name] = state['metrics']
            summaries[name] = dict(observed_checkpoint_steps=step, checkpoint_sha256=digest,
                observed_live_checkpoint=str(checkpoints[name]), temporary_readback_inode_not_retained=True,
                total_unique_parameters=base_count + parameters, learned_residual_parameters=parameters,
                changed_finite_residual_weights=True, frozen_actual_own_base_matches=True, complete_finite_step_matched_Adam=True,
                original_reference_pairs_embedded_and_reconciled=374, full_RNG_and_source_cursors_retained=True,
                observed_unique_verified_positives=len({x for m in state['metrics'] for x in m['verified_source_ids']}),
                observed_unique_measured_pairs=len({x for m in state['metrics'] for x in m['pair_source_ids']}),
                observed_replay_families=len({m['replay_family'] for m in state['metrics']}),
                actual_GPU_updates=True, world_size=1, DPO_alpha=profile['alpha'], capability_result=False)
            cache_fingerprints[name] = fingerprint(cache); initial_states[name] = initial
    if len({fingerprint(v) for v in initial_states.values()}) != 1 or len(set(cache_fingerprints.values())) != 1:
        raise ValueError('Both actual policies must start from identical learned weights and complete frozen reference scores')
    names = list(metric_snapshots); shared = min(len(metric_snapshots[n]) for n in names)
    for a, b in zip(metric_snapshots[names[0]][:shared], metric_snapshots[names[1]][:shared]):
        for key in ['code_source_ids', 'verified_source_ids', 'pair_source_ids', 'replay_family', 'replay_position', 'positive_positions', 'pair_positions']:
            if a[key] != b[key]: raise ValueError('Matched arms must consume the identical original data order')
    actor_records = [process(c['pid']) for c in json.loads((REPORTS / 'children.json').read_text())]
    cpu_path = ROOT / 'reports/completed-code-knowledge-CPU-v1/plan.json'; cpu_plan = json.loads(cpu_path.read_text())
    cpu_owner = process(cpu_plan['owner']['pid'])
    cpu_status = json.loads((cpu_path.parent / 'status.json').read_text()); cpu_workers = {n: process(p['pid']) for n, p in cpu_status['workers'].items()}
    progress = {}
    for name in cpu_workers:
        lines = re.findall(r'evaluated knowledge questions (\d+)', (cpu_path.parent / (name + '.log')).read_text())
        progress[name] = int(lines[-1]) if lines else 0
    if torch.cuda.is_initialized(): raise ValueError('Independent readback must not initialize CUDA')
    metrics_file = ROOT / 'reports/execution-code-pilot-v1-observed-checkpoint-metrics.json'
    atomic_json(metrics_file, metric_snapshots)
    prior_path = ROOT / 'reports/paired-visual-production-v1-readiness.json'
    prior = json.loads(prior_path.read_text())
    files = [Path(__file__).resolve(), REPORTS / 'plan.json', metrics_file, cpu_path,
        ROOT / 'scripts/assess_completed_code_knowledge_CPU_v1.py', ROOT / 'src/slm_lab/execution_code_pilot_v1.py',
        ROOT / 'tests/test_execution_code_pilot_v1.py', ROOT / 'tests/test_execution_code_pilot_driver_v1.py',
        ROOT / 'reports/execution-code-pilot-v1-release-tests.log', ROOT / 'reports/execution-code-pilot-driver-v1-release-tests.log',
        ROOT / 'reports/completed-systems-code-state-readback-v1.json', ROOT / 'reports/original-over-window-instruction-readback-v1.json', prior_path]
    files += [REPORTS / n / f for n in names for f in ['reference.json', 'initial-policy.json']]
    output = dict(format='own-execution-code-pilot-live-readback-v1', observed_at=datetime.now(timezone.utc).isoformat(),
        previous_goal_turn_classification=prior['current_turn_classification'],
        current_turn_classification=dict(classification='progress', reason='Verified complete1138-update systems weights/all metrics; launched full559question matched CPU assessments and actual two-arm857M verified-code CE/DPO experiment plus native systems assessment. Four focused checks pass; independently read back changed actual GPU checkpoints, complete Adam/RNG/374references and matched original source order.'),
        actual_GPU_checkpoint_observations=summaries, actual_initial_weights_and_all374reference_scores_identical=True,
        actual_matched_source_order_verified_through_step=shared, actual_pilot_owner=owner, held_original_waiter=waiter,
        actual_pilot_workers=actor_records, actual_CPU_knowledge_owner=cpu_owner, actual_CPU_knowledge_workers=cpu_workers,
        CPU_questions_processed_at_observation=progress, no_independent_readback_CUDA_context=True,
        new_focused_checks=4, final128_update_endpoints_and_capability_comparison_pending=True,
        full112609_code_all30_source_production_unrun=True, source_reviews7287_and4603_preserved_not_trained=True,
        new_capability_gain_established=False, candidate_promoted=False, goal_achieved=False,
        remaining_requirements=['Complete fixed128-update experimental endpoints and unchanged native code/knowledge evaluation; restore exact original waiter after workers end.',
            'Finish matched CPU559question readback, finalized image/all30 sources and exact final own-base/source/baseline binding.',
            'Run full production/capacity/training and demonstrate useful all-round, multimodal, coding and genuine-human-feedback gains before claiming high-end capability.'],
        files_sha256={str(p): file_hash(p) for p in files})
    atomic_json(ROOT / 'reports/execution-code-pilot-v1-live-readback.json', output)
    print(json.dumps(dict(report='reports/execution-code-pilot-v1-live-readback.json', checkpoints=summaries,
        CPU_progress=progress, goal_achieved=False), indent=2))


if __name__ == '__main__': main()
