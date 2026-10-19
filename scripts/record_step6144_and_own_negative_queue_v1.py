"""Record verified6144 training/held-input evidence with the whole goal intact."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/own-negative-queued-and-step6144-v1-review'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden evidence review required')
    parent_path = ROOT / 'reports/own-negative-contrast-queued-v1-review/review.json'
    review = json.loads(parent_path.read_text())
    proof_path = ROOT / 'reports/phase-joint-step6144-readback-v1/complete.json'; proof = json.loads(proof_path.read_text())
    if (proof['actual_completed_updates'] != 6144 or proof['actual_trained_targets'] != 212107871
            or not proof['native_and_resume_all_model_tensors_bit_exact'] or not proof['held_positions_contexts_and_token_bytes_rederived']
            or proof['held_step6144_NLL'] != .782702675787732): raise ValueError('Actual complete6144 state/held-input proof required')
    for evidence in [review, proof]:
        for name, digest in evidence['source_files_sha256'].items():
            if file_hash(name) != digest: raise ValueError('Immutable review/checkpoint evidence changed')
    phase_path = ROOT / 'reports/phase-joint-v3/plan.json'; phase = json.loads(phase_path.read_text())
    progress = json.loads((ROOT / 'reports/phase-joint-v3/progress.json').read_text())
    queue_paths = [ROOT / 'reports' / n for n in ['phase-human-preference-v4-queue.json', 'broad-anchored-code-v3-queue.json',
        'full-held-QA-pipeline-v1-queue.json', 'own-execution-contrast-full-CPU-v1-queue.json']]
    queues = [json.loads(p.read_text()) for p in queue_paths]
    def identity(owner):
        path = Path('/proc') / str(owner['pid']) / 'stat'; fields = path.read_text().split(') ', 1)[1].split()
        if fields[19] != owner['ticks'] or fields[0] in ['Z', 'X']: raise ValueError('Exact live process required')
        return dict(pid=owner['pid'], ticks=fields[19], state=fields[0])
    live = dict(main_owner=identity(phase['owner']), actual_published_main_updates=progress['actual_updates'],
        planned_main_updates=35284, waiting_successors={str(p.relative_to(ROOT)): identity(q['owner']) for p, q in zip(queue_paths, queues)})
    requirements = review['requirements']
    requirements['serious_full_model_training']['evidence'] += (' The new actual6144 checkpoint independently reconciles every native/resume weight, '
        'all317 language Adam states/clocks, finite moments/nonnegative second moments,212,107,871 actual supervised targets, original source positions, '
        'valid private CPU RNG states and retained four-worker CUDA RNG byte structures. Inactive vision/word/other heads remain exactly unchanged. '
        'The large mutable resume was observed/fingerprinted and not duplicated; permanent native weights plus compact evidence are retained. '
        'This reader performs no fresh neural loss forward, CUDA RNG restore or next-update replay.')
    requirements['any_code_correctness_and_efficiency']['evidence'] += (' On the exact same128 source-held packed4096-token windows and524,160 targets, '
        'producer NLL is0.8699451996362768(initial),0.8051151143154129(step2048),0.7935042203171179(step4096),0.782702675787732(step6144): '
        '10.028507989356206% lower than initial, perplexity2.187376052188782. Input positions, token bytes and protocol were independently checked. '
        'This is token-prediction progress, not a new native algorithmic correctness/efficiency or all-domain quality score; no per-window paired uncertainty exists.')
    assert len(requirements) == 10 and all(r['status'] == 'incomplete_for_requested_final_model' for r in requirements.values())
    at = datetime.now(timezone.utc).isoformat(); OUT.mkdir()
    files = [Path(__file__), parent_path, proof_path, phase_path, *queue_paths]
    result = dict(at=at, requirements=requirements, actual_runtime=live,
        latest_actual_checkpoint_readback=str(proof_path.relative_to(ROOT)),
        previous_goal_turn_classification='progress', current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True, no_remaining_scope_removed=True,
        new_unassisted_held_coding_or_all_domain_gain_measured=False, candidate_promoted=False, goal_achieved=False,
        source_files_sha256={str(p): file_hash(p) for p in files})
    atomic_json(OUT / 'review.json', result)
    master_path = ROOT / 'reports/completion-audit.json'; docs = [ROOT / 'README.md', ROOT / 'RESEARCH.md', master_path]
    protected = set(phase['source_files_sha256'])
    for q in queues: protected.update(q['source_files_sha256'])
    cpu_plan = json.loads((ROOT / 'reports/conditional-prefix-full-CPU-v1/plan.json').read_text()); protected.update(cpu_plan['source_files_sha256'])
    if any(str(p) in protected for p in docs): raise ValueError('Root documents source-bound; review retained without changes')
    with (ROOT / 'reports/completion-audit-before-step6144-own-negative-v1.json').open('xb') as stream: stream.write(master_path.read_bytes())
    master = json.loads(master_path.read_text()); master.update(updated_at=at,
        latest_full_scope_review=str((OUT / 'review.json').relative_to(ROOT)), requirements={k: v['evidence'] for k, v in requirements.items()},
        latest_actual_checkpoint_held_loss_readback=str(proof_path.relative_to(ROOT)), latest_actual_main_runtime=live,
        all_ten_final_requirements_incomplete=True, candidate_promoted=False, goal_achieved=False)
    atomic_json(master_path, master)
    paragraph = (f'Verified6,144-update snapshot, **{at}**: actual full-size state, all317 language Adam clocks, '
        '**212,107,871 supervised targets**, unchanged inactive image/word/other heads, and all held input bytes reconcile. '
        'On the same128 held packed windows/524,160 targets, producer NLL is **0.782703**, down **10.03%** from0.869945 initially '
        '(step2,048:0.805115; step4,096:0.793504). This is token prediction, not a new coding or broad knowledge score. '
        'The reader did not repeat neural loss computation or next-update CUDA replay. The own-negative40-update CPU comparison remains queued; '
        'all ten final requirements remain incomplete. [Readback](reports/phase-joint-step6144-readback-v1/complete.json), '
        '[full-scope review](reports/own-negative-queued-and-step6144-v1-review/review.json).\n\n')
    for path in docs[:2]:
        first, tail = path.read_text().split('\n', 1); path.write_text(first + '\n\n' + paragraph + tail.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT / 'review.json'), actual6144_held_NLL=proof['held_step6144_NLL'],
        relative_NLL_reduction=proof['relative_held_NLL_reduction'], goal_achieved=False)), flush=True)


if __name__ == '__main__': main()
