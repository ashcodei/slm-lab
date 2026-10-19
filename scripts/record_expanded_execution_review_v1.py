"""Preserve full goal scope while recording expanded execution-source progress."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/expanded-execution-and-phase3-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('Fresh CPU-only full-scope review required')
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    old_path = ROOT / 'reports/completed173-and-phase3-review-v1/review.json'
    old = json.loads(old_path.read_text())
    admission_path = ROOT / 'data/apps-execution-rewards-v1/admission.json'
    admission = json.loads(admission_path.read_text())
    native_path = ROOT / 'reports/expanded-execution-corpus-v1/complete.json'
    native = json.loads(native_path.read_text())
    cpu_path = ROOT / 'reports/anchored-execution-CPU-v1/complete.json'
    cpu = json.loads(cpu_path.read_text())
    four_path = ROOT / 'reports/anchored-execution-four-CPU-v1/complete.json'
    four = json.loads(four_path.read_text())
    if (not admission['scoring_admitted_to_training'] or native['all_original_sources'] != 3721
            or not four['all_four_next_update_weights_Adam_actions_objectives_and_private_CPU_RNGs_exact']
            or not four['single_rank_probability_failure_rejected_before_any_gradient_reduce_or_commit']):
        raise ValueError('Actual admitted sources and four-replica mechanics evidence required')
    plan_path = ROOT / 'reports/phase-joint-v3/plan.json'
    plan = json.loads(plan_path.read_text())
    human_path = ROOT / 'reports/phase-human-preference-v4-queue.json'
    human = json.loads(human_path.read_text())
    def live(owner):
        p = Path('/proc') / str(owner['pid'])
        fields = (p / 'stat').read_text().split(') ', 1)[1].split()
        if fields[19] != owner['ticks'] or fields[0] in ['Z', 'X']:
            raise ValueError('Exact live PID/start ticks required')
        return dict(pid=owner['pid'], ticks=fields[19], state=fields[0],
            argv=(p / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
    owner, human_owner = live(plan['owner']), live(human['owner'])
    children = json.loads((ROOT / 'reports/phase-joint-v3/children.json').read_text())
    workers = [live(child) for child in children if child['subject'].startswith('train-rank-')]
    if len(workers) != 4: raise ValueError('Four actual live full-source workers required')
    raw = (ROOT / 'reports/phase-joint-v3/training-metrics.jsonl').read_bytes()
    rows = []
    for line in raw.splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: break
    if [r['step'] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('Contiguous committed production journal required')
    if (ROOT / 'reports/phase-joint-v3/failure.json').exists():
        raise ValueError('Inspect actual failure before recording healthy running state')
    OUT.mkdir(); at = datetime.now(timezone.utc).isoformat()
    targets = int(sum(r['response_EOS_targets'] for r in rows))
    atomic_json(OUT / 'production-journal-prefix.json', rows)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used', '--format=csv,noheader'],
                         text=True, capture_output=True, check=True).stdout
    runtime = dict(at=at, owner=owner, four_live_workers=workers, human_queue_owner=human_owner,
        actual_updates=len(rows), planned_updates=35284, actual_targets=targets, planned_targets=467581587,
        phase=rows[-1]['phase'], GPU_observation=gpu, full_source_training_complete=False,
        human_full_size_production_updates=0, goal_achieved=False)
    atomic_json(OUT / 'runtime.json', runtime)
    requirements = old['requirements']
    requirements['public_data']['evidence'] += (' New admitted machine-feedback source covers all 3,548 already verified public APPS programs, '
        '17,990 original cases, and 17,811 cases that distinguish the no-op across 3,539 programs. Nine programs remain supervised-only. '
        'The combined original native API retains 3,721 sources and 584,218 reference/EOS targets. These are newly prepared feedback views of existing public sources, not a fresh raw collection.')
    requirements['serious_full_model_training']['evidence'] = (f'Both earlier code arms finished 173 updates and their native evaluations. '
        f'The actual 984,399,690-parameter broad policy now has {len(rows):,}/35,284 committed updates and {targets:,}/467,581,587 targets '
        'on all four local L40s after successful full-size GPU capacity/recovery admission. The new always-anchored execution kernel has real '
        'small-network native-source updates and exact private CPU recovery on one and four replicas, plus a coordinated single-worker '
        'probability-failure control. Its NN has only 1,110,722 parameters and horizon2. Full new-profile GPU admission, a complete owned '
        'production controller/source epoch, and model-quality comparison remain outstanding for that kernel.')
    requirements['human_feedback']['evidence'] += (' All new APPS rewards and no-op controls are machine feedback. '
        'They add zero human preference labels and do not replace the queued complete public-human DPO/chosen-CE comparison.')
    requirements['gpu_runtime']['evidence'] = ('The exact phase-v3 owner and all four full-source workers are verified live now on the '
        'original four L40s. The exact human-v4 owner is still waiting. New expanded-code proofs ran only on CPU and have not acquired '
        'the GPUs or changed any active or queued source/plan. Fresh full native context and up-to-3,059-action GPU capacity/recovery '
        'must pass before future expanded-code production.')
    requirements['any_code_correctness_and_efficiency']['evidence'] += (' The new kernel always retains complete original-code supervision '
        'regardless of zero, positive or varying rewards; analytic unequal-mask/global-denominator tests pass. This is tested mechanics, '
        'not improved code generation. New post-training has no full-size production updates or held capability results yet.')
    new = dict(at=at, previous_goal_turn_classification='progress', current_goal_turn_classification='progress',
        current_turn_progress=['Independent replay admitted every new APPS no-op disposition.',
            'All 3,721 complete original native source contexts/targets/EOS and adaptive training budgets were read back.',
            'Always-on original-code CE and fresh group-policy updates implemented without editing previous/live sources.',
            'Actual small-network one/four-CPU next-update weights/Adam/actions/probabilities/private-RNG recovery passed.',
            'Actual single-rank cached-probability fault rejected across all replicas before gradient reduction or commit.',
            'Full broad-policy training continues on all four L40s.'],
        requirements=requirements, current_broad_runtime=runtime, native_development_scores_unchanged=old['native_code'],
        new_APPS_reward_admission=admission['counts'], expanded_native_corpus=native['counts'],
        new_feedback_full_size_production_updates=0, new_kernel_CPU_counts=cpu['actual_counts'],
        full_new_profile_GPU_capacity_and_production_driver_pending=True,
        new_held_capability_gain_proven=False, no_remaining_scope_removed=True,
        all_ten_final_requirements_incomplete=True, candidate_promoted=False, goal_achieved=False)
    files = [Path(__file__), old_path, admission_path, native_path, cpu_path, four_path, plan_path, human_path,
        OUT / 'production-journal-prefix.json', OUT / 'runtime.json',
        ROOT / 'src/slm_lab/anchored_execution_step_v1.py', ROOT / 'src/slm_lab/expanded_execution_corpus_v1.py',
        ROOT / 'reports/apps-execution-reward-v1-tests.log', ROOT / 'reports/anchored-code-objective-v1-tests.log']
    new['source_files_sha256'] = {str(p): file_hash(p) for p in files}
    atomic_json(OUT / 'review.json', new)
    audit_path = ROOT / 'reports/completion-audit.json'; audit = json.loads(audit_path.read_text())
    backup = ROOT / 'reports/completion-audit-before-expanded-execution-v1.json'
    with backup.open('xb') as f: f.write(audit_path.read_bytes())
    audit.update(updated_at=at, goal_achieved=False, previous_goal_turn_classification='progress', current_goal_turn_classification='progress',
        requirements={k: v['evidence'] for k, v in requirements.items()},
        latest_full_scope_review=str((OUT / 'review.json').relative_to(ROOT)),
        live_joint_training=runtime, expanded_execution_coding=dict(original_sources=3721,
            original_reference_EOS_targets=584218, APPS_programs=3548, informative_APPS_programs=3539,
            complete_original_APPS_cases=17990, informative_APPS_cases=17811,
            all_controls_and_source_masks_independently_read_back=True, full_size_production_updates=0,
            CPU_kernel_proofs_passed=True, new_GPU_admission_and_owned_production_driver_pending=True),
        all_ten_final_requirements_incomplete=True, candidate_promoted=False)
    atomic_json(audit_path, audit)
    paragraph = (f'Expanded execution feedback prepared, **{at}**: all **3,548 existing public APPS training programs** now have '
        '**independently replayed no-op controls**. Of 17,990 original cases, **17,811** distinguish the no-op across '
        '**3,539 programs**; nine programs remain supervised-only. The combined native source API retains all '
        '**3,721 documented/APPS sources** and **584,218 unchanged original reference/EOS targets**, including the earlier bounded documented-function probes. '
        'The new execution kernel keeps complete reference-code CE on every update, including positive-reward groups. Analytic denominator/gradient tests '
        'and actual small-network one/four-CPU update/recovery/failure controls pass. **These are CPU mechanics on a 1,110,722-parameter random network with horizon2, '
        'not full-size GPU admission or a capability gain.** Fresh full-context/up-to-3,059-action GPU capacity and a complete owned production driver remain required. '
        'No new full-size code training has begun, and no active or queued source/plan was changed.\n\n'
        f'The actual broader model continues on all four local L40s: this snapshot records **{len(rows):,}/35,284 updates** and '
        f'**{targets:,}/467,581,587 targets**. Human-v4 still waits for the complete foundation and native assessments. '
        'Latest completed coding correctness remains **1/24 instruction tasks for both CE and RL, 0/24 completion**, with no demonstrated RL advantage. '
        '**All ten final requirements remain incomplete; no candidate is promoted.** See the [current full review](reports/expanded-execution-and-phase3-v1/review.json), '
        '[expanded source integration](reports/expanded-execution-corpus-v1/complete.json), '
        '[four-CPU proof](reports/anchored-execution-four-CPU-v1/complete.json), and '
        '[live broad training](reports/phase-joint-v3/progress.json). Older entries below are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        path = ROOT / name; header, rest = path.read_text().split('\n\n', 1)
        path.write_text(header + '\n\n' + paragraph + rest)
    print(json.dumps(dict(review=str(OUT / 'review.json'), main_updates=len(rows), main_targets=targets,
        coding_sources=3721, new_full_size_code_updates=0, all_ten_requirements_incomplete=True)), flush=True)


if __name__ == '__main__':
    main()
