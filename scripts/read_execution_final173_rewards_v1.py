"""New diagnostic rewards on a sealed prefix; applied old rewards stay intact."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/execution-final173-reward-replay-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU reward replay must hide all GPUs')
    if REPORTS.exists():
        raise ValueError('Preserve all earlier diagnostics')
    from slm_lab.bounded_code_reward_catalog_v1 import BoundedPublicCodeReward
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.execution_group_controlled_v2 import advantages
    from tokenizers import Tokenizer
    catalog = BoundedPublicCodeReward(ROOT)
    tokenizer = Tokenizer.from_file(str(ROOT / 'data/documented-doctest-verified-v1/tokenizer.json'))
    run = ROOT / 'reports/execution-single-update-recovery-v3'
    sealed = {}
    jobs = []
    for arm in ('verified_solution', 'execution_hybrid'):
        raw = (run / (arm+'-metrics.jsonl')).read_text()
        rows = list(map(json.loads, raw[:raw.rfind('\n')+1].splitlines()))[:173]
        if len(rows) != 173:
            raise ValueError('Both current arms must have completed all 173 updates')
        sealed[arm] = rows
        for i, row in enumerate(rows):
            rewards = [t['actual_execution_reward']['reward'] for t in row['trajectories']]
            if (row['step'] != i+1 or len(rewards) != 4 or row['global_rewards'] != rewards
                    or row['global_advantages'] != advantages(rewards)):
                raise ValueError('Every original applied source/reward group must reconstruct')
            for j, trajectory in enumerate(row['trajectories']):
                if tokenizer.decode(trajectory['response_ids']) != trajectory['own_response']:
                    raise ValueError('Only the actual own generated response may be replayed')
                jobs.append((arm, i+1, j, row['source_id'], trajectory))
    if any(a['source_id'] != b['source_id'] for a, b in zip(*sealed.values())):
        raise ValueError('Matched original source order required')
    REPORTS.mkdir()
    for arm, rows in sealed.items():
        with (REPORTS / (arm+'-sealed-prefix.jsonl')).open('x') as f:
            for row in rows:
                f.write(json.dumps(row)+'\n')
    def replay(job):
        arm, step, sample, key, trajectory = job
        previous = trajectory['actual_execution_reward']
        old = catalog.original.score(key, trajectory['own_response'])
        if old['reward'] != previous['reward']:
            raise ValueError('The actual applied original public reward must reproduce')
        new = catalog.score(key, trajectory['own_response'])
        return dict(arm=arm, step=step, global_sample=sample, source_id=key,
                    applied_original_reward=previous['reward'], replayed_original=old,
                    diagnostic_augmented=new, diagnostic_not_applied_to_current_training=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        observations = list(pool.map(replay, jobs))
    summaries = {}
    for arm in sealed:
        rows = [r for r in observations if r['arm'] == arm]
        old = [r['applied_original_reward'] for r in rows]
        new = [r['diagnostic_augmented']['reward'] for r in rows]
        summaries[arm] = dict(
            updates=173, inherited_updates=138, new_single_update_production_updates=35,
            supervised_target_updates=sum(r['complete_public_target_used'] for r in sealed[arm]),
            varying_reward_groups_with_applied_policy_scale=sum(r['policy_scale']==1 and len(set(r['global_rewards']))>1 for r in sealed[arm]),
            new_varying_reward_groups=sum(r['policy_scale']==1 and len(set(r['global_rewards']))>1 for r in sealed[arm][138:]), samples=len(rows), original_positive=sum(v > 0 for v in old),
            original_full=sum(v == 1 for v in old), original_mean=sum(old)/len(old),
            augmented_positive=sum(v > 0 for v in new), augmented_full=sum(v == 1 for v in new),
            augmented_mean=sum(new)/len(new),
            old_full_rejected_by_new_observations=sum(a == 1 and b < 1 for a, b in zip(old, new)),
            samples_with_new_probe_coverage=sum(r['diagnostic_augmented'].get('new_probe_count', 0) > 0 for r in rows),
            original_train_rewards_and_augmented_diagnostics_not_held_correctness=True)
    with (REPORTS / 'reexecutions.jsonl').open('x') as f:
        for row in observations:
            f.write(json.dumps(row)+'\n')
    files = [Path(__file__), run/'plan.json',
             ROOT/'src/slm_lab/bounded_code_probes_v1.py',
             ROOT/'src/slm_lab/bounded_code_reward_catalog_v1.py',
             ROOT/'reports/bounded-code-probes-v1/admission.json',
             *[REPORTS/(arm+'-sealed-prefix.jsonl') for arm in sealed],
             REPORTS/'reexecutions.jsonl']
    atomic_json(REPORTS/'complete.json', dict(
        at=datetime.now(timezone.utc).isoformat(), summaries=summaries,
        actual_own_responses_independently_reexecuted=len(observations),
        all_applied_original_rewards_reproduced=True, new_rewards_not_applied_to_live_training=True,
        human_preference_labels=0, external_model_teacher=False, new_neural_updates=0,
        source_files_sha256={str(p): file_hash(p) for p in files},
        finite_reference_observations_not_formal_domain_or_general_correctness=True,
        prequential_training_samples_not_final_checkpoint_capability=True,held_quality_not_measured=True, candidate_promoted=False, goal_achieved=False))
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
