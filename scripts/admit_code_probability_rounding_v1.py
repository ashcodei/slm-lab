"""Read back actual FP32/FP64 diagnostic vectors and admit a bounded guard."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/code-probability-rounding-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Numerical readback must hide every GPU')
    if REPORTS.exists():
        raise ValueError('Preserve earlier complete or partial calibration')
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    torch.set_num_threads(1)
    run = ROOT/'reports/execution-probability-diagnostic-v7'
    plan = json.loads((run/'plan.json').read_text())
    complete = json.loads((run/'complete.json').read_text())
    if not complete['disposable_numerical_diagnostic_only'] or len(complete['proofs']) != 4:
        raise ValueError('Actual completed four-GPU full-size numerical diagnostic required')
    FileBindings(plan['source_files_sha256']).check(full=True)
    bindings = {str(Path(__file__)):file_hash(__file__)}
    for path in [run/'plan.json',run/'complete.json',run/'restored.json']:
        bindings[str(path)] = file_hash(path)
    maximum = maximum64 = maximum32reference = 0.
    failed = observations = actions = 0
    config = counts = None
    checked = []
    for rank,proof in enumerate(sorted(complete['proofs'],key=lambda r:r['physical_GPU'])):
        if (proof['physical_GPU'] != rank or proof['stop']['actual_completed'] != 138
                or not proof['diagnostic_step139_not_committed']
                or not proof['old_recorded_prefix_behavior_and_rewards_reproduced']):
            raise ValueError('All original next behavior groups must reproduce through138 before the uncommitted139attempt')
        raw_path = Path(proof['raw_original_FP32_comparisons'])
        checkpoint = Path(proof['actor_Adam_and_RNG_checkpoint_before_rejected_attempt'])
        if (file_hash(raw_path) != proof['raw_comparison_sha256']
                or file_hash(checkpoint) != proof['actor_Adam_and_RNG_checkpoint_sha256']):
            raise ValueError('Every original raw probability and pre-attempt model state byte required')
        raw = torch.load(raw_path,map_location='cpu',weights_only=True,mmap=True)
        state = torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        model = state['private_replica_state']['model']
        if config is None:
            config,counts = model['config'],model['actual_counts']
        if (config != model['config'] or counts != model['actual_counts']
                or counts['actual_unique'] != 999706442):
            raise ValueError('All references must use the actual same full counted student architecture')
        rows = raw['observations']
        for row in rows:
            cached = torch.tensor(row['cached_behavior_logps'],dtype=torch.float32)
            full = torch.tensor(row['full_behavior_logps'],dtype=torch.float32)
            logits = row['actual_full_FP32_logits']
            selected = logits.log_softmax(-1).gather(1,torch.tensor(row['response_ids'])[:,None]).squeeze(1)
            difference = float((cached-full).abs().max())
            accepted = bool(torch.allclose(full,cached,atol=1e-4,rtol=1e-5))
            if (difference != row['max_absolute_action_logp_difference']
                    or accepted != row['original_allclose_passed']
                    or not torch.allclose(selected,full,atol=1e-5,rtol=1e-5)):
                raise ValueError('Every stored full FP32 logit, selected probability and old guard verdict must reconstruct')
            maximum = max(maximum,difference)
            failed += not accepted
            observations += 1
            actions += len(cached)
        worst = max(rows,key=lambda r:r['max_absolute_action_logp_difference'])
        reference = proof['same_student_weight_numerical_reference']
        if (reference['actual_recorded_FP32_cached_action_logps'] != worst['cached_behavior_logps']
                or reference['actual_recorded_FP32_full_action_logps'] != worst['full_behavior_logps']
                or not reference['recorded_cached_FP32_replay_exact']):
            raise ValueError('The same actual stopped weights must reproduce the original cached policy exactly')
        full64 = torch.tensor(reference['FP64_full_action_logps'],dtype=torch.float64)
        cached64 = torch.tensor(reference['FP64_cached_action_logps'],dtype=torch.float64)
        discrepancy64 = float((full64-cached64).abs().max())
        discrepancy32 = max(float((torch.tensor(worst['full_behavior_logps'],dtype=torch.float64)-full64).abs().max()),
                            float((torch.tensor(worst['cached_behavior_logps'],dtype=torch.float64)-cached64).abs().max()))
        if discrepancy64 > 1e-8 or discrepancy32 > 1e-3:
            raise ValueError('FP64 equation reference must agree between routes; FP32 deviations must stay bounded')
        maximum64 = max(maximum64,discrepancy64)
        maximum32reference = max(maximum32reference,discrepancy32)
        checked.append(dict(rank=rank,original_guard_failures=sum(not r['original_allclose_passed'] for r in rows),
            FP64_full_cached_max_difference=discrepancy64,FP32_vs_FP64_max_difference=discrepancy32))
        for path in [raw_path,checkpoint,run/f'diagnostic-{rank}.json']:
            bindings[str(path)] = file_hash(path)
    bound = math.ceil(max(2e-4,4*maximum)*1e6)/1e6
    if bound > 1e-3:
        raise ValueError('Observed rounding exceeds the hard0.001log-probability guard budget; investigate further')
    REPORTS.mkdir()
    admission = dict(format='own-actual-FP32-cache-full-rounding-guard-v1',at=datetime.now(timezone.utc).isoformat(),
        actual_counts=counts,config=config,observed_own_trajectories=observations,observed_own_action_tokens=actions,
        original_guard_rejections_reproduced=failed,maximum_observed_FP32_cache_full_logp_difference=maximum,
        maximum_observed_FP64_cache_full_logp_difference=maximum64,maximum_observed_FP32_vs_FP64_logp_difference=maximum32reference,
        admitted_absolute_action_logp_bound=bound,relative_action_probability_error_bound=math.expm1(bound),
        raw_vector_and_original_logit_aggregates_independently_reconstructed=True,
        FP64_numerical_neural_results_read_back_not_independently_reexecuted_here=True,
        full_size_four_L40_diagnostic_used=True,checks=checked,
        guard_is_runtime_fail_closed_not_a_future_error_or_accuracy_certificate=True,
        original_sources_and_original_failed_trial_unchanged=True,
        production_use_requires_fresh_full_profile_GPU_capacity=True,
        new_neural_updates=0,quality_gain_proven=False,source_files_sha256=bindings,goal_achieved=False)
    atomic_json(REPORTS/'admission.json',admission)
    print(json.dumps({k:admission[k] for k in ['observed_own_trajectories','observed_own_action_tokens',
        'original_guard_rejections_reproduced','maximum_observed_FP32_cache_full_logp_difference',
        'maximum_observed_FP64_cache_full_logp_difference','admitted_absolute_action_logp_bound']}),flush=True)


if __name__ == '__main__':
    main()
