"""Own-small fault reproduction; never a production model or GPU process."""
import os
import json
from pathlib import Path
import sys


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19:
        raise ValueError('CUDA-hidden idle own-small probe required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
    import torch
    from slm_lab.batched_own_native_rollouts_v1 import sample
    from test_frozen_linear_cast_cache_v2 import small
    actor = small()
    original = actor.forward
    def nonfinite_full_forward(*args, **kwargs):
        result = original(*args, **kwargs)
        return dict(result, logits=torch.full_like(result['logits'], float('nan')))
    actor.forward = nonfinite_full_forward
    report = sample(actor, [1, 5, 7], seeds=[19, 31, 47, 79], horizon=3,
        purpose='CPU-mechanics', maximum_cache_bytes=8 * 1024**2)
    bad = any(not __import__('math').isfinite(p) for t in report['trajectories'] for p in t['actual_full_forward_behavior_logps'])
    value = dict(complete=True, deliberately_injected_nonfinite_full_forward=True,
        actual_own_small_parameters=actor.counts()['actual_unique'],
        actual_generated_actions_preserved=[t['actual_generated_token_ids'] for t in report['trajectories']],
        actual_nonfinite_selected_full_probabilities=bad,
        original_v1_reported_behavior_rescore_passed=report['behavior_rescore_passed'],
        original_v1_reported_max_absolute_difference=report['cached_full_max_absolute_difference'],
        actual_v1_gap_observed=bad and report['behavior_rescore_passed'],
        new_production_updates=0, GPUcontexts=0, no_existing_worker_or_weight_modified=True)
    path = Path('reports/batched-rollout-v1-nonfinite-rescore-gap.json')
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps(value))
