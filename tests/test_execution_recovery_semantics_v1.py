import copy
import json
from pathlib import Path

from slm_lab.execution_recovery_semantics_v1 import same_executed_update

ROOT = Path(__file__).resolve().parents[1]


def test_real_positive_training_record_replay_preserves_every_outcome_but_elapsed_wall_time():
    path = ROOT/'reports/execution-prefix96-augmented-v1/execution_hybrid-sealed-prefix.jsonl'
    rows = [json.loads(line) for line in path.open()]
    original = next(r for r in rows if 1. in r['global_rewards'])
    changed = copy.deepcopy(original)
    index = next(i for i,r in enumerate(changed['trajectories']) if r['actual_execution_reward']['reward'] == 1.)
    execution = changed['trajectories'][index]['actual_execution_reward']['raw_candidate_execution']['execution']
    execution['seconds'] += .125
    assert changed != original and same_executed_update(original, changed)
    # Actual policy behavior, reward, execution status and observable outputs
    # remain part of the comparison, including a program printing "seconds".
    for mutate in [lambda r:r['global_rewards'].__setitem__(index, .5),
        lambda r:r['trajectories'][index]['response_ids'].__setitem__(0, 3),
        lambda r:r['trajectories'][index]['actual_cached_behavior_logps'].__setitem__(0, -.1),
        lambda r:r['trajectories'][index]['actual_execution_reward']['raw_candidate_execution']['execution'].update(returncode=1),
        lambda r:r['trajectories'][index]['actual_execution_reward']['raw_candidate_execution']['raw_results'][1].update(stdout='seconds\n')]:
        bad = copy.deepcopy(changed)
        mutate(bad)
        assert not same_executed_update(original, bad)
