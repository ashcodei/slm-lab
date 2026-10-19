import copy
import json

import pytest

from slm_lab.corpus import file_hash
from slm_lab.dense_human_preference_evaluation_v1 import evaluate
from slm_lab.held_human_shards_v1 import evaluate_shard, reconcile
from test_dense_human_preference_evaluation_v1 import held
from test_own_dense_actor_v1 import actor


def profile():
    return dict(checkpoint_sha256='a'*64, run_plan_sha256='b'*64,
        human_manifest_sha256='c'*64, frozen_dense_sha256='d'*64,
        inference_device_type='cpu', inference_precision='float32')


def test_all_original_held_pair_neural_scores_and_length_controls_equal_serial(tmp_path):
    model, data = actor(), held()
    serial = evaluate(model, data, tmp_path/'serial.json')
    paths = [tmp_path/f'shard-{rank}.json' for rank in range(2)]
    for rank, path in enumerate(paths):
        evaluate_shard(model, data, path, rank=rank, world=2, provenance=profile())
    merged = reconcile(data, paths, tmp_path/'merged.json')
    assert merged['results'] == serial['results']
    assert merged['overall'] == serial['overall']
    assert merged['by_source'] == serial['by_source']
    assert merged['overall']['pairs'] == 2
    assert merged['overall']['implicit_preference_ties'] == 2


def test_held_role_duplicate_or_changed_policy_shards_and_changed_scores_reject(tmp_path):
    model, data = actor(), held()
    data.role = 'train'
    with pytest.raises(ValueError, match='evaluation-only'):
        evaluate_shard(model, data, tmp_path/'invalid.json', rank=0, world=2, provenance=profile())
    data.role = 'valid'
    paths = [tmp_path/f'shard-{rank}.json' for rank in range(2)]
    for rank, path in enumerate(paths):
        evaluate_shard(model, data, path, rank=rank, world=2, provenance=profile())
    with pytest.raises(ValueError):
        reconcile(data, [paths[0], paths[0]], tmp_path/'duplicate.json')
    original = json.loads(paths[1].read_text())
    changed = copy.deepcopy(original)
    changed['profile']['checkpoint_sha256'] = 'e'*64
    paths[1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='policy'):
        reconcile(data, paths, tmp_path/'different-policy.json')
    changed = copy.deepcopy(original)
    changed['results'][0]['implicit_preference_credit'] = 1.
    journal = paths[1].with_suffix('.jsonl')
    journal.write_text(''.join(json.dumps(r)+'\n' for r in changed['results']))
    changed['raw_journal_sha256'] = file_hash(journal)
    paths[1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='score'):
        reconcile(data, paths, tmp_path/'changed-score.json')
