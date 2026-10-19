import copy
import json

import pytest
import torch

from slm_lab.corpus import file_hash
from slm_lab.point_grounding_evaluation_v1 import state_digest
from slm_lab.whole_human_reward_inference_v1 import attach, expected_sources, export, frozen_general_reward
from slm_lab.whole_human_reward_v1 import FEATURE_ROUTE, RewardHeadStage
from test_whole_human_reward_v1 import setup, SOURCES


def test_actual_completed_reward_attachment_preserves_all_other_weights_and_rejects_bad_provenance(tmp_path):
    model, data, shards, features, options, optimizer = setup()
    cold = copy.deepcopy(model)
    stage = RewardHeadStage(model, optimizer, features, options, SOURCES)
    for _ in range(options['steps']): stage.update()
    plan = dict(training_options={k:v for k,v in options.items() if k != 'learning_rate'},
        arms=dict(actual=dict(learning_rate=options['learning_rate'])), initial_model_plan=dict(actual_counts=model.counts()),
        source_population=5, held_source_population=5, source_manifest_sha256='a'*64,
        source_admission_sha256='b'*64, frozen_encoder_sha256=state_digest(model.original.language.base))
    plan_path = tmp_path/'plan.json'; plan_path.write_text(json.dumps(plan)); plan_sha = file_hash(plan_path)
    admission = dict(feature_route=FEATURE_ROUTE, train_only_head_optimization_admitted=True,
        complete_original_train_pairs=5, complete_original_held_pairs=5,
        feature_sources={k:v for k,v in expected_sources(plan, plan_sha, 'd'*64).items() if k != 'feature_admission_sha256'})
    admission_path = tmp_path/'admission.json'; admission_path.write_text(json.dumps(admission)); admission_sha = file_hash(admission_path)
    checkpoint = tmp_path/'head.pt'; torch.save(export(model, plan, plan_sha, admission_sha, 'actual', stage.completed), checkpoint)
    fixed = [(p, p.detach().clone(), p._version, p.data_ptr()) for n,p in cold.named_parameters() if '.reward.' not in n]
    metadata = attach(cold, checkpoint, plan_path, admission_path, 'actual',
        expected_checkpoint_sha256=file_hash(checkpoint), expected_plan_sha256=plan_sha,
        expected_feature_admission_sha256=admission_sha)
    assert metadata['additional_backbones'] == 0 and metadata['changed_parameter_tensors'] == 3
    ids = torch.tensor([data.sequence(0, 'chosen')[0]])
    assert torch.equal(frozen_general_reward(model, ids), frozen_general_reward(cold, ids))
    assert all(torch.equal(p, t) and p._version == version and p.data_ptr() == pointer for p,t,version,pointer in fixed)
    for kind in ('route', 'prefix', 'nonfinite'):
        bad = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if kind == 'route': bad['feature_route'] = 'aligned'
        elif kind == 'prefix': bad['actual_human_reward_updates'] -= 1
        else: next(iter(bad['reward'].values())).fill_(float('nan'))
        path = tmp_path/(kind+'.pt'); torch.save(bad, path)
        before = [(p.detach().clone(), p._version) for p in cold.parameters()]
        with pytest.raises(ValueError):
            attach(cold, path, plan_path, admission_path, 'actual', expected_checkpoint_sha256=file_hash(path),
                expected_plan_sha256=plan_sha, expected_feature_admission_sha256=admission_sha)
        assert all(torch.equal(p, t) and p._version == version for p,(t,version) in zip(cold.parameters(), before))
    wrong = copy.deepcopy(cold)
    with torch.no_grad(): next(wrong.original.language.base.parameters()).add_(.1)
    with pytest.raises(ValueError, match='same fixed original general encoder'):
        attach(wrong, checkpoint, plan_path, admission_path, 'actual', expected_checkpoint_sha256=file_hash(checkpoint),
            expected_plan_sha256=plan_sha, expected_feature_admission_sha256=admission_sha)
