import copy
import json
import math

import pytest
import torch

from slm_lab.dense_human_preference_evaluation_v1 import evaluate,readback
from test_dense_human_preference_v1 import pairs
from test_own_dense_actor_v1 import actor


def held():
    data=pairs();data.role='valid';data.purpose='evaluation'
    for row in data.rows:row.update(split='valid',original_split='test',source_blob='helpful-base-test.jsonl.gz')
    return data


def test_all_complete_held_pairs_neural_scores_ties_and_length_controls_reconstruct_without_gradients(tmp_path):
    model=actor();data=held();path=tmp_path/'held.json';report=evaluate(model,data,path)
    assert report['overall']['pairs']==2 and report['overall']['implicit_preference_accuracy']==.5
    assert report['overall']['implicit_preference_ties']==2
    assert report['overall']['pair_mean_DPO']==pytest.approx(math.log(2))
    assert report['overall']['shorter_response_prior_accuracy']==.5
    assert report['overall']['complete_response_EOS_targets']==11
    assert readback(data,path)['overall']==report['overall']
    assert all(p.grad is None for p in model.parameters())
    with torch.no_grad():
        for row in report['results']:
            for field in ('chosen','rejected'):
                b=data.batch([row['original_held_position']],field,'cpu');labels=b['labels'][:,1:];mask=labels!=-100
                logp=model(b['input_ids'])['logits'][:,:-1].float().log_softmax(-1)[mask].gather(1,labels[mask,None]).sum()
                assert row['policy_'+field+'_logp']==pytest.approx(float(logp),rel=2e-6)


def test_training_roles_overwrite_changed_targets_scores_and_missing_pairs_are_rejected(tmp_path):
    model=actor();data=held();data.role='train'
    with pytest.raises(ValueError,match='evaluation-only'):evaluate(model,data,tmp_path/'bad.json')
    assert not list(tmp_path.iterdir());data.role='valid'
    path=tmp_path/'held.json';original=evaluate(model,data,path)
    with pytest.raises(ValueError,match='Preserve'):evaluate(model,data,path)
    for mutate in [lambda r:r['results'][0].update(chosen_target_ids_sha256='0'*64),
                   lambda r:r['results'][0].update(implicit_preference_credit=1.),lambda r:r['results'].pop()]:
        bad=copy.deepcopy(original);mutate(bad);path.write_text(json.dumps(bad))
        path.with_suffix('.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in bad['results']))
        with pytest.raises(ValueError):readback(data,path)
