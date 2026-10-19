import copy
import json

import pytest
import torch
from torch.nn import functional as F

from slm_lab.dense_foundation_held_v1 import DenseHeldSources, evaluate, readback, target_digest
from test_own_dense_growth_v1 import model as make_model


def held():
    d = object.__new__(DenseHeldSources)
    d.populations = dict(education=2, variety=1, human=1)
    sequences = dict(education=[([1,4,5,6,7],1),([1,7,2],1)],
                     variety=[([1,5,6,7,8,2],4)], human=[([1,8,9,2],2)])
    def batch(name, position, device='cpu'):
        ids, start = sequences[name][position]
        return dict(input_ids=torch.tensor([ids],device=device),
            labels=torch.tensor([[-100]*start+ids[start:]],device=device),
            attention_mask=torch.ones(1,len(ids),device=device,dtype=torch.bool))
    def info(name, position):
        b=batch(name,position); targets=b['labels'][:,1:]; targets=targets[targets!=-100].tolist()
        return dict(family=name,original_held_position=position,complete_context_tokens=b['input_ids'].numel(),
            actual_next_or_response_EOS_targets=len(targets),target_ids_sha256=target_digest(targets))
    d.batch=batch; d.information=info; return d


def test_every_complete_held_source_target_and_tail_matches_independent_neural_oracle(tmp_path):
    m,_=make_model();m.eval();data=held();path=tmp_path/'held.json';report=evaluate(m,data,path)
    nll=[];correct=0
    with torch.no_grad():
        for name in data.names:
            for position in range(data.populations[name]):
                b=data.batch(name,position);out=m(b['input_ids'])['logits'][:,:-1];labels=b['labels'][:,1:];mask=labels!=-100
                nll.extend(F.cross_entropy(out[mask].float(),labels[mask],reduction='none').tolist())
                correct+=int((out.argmax(-1)[mask]==labels[mask]).sum())
    assert report['overall']['examples']==4 and len(nll)==10
    assert report['overall']['token_mean_NLL']==pytest.approx(sum(nll)/10)
    assert report['overall']['teacher_argmax_accuracy']==correct/10
    assert readback(data,path)['overall']==report['overall']
    assert all(p.grad is None for p in m.parameters())


def test_changed_source_mask_order_journal_and_aggregate_are_rejected(tmp_path):
    m,_=make_model();data=held();path=tmp_path/'held.json';original=evaluate(m,data,path)
    for mutation in [lambda r:r['results'][0].update(target_ids_sha256='a'*64),
                     lambda r:r['results'].reverse(),
                     lambda r:r['overall'].update(token_mean_NLL=0.)]:
        bad=copy.deepcopy(original);mutation(bad);path.write_text(json.dumps(bad))
        path.with_suffix('.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in bad['results']))
        with pytest.raises(ValueError):readback(data,path)
    path.write_text(json.dumps(original));path.with_suffix('.jsonl').write_text('')
    with pytest.raises(ValueError):readback(data,path)


def test_held_role_required_before_forward_and_existing_deliveries_preserved(tmp_path):
    m,_=make_model();data=held();data.role='train'
    with pytest.raises(ValueError,match='evaluation'):evaluate(m,data,tmp_path/'wrong.json')
    assert not list(tmp_path.iterdir())
    data.role='valid';path=tmp_path/'held.json';evaluate(m,data,path)
    with pytest.raises(ValueError,match='Preserve'):evaluate(m,data,path)
