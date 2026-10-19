import copy
import json

import pytest

from slm_lab.dense_word_evaluation_v1 import evaluate
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.shared_word_evidence_v1 import readback_held
from test_shared_word_evaluation_v1 import data
from test_own_dense_growth_v1 import model as make_model
from test_own_aligned_assistant_v1 import original_signature


class Tokenizer:
    def decode(self,ids,skip_special_tokens=True):return ' '.join(str(v) for v in ids if v not in (0,1,2))


@pytest.mark.parametrize('policy',[False,True])
def test_every_held_target_and_raw_CTC_native_control_scores_reconcile_for_dense_and_actor(tmp_path,policy):
    base,_=make_model();model=OwnDenseActor(base,rank=2).eval() if policy else base
    held=data();before=original_signature(model);path=tmp_path/'held.json'
    report=evaluate(model,held,Tokenizer(),path,batch_size=2,generation_budget=3)
    assert report['actual_counts']==model.counts() and report['overall']['examples']==len(held)
    assert [r['source_position'] for r in report['results']]==list(range(len(held)))
    assert all(set(r['conditions'])=={'proper','blank','wrong'} for r in report['results'])
    audited=readback_held(held,Tokenizer(),path,3,ctc_frames=40)
    assert audited['all_raw_paths_tokens_literal_scores_and_original_source_roles_reconstructed']
    assert before==original_signature(model) and all(p.grad is None for p in model.parameters())


def test_train_roles_changed_delivery_scores_and_overwrite_reject(tmp_path):
    model,_=make_model();held=data();held.role='train'
    with pytest.raises(ValueError,match='held'):evaluate(model,held,Tokenizer(),tmp_path/'bad.json')
    assert not list(tmp_path.iterdir());held.role='valid';path=tmp_path/'held.json'
    original=evaluate(model,held,Tokenizer(),path,batch_size=2,generation_budget=3)
    with pytest.raises(ValueError,match='Preserve'):evaluate(model,held,Tokenizer(),path)
    bad=copy.deepcopy(original);bad['results'][0]['conditions']['proper']['CTC']['generated_UTF8_bytes'].append(7)
    path.write_text(json.dumps(bad));path.with_suffix('.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in bad['results']))
    with pytest.raises(ValueError,match='collapse'):readback_held(held,Tokenizer(),path,3,ctc_frames=40)
