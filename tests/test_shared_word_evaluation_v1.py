import copy
import json

import numpy as np
import pytest
import torch

from slm_lab.shared_word_evaluation_v1 import ctc_records,evaluate,literal_scores,summarize,trim_response,wrong_permutation
from test_shared_word_stage_v1 import model,synthetic_words


def data():
    d=synthetic_words();d.role='valid';d.purpose='evaluation';cursor=0;sequences=[]
    for i,row in enumerate(d.rows):
        n=row['native']['length'];seq=[1,8,9]+[4+i]*(n-4)+[2];sequences.append(seq)
        row.update(id=str(i),group_id='source-photo-'+str(i),source_RGB_pixels_sha256=str(i)*64,
                   stored_image_sha256=str(i+1)*64,byte_family='a' if i<2 else 'b')
        row['native']['offset']=cursor;cursor+=n
    d.tokens=np.array(sum(sequences,[]),dtype='<u2');old_batch=d.batch
    def batch(positions,device='cpu'):
        result=old_batch(positions,device)
        for i,p in enumerate(positions):
            seq=sequences[p];result['input_ids'][i,:len(seq)]=torch.tensor(seq,device=device)
            result['labels'][i,3:len(seq)]=torch.tensor(seq[3:],device=device)
        return result
    d.batch=batch
    return d


def test_wrong_donors_are_reproducible_one_to_one_with_same_family_and_different_photos_pixels_and_words():
    d=data();mapping=wrong_permutation(d.rows)
    assert mapping==wrong_permutation(d.rows) and sorted(mapping)==list(range(len(d)))
    for i,j in enumerate(mapping):
        a,b=d.rows[i],d.rows[j]
        assert a['byte_family']==b['byte_family'] and a['group_id']!=b['group_id']
        assert a['response']!=b['response'] and a['source_RGB_pixels_sha256']!=b['source_RGB_pixels_sha256']
    broken=copy.deepcopy(d.rows)
    for row in broken:row['group_id']='same-original-photo'
    with pytest.raises(ValueError,match='derangement'):wrong_permutation(broken)


def test_native_UTF8_CTC_repeated_bytes_and_literal_unicode_scoring_preserve_delivered_outputs():
    path=[97,256,97,195,169];logits=torch.full((1,5,257),-20.)
    for i,value in enumerate(path):logits[0,i,value]=20.
    row=ctc_records(logits,['aaé'])[0]
    assert row['path']==path and row['generated_UTF8_bytes']==[97,97,195,169]
    assert row['generated']=='aaé' and row['valid_UTF8'] and row['exact'] and row['byte_edits']==0
    assert not literal_scores('é','e\u0301')['exact'] and not literal_scores('a','a ')['exact']
    assert trim_response([5,2,2],3)==([5,2],'EOS') and trim_response([5,6,7],3)==([5,6,7],'length')
    with pytest.raises(ValueError,match='EOS padding'):trim_response([5,2,7],3)
    with pytest.raises(ValueError,match='exact declared'):trim_response([5],3)
    with pytest.raises(ValueError,match='fit'):ctc_records(torch.zeros(1,2,257),['aaa'])


def test_actual_tiny_shared_model_evaluates_every_held_target_and_all_three_controls_without_gradients(tmp_path):
    m=model();d=data();before=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in m.parameters()]
    class Tokenizer:
        def decode(self,ids,skip_special_tokens=True):return ' '.join(str(v) for v in ids if v not in (0,1,2))
    output=tmp_path/'all-held.json'
    report=evaluate(m,d,Tokenizer(),output,batch_size=2,generation_budget=3)
    assert report['overall']['examples']==len(d) and [r['source_position'] for r in report['results']]==list(range(len(d)))
    assert len(output.with_suffix('.jsonl').read_text().splitlines())==len(d)
    assert report['overall']==summarize(report['results']) and json.loads(output.read_text())==report
    for p,row in enumerate(report['results']):
        assert row['native_prompt_ids']==[1,8,9] and row['expected']==d.rows[p]['response']
        for condition,record in row['conditions'].items():
            target=[4+p]*(d.rows[p]['native']['length']-4)+[2]
            assert record['native']['target_ids']==target
            assert len(record['native']['per_target_NLL'])==len(target)
            assert record['CTC']['reference_bytes']==len(row['expected'].encode('utf-8'))
    assert before==[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in m.parameters()]
    assert all(p.grad is None for p in m.parameters())
    from slm_lab.shared_word_evidence_v1 import readback_held
    audited=readback_held(d,Tokenizer(),output,3,ctc_frames=40)
    assert audited['all_raw_paths_tokens_literal_scores_and_original_source_roles_reconstructed']
    bad=copy.deepcopy(report);bad['results'][1]['conditions']['proper']['native']['character_edits']+=1
    output.write_text(json.dumps(bad))
    output.with_suffix('.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in bad['results']))
    with pytest.raises(ValueError,match='edit score'):readback_held(d,Tokenizer(),output,3,ctc_frames=40)
    d.role='train'
    with pytest.raises(ValueError,match='held'):evaluate(m,d,Tokenizer(),tmp_path/'bad.json')
