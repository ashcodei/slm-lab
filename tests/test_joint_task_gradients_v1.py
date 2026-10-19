"""Actual shared-model tasks, full-source code, independent loss oracles.

Image/word fixtures are synthetic controls. Their gradients and disposable
updates do not establish real visual/coding capability or full-GPU capacity.
"""
import copy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.deterministic_positions_v1 import install
from slm_lab.joint_code_task_gradients_v1 import gradients as code_gradients
from slm_lab.joint_visual_task_gradients_v1 import native_gradients, word_gradients, photo_gradients
from slm_lab.mixed_coding_SFT_data_v1 import MixedCodingSFTData, FAMILIES
from slm_lab.model import ModelConfig
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone
from slm_lab.own_actor_vision_alignment_v1 import pair_positives, symmetric_pair_loss
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.shared_joint_actor_vision_v1 import SharedJointActorVisionV1
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest

ROOT=Path(__file__).resolve().parents[1]


def view():
    torch.set_num_threads(1)
    config=ModelConfig(vocab_size=32768,hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=4096,vision_hidden_size=16,vision_intermediate_size=32,vision_layers=1,vision_heads=4,image_size=32,patch_size=16,pool_size=2)
    base=install(OwnDenseAssistant(new_backbone(config,991901),reward_hidden=8,word_temporal=4,pointing_rank=8,
        router_hidden=8,detector_hidden=4,word_expansion=1)).eval()
    return SharedJointActorVisionV1(OwnDenseActor(base,rank=2,seed=991902)).eval()


def fixture(count=3):
    rows=[]
    for i in range(count):rows.append([1,5,6,14+i,24+i,2])
    ids=torch.tensor(rows);labels=ids.clone();labels[:,:3]=-100
    pixels=torch.linspace(-1.,1.,count*3*32*32).reshape(count,3,32,32)
    batch=dict(input_ids=ids,labels=labels,attention_mask=torch.ones_like(ids,dtype=torch.bool),pixel_values=pixels)
    source=dict(source_role='train',purpose='optimization',source_sha256='a'*64,positions=[17+3*i for i in range(count)],
        context_lengths=[6]*count,response_EOS_targets=[3]*count,image_size=32,negative_context_lengths=[0]*count,negative_response_EOS_targets=[0]*count)
    return batch,source


def word_fixture():
    batch,source=fixture(2);texts=['a','aa']
    word=dict(word_pixels=torch.linspace(-1.,1.,2*3*32*64).reshape(2,3,32,64),texts=texts)
    source.update(word_height=32,word_width=64,UTF8_bytes=[1,2],UTF8_sha256=[hashlib.sha256(t.encode()).hexdigest() for t in texts],CTC_frames=4)
    return batch,word,source


def photo_fixture():
    batch,source=fixture(9);batch['pixel_values'][1]=batch['pixel_values'][0]
    batch['input_ids'][2,3:]=batch['input_ids'][1,3:];batch['labels'][2,3:]=batch['labels'][1,3:]
    ids=torch.cat([torch.ones(9,1,dtype=torch.long),batch['labels'][:,3:]],1)
    matching=dict(text_ids=ids,text_mask=torch.ones_like(ids,dtype=torch.bool))
    source.update(pixel_keys=['same','same','other']+['image'+str(i) for i in range(3,9)],
        captions=['cat','animal','animal']+['caption'+str(i) for i in range(3,9)])
    return batch,matching,source


def correction_fixture():
    batch,source=fixture(3)
    negative=dict(input_ids=torch.tensor([[1,5,6,34,35,36,37,2]]),labels=torch.tensor([[-100,-100,-100,34,35,36,37,2]]),
        attention_mask=torch.ones(1,8,dtype=torch.bool),pixel_values=batch['pixel_values'][1:2].clone())
    source['negative_context_lengths'][1]=8;source['negative_response_EOS_targets'][1]=5
    return batch,{1:negative},source


def derivative(current,loss):
    named=current.named_selected();values=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True)
    return {n:g for (n,_),g in zip(named,values)}


def compare(actual,expected,tolerance=5e-6):
    for name in actual:
        if expected[name] is None:assert actual[name] is None,name
        else:assert actual[name] is not None and float((actual[name]-expected[name]).abs().max())<tolerance,name


@pytest.fixture(scope='module')
def code():return MixedCodingSFTData(ROOT)


def code_positions(data):
    return [min((p for p in data.slots if data.information(p)['family']==family),key=lambda p:data.information(p)['complete_context_tokens']) for family in FAMILIES]


@pytest.mark.parametrize('profile',['plain_tokens','balanced_sources'])
def test_actual_original_whole_code_gradients_and_source_weights_match_independent_scalar_oracle(code,profile,tmp_path):
    current=view();oracle=copy.deepcopy(current);positions=code_positions(code)
    before=tensor_digest(current.named_parameters()),value_digest(capture_rng('cpu'))
    actual,record=code_gradients(current,code,positions,profile=profile,maximum_examples=2)
    denominator=sum(code.information(p)['response_EOS_targets']*code.weight(profile,p) for p in positions)
    loss=0.
    for p in positions:
        b=code.batch([p]);labels=b['labels'][:,1:];active=labels!=-100
        out=oracle.text(**{k:v for k,v in b.items() if k!='labels'})['logits'][:,:-1]
        loss=loss+F.cross_entropy(out.float()[active],labels[active],reduction='sum')*code.weight(profile,p)/denominator
    compare(actual,derivative(oracle,loss))
    assert record['original_global_weighted_targets']==pytest.approx(denominator)
    assert record['local_response_EOS_targets']==sum(code.information(p)['response_EOS_targets'] for p in positions)
    assert record['synthetic_tests_used_as_rewards'] is False
    assert before==(tensor_digest(current.named_parameters()),value_digest(capture_rng('cpu')))
    assert all(p.grad is None for p in current.parameters()) and not torch.cuda.is_initialized()
    (tmp_path/'actual-real-code-source-gradient.json').write_text(json.dumps(dict(profile=profile,positions=positions,
        original_family_names=[code.information(p)['family'] for p in positions],actual_small_model_counts=current.actor.counts(),
        source_training_role_and_whole_native_targets_checked=True,independent_gradient_oracle_passed=True,
        full_size_GPUcapacity_or_quality_gain=False),indent=2))


def test_corrective_native_CE_and_every_original_negative_EOS_match_independent_length_normalized_ranking():
    current=view();oracle=copy.deepcopy(current);batch,negatives,source=correction_fixture()
    actual,record=native_gradients(current,batch,source,negatives=negatives,maximum_examples=2,correction_weight=.2)
    out=oracle.native_image(**{k:v for k,v in batch.items() if k!='labels'})['logits'][:,:-1]
    labels=batch['labels'][:,1:];mask=labels!=-100;loss=F.cross_entropy(out.float()[mask],labels[mask],reduction='sum')/9
    a=F.cross_entropy(out[1].float()[mask[1]],labels[1][mask[1]],reduction='sum')/3
    n=negatives[1];other=oracle.native_image(**{k:v for k,v in n.items() if k!='labels'})['logits'][:,:-1]
    target=n['labels'][:,1:];keep=target!=-100;b=F.cross_entropy(other.float()[keep],target[keep],reduction='sum')/5
    loss=loss+.2*F.softplus(a-b)
    compare(actual,derivative(oracle,loss))
    assert record['local_response_EOS_targets']==9 and record['local_negative_response_EOS_targets']==5
    assert any(v is not None and bool(v.abs().sum()>0) for n,v in actual.items() if n.startswith('actor.'))
    assert all(p.grad is None for p in current.parameters())


def test_complete_unequal_UTF8_CTC_and_native_word_CE_match_independent_per_word_objective():
    current=view();oracle=copy.deepcopy(current);batch,word,source=word_fixture()
    actual,record=word_gradients(current,batch,word,source,maximum_examples=1)
    ctc=0.
    for i,text in enumerate(word['texts']):
        logits=oracle.word_logits(word['word_pixels'][i:i+1]).float().log_softmax(-1).transpose(0,1)
        raw=F.ctc_loss(logits,torch.tensor(list(text.encode())),torch.tensor([4]),torch.tensor([len(text.encode())]),blank=256,reduction='sum')
        ctc=ctc+raw/len(text.encode())/2
    output=oracle.native_image(**{k:v for k,v in batch.items() if k!='labels'})['logits'][:,:-1]
    target=batch['labels'][:,1:];keep=target!=-100
    loss=ctc+F.cross_entropy(output.float()[keep],target[keep],reduction='sum')/6
    compare(actual,derivative(oracle,loss))
    assert record['local_UTF8_bytes']==3 and record['local_response_EOS_targets']==6
    assert any(v is not None and bool(v.abs().sum()>0) for n,v in actual.items() if n.startswith('actor.'))
    assert 'word.height.bias' not in actual and all(p.grad is None for p in current.parameters())


@pytest.mark.parametrize('bf16',[False,True])
@pytest.mark.parametrize('outer',[False,True])
def test_cached_global_photo_matches_independent_attached_chunk_graphs_with_actor_gradients(bf16,outer):
    current=view();oracle=copy.deepcopy(current);batch,matching,source=photo_fixture()
    amp=(lambda:torch.autocast('cpu',dtype=torch.bfloat16)) if bf16 else None
    outer_context=(lambda:torch.autocast('cpu',dtype=torch.bfloat16)) if bf16 and outer else nullcontext
    with outer_context():
        actual,record=photo_gradients(current,batch,matching,source,maximum_examples=2,amp=amp,temperature=.7)
    reference_amp=(lambda:torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=False)) if bf16 else nullcontext
    image_parts=[];text_parts=[];loss=0.
    for start in range(0,9,2):
        slots=list(range(start,min(start+2,9)));native={k:v[slots] for k,v in batch.items()}
        # Independent encoder graphs must not share a cached BF16 weight
        # cast between the detached text anchor and caption supervision.
        with reference_amp():
            images,texts=oracle.alignment.features(matching['text_ids'][slots],matching['text_mask'][slots],native['pixel_values'])
            ce=oracle.native_image(**native)['loss']
        image_parts.append(images);text_parts.append(texts);loss=loss+ce*len(slots)/9
    with torch.autocast('cpu',enabled=False):
        pairs=symmetric_pair_loss(torch.cat(image_parts),torch.cat(text_parts),pair_positives(source['pixel_keys'],source['captions']),temperature=.7)
        expected=derivative(oracle,loss+.1*pairs)
    compare(actual,expected)
    assert record['actual_cache_encoder_calls']==record['actual_replay_encoder_calls']==5
    assert record['replayed_source_features_bitexact'] and record['complete_global_matching']['actual_global_positive_cells']>9
    assert any(v is not None and bool(v.abs().sum()>0) for n,v in actual.items() if n.startswith('actor.'))
    assert all(p.grad is None for p in current.parameters()) and not torch.cuda.is_initialized()


@pytest.mark.parametrize('case',['held','clipping','missing_negative','changed_prompt','changed_word','changed_caption'])
def test_invalid_complete_source_rejects_before_model_or_RNG_mutation(case):
    current=view();batch,negative,source=correction_fixture();kwargs={};kind='native'
    if case=='held':source['source_role']='valid'
    elif case=='clipping':kwargs['maximum_padded_tokens']=8
    elif case=='missing_negative':negative={}
    elif case=='changed_prompt':negative[1]['input_ids'][0,1]=9
    elif case=='changed_word':
        batch,word,source=word_fixture();word['texts'][1]='ab';kind='word'
    else:
        batch,matching,source=photo_fixture();matching['text_ids'][0,1]+=1;kind='photo'
    before=tensor_digest(current.named_parameters()),value_digest(capture_rng('cpu'))
    with pytest.raises(ValueError):
        if kind=='native':native_gradients(current,batch,source,negatives=negative,**kwargs)
        elif kind=='word':word_gradients(current,batch,word,source)
        else:photo_gradients(current,batch,matching,source)
    assert before==(tensor_digest(current.named_parameters()),value_digest(capture_rng('cpu')))
    assert all(p.grad is None for p in current.parameters())
